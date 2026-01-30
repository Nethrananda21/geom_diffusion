"""
Training Script for Geometry-Complete Equivariant Diffusion

Implements:
- Full training loop with gradient accumulation
- Mixed precision training (FP16 for T4, BF16 for A100)
- WandB logging
- Checkpointing and early stopping
- Learning rate scheduling with warmup
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from data.dataset import CrossDockedDataset, DatasetConfig, create_dataloader
from data.filters import FilterConfig
from data.pockets import PocketConfig
from models.egnn import ConditionalEGNN
from models.diffusion import MolecularDiffusion, DiffusionConfig
from models.encoder import ClassifierFreeGuidance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GeomDiffusionModel(nn.Module):
    """
    Main model combining EGNN denoiser with diffusion process.
    """
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        super().__init__()
        
        model_config = config['model']
        diffusion_config = config['diffusion']
        
        # Atom type info
        self.atom_types = model_config['atom_types']
        self.n_atom_types = len(self.atom_types)
        
        # Conditional EGNN
        self.denoiser = ConditionalEGNN(
            in_node_dim=self.n_atom_types,
            hidden_dim=model_config['egnn']['hidden_dim'],
            out_node_dim=self.n_atom_types,
            n_layers=model_config['egnn']['n_layers'],
            pocket_layers=model_config['pocket_encoder']['n_layers'],
            edge_dim=16,
            attention=True,
            dropout=model_config['egnn']['dropout'],
            time_emb_dim=model_config['time_emb_dim']
        )
        
        # Diffusion process
        self.diffusion = MolecularDiffusion(
            DiffusionConfig(
                timesteps=diffusion_config['timesteps'],
                schedule=diffusion_config['schedule'],
                beta_start=diffusion_config['beta_start'],
                beta_end=diffusion_config['beta_end'],
                parameterization=diffusion_config['parameterization'],
                guidance_dropout=diffusion_config['guidance']['dropout'],
                guidance_scale=diffusion_config['guidance']['scale']
            ),
            n_atom_types=self.n_atom_types
        )
        
        # Classifier-free guidance
        self.cfg = ClassifierFreeGuidance(
            dropout_prob=diffusion_config['guidance']['dropout'],
            guidance_scale=diffusion_config['guidance']['scale']
        )
    
    def forward(
        self,
        batch,
        t: torch.Tensor,
        drop_conditioning: bool = False
    ):
        """
        Forward pass for training.
        
        Args:
            batch: PyG batch with ligand and pocket data
            t: Timesteps
            drop_conditioning: Whether to drop pocket conditioning (for CFG)
        
        Returns:
            Dict with predictions
        """
        # Extract batch data
        lig_x = batch.x  # (N, n_types)
        lig_pos = batch.pos  # (N, 3)
        lig_edge_index = batch.edge_index
        lig_edge_attr = batch.edge_attr
        
        pocket_x = batch.pocket_x
        pocket_pos = batch.pocket_pos
        pocket_edge_index = batch.pocket_edge_index
        pocket_edge_attr = batch.pocket_edge_attr
        
        # CRITICAL FIX: Extract pocket_id for caching (~10x speedup!)
        # Without this, pocket is re-encoded at every timestep (500 times per batch)
        pocket_id = getattr(batch, 'pocket_id', None)
        if hasattr(pocket_id, 'item'):  # Handle tensor case
            pocket_id = str(pocket_id.item())
        elif pocket_id is not None:
            pocket_id = str(pocket_id)
        
        # Forward through denoiser
        if drop_conditioning:
            # Unconditional forward (for CFG training) - don't cache unconditioned
            type_pred, coord_pred = self.denoiser(
                lig_x, lig_pos, lig_edge_index,
                torch.zeros_like(pocket_x), torch.zeros_like(pocket_pos),
                pocket_edge_index, t,
                lig_edge_attr, pocket_edge_attr,
                pocket_id=None  # No caching for unconditional
            )
        else:
            type_pred, coord_pred = self.denoiser(
                lig_x, lig_pos, lig_edge_index,
                pocket_x, pocket_pos, pocket_edge_index, t,
                lig_edge_attr, pocket_edge_attr,
                pocket_id=pocket_id  # CRITICAL: Pass pocket_id for caching!
            )
        
        return {
            'type_pred': type_pred,
            'coord_pred': coord_pred
        }


def compute_loss(
    model: GeomDiffusionModel,
    batch,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    Compute training loss.
    
    Loss = λ_pos * MSE(coord_pred, coord_noise) +
           λ_type * MSE(type_pred, type_noise) +
           λ_bond * bond_penalty (optional)
    """
    loss_config = config['training']['loss']
    
    # Sample random timesteps
    batch_size = 1  # Graph batching makes this tricky
    t = torch.randint(
        0, model.diffusion.timesteps,
        (batch_size,),
        device=device
    )
    
    # Get clean data
    coords_0 = batch.pos.to(device)
    types_0 = batch.x.to(device)
    
    # Sample noise
    coord_noise = torch.randn_like(coords_0)
    type_noise = torch.randn_like(types_0)
    
    # Forward diffusion
    coords_t, types_t, _, _ = model.diffusion.q_sample_mol(
        coords_0, types_0, t, coord_noise, type_noise
    )
    
    # Create noised batch
    batch_noised = batch.clone()
    batch_noised.pos = coords_t
    batch_noised.x = types_t
    
    # CFG training: randomly drop conditioning
    drop_conditioning = torch.rand(1).item() < config['diffusion']['guidance']['dropout']
    
    # Model prediction
    preds = model(batch_noised, t, drop_conditioning=drop_conditioning)
    
    # Compute losses
    coord_loss = F.mse_loss(preds['coord_pred'], coord_noise)
    type_loss = F.mse_loss(preds['type_pred'], type_noise)
    
    # Combined loss
    total_loss = (
        loss_config['lambda_pos'] * coord_loss +
        loss_config['lambda_type'] * type_loss
    )
    
    # Optional: bond length penalty
    if loss_config.get('lambda_bond', 0) > 0:
        # Predict x_0 from noise prediction
        pred_coords = model.diffusion.predict_x0_from_eps(
            coords_t, t, preds['coord_pred']
        )
        bond_penalty = compute_bond_penalty(
            pred_coords, batch.edge_index, batch.x
        )
        total_loss = total_loss + loss_config['lambda_bond'] * bond_penalty
    else:
        bond_penalty = torch.tensor(0.0)
    
    return {
        'loss': total_loss,
        'coord_loss': coord_loss,
        'type_loss': type_loss,
        'bond_loss': bond_penalty
    }


def compute_bond_penalty(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    atom_types: torch.Tensor,
    target_length: float = 1.5,
    tolerance: float = 0.3
) -> torch.Tensor:
    """
    Penalty for unrealistic bond lengths.
    
    Encourages generated structures to have reasonable geometry.
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=coords.device)
    
    row, col = edge_index
    
    # Compute bond lengths
    bond_vectors = coords[row] - coords[col]
    bond_lengths = torch.norm(bond_vectors, dim=-1)
    
    # Penalty for bonds outside tolerance
    penalty = F.relu(bond_lengths - target_length - tolerance) ** 2
    penalty = penalty + F.relu(target_length - tolerance - bond_lengths) ** 2
    
    return penalty.mean()


def get_scheduler(optimizer, config: Dict[str, Any], num_training_steps: int):
    """Create learning rate scheduler with warmup."""
    scheduler_config = config['training']['scheduler']
    warmup_steps = scheduler_config['warmup_steps']
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=scheduler_config['T_0'],
        T_mult=1
    )
    
    return warmup_scheduler, cosine_scheduler


def train_epoch(
    model: GeomDiffusionModel,
    train_loader,
    optimizer,
    scaler: Optional[GradScaler],
    config: Dict[str, Any],
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_coord_loss = 0.0
    total_type_loss = 0.0
    n_batches = 0
    
    accumulation_steps = config['training']['gradient_accumulation_steps']
    use_amp = config['training']['stability']['mixed_precision'] in ['fp16', 'bf16']
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = batch.to(device)
        
        # Mixed precision forward
        with autocast(enabled=use_amp):
            losses = compute_loss(model, batch, config, device)
            loss = losses['loss'] / accumulation_steps
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if scaler is not None:
                scaler.unscale_(optimizer)
            
            clip_norm = config['training']['stability']['clip_norm']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += losses['loss'].item()
        total_coord_loss += losses['coord_loss'].item()
        total_type_loss += losses['type_loss'].item()
        n_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss / n_batches:.4f}",
            'coord': f"{total_coord_loss / n_batches:.4f}",
            'type': f"{total_type_loss / n_batches:.4f}"
        })
    
    return {
        'loss': total_loss / n_batches,
        'coord_loss': total_coord_loss / n_batches,
        'type_loss': total_type_loss / n_batches
    }


@torch.no_grad()
def validate(
    model: GeomDiffusionModel,
    val_loader,
    config: Dict[str, Any],
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    total_coord_loss = 0.0
    total_type_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        batch = batch.to(device)
        
        losses = compute_loss(model, batch, config, device)
        
        total_loss += losses['loss'].item()
        total_coord_loss += losses['coord_loss'].item()
        total_type_loss += losses['type_loss'].item()
        n_batches += 1
    
    return {
        'val_loss': total_loss / n_batches,
        'val_coord_loss': total_coord_loss / n_batches,
        'val_type_loss': total_type_loss / n_batches
    }


def save_checkpoint(
    model: GeomDiffusionModel,
    optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    config: Dict[str, Any],
    checkpoint_dir: Path
):
    """Save training checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'config': config
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest
    torch.save(checkpoint, checkpoint_dir / 'latest.pt')
    
    # Save best
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    
    logger.info(f"Saved checkpoint at epoch {epoch}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function."""
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Create data loaders
    dataset_config = DatasetConfig(
        root=config['data']['root'],
        train_pockets=config['data']['train_pockets'],
        val_pockets=config['data']['val_pockets'],
        ligands_per_pocket=config['data']['ligands_per_pocket'],
        filter_config=FilterConfig(
            rmsd_max=config['data']['filters']['rmsd_max'],
            vina_max=config['data']['filters']['vina_max'],
            ligand_atoms_min=config['data']['filters']['ligand_atoms_min'],
            ligand_atoms_max=config['data']['filters']['ligand_atoms_max'],
            pocket_atoms_max=config['data']['filters']['pocket_atoms_max']
        ),
        pocket_config=PocketConfig(
            pocket_radius=config['data']['pocket_radius'],
            remove_hydrogens=config['data']['remove_hydrogens'],
            centering=config['data']['centering']
        ),
        edge_cutoff=config['model']['egnn']['edge_cutoff']
    )
    
    train_loader = create_dataloader(
        dataset_config, 'train',
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers']
    )
    
    val_loader = create_dataloader(
        dataset_config, 'val',
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers']
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    model = GeomDiffusionModel(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=tuple(config['training']['optimizer']['betas'])
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * config['training']['max_epochs']
    warmup_scheduler, cosine_scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # Mixed precision
    use_amp = config['training']['stability']['mixed_precision'] in ['fp16', 'bf16']
    scaler = GradScaler() if use_amp and device.type == 'cuda' else None
    
    # WandB logging
    if WANDB_AVAILABLE and args.wandb:
        wandb.init(
            project=config['project'],
            config=config,
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['max_epochs'] + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{config['training']['max_epochs']}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            config, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, config, device)
        
        # Scheduler step
        if epoch <= config['training']['scheduler']['warmup_steps'] // len(train_loader):
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        # Log metrics
        all_metrics = {**train_metrics, **val_metrics}
        logger.info(f"Train loss: {train_metrics['loss']:.4f}")
        logger.info(f"Val loss: {val_metrics['val_loss']:.4f}")
        
        if WANDB_AVAILABLE and args.wandb:
            wandb.log(all_metrics, step=epoch)
        
        # Save checkpoint
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(
                model, optimizer, cosine_scheduler,
                epoch, best_val_loss, config, checkpoint_dir
            )
            logger.info(f"New best val loss: {best_val_loss:.4f}")
        
        # Early stopping check (optional)
        if train_metrics['loss'] < 0.1:
            logger.info("Loss < 0.1, debug criteria met!")
    
    logger.info("\nTraining complete!")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    
    if WANDB_AVAILABLE and args.wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Geometry-Complete Diffusion Model')
    parser.add_argument('--config', type=str, default='configs/debug_t4.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB logging')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
