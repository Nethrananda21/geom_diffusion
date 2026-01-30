"""
Frozen Pocket Encoder

Optimization module for encoding protein pockets once per sample,
then reusing the encoding across all diffusion timesteps.

Key Optimization:
- Pocket is static (doesn't change during diffusion)
- Encode once at t=T, cache, reuse at t=T-1, ..., t=0
- Provides ~2x speedup during training and inference
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from torch import Tensor

from .egnn import EGNN, EGNN_Layer


class FrozenPocketEncoder(nn.Module):
    """
    Shallow EGNN encoder for pocket representation.
    
    Design Choices:
    - 2-3 layers sufficient (pocket is conditioning, not main target)
    - No dropout (pocket should be deterministic)
    - Caching enabled for efficiency
    """
    
    def __init__(
        self,
        in_node_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        edge_dim: int = 16,
        attention: bool = True
    ):
        """
        Args:
            in_node_dim: Input pocket node dimension
            hidden_dim: Hidden dimension
            out_dim: Output embedding dimension
            n_layers: Number of EGNN layers (2-3 recommended)
            edge_dim: Edge feature dimension
            attention: Use attention mechanism
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        # Input projection
        self.node_embed = nn.Linear(in_node_dim, hidden_dim)
        
        # EGNN layers
        self.layers = nn.ModuleList([
            EGNN_Layer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                attention=attention,
                normalize=True,
                tanh=False
            )
            for _ in range(n_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        
        # Cache storage
        self._cache: Dict[str, Tensor] = {}
        self._cache_enabled = True
    
    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        cache_key: Optional[str] = None
    ) -> Tensor:
        """
        Encode pocket with optional caching.
        
        Args:
            h: (M, in_dim) pocket node features
            x: (M, 3) pocket coordinates
            edge_index: (2, E) pocket edge indices
            edge_attr: (E, edge_dim) pocket edge features
            cache_key: Optional key for caching (e.g., pocket_id)
        
        Returns:
            (M, out_dim) pocket node embeddings
        """
        # Check cache
        if cache_key is not None and cache_key in self._cache and self._cache_enabled:
            return self._cache[cache_key]
        
        # Embed nodes
        h = self.node_embed(h)
        
        # Process through layers
        for layer, ln in zip(self.layers, self.layer_norms):
            h, x = layer(h, x, edge_index, edge_attr)
            h = ln(h)
        
        # Output projection
        out = self.out_proj(h)
        
        # Cache result
        if cache_key is not None and self._cache_enabled:
            self._cache[cache_key] = out.detach()
        
        return out
    
    def clear_cache(self):
        """Clear all cached embeddings."""
        self._cache.clear()
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
    
    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)


class PocketConditioner(nn.Module):
    """
    Full pocket conditioning module.
    
    Combines:
    1. Frozen pocket encoder (cached embeddings)
    2. Cross-attention from ligand to pocket
    3. Pocket context injection into ligand features
    """
    
    def __init__(
        self,
        pocket_dim: int,
        ligand_dim: int,
        hidden_dim: int,
        n_attention_heads: int = 4,
        dropout: float = 0.1,
        n_encoder_layers: int = 2,
        edge_dim: int = 16
    ):
        """
        Args:
            pocket_dim: Pocket input dimension
            ligand_dim: Ligand feature dimension
            hidden_dim: Hidden dimension
            n_attention_heads: Number of attention heads
            dropout: Dropout rate
            n_encoder_layers: Pocket encoder depth
            edge_dim: Edge feature dimension
        """
        super().__init__()
        
        # Pocket encoder
        self.pocket_encoder = FrozenPocketEncoder(
            in_node_dim=pocket_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            n_layers=n_encoder_layers,
            edge_dim=edge_dim
        )
        
        # Cross-attention: ligand attends to pocket
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Projections
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, ligand_dim)
        
        # Global pocket summary
        self.pocket_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        ligand_h: Tensor,
        pocket_h: Tensor,
        pocket_x: Tensor,
        pocket_edge_index: Tensor,
        pocket_edge_attr: Optional[Tensor] = None,
        pocket_id: Optional[str] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply pocket conditioning to ligand features.
        
        Args:
            ligand_h: (N, ligand_dim) ligand node features
            pocket_h: (M, pocket_dim) pocket node features
            pocket_x: (M, 3) pocket coordinates
            pocket_edge_index: (2, E) pocket edges
            pocket_edge_attr: (E, edge_dim) pocket edge features
            pocket_id: Optional ID for caching
        
        Returns:
            Tuple of:
                - (N, ligand_dim) conditioned ligand features
                - (hidden_dim,) global pocket context
        """
        # Encode pocket (cached)
        pocket_emb = self.pocket_encoder(
            pocket_h, pocket_x, pocket_edge_index, pocket_edge_attr,
            cache_key=pocket_id
        )  # (M, hidden_dim)
        
        # Project ligand to hidden dim
        ligand_emb = self.ligand_proj(ligand_h)  # (N, hidden_dim)
        
        # Cross-attention (ligand attends to pocket)
        # Add batch dimension for attention
        ligand_q = ligand_emb.unsqueeze(0)  # (1, N, hidden_dim)
        pocket_kv = pocket_emb.unsqueeze(0)  # (1, M, hidden_dim)
        
        attn_out, _ = self.cross_attention(ligand_q, pocket_kv, pocket_kv)
        attn_out = attn_out.squeeze(0)  # (N, hidden_dim)
        
        # Residual + norm
        ligand_emb = self.norm1(ligand_emb + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(ligand_emb)
        ligand_emb = self.norm2(ligand_emb + ffn_out)
        
        # Project back to ligand dimension
        ligand_out = self.out_proj(ligand_emb)  # (N, ligand_dim)
        
        # Global pocket context (mean pooling)
        pocket_global = self.pocket_pool(pocket_emb.mean(dim=0))  # (hidden_dim,)
        
        return ligand_out, pocket_global
    
    def clear_pocket_cache(self):
        """Clear pocket encoder cache."""
        self.pocket_encoder.clear_cache()


class ClassifierFreeGuidance:
    """
    Classifier-Free Guidance for conditional generation.
    
    During training:
        - Randomly drop pocket conditioning with probability p (e.g., 10%)
        - This trains model to work both conditionally and unconditionally
    
    During inference:
        - Run both conditional and unconditional forward passes
        - Blend: output = unconditional + scale * (conditional - unconditional)
        - scale > 1 increases pocket adherence
    """
    
    def __init__(
        self,
        dropout_prob: float = 0.1,
        guidance_scale: float = 2.0
    ):
        """
        Args:
            dropout_prob: Probability of dropping conditioning during training
            guidance_scale: Guidance scale at inference (>1 for stronger conditioning)
        """
        self.dropout_prob = dropout_prob
        self.guidance_scale = guidance_scale
    
    def training_mask(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tensor:
        """
        Generate mask for training (1 = use conditioning, 0 = drop).
        
        Returns:
            (batch_size,) boolean mask
        """
        return torch.rand(batch_size, device=device) > self.dropout_prob
    
    def apply_guidance(
        self,
        cond_output: Tensor,
        uncond_output: Tensor
    ) -> Tensor:
        """
        Apply classifier-free guidance at inference.
        
        output = uncond + scale * (cond - uncond)
        
        Args:
            cond_output: Model output with conditioning
            uncond_output: Model output without conditioning
        
        Returns:
            Guided output
        """
        return uncond_output + self.guidance_scale * (cond_output - uncond_output)
