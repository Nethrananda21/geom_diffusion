"""
Molecule Generation Script for Trained Diffusion Model
Generates novel drug-like molecules conditioned on protein pockets
"""

import torch
import numpy as np
import pickle
import argparse
import yaml
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import model components
from train import DiffusionModel


def load_model(checkpoint_path, config_path, device):
    """Load trained model from checkpoint"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    model = DiffusionModel(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    return model, config


def get_beta_schedule(timesteps, schedule='cosine'):
    """Get diffusion beta schedule"""
    if schedule == 'cosine':
        s = 0.008
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.02)
    else:
        return np.linspace(0.0001, 0.02, timesteps)


@torch.no_grad()
def generate_molecules(model, pocket_data, config, device, num_samples=10, num_atoms=25):
    """
    Generate molecules using reverse diffusion process
    
    Args:
        model: Trained diffusion model
        pocket_data: Dict with pocket_coords and pocket_types
        config: Model config
        device: torch device
        num_samples: Number of molecules to generate
        num_atoms: Number of atoms per molecule
    
    Returns:
        List of generated molecule data
    """
    model.eval()
    timesteps = config['diffusion']['timesteps']
    betas = get_beta_schedule(timesteps, config['diffusion'].get('schedule', 'cosine'))
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    
    # Convert to tensors
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    alphas = torch.tensor(alphas, dtype=torch.float32, device=device)
    alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)
    
    # Prepare pocket
    pocket_coords = torch.tensor(pocket_data['pocket_coords'], dtype=torch.float32, device=device)
    pocket_types = torch.tensor(pocket_data['pocket_types'], dtype=torch.float32, device=device)
    
    # Expand for batch
    pocket_coords = pocket_coords.unsqueeze(0).expand(num_samples, -1, -1)
    pocket_types = pocket_types.unsqueeze(0).expand(num_samples, -1, -1)
    
    # Initialize random ligand (pure noise)
    # Start from pocket center with noise
    pocket_center = pocket_coords.mean(dim=1, keepdim=True)
    ligand_coords = pocket_center + torch.randn(num_samples, num_atoms, 3, device=device) * 2.0
    ligand_types = torch.randn(num_samples, num_atoms, 10, device=device)  # 10 atom types
    
    generated = []
    
    # Reverse diffusion
    for t in tqdm(reversed(range(timesteps)), desc="Generating", total=timesteps):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        
        # Build batch for model
        batch = {
            'ligand_coords': ligand_coords,
            'ligand_types': ligand_types,
            'pocket_coords': pocket_coords,
            'pocket_types': pocket_types,
            'ligand_mask': torch.ones(num_samples, num_atoms, device=device),
            'pocket_mask': torch.ones(num_samples, pocket_coords.size(1), device=device),
        }
        
        # Predict noise
        with torch.cuda.amp.autocast(enabled=True):
            preds = model(batch, t_batch, drop_conditioning=False)
        
        type_pred = preds['type_pred']
        coord_pred = preds['coord_pred']
        
        # Get diffusion parameters
        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        beta_t = betas[t]
        
        if t > 0:
            alpha_cumprod_prev = alphas_cumprod[t - 1]
            
            # Posterior mean coefficient
            coef1 = beta_t * torch.sqrt(alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            coef2 = (1 - alpha_cumprod_prev) * torch.sqrt(alpha_t) / (1 - alpha_cumprod_t)
            
            # Update coordinates
            pred_x0_coords = (ligand_coords - torch.sqrt(1 - alpha_cumprod_t) * coord_pred) / torch.sqrt(alpha_cumprod_t)
            mean_coords = coef1 * pred_x0_coords + coef2 * ligand_coords
            
            # Add noise
            noise = torch.randn_like(ligand_coords) * torch.sqrt(beta_t)
            ligand_coords = mean_coords + noise
            
            # Update types similarly
            pred_x0_types = (ligand_types - torch.sqrt(1 - alpha_cumprod_t) * type_pred) / torch.sqrt(alpha_cumprod_t)
            mean_types = coef1 * pred_x0_types + coef2 * ligand_types
            ligand_types = mean_types + torch.randn_like(ligand_types) * torch.sqrt(beta_t)
        else:
            # Final step - no noise
            ligand_coords = (ligand_coords - torch.sqrt(1 - alpha_cumprod_t) * coord_pred) / torch.sqrt(alpha_cumprod_t)
            ligand_types = (ligand_types - torch.sqrt(1 - alpha_cumprod_t) * type_pred) / torch.sqrt(alpha_cumprod_t)
    
    # Convert to molecules
    atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'Other']
    
    for i in range(num_samples):
        coords = ligand_coords[i].cpu().numpy()
        types = ligand_types[i].cpu().numpy()
        
        # Get atom types from one-hot
        atom_indices = types.argmax(axis=1)
        atoms = [atom_types[idx] if idx < len(atom_types) else 'C' for idx in atom_indices]
        
        generated.append({
            'coords': coords,
            'atoms': atoms,
            'types_raw': types
        })
    
    return generated


def coords_to_mol(coords, atoms, add_bonds=True):
    """Convert coordinates and atom types to RDKit molecule"""
    mol = Chem.RWMol()
    
    # Add atoms
    atom_map = {'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15, 'F': 9, 'Cl': 17, 'Br': 35, 'I': 53, 'Other': 6}
    
    conf = Chem.Conformer(len(coords))
    for i, (coord, atom) in enumerate(zip(coords, atoms)):
        atomic_num = atom_map.get(atom, 6)
        idx = mol.AddAtom(Chem.Atom(atomic_num))
        conf.SetAtomPosition(idx, coord.tolist())
    
    mol.AddConformer(conf)
    
    if add_bonds:
        # Add bonds based on distance
        bond_thresholds = {
            (6, 6): 1.7, (6, 7): 1.6, (6, 8): 1.6, (6, 16): 2.0,
            (7, 7): 1.5, (7, 8): 1.5, (8, 8): 1.6,
            'default': 1.8
        }
        
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.linalg.norm(coords[i] - coords[j])
                
                an1 = atom_map.get(atoms[i], 6)
                an2 = atom_map.get(atoms[j], 6)
                key = tuple(sorted([an1, an2]))
                threshold = bond_thresholds.get(key, bond_thresholds['default'])
                
                if dist < threshold:
                    try:
                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                    except:
                        pass
    
    try:
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except:
        return mol.GetMol()


def evaluate_molecules(molecules):
    """Evaluate generated molecules"""
    results = {
        'total': len(molecules),
        'valid': 0,
        'valid_smiles': [],
        'qed_scores': [],
        'mw': [],
        'logp': []
    }
    
    for mol_data in molecules:
        mol = coords_to_mol(mol_data['coords'], mol_data['atoms'])
        
        try:
            smiles = Chem.MolToSmiles(mol)
            if smiles and len(smiles) > 3:
                results['valid'] += 1
                results['valid_smiles'].append(smiles)
                
                # Calculate properties
                try:
                    qed = Descriptors.qed(mol)
                    mw = Descriptors.MolWt(mol)
                    logp = Descriptors.MolLogP(mol)
                    results['qed_scores'].append(qed)
                    results['mw'].append(mw)
                    results['logp'].append(logp)
                except:
                    pass
        except:
            pass
    
    # Calculate statistics
    validity = results['valid'] / results['total'] * 100
    print(f"\n📊 Generation Results:")
    print(f"   Valid molecules: {results['valid']}/{results['total']} ({validity:.1f}%)")
    
    if results['qed_scores']:
        print(f"   Mean QED: {np.mean(results['qed_scores']):.3f}")
        print(f"   Mean MW: {np.mean(results['mw']):.1f}")
        print(f"   Mean LogP: {np.mean(results['logp']):.2f}")
    
    # Uniqueness
    unique = len(set(results['valid_smiles']))
    if results['valid'] > 0:
        uniqueness = unique / results['valid'] * 100
        print(f"   Unique: {unique}/{results['valid']} ({uniqueness:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate molecules from trained model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_t4max/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/t4_max.yaml',
                        help='Path to config file')
    parser.add_argument('--data', type=str, default='/content/data/crossdocked/val_data.pkl',
                        help='Path to validation data (for pocket)')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of molecules to generate')
    parser.add_argument('--num_atoms', type=int, default=25,
                        help='Number of atoms per molecule')
    parser.add_argument('--output', type=str, default='generated_molecules.sdf',
                        help='Output SDF file')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, args.config, device)
    
    # Load a pocket from validation data
    print(f"📂 Loading pocket from {args.data}")
    with open(args.data, 'rb') as f:
        val_data = pickle.load(f)
    
    # Use first pocket
    pocket_data = val_data[0]
    print(f"   Pocket: {pocket_data.get('pocket_id', 'unknown')}")
    print(f"   Pocket atoms: {len(pocket_data['pocket_coords'])}")
    
    # Generate molecules
    print(f"\n🧬 Generating {args.num_samples} molecules...")
    molecules = generate_molecules(
        model, pocket_data, config, device,
        num_samples=args.num_samples,
        num_atoms=args.num_atoms
    )
    
    # Evaluate
    results = evaluate_molecules(molecules)
    
    # Save valid molecules
    if results['valid_smiles']:
        print(f"\n💾 Saving to {args.output}")
        writer = Chem.SDWriter(args.output)
        for i, mol_data in enumerate(molecules):
            mol = coords_to_mol(mol_data['coords'], mol_data['atoms'])
            if mol:
                mol.SetProp('_Name', f'Generated_{i}')
                try:
                    writer.write(mol)
                except:
                    pass
        writer.close()
        print(f"   ✅ Saved {results['valid']} molecules")
    
    # Print sample SMILES
    print(f"\n📝 Sample SMILES:")
    for smi in results['valid_smiles'][:5]:
        print(f"   {smi}")


if __name__ == '__main__':
    main()
