"""
CrossDocked2020 Preprocessing Script

Converts raw CrossDocked2020 files (.sdf, .pdb) to the indexed .pkl format
expected by the dataset loader.

Usage:
    python preprocess_crossdocked.py \
        --data_dir /path/to/crossdocked2020 \
        --output_dir ./data/processed \
        --config configs/debug_t4.yaml

Expected Input Structure:
    crossdocked2020/
    ├── index.pkl (or split information)
    ├── <pdb_id>/
    │   ├── *_pocket.pdb
    │   └── *_ligand.sdf
    └── ...

Output Format:
    processed/
    ├── index.pkl (metadata + file paths)
    ├── train_data.pkl
    ├── val_data.pkl
    └── stats.json
"""

import os
import sys
import pickle
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data.filters import QualityFilter, FilterConfig
from data.diversity import select_diverse_ligands, compute_fingerprints
from data.pockets import PocketProcessor, PocketConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_pdb_pocket(pdb_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Parse pocket PDB file to extract coordinates and elements.
    
    Args:
        pdb_path: Path to pocket PDB file
    
    Returns:
        Tuple of (coords, elements)
    """
    coords = []
    elements = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Extract coordinates (columns 31-54)
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                
                # Extract element (columns 77-78, or infer from atom name)
                if len(line) >= 78:
                    elem = line[76:78].strip()
                else:
                    # Infer from atom name (columns 13-16)
                    atom_name = line[12:16].strip()
                    elem = ''.join(c for c in atom_name if c.isalpha())[:1]
                
                elements.append(elem if elem else 'C')
    
    return np.array(coords, dtype=np.float32), elements


def parse_sdf_ligand(sdf_path: str) -> Optional[Tuple[Chem.Mol, np.ndarray, List[str]]]:
    """
    Parse ligand SDF file.
    
    Args:
        sdf_path: Path to ligand SDF file
    
    Returns:
        Tuple of (mol, coords, elements) or None if failed
    """
    try:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        mol = suppl[0]
        
        if mol is None:
            return None
        
        # Try to sanitize
        try:
            Chem.SanitizeMol(mol)
        except:
            pass  # Continue with unsanitized mol
        
        # Get conformer
        conf = mol.GetConformer()
        coords = np.array([
            [conf.GetAtomPosition(i).x,
             conf.GetAtomPosition(i).y,
             conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())
        ], dtype=np.float32)
        
        elements = [atom.GetSymbol() for atom in mol.GetAtoms()]
        
        return mol, coords, elements
    
    except Exception as e:
        logger.debug(f"Failed to parse {sdf_path}: {e}")
        return None


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Extract RMSD and Vina score from CrossDocked2020 filename format.
    
    Expected format: <receptor>_<ligand>_<vina>_<rmsd>.sdf
    Example: 1a4w_B_rec_1a4w_bax_lig_tt_min_-6.123_0.456.sdf
    """
    try:
        # Split by underscore and extract last two parts
        parts = filename.replace('.sdf', '').split('_')
        
        # Last two parts are typically rmsd and vina score
        rmsd = float(parts[-1])
        vina = float(parts[-2])
        
        return {'rmsd': rmsd, 'vina_score': vina}
    except:
        return {'rmsd': 0.0, 'vina_score': -6.0}  # Default values


def find_pocket_ligand_pairs(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Find all pocket-ligand pairs in CrossDocked2020 directory.
    
    Returns:
        List of dicts with keys: pocket_path, ligand_path, pocket_id, metadata
    """
    pairs = []
    
    # Walk through directory
    for root, dirs, files in os.walk(data_dir):
        root_path = Path(root)
        
        # Find pocket files
        pocket_files = [f for f in files if f.endswith('_pocket.pdb') or f.endswith('pocket.pdb')]
        ligand_files = [f for f in files if f.endswith('.sdf')]
        
        for pocket_file in pocket_files:
            pocket_path = root_path / pocket_file
            pocket_id = pocket_file.replace('_pocket.pdb', '').replace('.pdb', '')
            
            # Find matching ligands (same receptor)
            for ligand_file in ligand_files:
                ligand_path = root_path / ligand_file
                metadata = extract_metadata_from_filename(ligand_file)
                
                pairs.append({
                    'pocket_path': str(pocket_path),
                    'ligand_path': str(ligand_path),
                    'pocket_id': pocket_id,
                    'ligand_id': ligand_file.replace('.sdf', ''),
                    **metadata
                })
    
    logger.info(f"Found {len(pairs)} pocket-ligand pairs")
    return pairs


def process_pairs(
    pairs: List[Dict[str, Any]],
    filter_config: FilterConfig,
    pocket_config: PocketConfig,
    n_ligands_per_pocket: int = 20
) -> List[Dict[str, Any]]:
    """
    Process and filter pocket-ligand pairs.
    
    Applies:
    1. Quality filtering
    2. Pocket processing (truncation, centering)
    3. Diversity selection per pocket
    
    Returns:
        List of processed samples ready for training
    """
    quality_filter = QualityFilter(filter_config)
    pocket_processor = PocketProcessor(pocket_config)
    
    # Group pairs by pocket
    pocket_to_ligands = defaultdict(list)
    for pair in pairs:
        pocket_to_ligands[pair['pocket_id']].append(pair)
    
    logger.info(f"Processing {len(pocket_to_ligands)} unique pockets")
    
    processed_samples = []
    
    for pocket_id, ligand_pairs in tqdm(pocket_to_ligands.items(), desc="Processing pockets"):
        # Parse pocket once
        pocket_path = ligand_pairs[0]['pocket_path']
        try:
            pocket_coords, pocket_elements = parse_pdb_pocket(pocket_path)
        except Exception as e:
            logger.debug(f"Failed to parse pocket {pocket_path}: {e}")
            continue
        
        if len(pocket_coords) == 0:
            continue
        
        valid_ligands = []
        
        # Process each ligand
        for pair in ligand_pairs:
            result = parse_sdf_ligand(pair['ligand_path'])
            if result is None:
                continue
            
            mol, lig_coords, lig_elements = result
            
            # Quality filter
            sample = {
                'rmsd': pair['rmsd'],
                'vina_score': pair['vina_score'],
                'resolution': 2.0,  # Default if not available
                'ligand_mol': mol,
                'ligand_coords': lig_coords,
                'pocket_coords': pocket_coords
            }
            
            passed, reason = quality_filter.apply(sample)
            if not passed:
                continue
            
            # Process pocket (truncation, centering)
            try:
                processed = pocket_processor.process(
                    pocket_coords.copy(),
                    pocket_elements.copy(),
                    lig_coords,
                    lig_elements
                )
            except Exception as e:
                logger.debug(f"Processing failed: {e}")
                continue
            
            valid_ligands.append({
                'mol': mol,
                'pair': pair,
                'processed': processed
            })
        
        if len(valid_ligands) == 0:
            continue
        
        # Diversity selection
        if len(valid_ligands) > n_ligands_per_pocket:
            try:
                mols = [v['mol'] for v in valid_ligands]
                selected_indices = select_diverse_ligands(
                    mols, 
                    n_select=n_ligands_per_pocket
                )
                valid_ligands = [valid_ligands[i] for i in selected_indices]
            except Exception as e:
                logger.debug(f"Diversity selection failed: {e}")
                valid_ligands = valid_ligands[:n_ligands_per_pocket]
        
        # Add to processed samples
        for item in valid_ligands:
            processed = item['processed']
            pair = item['pair']
            
            processed_samples.append({
                'pocket_id': pocket_id,
                'ligand_id': pair['ligand_id'],
                'rmsd': pair['rmsd'],
                'vina_score': pair['vina_score'],
                'ligand_coords': processed['ligand_coords'],
                'ligand_types': processed['ligand_types'],
                'ligand_elements': processed['ligand_elements'],
                'pocket_coords': processed['pocket_coords'],
                'pocket_types': processed['pocket_types'],
                'pocket_elements': processed['pocket_elements'],
                'com': processed['com'],
                'n_ligand_atoms': processed['n_ligand_atoms'],
                'n_pocket_atoms': processed['n_pocket_atoms']
            })
    
    # Print filter stats
    stats = quality_filter.get_stats()
    logger.info(f"Filter stats: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.1f}%)")
    
    return processed_samples


def split_by_pocket(
    samples: List[Dict],
    train_pockets: int,
    val_pockets: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples by pocket to prevent data leakage.
    
    CRITICAL: Same pocket must NOT appear in both train and val!
    """
    # Get unique pocket IDs
    pocket_ids = list(set(s['pocket_id'] for s in samples))
    np.random.shuffle(pocket_ids)
    
    total_pockets = train_pockets + val_pockets
    if len(pocket_ids) < total_pockets:
        logger.warning(f"Only {len(pocket_ids)} pockets available, need {total_pockets}")
        train_pockets = int(len(pocket_ids) * 0.8)
        val_pockets = len(pocket_ids) - train_pockets
    
    train_pocket_ids = set(pocket_ids[:train_pockets])
    val_pocket_ids = set(pocket_ids[train_pockets:train_pockets + val_pockets])
    
    train_samples = [s for s in samples if s['pocket_id'] in train_pocket_ids]
    val_samples = [s for s in samples if s['pocket_id'] in val_pocket_ids]
    
    logger.info(f"Split: {len(train_samples)} train samples ({train_pockets} pockets), "
                f"{len(val_samples)} val samples ({val_pockets} pockets)")
    
    return train_samples, val_samples


def main(args):
    """Main preprocessing function."""
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(config.get('seed', 42))
    
    # Find all pairs
    data_dir = Path(args.data_dir)
    pairs = find_pocket_ligand_pairs(data_dir)
    
    if len(pairs) == 0:
        logger.error("No pocket-ligand pairs found!")
        return
    
    # Create filter and pocket configs
    filter_cfg = FilterConfig(
        rmsd_max=config['data']['filters']['rmsd_max'],
        vina_max=config['data']['filters']['vina_max'],
        ligand_atoms_min=config['data']['filters']['ligand_atoms_min'],
        ligand_atoms_max=config['data']['filters']['ligand_atoms_max'],
        pocket_atoms_max=config['data']['filters']['pocket_atoms_max']
    )
    
    pocket_cfg = PocketConfig(
        pocket_radius=config['data']['pocket_radius'],
        remove_hydrogens=config['data']['remove_hydrogens'],
        centering=config['data']['centering'],
        max_pocket_atoms=config['data']['filters']['pocket_atoms_max']
    )
    
    # Process pairs
    processed = process_pairs(
        pairs,
        filter_cfg,
        pocket_cfg,
        n_ligands_per_pocket=config['data']['ligands_per_pocket']
    )
    
    logger.info(f"Total processed samples: {len(processed)}")
    
    # Split by pocket
    train_samples, val_samples = split_by_pocket(
        processed,
        train_pockets=config['data']['train_pockets'],
        val_pockets=config['data']['val_pockets']
    )
    
    # Save processed data
    with open(output_dir / 'train_data.pkl', 'wb') as f:
        pickle.dump(train_samples, f)
    logger.info(f"Saved {len(train_samples)} train samples to {output_dir / 'train_data.pkl'}")
    
    with open(output_dir / 'val_data.pkl', 'wb') as f:
        pickle.dump(val_samples, f)
    logger.info(f"Saved {len(val_samples)} val samples to {output_dir / 'val_data.pkl'}")
    
    # Save index (metadata)
    index = {
        'train_pockets': list(set(s['pocket_id'] for s in train_samples)),
        'val_pockets': list(set(s['pocket_id'] for s in val_samples)),
        'n_train': len(train_samples),
        'n_val': len(val_samples),
        'filter_config': filter_cfg.__dict__,
        'pocket_config': pocket_cfg.__dict__
    }
    
    with open(output_dir / 'index.pkl', 'wb') as f:
        pickle.dump(index, f)
    
    # Save stats
    stats = {
        'total_pairs': len(pairs),
        'processed_samples': len(processed),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'train_pockets': len(index['train_pockets']),
        'val_pockets': len(index['val_pockets']),
        'avg_ligand_atoms': np.mean([s['n_ligand_atoms'] for s in processed]),
        'avg_pocket_atoms': np.mean([s['n_pocket_atoms'] for s in processed])
    }
    
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("Preprocessing complete!")
    logger.info(f"Stats: {json.dumps(stats, indent=2)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess CrossDocked2020')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CrossDocked2020 directory')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for processed files')
    parser.add_argument('--config', type=str, default='configs/debug_t4.yaml',
                        help='Config file for filter/pocket settings')
    
    args = parser.parse_args()
    main(args)
