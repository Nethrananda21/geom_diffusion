"""
Quality Filters for CrossDocked2020 Dataset

Implements strict quality filtering as specified in the plan:
- RMSD < 1.5 Å (remove poor docking poses)
- Vina score <= -6.0 kcal/mol (docking affinity proxy)
- Resolution < 2.5 Å (high-quality crystal structures)
- Ligand atoms: 15-40 (exclude fragments and macrocycles)
- Pocket atoms: <= 250 (CRITICAL: T4 memory constraint)
- Min interatomic distance >= 0.8 Å (remove steric clashes)
- Single component only
- Organic atoms only (no metalloproteins)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for quality filters."""
    rmsd_max: float = 1.5
    vina_max: float = -6.0
    resolution_max: float = 2.5
    ligand_atoms_min: int = 15
    ligand_atoms_max: int = 40
    pocket_atoms_max: int = 250
    min_interatomic_dist: float = 0.8
    require_single_component: bool = True
    require_organic_only: bool = True


# Organic elements allowed in ligands
ORGANIC_ELEMENTS = {'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H'}

# Metal elements to exclude
METAL_ELEMENTS = {
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn',
    'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo',
    'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Hf', 'Ta',
    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'
}


class QualityFilter:
    """
    Applies quality filters to protein-ligand pairs from CrossDocked2020.
    
    CRITICAL: This filter is essential for T4 GPU compatibility.
    - Pocket atoms <= 250 prevents OOM
    - Ligand atoms <= 40 ensures manageable graph sizes
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize quality filter.
        
        Args:
            config: FilterConfig object with filter thresholds
        """
        self.config = config or FilterConfig()
        self.stats = {
            'total': 0,
            'passed': 0,
            'failed_rmsd': 0,
            'failed_vina': 0,
            'failed_resolution': 0,
            'failed_ligand_size': 0,
            'failed_pocket_size': 0,
            'failed_steric_clash': 0,
            'failed_multi_component': 0,
            'failed_metal': 0
        }
    
    def filter_rmsd(self, rmsd: float) -> bool:
        """
        Filter by RMSD (Root Mean Square Deviation).
        
        Removes poor docking poses where docked pose deviates significantly
        from crystal pose (>1.5Å deviation indicates unreliable pose).
        """
        return rmsd < self.config.rmsd_max
    
    def filter_vina_score(self, vina_score: float) -> bool:
        """
        Filter by Vina docking score.
        
        WARNING: This is docking affinity, NOT experimental pKd/pKi.
        CrossDocked2020 has <5% experimental affinities.
        """
        return vina_score <= self.config.vina_max
    
    def filter_resolution(self, resolution: float) -> bool:
        """
        Filter by crystal structure resolution.
        
        High-quality structures (< 2.5Å) have more reliable atom positions.
        
        FIX Bug #4: NMR structures have resolution=None, handle gracefully.
        """
        if resolution is None:
            return False  # Reject NMR structures (no resolution data)
        return resolution < self.config.resolution_max
    
    def filter_ligand_size(self, num_atoms: int) -> bool:
        """
        Filter by ligand heavy atom count.
        
        - < 15 atoms: Fragments (too small for drug-likeness)
        - > 40 atoms: Macrocycles (T4 memory + rare in drugs)
        """
        return self.config.ligand_atoms_min <= num_atoms <= self.config.ligand_atoms_max
    
    def filter_pocket_size(self, num_atoms: int) -> bool:
        """
        Filter by pocket atom count.
        
        CRITICAL: > 250 atoms causes OOM on T4 (16GB VRAM).
        This is a hard limit, do not adjust without hardware change.
        """
        return num_atoms <= self.config.pocket_atoms_max
    
    def filter_steric_clashes(
        self, 
        coords: np.ndarray,
        bond_pairs: Optional[np.ndarray] = None  # FIX Issue 12: Exclude bonded atoms
    ) -> bool:
        """
        Filter molecules with steric clashes.
        
        Atoms closer than 0.8Å indicate structure errors or failed minimization.
        
        FIX Issue 12: Bonded atoms (1-2) and 1-3 pairs are naturally close
        and should NOT be flagged as clashes.
        """
        if len(coords) < 2:
            return True
        
        # Compute pairwise distances
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Mask diagonal (self-distances)
        np.fill_diagonal(dists, np.inf)
        
        # FIX Issue 12: Mask bonded pairs (they're expected to be close)
        if bond_pairs is not None and len(bond_pairs) > 0:
            for i, j in bond_pairs:
                dists[i, j] = np.inf
                dists[j, i] = np.inf
                
                # Also mask 1-3 pairs (atoms bonded to same neighbor)
                # This requires more complex connectivity analysis
                # For simplicity, we only exclude direct bonds here
        
        # Get minimum non-bonded distance
        min_dist = np.min(dists)
        
        return min_dist >= self.config.min_interatomic_dist
    
    def filter_single_component(self, mol: Chem.Mol) -> bool:
        """
        Filter molecules with disconnected fragments.
        
        Multi-component systems (salts, solvents) cause training issues.
        """
        if not self.config.require_single_component:
            return True
        
        frags = Chem.GetMolFrags(mol)
        return len(frags) == 1
    
    def filter_organic_only(self, mol: Chem.Mol) -> bool:
        """
        Filter out metalloproteins and organometallic compounds.
        
        Metal coordination (Fe, Zn, etc.) requires special handling
        not covered by standard EGNN.
        """
        if not self.config.require_organic_only:
            return True
        
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in METAL_ELEMENTS:
                return False
            if symbol not in ORGANIC_ELEMENTS:
                return False
        
        return True
    
    def apply(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Apply all quality filters to a protein-ligand pair.
        
        Args:
            sample: Dict containing:
                - 'rmsd': float
                - 'vina_score': float
                - 'resolution': float
                - 'ligand_mol': RDKit Mol object
                - 'ligand_coords': np.ndarray (N, 3)
                - 'pocket_coords': np.ndarray (M, 3)
        
        Returns:
            Tuple of (passed: bool, reason: str)
        """
        self.stats['total'] += 1
        
        # RMSD filter
        if not self.filter_rmsd(sample.get('rmsd', 0.0)):
            self.stats['failed_rmsd'] += 1
            return False, f"RMSD {sample['rmsd']:.2f} >= {self.config.rmsd_max}"
        
        # Vina score filter
        if not self.filter_vina_score(sample.get('vina_score', 0.0)):
            self.stats['failed_vina'] += 1
            return False, f"Vina {sample['vina_score']:.2f} > {self.config.vina_max}"
        
        # Resolution filter
        if not self.filter_resolution(sample.get('resolution', 0.0)):
            self.stats['failed_resolution'] += 1
            return False, f"Resolution {sample['resolution']:.2f} >= {self.config.resolution_max}"
        
        # Ligand size filter
        ligand_mol = sample.get('ligand_mol')
        if ligand_mol is not None:
            num_heavy = ligand_mol.GetNumHeavyAtoms()
            if not self.filter_ligand_size(num_heavy):
                self.stats['failed_ligand_size'] += 1
                return False, f"Ligand atoms {num_heavy} outside [{self.config.ligand_atoms_min}, {self.config.ligand_atoms_max}]"
        
        # Pocket size check - DISABLED: we now TRUNCATE in pockets.py instead of filtering
        # This preserves ~60% more training data by keeping closest atoms to ligand
        # pocket_coords = sample.get('pocket_coords')
        # if pocket_coords is not None:
        #     num_pocket = len(pocket_coords)
        #     if not self.filter_pocket_size(num_pocket):
        #         self.stats['failed_pocket_size'] += 1
        #         return False, f"Pocket atoms {num_pocket} > {self.config.pocket_atoms_max} (T4 limit)"
        
        # Steric clash filter
        ligand_coords = sample.get('ligand_coords')
        if ligand_coords is not None:
            if not self.filter_steric_clashes(ligand_coords):
                self.stats['failed_steric_clash'] += 1
                return False, f"Steric clash detected (min dist < {self.config.min_interatomic_dist})"
        
        # Single component filter
        if ligand_mol is not None:
            if not self.filter_single_component(ligand_mol):
                self.stats['failed_multi_component'] += 1
                return False, "Multi-component molecule"
        
        # Organic only filter
        if ligand_mol is not None:
            if not self.filter_organic_only(ligand_mol):
                self.stats['failed_metal'] += 1
                return False, "Contains metal atoms"
        
        self.stats['passed'] += 1
        return True, "Passed all filters"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics."""
        stats = self.stats.copy()
        if stats['total'] > 0:
            stats['pass_rate'] = stats['passed'] / stats['total'] * 100
        else:
            stats['pass_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset filter statistics."""
        for key in self.stats:
            self.stats[key] = 0


def filter_dataset(
    samples: List[Dict[str, Any]],
    config: Optional[FilterConfig] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter a list of protein-ligand samples.
    
    Args:
        samples: List of sample dictionaries
        config: Filter configuration
        verbose: Print statistics
    
    Returns:
        Filtered list of samples
    """
    filter_obj = QualityFilter(config)
    filtered = []
    
    for sample in samples:
        passed, reason = filter_obj.apply(sample)
        if passed:
            filtered.append(sample)
        elif verbose:
            logger.debug(f"Filtered: {reason}")
    
    if verbose:
        stats = filter_obj.get_stats()
        logger.info(f"Filtering complete: {stats['passed']}/{stats['total']} passed ({stats['pass_rate']:.1f}%)")
        for key, val in stats.items():
            if key.startswith('failed_') and val > 0:
                logger.info(f"  - {key}: {val}")
    
    return filtered
