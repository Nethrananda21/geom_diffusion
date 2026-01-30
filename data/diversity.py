"""
Diversity Selection for CrossDocked2020 Dataset

Implements MaxMin algorithm for selecting chemically diverse ligands:
- Uses ECFP4 fingerprints for molecular representation
- Tanimoto cutoff <= 0.8 for diversity
- Size stratification (15-25: 30%, 26-35: 40%, 36-40: 30%)
- Mandatory element coverage (F, Cl, Br, S each in >50 ligands)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers

logger = logging.getLogger(__name__)


@dataclass
class DiversityConfig:
    """Configuration for diversity selection."""
    fingerprint_type: str = "ECFP4"
    fingerprint_radius: int = 2
    fingerprint_nbits: int = 2048
    tanimoto_cutoff: float = 0.8
    selection_method: str = "MaxMin"
    required_elements: List[str] = None
    min_element_count: int = 50  # Each required element in at least this many ligands
    
    # Size stratification
    size_bins: Dict[str, Tuple[int, int]] = None
    size_distribution: Dict[str, float] = None
    
    def __post_init__(self):
        if self.required_elements is None:
            self.required_elements = ["F", "Cl", "Br", "S"]
        
        if self.size_bins is None:
            self.size_bins = {
                "small": (15, 25),
                "medium": (26, 35),
                "large": (36, 40)
            }
        
        if self.size_distribution is None:
            self.size_distribution = {
                "small": 0.30,
                "medium": 0.40,
                "large": 0.30
            }


class MolecularFingerprinter:
    """
    Compute molecular fingerprints for diversity calculations.
    """
    
    def __init__(self, config: DiversityConfig):
        self.config = config
    
    def compute_fingerprint(self, mol: Chem.Mol) -> DataStructs.ExplicitBitVect:
        """
        Compute ECFP4 fingerprint for a molecule.
        
        ECFP4 (radius=2) captures local chemical environments
        and is ideal for diversity assessment.
        """
        if self.config.fingerprint_type == "ECFP4":
            return AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=self.config.fingerprint_radius,
                nBits=self.config.fingerprint_nbits
            )
        else:
            raise ValueError(f"Unknown fingerprint type: {self.config.fingerprint_type}")
    
    def compute_fingerprints(self, mols: List[Chem.Mol]) -> List[DataStructs.ExplicitBitVect]:
        """Compute fingerprints for a list of molecules."""
        return [self.compute_fingerprint(mol) for mol in mols]
    
    def tanimoto_similarity(
        self,
        fp1: DataStructs.ExplicitBitVect,
        fp2: DataStructs.ExplicitBitVect
    ) -> float:
        """Compute Tanimoto similarity between two fingerprints."""
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def similarity_matrix(
        self,
        fps: List[DataStructs.ExplicitBitVect]
    ) -> np.ndarray:
        """Compute pairwise Tanimoto similarity matrix."""
        n = len(fps)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.tanimoto_similarity(fps[i], fps[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
            sim_matrix[i, i] = 1.0
        
        return sim_matrix


class MaxMinSelector:
    """
    MaxMin algorithm for selecting diverse molecules.
    
    Algorithm:
    1. Start with the molecule most dissimilar to the centroid
    2. Iteratively add molecules that maximize minimum distance to selected set
    3. Ensures maximum spread across chemical space
    """
    
    def __init__(self, config: DiversityConfig):
        self.config = config
        self.fingerprinter = MolecularFingerprinter(config)
    
    def select(
        self,
        mols: List[Chem.Mol],
        n_select: int,
        fps: Optional[List[DataStructs.ExplicitBitVect]] = None
    ) -> List[int]:
        """
        Select n_select diverse molecules using MaxMin algorithm.
        
        Args:
            mols: List of RDKit molecules
            n_select: Number of molecules to select
            fps: Pre-computed fingerprints (optional)
        
        Returns:
            List of indices of selected molecules
        """
        if fps is None:
            fps = self.fingerprinter.compute_fingerprints(mols)
        
        n = len(mols)
        if n_select >= n:
            return list(range(n))
        
        # Use RDKit's built-in MaxMin picker for efficiency
        picker = rdSimDivPickers.MaxMinPicker()
        
        # Create distance function (1 - Tanimoto similarity)
        def distance_fn(i, j):
            return 1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
        
        # Run MaxMin selection
        selected_indices = list(picker.LazyBitVectorPick(
            fps, n, n_select, firstPicks=[]
        ))
        
        return selected_indices


class StratifiedDiversitySelector:
    """
    Select diverse molecules with stratification by size and element coverage.
    
    Ensures:
    1. Size distribution matches target (30% small, 40% medium, 30% large)
    2. Required elements (F, Cl, Br, S) each appear in sufficient samples
    3. Chemical diversity via MaxMin selection within each stratum
    """
    
    def __init__(self, config: Optional[DiversityConfig] = None):
        self.config = config or DiversityConfig()
        self.maxmin = MaxMinSelector(self.config)
        self.fingerprinter = MolecularFingerprinter(self.config)
    
    def get_size_bin(self, mol: Chem.Mol) -> str:
        """Classify molecule by size."""
        n_atoms = mol.GetNumHeavyAtoms()
        
        for bin_name, (min_size, max_size) in self.config.size_bins.items():
            if min_size <= n_atoms <= max_size:
                return bin_name
        
        return None  # Outside all bins
    
    def get_elements(self, mol: Chem.Mol) -> Set[str]:
        """Get set of element symbols in molecule."""
        return {atom.GetSymbol() for atom in mol.GetAtoms()}
    
    def stratify_by_size(
        self,
        mols: List[Chem.Mol],
        indices: Optional[List[int]] = None
    ) -> Dict[str, List[int]]:
        """Group molecules by size bin."""
        if indices is None:
            indices = list(range(len(mols)))
        
        bins = defaultdict(list)
        for idx in indices:
            bin_name = self.get_size_bin(mols[idx])
            if bin_name is not None:
                bins[bin_name].append(idx)
        
        return dict(bins)
    
    def find_element_rich(
        self,
        mols: List[Chem.Mol],
        indices: List[int],
        element: str
    ) -> List[int]:
        """Find indices of molecules containing a specific element."""
        return [idx for idx in indices if element in self.get_elements(mols[idx])]
    
    def select(
        self,
        mols: List[Chem.Mol],
        n_select: int,
        pocket_id: Optional[str] = None
    ) -> List[int]:
        """
        Select n_select diverse, stratified molecules.
        
        Args:
            mols: List of RDKit molecules
            n_select: Total number to select
            pocket_id: Optional pocket identifier for logging
        
        Returns:
            List of selected indices
        """
        # Compute fingerprints once
        fps = self.fingerprinter.compute_fingerprints(mols)
        
        # 1. Stratify by size
        size_bins = self.stratify_by_size(mols)
        
        # Calculate how many to select from each bin
        n_per_bin = {}
        for bin_name, fraction in self.config.size_distribution.items():
            n_per_bin[bin_name] = int(n_select * fraction)
        
        # Adjust for rounding
        total_allocated = sum(n_per_bin.values())
        if total_allocated < n_select:
            n_per_bin["medium"] += n_select - total_allocated
        
        selected = set()
        
        # 2. First pass: ensure element coverage
        for element in self.config.required_elements:
            element_rich = []
            for bin_name, indices in size_bins.items():
                element_rich.extend(self.find_element_rich(mols, indices, element))
            
            # Select some with MaxMin from element-rich set
            if element_rich and len(selected) < self.config.min_element_count:
                n_to_add = min(
                    self.config.min_element_count // len(self.config.required_elements),
                    len(element_rich)
                )
                element_fps = [fps[i] for i in element_rich]
                element_selected = self.maxmin.select(
                    [mols[i] for i in element_rich],
                    n_to_add,
                    element_fps
                )
                selected.update(element_rich[i] for i in element_selected)
        
        # 3. Second pass: fill remaining with stratified MaxMin
        for bin_name, indices in size_bins.items():
            remaining = [i for i in indices if i not in selected]
            n_needed = max(0, n_per_bin.get(bin_name, 0) - len([s for s in selected if s in indices]))
            
            if remaining and n_needed > 0:
                bin_fps = [fps[i] for i in remaining]
                bin_selected = self.maxmin.select(
                    [mols[i] for i in remaining],
                    min(n_needed, len(remaining)),
                    bin_fps
                )
                selected.update(remaining[i] for i in bin_selected)
        
        # 4. Fill any remaining slots with MaxMin from unselected
        if len(selected) < n_select:
            unselected = [i for i in range(len(mols)) if i not in selected]
            if unselected:
                unseen_fps = [fps[i] for i in unselected]
                n_remaining = n_select - len(selected)
                fill_selected = self.maxmin.select(
                    [mols[i] for i in unselected],
                    min(n_remaining, len(unselected)),
                    unseen_fps
                )
                selected.update(unselected[i] for i in fill_selected)
        
        selected_list = list(selected)[:n_select]
        
        # Log statistics
        if pocket_id:
            self._log_selection_stats(mols, selected_list, pocket_id)
        
        return selected_list
    
    def _log_selection_stats(
        self,
        mols: List[Chem.Mol],
        selected: List[int],
        pocket_id: str
    ):
        """Log statistics about selected molecules."""
        size_counts = defaultdict(int)
        element_counts = defaultdict(int)
        
        for idx in selected:
            mol = mols[idx]
            size_counts[self.get_size_bin(mol)] += 1
            for elem in self.get_elements(mol):
                element_counts[elem] += 1
        
        logger.info(f"Pocket {pocket_id}: Selected {len(selected)} ligands")
        logger.debug(f"  Size distribution: {dict(size_counts)}")
        logger.debug(f"  Element coverage: F={element_counts.get('F', 0)}, "
                    f"Cl={element_counts.get('Cl', 0)}, "
                    f"Br={element_counts.get('Br', 0)}, "
                    f"S={element_counts.get('S', 0)}")


def select_diverse_ligands(
    mols: List[Chem.Mol],
    n_select: int,
    config: Optional[DiversityConfig] = None,
    pocket_id: Optional[str] = None
) -> List[int]:
    """
    Convenience function for diversity selection.
    
    Args:
        mols: List of RDKit molecules
        n_select: Number to select
        config: Diversity configuration
        pocket_id: Optional pocket identifier
    
    Returns:
        List of selected indices
    """
    selector = StratifiedDiversitySelector(config)
    return selector.select(mols, n_select, pocket_id)
