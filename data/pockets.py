"""
Pocket Processing for CrossDocked2020 Dataset

Implements pocket extraction and preprocessing:
- Center of mass centering (shift to origin)
- Pocket definition as residues within 6Å of ligand
- Heavy atom only (remove hydrogens for T4 memory)
- Atom type encoding
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

from rdkit import Chem
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


@dataclass
class PocketConfig:
    """Configuration for pocket processing."""
    pocket_radius: float = 6.0  # Angstroms around ligand
    remove_hydrogens: bool = True
    centering: str = "center_of_mass"  # or "ligand_centroid"
    max_pocket_atoms: int = 250  # CRITICAL: T4 memory limit - truncate, don't filter!
    
    # Atom type encoding
    atom_types: List[str] = None
    
    def __post_init__(self):
        if self.atom_types is None:
            self.atom_types = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"]


# Standard covalent radii for bond inference (in Angstroms)
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39
}


class PocketProcessor:
    """
    Process protein pockets for diffusion model input.
    
    Key operations:
    1. Extract pocket atoms within cutoff distance of ligand
    2. TRUNCATE to max_pocket_atoms (keep closest to ligand COM)
    3. Center system at center of mass
    4. Encode atom types as one-hot vectors
    5. Remove hydrogens for memory efficiency
    """
    
    def __init__(self, config: Optional[PocketConfig] = None):
        self.config = config or PocketConfig()
        self.atom_to_idx = {a: i for i, a in enumerate(self.config.atom_types)}
        self.n_atom_types = len(self.config.atom_types)
    
    def extract_pocket(
        self,
        protein_coords: np.ndarray,
        protein_elements: List[str],
        ligand_coords: np.ndarray,
        residue_ids: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, List[str], Optional[List[int]]]:
        """
        Extract pocket atoms within cutoff distance of ligand.
        
        Args:
            protein_coords: (N, 3) protein atom coordinates
            protein_elements: List of N element symbols
            ligand_coords: (M, 3) ligand atom coordinates
            residue_ids: Optional list of residue IDs for each protein atom
        
        Returns:
            Tuple of (pocket_coords, pocket_elements, pocket_residue_ids)
        """
        # Compute distances from each protein atom to nearest ligand atom
        distances = cdist(protein_coords, ligand_coords)
        min_distances = np.min(distances, axis=1)
        
        # Select atoms within pocket radius
        mask = min_distances <= self.config.pocket_radius
        
        pocket_coords = protein_coords[mask]
        pocket_elements = [protein_elements[i] for i in range(len(protein_elements)) if mask[i]]
        
        if residue_ids is not None:
            pocket_residue_ids = [residue_ids[i] for i in range(len(residue_ids)) if mask[i]]
        else:
            pocket_residue_ids = None
        
        return pocket_coords, pocket_elements, pocket_residue_ids
    
    def remove_hydrogens(
        self,
        coords: np.ndarray,
        elements: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Remove hydrogen atoms from coordinate array.
        
        This is CRITICAL for T4 memory:
        - Reduces atom count by ~40-50%
        - Heavy atoms sufficient for structure learning
        """
        mask = np.array([e != 'H' for e in elements])
        return coords[mask], [e for e, m in zip(elements, mask) if m]
    
    def center_at_com(
        self,
        ligand_coords: np.ndarray,
        pocket_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Center system at center of mass of ligand.
        
        Returns:
            Tuple of (centered_ligand, centered_pocket, com)
        """
        com = np.mean(ligand_coords, axis=0)
        return ligand_coords - com, pocket_coords - com, com
    
    def center_at_pocket_com(
        self,
        ligand_coords: np.ndarray,
        pocket_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Center system at center of mass of pocket.
        """
        com = np.mean(pocket_coords, axis=0)
        return ligand_coords - com, pocket_coords - com, com
    
    def encode_atom_types(
        self,
        elements: List[str]
    ) -> np.ndarray:
        """
        Encode elements as one-hot vectors.
        
        Args:
            elements: List of element symbols
        
        Returns:
            (N, n_atom_types) one-hot encoded array
        """
        one_hot = np.zeros((len(elements), self.n_atom_types), dtype=np.float32)
        
        for i, elem in enumerate(elements):
            if elem in self.atom_to_idx:
                one_hot[i, self.atom_to_idx[elem]] = 1.0
            else:
                # Unknown element - use carbon as default
                logger.warning(f"Unknown element {elem}, encoding as C")
                one_hot[i, self.atom_to_idx['C']] = 1.0
        
        return one_hot
    
    def decode_atom_types(
        self,
        one_hot: np.ndarray
    ) -> List[str]:
        """
        Decode one-hot vectors back to element symbols.
        
        Args:
            one_hot: (N, n_atom_types) one-hot encoded array
        
        Returns:
            List of element symbols
        """
        indices = np.argmax(one_hot, axis=1)
        return [self.config.atom_types[i] for i in indices]
    
    def truncate_pocket(
        self,
        pocket_coords: np.ndarray,
        pocket_elements: List[str],
        ligand_coords: np.ndarray,
        max_atoms: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Truncate pocket to max_atoms by keeping atoms closest to ligand COM.
        
        CRITICAL FIX: Instead of FILTERING OUT large pockets (losing ~60% of data),
        we TRUNCATE them by keeping the closest atoms to the ligand.
        
        Args:
            pocket_coords: (N, 3) pocket atom coordinates
            pocket_elements: List of N element symbols
            ligand_coords: (M, 3) ligand coordinates
            max_atoms: Maximum atoms to keep (default: config.max_pocket_atoms)
        
        Returns:
            Tuple of (truncated_coords, truncated_elements)
        """
        if max_atoms is None:
            max_atoms = self.config.max_pocket_atoms
        
        if len(pocket_coords) <= max_atoms:
            return pocket_coords, pocket_elements
        
        # Compute ligand center of mass
        ligand_com = np.mean(ligand_coords, axis=0)
        
        # Compute distances from pocket atoms to ligand COM
        distances = np.linalg.norm(pocket_coords - ligand_com, axis=1)
        
        # Get indices of closest atoms
        closest_indices = np.argsort(distances)[:max_atoms]
        
        # Sort indices to maintain original order (helps with residue grouping)
        closest_indices = np.sort(closest_indices)
        
        truncated_coords = pocket_coords[closest_indices]
        truncated_elements = [pocket_elements[i] for i in closest_indices]
        
        logger.debug(f"Truncated pocket from {len(pocket_coords)} to {max_atoms} atoms")
        
        return truncated_coords, truncated_elements

    def process(
        self,
        protein_coords: np.ndarray,
        protein_elements: List[str],
        ligand_coords: np.ndarray,
        ligand_elements: List[str],
        residue_ids: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline for a protein-ligand pair.
        
        Args:
            protein_coords: (N, 3) protein atom coordinates
            protein_elements: List of N element symbols
            ligand_coords: (M, 3) ligand atom coordinates
            ligand_elements: List of M element symbols
            residue_ids: Optional residue IDs
        
        Returns:
            Dict containing:
                - 'ligand_coords': (M', 3) centered ligand coordinates
                - 'ligand_types': (M', n_types) one-hot atom types
                - 'pocket_coords': (P, 3) centered pocket coordinates
                - 'pocket_types': (P, n_types) one-hot atom types
                - 'com': (3,) original center of mass
        """
        # 1. Remove hydrogens if configured
        if self.config.remove_hydrogens:
            ligand_coords, ligand_elements = self.remove_hydrogens(
                ligand_coords, ligand_elements
            )
            protein_coords, protein_elements = self.remove_hydrogens(
                protein_coords, protein_elements
            )
        
        # 2. Extract pocket atoms (within radius of ligand)
        pocket_coords, pocket_elements, _ = self.extract_pocket(
            protein_coords, protein_elements, ligand_coords, residue_ids
        )
        
        # 3. TRUNCATE pocket to max_atoms (CRITICAL: keep closest atoms, don't discard!)
        pocket_coords, pocket_elements = self.truncate_pocket(
            pocket_coords, pocket_elements, ligand_coords
        )
        
        # 4. Center at COM
        if self.config.centering == "center_of_mass":
            ligand_coords, pocket_coords, com = self.center_at_com(
                ligand_coords, pocket_coords
            )
        elif self.config.centering == "pocket_com":
            ligand_coords, pocket_coords, com = self.center_at_pocket_com(
                ligand_coords, pocket_coords
            )
        else:
            com = np.zeros(3)
        
        # 5. Encode atom types
        ligand_types = self.encode_atom_types(ligand_elements)
        pocket_types = self.encode_atom_types(pocket_elements)
        
        return {
            'ligand_coords': ligand_coords.astype(np.float32),
            'ligand_types': ligand_types,
            'ligand_elements': ligand_elements,
            'pocket_coords': pocket_coords.astype(np.float32),
            'pocket_types': pocket_types,
            'pocket_elements': pocket_elements,
            'com': com.astype(np.float32),
            'n_ligand_atoms': len(ligand_coords),
            'n_pocket_atoms': len(pocket_coords)
        }


def compute_adjacency_matrix(
    coords: np.ndarray,
    elements: List[str],
    tolerance: float = 0.5
) -> np.ndarray:
    """
    Compute adjacency matrix based on covalent radii.
    
    Two atoms are bonded if:
        distance <= (cov_radius_1 + cov_radius_2) + tolerance
    
    Args:
        coords: (N, 3) atom coordinates
        elements: List of N element symbols
        tolerance: Bond tolerance in Angstroms
    
    Returns:
        (N, N) binary adjacency matrix
    """
    n = len(coords)
    distances = cdist(coords, coords)
    
    # Get covalent radius for each atom
    radii = np.array([COVALENT_RADII.get(e, 1.5) for e in elements])
    
    # Compute cutoff matrix
    cutoffs = radii[:, np.newaxis] + radii[np.newaxis, :] + tolerance
    
    # Create adjacency matrix (no self-loops)
    adj = (distances <= cutoffs) & (distances > 0.01)
    
    return adj.astype(np.float32)


def coords_to_mol(
    coords: np.ndarray,
    elements: List[str],
    tolerance: float = 0.5,
    infer_bond_orders: bool = True  # FIX Bug #13: Enable bond order inference
) -> Chem.Mol:
    """
    Convert coordinates and elements to RDKit molecule.
    
    Uses distance-based bond inference with covalent radii.
    
    FIX Bug #13: Now infers proper bond orders (single/double/triple/aromatic)
    instead of hardcoding all bonds as SINGLE. This is critical for:
    - Aromatic systems (benzene rings)
    - Carbonyl groups (C=O)
    - Correct valency validation
    
    Args:
        coords: (N, 3) atom coordinates
        elements: List of N element symbols
        tolerance: Bond tolerance in Angstroms
        infer_bond_orders: If True, use RDKit to infer double/triple/aromatic bonds
    
    Returns:
        RDKit Mol object (may fail sanitization)
    """
    from rdkit.Geometry import Point3D
    from rdkit.Chem import AllChem
    
    # Create editable molecule
    mol = Chem.RWMol()
    
    # Add atoms with explicit hydrogens tracking
    for elem in elements:
        atom = Chem.Atom(elem)
        mol.AddAtom(atom)
    
    # Add conformer with coordinates
    conf = Chem.Conformer(len(coords))
    for i, coord in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*coord.tolist()))
    mol.AddConformer(conf)
    
    # Add bonds based on distance
    adj = compute_adjacency_matrix(coords, elements, tolerance)
    
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if adj[i, j]:
                mol.AddBond(i, j, Chem.BondType.SINGLE)  # Initial assignment
    
    mol = mol.GetMol()
    
    # FIX Bug #13: Infer proper bond orders
    if infer_bond_orders:
        try:
            # Method 1: Use RDKit's DetermineBonds (RDKit >= 2022.03)
            # This uses xyz2mol algorithm to infer bond orders from geometry
            from rdkit.Chem import rdDetermineBonds
            mol_copy = Chem.RWMol(mol)
            rdDetermineBonds.DetermineBonds(mol_copy, useHueckel=True)
            mol = mol_copy.GetMol()
        except (ImportError, AttributeError, TypeError):
            # FIX Medium #5: Added TypeError for intermediate RDKit versions
            # Method 2: Fallback - try to sanitize and assign from valence
            try:
                # Remove all bonds and let AssignBondOrdersFromTemplate work
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS)
                
                # Try to perceive aromaticity
                Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_MDL)
            except Exception:
                pass  # Keep single bonds as fallback
        except Exception as e:
            logger.debug(f"Bond order inference failed: {e}, keeping single bonds")
    
    return mol


def sanitize_mol(mol: Chem.Mol, relax: bool = False) -> Optional[Chem.Mol]:
    """
    Sanitize molecule with RDKit and optionally relax with MMFF94.
    
    Args:
        mol: RDKit molecule
        relax: Whether to run MMFF94 minimization
    
    Returns:
        Sanitized molecule or None if failed
    """
    try:
        Chem.SanitizeMol(mol)
        
        if relax:
            try:
                from rdkit.Chem import AllChem
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception as e:
                logger.warning(f"MMFF optimization failed: {e}")
        
        return mol
    except Exception as e:
        logger.debug(f"Sanitization failed: {e}")
        return None


def get_largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """
    Get the largest connected fragment from a molecule.
    
    Useful for post-processing generated molecules that may have
    disconnected fragments.
    """
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) == 1:
        return mol
    
    # Return fragment with most heavy atoms
    return max(frags, key=lambda f: f.GetNumHeavyAtoms())
