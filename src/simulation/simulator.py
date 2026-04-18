"""X-ray diffraction simulator for generating diffraction patterns from PDB structures."""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter

try:
    from Bio.PDB import PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.config_loader import SimulationConfig


# Atomic scattering factors (simplified - using number of electrons)
ATOMIC_ELECTRONS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
    'FE': 26, 'ZN': 30, 'MG': 12, 'CA': 20, 'MN': 25,
    'CU': 29, 'SE': 34, 'CL': 17, 'NA': 11, 'K': 19,
}


class XRaySimulator:
    """
    X-ray diffraction pattern simulator.
    
    Generates 2D diffraction patterns from PDB structures using FFT.
    Supports oversampling for phase retrieval requirements.
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        Initialize the simulator.
        
        Args:
            config: Simulation configuration. Uses defaults if None.
        """
        self.config = config or SimulationConfig()
        if HAS_BIOPYTHON:
            self.parser = PDBParser(QUIET=True)
        else:
            self.parser = None
    
    def load_pdb_to_density(
        self,
        pdb_path: Union[str, Path],
        grid_size: Optional[int] = None,
        resolution: Optional[float] = None,
        projection_axis: int = 2
    ) -> np.ndarray:
        """
        Convert PDB structure to 2D electron density projection.
        
        Args:
            pdb_path: Path to PDB file.
            grid_size: Output grid size. Uses config if None.
            resolution: Angstroms per pixel. Uses config if None.
            projection_axis: Axis to project along (0=x, 1=y, 2=z).
        
        Returns:
            2D electron density map (grid_size x grid_size).
        
        Raises:
            ImportError: If BioPython is not installed.
            FileNotFoundError: If PDB file doesn't exist.
            ValueError: If PDB file has no atoms.
        """
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython is required for PDB parsing. Install with: pip install biopython")
        
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        grid_size = grid_size or self.config.grid_size
        resolution = resolution or self.config.resolution
        
        # Parse PDB structure
        structure = self.parser.get_structure("protein", str(pdb_path))
        
        # Extract atom coordinates and weights
        coords = []
        weights = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coord = atom.get_coord()
                        element = atom.element.upper().strip()
                        weight = ATOMIC_ELECTRONS.get(element, 6)  # Default to carbon
                        coords.append(coord)
                        weights.append(weight)
        
        if len(coords) == 0:
            raise ValueError(f"No atoms found in PDB file: {pdb_path}")
        
        coords = np.array(coords)
        weights = np.array(weights)
        
        # Center coordinates
        coords = coords - coords.mean(axis=0)
        
        # Project to 2D
        axes = [0, 1, 2]
        axes.remove(projection_axis)
        coords_2d = coords[:, axes]
        
        # Scale to grid
        max_extent = np.abs(coords_2d).max()
        scale = (grid_size / 2 - 2) / max_extent if max_extent > 0 else 1.0
        coords_scaled = coords_2d * scale + grid_size / 2
        
        # Create density map
        density = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        for (x, y), w in zip(coords_scaled, weights):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                density[iy, ix] += w
        
        # Apply Gaussian blur to simulate electron cloud
        sigma = 1.5 / resolution  # ~1.5 Angstrom blur
        density = gaussian_filter(density, sigma=sigma)
        
        # Normalize
        if density.max() > 0:
            density = density / density.max()
        
        return density

    def generate_diffraction(
        self,
        density: np.ndarray,
        oversampling_ratio: Optional[float] = None,
        apply_shift: bool = True
    ) -> np.ndarray:
        """
        Generate diffraction pattern from electron density using FFT.
        
        Applies oversampling (zero-padding) to satisfy phase retrieval
        oversampling condition (ratio >= 2 for 2D).
        
        Args:
            density: 2D electron density map.
            oversampling_ratio: Oversampling ratio. Uses config if None.
            apply_shift: Whether to shift zero-frequency to center.
        
        Returns:
            2D diffraction intensity pattern.
        """
        oversampling_ratio = oversampling_ratio or self.config.oversampling_ratio
        
        # Validate input
        if density.ndim != 2:
            raise ValueError(f"Expected 2D density, got {density.ndim}D")
        
        original_size = density.shape[0]
        
        # Apply oversampling via zero-padding
        if oversampling_ratio > 1.0:
            padded_size = int(original_size * oversampling_ratio)
            padded = np.zeros((padded_size, padded_size), dtype=density.dtype)
            offset = (padded_size - original_size) // 2
            padded[offset:offset + original_size, offset:offset + original_size] = density
        else:
            padded = density.copy()
        
        # Compute FFT
        f_map = fft2(padded)
        
        # Shift zero-frequency to center
        if apply_shift:
            f_map = fftshift(f_map)
        
        # Compute intensity (|F|^2)
        intensity = np.abs(f_map) ** 2
        
        return intensity
    
    def apply_beam_stop(
        self,
        pattern: np.ndarray,
        radius: Optional[int] = None,
        fill_value: float = 0.0
    ) -> np.ndarray:
        """
        Apply beam stop mask to diffraction pattern.
        
        Args:
            pattern: 2D diffraction pattern.
            radius: Beam stop radius in pixels. Uses config if None.
            fill_value: Value to fill masked region.
        
        Returns:
            Masked diffraction pattern.
        """
        radius = radius or self.config.beam_stop_radius
        
        result = pattern.copy()
        h, w = pattern.shape
        cy, cx = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        result[mask] = fill_value
        
        return result
    
    def generate_from_density(
        self,
        density: np.ndarray,
        apply_beam_stop: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate clean diffraction pattern from density.
        
        Convenience method combining diffraction generation and beam stop.
        
        Args:
            density: 2D electron density map.
            apply_beam_stop: Whether to apply beam stop mask.
        
        Returns:
            Tuple of (diffraction_pattern, beam_stop_mask).
        """
        pattern = self.generate_diffraction(density)
        
        if apply_beam_stop:
            h, w = pattern.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= self.config.beam_stop_radius ** 2
            pattern = self.apply_beam_stop(pattern)
        else:
            mask = np.zeros(pattern.shape, dtype=bool)
        
        return pattern, mask
    
    def create_synthetic_density(
        self,
        grid_size: Optional[int] = None,
        shape: str = "circle",
        **kwargs
    ) -> np.ndarray:
        """
        Create synthetic electron density for testing.
        
        Args:
            grid_size: Output grid size.
            shape: Shape type ("circle", "ellipse", "gaussian", "random").
            **kwargs: Shape-specific parameters.
        
        Returns:
            2D synthetic density map.
        """
        grid_size = grid_size or self.config.grid_size
        density = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        y, x = np.ogrid[:grid_size, :grid_size]
        cy, cx = grid_size // 2, grid_size // 2
        
        if shape == "circle":
            radius = kwargs.get("radius", grid_size // 4)
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
            density[mask] = 1.0
            
        elif shape == "ellipse":
            a = kwargs.get("a", grid_size // 3)
            b = kwargs.get("b", grid_size // 5)
            mask = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
            density[mask] = 1.0
            
        elif shape == "gaussian":
            sigma = kwargs.get("sigma", grid_size // 8)
            density = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
            
        elif shape == "random":
            # Random blobs
            n_blobs = kwargs.get("n_blobs", 5)
            blob_size = kwargs.get("blob_size", grid_size // 10)
            rng = np.random.default_rng(kwargs.get("seed", 42))
            
            for _ in range(n_blobs):
                bx = rng.integers(blob_size, grid_size - blob_size)
                by = rng.integers(blob_size, grid_size - blob_size)
                br = rng.integers(blob_size // 2, blob_size)
                mask = (x - bx) ** 2 + (y - by) ** 2 <= br ** 2
                density[mask] = rng.uniform(0.5, 1.0)
            
            density = gaussian_filter(density, sigma=2)
        
        # Normalize
        if density.max() > 0:
            density = density / density.max()
        
        return density
