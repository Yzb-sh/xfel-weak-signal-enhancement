"""
Dataset module for DeepPhase-X.

Provides HDF5 data storage and PyTorch Dataset classes.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import torch
    from torch.utils.data import Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def save_to_hdf5(
    data: Dict[str, np.ndarray],
    filepath: str,
    compression: str = "gzip",
    compression_opts: int = 4
) -> None:
    """
    Save data dictionary to HDF5 file.
    
    Args:
        data: Dictionary mapping group/dataset names to numpy arrays
        filepath: Path to save the HDF5 file
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_opts: Compression level (1-9 for gzip)
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 operations")
    
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, "w") as f:
        for key, value in data.items():
            if compression:
                f.create_dataset(
                    key,
                    data=value,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=True
                )
            else:
                f.create_dataset(key, data=value)


def load_from_hdf5(filepath: str, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Load data from HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
        keys: Optional list of keys to load (loads all if None)
        
    Returns:
        Dictionary mapping dataset names to numpy arrays
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for HDF5 operations")
    
    data = {}
    
    with h5py.File(filepath, "r") as f:
        if keys is None:
            keys = list(f.keys())
        
        for key in keys:
            if key in f:
                data[key] = f[key][:]
    
    return data


class DiffractionDataset:
    """
    PyTorch Dataset for diffraction patterns.
    
    Supports dual-channel input (I, √I) for physics-guided U-Net.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        clean_key: str = "clean",
        noisy_key: str = "noisy_analytic",
        transform: Optional[callable] = None,
        dual_channel: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            hdf5_path: Path to HDF5 file
            clean_key: Key for clean patterns
            noisy_key: Key for noisy patterns
            transform: Optional transform to apply
            dual_channel: If True, return (I, √I) as input
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required")
        
        self.hdf5_path = hdf5_path
        self.clean_key = clean_key
        self.noisy_key = noisy_key
        self.transform = transform
        self.dual_channel = dual_channel
        
        # Get dataset length
        with h5py.File(hdf5_path, "r") as f:
            self.length = len(f[noisy_key])
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Get a sample.
        
        Returns:
            Tuple of (input, target) where:
            - input: (2, H, W) if dual_channel else (1, H, W)
            - target: (1, H, W) clean pattern
        """
        with h5py.File(self.hdf5_path, "r") as f:
            noisy = f[self.noisy_key][idx].astype(np.float32)
            clean = f[self.clean_key][idx].astype(np.float32)
        
        if self.transform:
            noisy = self.transform(noisy)
            clean = self.transform(clean)
        
        if self.dual_channel:
            # Create dual-channel input: (I, √I)
            intensity = noisy
            amplitude = np.sqrt(np.maximum(noisy, 0))
            input_data = np.stack([intensity, amplitude], axis=0)
        else:
            input_data = noisy[np.newaxis, ...]
        
        target = clean[np.newaxis, ...]
        
        if HAS_TORCH:
            import torch
            return torch.from_numpy(input_data), torch.from_numpy(target)
        
        return input_data, target
