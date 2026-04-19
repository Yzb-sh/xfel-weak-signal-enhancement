"""
Diffraction Simulator - FFT-based diffraction pattern generation.

This module implements a lightweight FFT-based diffraction simulator
for 2D projection density maps. It includes physical scale calibration.

Reference: Only the core FFT logic is referenced from src/physics/simulator.py
All PDB parsing, oversampling, and zero-padding logic is removed.
"""

import numpy as np
from typing import Dict, Any, Optional

from .bio_config import EXP_CONFIG
from .backend import get_xp, to_gpu, to_cpu


class DiffractionSimulator:
    """
    Lightweight FFT-based diffraction simulator.

    Computes diffraction intensity from 2D projection density maps
    using Fourier transform with physical scale calibration.
    """

    def __init__(self, use_gpu=False):
        """Initialize the diffraction simulator.

        Args:
            use_gpu: If True, use CuPy/CUDA for FFT computation.
        """
        self.train_size = EXP_CONFIG['train_size']
        self._physical_params = None
        self.use_gpu = use_gpu

        if use_gpu:
            import cupy as xp
            from cupy.fft import fft2, fftshift
        else:
            import numpy as xp
            from numpy.fft import fft2, fftshift

        self.xp = xp
        self.fft2 = fft2
        self.fftshift = fftshift

    def simulate(self, obj: np.ndarray) -> np.ndarray:
        """
        Compute diffraction intensity from object density.

        Args:
            obj: 2D projection density map (585, 585).

        Returns:
            I_clean: Diffraction intensity (585, 585), non-negative float32.
        """
        # Validate input
        if obj.shape != (self.train_size, self.train_size):
            raise ValueError(f"Expected shape ({self.train_size}, {self.train_size}), got {obj.shape}")

        xp = self.xp
        if self.use_gpu:
            obj = xp.asarray(obj)

        # Compute Fourier transform
        # A = FFT2(obj) gives complex amplitude
        A = self.fft2(obj)

        # Shift zero frequency to center
        A = self.fftshift(A)

        # Compute intensity (|A|^2)
        I_clean = xp.abs(A) ** 2

        # Ensure non-negative (should always be true for |A|^2)
        I_clean = xp.maximum(I_clean, 0).astype(xp.float32)

        if self.use_gpu:
            I_clean = I_clean.get()

        return I_clean

    def compute_physical_scale(self) -> Dict[str, Any]:
        """
        Compute and return physical scale parameters.

        The physical scale relates the FFT grid to real-space dimensions.

        Returns:
            Dictionary with physical scale parameters.
        """
        if self._physical_params is not None:
            return self._physical_params

        train_size = EXP_CONFIG['train_size']
        dx_real = EXP_CONFIG['detector_target_pixel_size']

        # Real space pixel size (meters)
        # This is the physical size represented by each pixel in the object

        # Reciprocal space pixel size (1/meters)
        # dq = 1 / (N * dx) where N is the grid size
        dq = 1.0 / (train_size * dx_real)

        # Maximum spatial frequency (1/meters)
        # q_max = dq * (N/2) - maximum frequency that can be represented
        q_max = dq * (train_size // 2)

        self._physical_params = {
            'train_size': train_size,
            'dx_real_m': dx_real,
            'dx_real_um': dx_real * 1e6,
            'dq_inv_m': dq,
            'dq_nm_inv': dq * 1e-9,
            'q_max_inv_m': q_max,
            'q_max_nm_inv': q_max * 1e-9,
        }

        return self._physical_params

    def get_q_coordinates(self) -> np.ndarray:
        """
        Get q (spatial frequency) coordinates for the diffraction pattern.

        Returns:
            2D array of q values (1/m) corresponding to each pixel.
        """
        params = self.compute_physical_scale()
        dq = params['dq_inv_m']

        # Create coordinate arrays
        q_1d = np.fft.fftshift(np.fft.fftfreq(self.train_size, d=params['dx_real_m']))
        q_2d = np.sqrt(np.outer(q_1d, q_1d) ** 2)

        return q_2d

    def get_radial_profile(self, intensity: np.ndarray) -> tuple:
        """
        Compute radial average of the diffraction pattern.

        Args:
            intensity: Diffraction intensity pattern.

        Returns:
            Tuple of (q_values, radial_profile) where q_values are in 1/m.
        """
        params = self.compute_physical_scale()
        dq = params['dq_inv_m']

        # Create radial distance array
        center = self.train_size // 2
        y, x = np.ogrid[:self.train_size, :self.train_size]
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2).astype(int)

        # Radial average
        radial_sum = np.bincount(r.ravel(), intensity.ravel())
        radial_count = np.bincount(r.ravel())
        radial_profile = radial_sum / (radial_count + 1e-10)

        # q values corresponding to each radius
        q_values = np.arange(len(radial_profile)) * dq

        return q_values, radial_profile


def simulate_diffraction(obj: np.ndarray) -> np.ndarray:
    """
    Convenience function to simulate diffraction from an object.

    Args:
        obj: 2D projection density map (585, 585).

    Returns:
        Diffraction intensity pattern (585, 585).
    """
    simulator = DiffractionSimulator()
    return simulator.simulate(obj)


if __name__ == "__main__":
    # Test the simulator
    import matplotlib.pyplot as plt
    from pathlib import Path

    from .bio_sample_generator import generate_bio_sample
    from .bio_config import print_physical_report

    print("Testing DiffractionSimulator...")
    print_physical_report()

    # Generate a sample
    obj = generate_bio_sample(seed=42)

    # Simulate diffraction
    simulator = DiffractionSimulator()
    I_clean = simulator.simulate(obj)

    # Get physical parameters
    params = simulator.compute_physical_scale()
    print(f"\nPhysical scale parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Get radial profile
    q_values, radial_profile = simulator.get_radial_profile(I_clean)

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Object
    axes[0].imshow(obj, cmap='gray')
    axes[0].set_title('Object (Real Space)')
    axes[0].axis('off')

    # Diffraction pattern (log scale)
    I_log = np.log10(1 + I_clean)
    im = axes[1].imshow(I_log, cmap='inferno')
    axes[1].set_title('Diffraction Intensity (log scale)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='log10(1 + I)')

    # Radial profile
    valid = radial_profile > 0
    axes[2].semilogy(q_values[valid] * 1e-9, radial_profile[valid])
    axes[2].set_xlabel('q (nm⁻¹)')
    axes[2].set_ylabel('Intensity (arb. units)')
    axes[2].set_title('Radial Profile')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(__file__).parent / 'test_diffraction.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nDiffraction pattern shape: {I_clean.shape}")
    print(f"Intensity range: [{I_clean.min():.2e}, {I_clean.max():.2e}]")
    print(f"Test output saved to {output_path}")
