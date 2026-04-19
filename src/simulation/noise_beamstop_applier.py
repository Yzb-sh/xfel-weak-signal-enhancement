"""
Noise and Beamstop Applier - Add physical noise and beam stop mask.

This module:
1. Applies Poisson + Gaussian noise (reusing src/physics/noise_model.py)
2. Adds bad pixels and bad lines
3. Loads and applies beam stop mask from .mat file
4. Creates gradient transition at beam stop edges
"""

import numpy as np
from scipy.io import loadmat
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from .noise_model import AnalyticNoiseModel
from .bio_config import EXP_CONFIG, BASE_DIR
from .backend import get_xp, get_ndimage, to_gpu, to_cpu


class NoiseAndBeamstopApplier:
    """
    Apply physical noise and beam stop mask to diffraction patterns.

    Noise model includes:
    - Poisson shot noise (photon counting statistics)
    - Gaussian readout noise
    - Bad pixels
    - Bad lines

    Beam stop:
    - Loaded from .mat file
    - Gradient transition at edges
    """

    def __init__(self, seed: Optional[int] = None, use_gpu: bool = False):
        """
        Initialize the noise and beamstop applier.

        Args:
            seed: Random seed for reproducibility.
            use_gpu: If True, use CuPy/CUDA for beamstop operations.
        """
        self.rng = np.random.default_rng(seed)
        self.noise_model = AnalyticNoiseModel(seed=seed)
        self.grid_size = EXP_CONFIG['train_size']
        self._beamstop_mask = None
        self._beamstop_gradient = None
        self.use_gpu = use_gpu

        self.xp = get_xp(use_gpu)
        self.ndimage = get_ndimage(use_gpu)

    def apply(
        self,
        I_normalized: np.ndarray,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
        """
        Apply noise and beam stop to normalized intensity.

        Args:
            I_normalized: Normalized intensity (sum = 1).
            seed: Optional seed to override initialization seed.

        Returns:
            Tuple of:
                - I_noisy: Noisy intensity with beam stop applied
                - combined_mask: Boolean mask (beamstop | bad pixels | bad lines)
                - I_sc: Total photon count used for this sample
                - metadata: Dictionary with noise parameters
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.noise_model = AnalyticNoiseModel(seed=seed)

        metadata = {}

        # Step 1: Apply Poisson + Gaussian noise
        # Log-uniform sampling: ensures uniform coverage across orders of magnitude
        log_low, log_high = np.log10(EXP_CONFIG['poisson_I_sc_range'][0]), np.log10(EXP_CONFIG['poisson_I_sc_range'][1])
        I_sc = 10 ** self.rng.uniform(log_low, log_high)
        I_noisy = self.noise_model.add_poisson_gaussian(
            I_normalized,
            exposure_level=I_sc,
            readout_noise=EXP_CONFIG['gaussian_read_noise_sigma'],
            clip_negative=False  # Don't clip yet, bad pixels might make it negative
        )
        metadata['I_sc'] = I_sc
        metadata['readout_noise_sigma'] = EXP_CONFIG['gaussian_read_noise_sigma']

        # Step 2: Add bad pixels
        I_noisy, bad_pixel_mask = self._add_bad_pixels(I_noisy)
        metadata['n_bad_pixels'] = np.sum(bad_pixel_mask)

        # Step 3: Add bad lines
        I_noisy, bad_line_mask = self._add_bad_lines(I_noisy)
        metadata['n_bad_line_pixels'] = np.sum(bad_line_mask)

        # Step 4: Clip negative values
        I_noisy = np.maximum(I_noisy, 0).astype(np.float32)

        # Step 5: Apply beam stop
        I_noisy, beamstop_mask = self._apply_beamstop(I_noisy)
        metadata['beamstop_masked_pixels'] = np.sum(beamstop_mask)

        # Combine all masks
        combined_mask = beamstop_mask | bad_pixel_mask | bad_line_mask

        return I_noisy, combined_mask, I_sc, metadata

    def apply_noise_only(
        self,
        I_normalized: np.ndarray,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, float, Dict[str, Any], np.ndarray]:
        """
        Apply Poisson + Gaussian noise only, without beam stop mask.

        This method is used when you want to add noise before applying
        random detector masks (RandomMaskApplier), following the correct
        physical pipeline: Normalize -> Noise -> Defects -> Beamstop.

        Args:
            I_normalized: Normalized intensity (sum = 1).
            seed: Optional seed to override initialization seed.

        Returns:
            Tuple of:
                - I_noisy: Noisy intensity without beam stop
                - I_sc: Total photon count used for this sample
                - metadata: Dictionary with noise parameters
                - defect_mask: Boolean mask of bad pixels and bad lines
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.noise_model = AnalyticNoiseModel(seed=seed)

        metadata = {}

        # Step 1: Apply Poisson + Gaussian noise
        # Log-uniform sampling: ensures uniform coverage across orders of magnitude
        log_low, log_high = np.log10(EXP_CONFIG['poisson_I_sc_range'][0]), np.log10(EXP_CONFIG['poisson_I_sc_range'][1])
        I_sc = 10 ** self.rng.uniform(log_low, log_high)
        I_noisy = self.noise_model.add_poisson_gaussian(
            I_normalized,
            exposure_level=I_sc,
            readout_noise=EXP_CONFIG['gaussian_read_noise_sigma'],
            clip_negative=False  # Don't clip yet, bad pixels might make it negative
        )
        metadata['I_sc'] = I_sc
        metadata['readout_noise_sigma'] = EXP_CONFIG['gaussian_read_noise_sigma']

        # Step 2: Add bad pixels
        I_noisy, bad_pixel_mask = self._add_bad_pixels(I_noisy)
        metadata['n_bad_pixels'] = np.sum(bad_pixel_mask)

        # Step 3: Add bad lines
        I_noisy, bad_line_mask = self._add_bad_lines(I_noisy)
        metadata['n_bad_line_pixels'] = np.sum(bad_line_mask)

        # Step 4: Clip negative values
        I_noisy = np.maximum(I_noisy, 0).astype(np.float32)

        # Combine defect mask (bad pixels + bad lines)
        defect_mask = bad_pixel_mask | bad_line_mask

        return I_noisy, I_sc, metadata, defect_mask

    def apply_beamstop_only(
        self,
        I_noisy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply beam stop mask only, without adding noise.

        This method is used when you want to apply the beam stop mask
        after noise and random detector masks have been added, following
        the correct physical pipeline: Normalize -> Noise -> Defects -> Beamstop.

        Args:
            I_noisy: Noisy intensity pattern.

        Returns:
            Tuple of:
                - I_masked: Intensity with beam stop applied
                - beamstop_mask: Boolean mask where True = blocked
                - metadata: Dictionary with beamstop information
        """
        I_masked, beamstop_mask = self._apply_beamstop(I_noisy)
        metadata = {
            'beamstop_masked_pixels': int(np.sum(beamstop_mask))
        }
        return I_masked, beamstop_mask, metadata

    def _add_bad_pixels(
        self,
        I: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random bad pixels.

        Args:
            I: Input intensity.

        Returns:
            Tuple of (intensity with bad pixels, bad pixel mask).
        """
        bad_pixel_mask = self.rng.random((self.grid_size, self.grid_size)) < EXP_CONFIG['bad_pixel_prob']

        I_with_bad = I.copy()

        # Bad pixels can be either dead (0) or hot (very high)
        for y, x in zip(*np.where(bad_pixel_mask)):
            if self.rng.random() > 0.5:
                # Dead pixel
                I_with_bad[y, x] = 0
            else:
                # Hot pixel
                I_with_bad[y, x] = self.rng.uniform(1e4, 1e5)

        return I_with_bad, bad_pixel_mask

    def _add_bad_lines(
        self,
        I: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add random bad lines (row or column defects).

        Args:
            I: Input intensity.

        Returns:
            Tuple of (intensity with bad lines, bad line mask).
        """
        bad_line_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        I_with_bad = I.copy()

        # Check if we should add bad lines
        if self.rng.random() < EXP_CONFIG['bad_line_prob']:
            # Decide: row or column
            if self.rng.random() > 0.5:
                # Bad row
                bad_row = self.rng.integers(0, self.grid_size)
                bad_line_mask[bad_row, :] = True
            else:
                # Bad column
                bad_col = self.rng.integers(0, self.grid_size)
                bad_line_mask[:, bad_col] = True

            # Apply bad line (set to 0)
            I_with_bad[bad_line_mask] = 0

        return I_with_bad, bad_line_mask

    def load_beamstop_mask(self) -> np.ndarray:
        """
        Load beam stop mask from .mat file.

        Returns:
            Binary beam stop mask where 1 = blocked, 0 = clear.
        """
        if self._beamstop_mask is not None:
            return self._beamstop_mask

        mask_file = EXP_CONFIG['beamstop_mask_file']

        if not mask_file.exists():
            raise FileNotFoundError(f"Beam stop mask file not found: {mask_file}")

        # Load .mat file
        mat_data = loadmat(str(mask_file))

        # Try common variable names
        for key in ['mask', 'beamstop', 'beamstop_mask', 'M']:
            if key in mat_data:
                self._beamstop_mask = mat_data[key].astype(np.float32)
                break
        else:
            # Use the first array-like variable
            for key, value in mat_data.items():
                if not key.startswith('_') and isinstance(value, np.ndarray):
                    if value.shape == (self.grid_size, self.grid_size):
                        self._beamstop_mask = value.astype(np.float32)
                        break
            else:
                raise KeyError(f"Could not find mask data in {mask_file}. Available keys: {list(mat_data.keys())}")

        # Ensure binary
        self._beamstop_mask = (self._beamstop_mask > 0.5).astype(np.float32)

        return self._beamstop_mask

    def _create_beamstop_gradient(
        self,
        base_mask,
        gradient_width: Optional[int] = None
    ):
        """
        Create gradient transition at beam stop edges.

        Args:
            base_mask: Binary beam stop mask (numpy or cupy array).
            gradient_width: Width of gradient in pixels. Random if None.

        Returns:
            Gradient mask (0 = fully blocked, 1 = fully clear).
        """
        xp = self.xp
        ndimage = self.ndimage

        # Decide whether to apply gradient or use hard edge
        if gradient_width is None:
            if self.rng.random() < EXP_CONFIG.get('beamstop_no_gradient_prob', 0.0):
                # No gradient — hard edge (binary mask)
                gradient_mask = (1 - base_mask).astype(xp.float32)
                return gradient_mask
            gradient_width = self.rng.integers(*EXP_CONFIG['beamstop_gradient_width_range'])

        # Distance from blocked region
        distance = ndimage.distance_transform_edt(1 - base_mask)

        # Create gradient
        gradient_mask = xp.clip(distance / gradient_width, 0, 1).astype(xp.float32)

        # Blocked region is 0
        gradient_mask[base_mask > 0.5] = 0

        return gradient_mask

    def _apply_beamstop(
        self,
        I: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply beam stop mask with gradient transition.

        Args:
            I: Input intensity.

        Returns:
            Tuple of (masked intensity, final beamstop mask).
        """
        xp = self.xp

        # Load base mask (always from CPU .mat file)
        base_mask_cpu = self.load_beamstop_mask()

        # Transfer to GPU if needed
        if self.use_gpu:
            base_mask = xp.asarray(base_mask_cpu)
        else:
            base_mask = base_mask_cpu

        # Create gradient
        gradient_mask = self._create_beamstop_gradient(base_mask)

        # Transfer I to GPU if needed
        if self.use_gpu:
            I_device = xp.asarray(I)
        else:
            I_device = I

        # Apply gradient
        I_masked = I_device * gradient_mask

        # Final binary mask for output
        final_mask = base_mask > 0.5

        # Transfer back to CPU
        if self.use_gpu:
            I_masked = I_masked.get()
            final_mask = final_mask.get()

        return I_masked.astype(np.float32), final_mask


def apply_noise_and_beamstop(
    I_normalized: np.ndarray,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convenience function to apply noise and beam stop.

    Args:
        I_normalized: Normalized intensity (sum = 1).
        seed: Random seed.

    Returns:
        Tuple of (noisy_intensity, combined_mask, I_sc).
    """
    applier = NoiseAndBeamstopApplier(seed=seed)
    I_noisy, combined_mask, I_sc, _ = applier.apply(I_normalized)
    return I_noisy, combined_mask, I_sc


if __name__ == "__main__":
    # Test the noise and beamstop applier
    import matplotlib.pyplot as plt

    from .bio_sample_generator import generate_bio_sample
    from .bio_diffraction_simulator import simulate_diffraction
    from .intensity_normalizer import normalize_intensity

    print("Testing NoiseAndBeamstopApplier...")

    # Generate sample
    obj = generate_bio_sample(seed=42)
    I_clean = simulate_diffraction(obj)
    I_norm = normalize_intensity(I_clean)

    print(f"Normalized intensity sum: {np.sum(I_norm):.10f}")

    # Apply noise and beamstop
    applier = NoiseAndBeamstopApplier(seed=42)
    I_noisy, beamstop_mask, I_sc, metadata = applier.apply(I_norm)

    print(f"\nNoise parameters:")
    print(f"  I_sc (total photons): {I_sc:.2e}")
    print(f"  Bad pixels: {metadata['n_bad_pixels']}")
    print(f"  Bad line pixels: {metadata['n_bad_line_pixels']}")
    print(f"  Beamstop pixels: {metadata['beamstop_masked_pixels']}")

    # Validate Poisson statistics (in non-masked region)
    valid_region = ~beamstop_mask
    if np.any(valid_region):
        var_valid = np.var(I_noisy[valid_region])
        mean_valid = np.mean(I_noisy[valid_region])
        ratio = var_valid / (mean_valid + 1e-10)
        print(f"\nPoisson statistics (non-beamstop region):")
        print(f"  Variance: {var_valid:.2f}")
        print(f"  Mean: {mean_valid:.2f}")
        print(f"  Var/Mean ratio: {ratio:.3f} (expected ~1 for pure Poisson)")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original object
    axes[0, 0].imshow(obj, cmap='gray')
    axes[0, 0].set_title('Object (Real Space)')
    axes[0, 0].axis('off')

    # Clean diffraction
    im = axes[0, 1].imshow(np.log10(1 + I_clean), cmap='inferno')
    axes[0, 1].set_title('Clean Diffraction (log)')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1])

    # Noisy diffraction
    im = axes[0, 2].imshow(np.log10(1 + I_noisy), cmap='inferno')
    axes[0, 2].set_title(f'Noisy Diffraction (log)\nI_sc={I_sc:.1e}')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])

    # Beamstop mask
    axes[1, 0].imshow(beamstop_mask.astype(float), cmap='gray')
    axes[1, 0].set_title('Beamstop Mask')
    axes[1, 0].axis('off')

    # Difference (noise visualization)
    diff = I_noisy - I_clean * I_sc * np.sum(I_norm) / np.sum(I_clean)
    valid = ~beamstop_mask
    diff_valid = diff.copy()
    diff_valid[beamstop_mask] = 0
    im = axes[1, 1].imshow(diff_valid, cmap='RdBu_r', vmin=-np.percentile(np.abs(diff_valid), 99), vmax=np.percentile(np.abs(diff_valid), 99))
    axes[1, 1].set_title('Noise (Noisy - Expected)')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])

    # Histogram of noisy values
    axes[1, 2].hist(I_noisy[valid].flatten(), bins=100, alpha=0.7, log=True)
    axes[1, 2].set_xlabel('Intensity')
    axes[1, 2].set_ylabel('Count (log)')
    axes[1, 2].set_title('Intensity Histogram (non-beamstop)')
    axes[1, 2].axvline(np.mean(I_noisy[valid]), color='r', linestyle='--', label=f'Mean: {np.mean(I_noisy[valid]):.1f}')
    axes[1, 2].legend()

    plt.tight_layout()
    output_path = BASE_DIR / 'test_noise_beamstop.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nTest output saved to {output_path}")
