"""
Intensity Normalizer - Normalize diffraction intensity before adding noise.

This module normalizes the diffraction intensity so that the sum equals 1.
This is CRITICAL for physically correct Poisson noise: after normalization,
the intensity represents the probability distribution of scattered photons,
and the exposure level (I_sc) has the clear physical meaning of total photon count.

IMPORTANT: This normalization MUST be applied BEFORE adding Poisson noise.
"""

import numpy as np
from typing import Optional

from config import EXP_CONFIG


class IntensityNormalizer:
    """
    Normalize diffraction intensity for physically correct noise simulation.

    The normalization ensures that:
    - Total intensity = 1 (probability distribution)
    - I_sc (exposure level) = total scattered photon count
    - expected_photons = I_normalized * I_sc
    """

    def __init__(self):
        """Initialize the intensity normalizer."""
        self.train_size = EXP_CONFIG['train_size']

    def normalize(self, I_clean: np.ndarray) -> np.ndarray:
        """
        Normalize intensity so that sum equals 1.

        Physical meaning:
        - I_clean_normalized represents the probability distribution
          of where scattered photons land on the detector
        - The sum being 1 ensures that when multiplied by I_sc (total photons),
          we get the correct expected photon count at each pixel

        Args:
            I_clean: Diffraction intensity pattern (585, 585), non-negative.

        Returns:
            I_normalized: Normalized intensity with sum = 1.
        """
        # Validate input
        if I_clean.shape != (self.train_size, self.train_size):
            raise ValueError(f"Expected shape ({self.train_size}, {self.train_size}), got {I_clean.shape}")

        # Calculate total intensity
        total = np.sum(I_clean)

        # Handle edge case of zero intensity
        if total < 1e-10:
            raise ValueError("Total intensity is too close to zero for normalization")

        # Normalize
        I_normalized = I_clean / total

        # Ensure result is float32
        I_normalized = I_normalized.astype(np.float32)

        return I_normalized

    def denormalize(
        self,
        I_normalized: np.ndarray,
        original_total: float
    ) -> np.ndarray:
        """
        Reverse the normalization (for reference/testing).

        Args:
            I_normalized: Normalized intensity with sum = 1.
            original_total: Original total intensity to restore.

        Returns:
            Denormalized intensity.
        """
        return (I_normalized * original_total).astype(np.float32)

    def validate_normalization(self, I_normalized: np.ndarray) -> dict:
        """
        Validate that normalization is correct.

        Args:
            I_normalized: Normalized intensity.

        Returns:
            Dictionary with validation results.
        """
        total = np.sum(I_normalized)
        is_valid = np.isclose(total, 1.0, rtol=1e-5)

        return {
            'is_valid': is_valid,
            'total': total,
            'expected_total': 1.0,
            'relative_error': abs(total - 1.0) / 1.0,
            'min_value': np.min(I_normalized),
            'max_value': np.max(I_normalized),
        }


def normalize_intensity(I_clean: np.ndarray) -> np.ndarray:
    """
    Convenience function to normalize diffraction intensity.

    Args:
        I_clean: Diffraction intensity pattern.

    Returns:
        Normalized intensity with sum = 1.
    """
    normalizer = IntensityNormalizer()
    return normalizer.normalize(I_clean)


if __name__ == "__main__":
    # Test the normalizer
    import matplotlib.pyplot as plt
    from pathlib import Path

    from bio_sample_generator import generate_bio_sample
    from diffraction_simulator import simulate_diffraction

    print("Testing IntensityNormalizer...")

    # Generate and simulate
    obj = generate_bio_sample(seed=42)
    I_clean = simulate_diffraction(obj)

    print(f"\nBefore normalization:")
    print(f"  Total intensity: {np.sum(I_clean):.2e}")
    print(f"  Min: {I_clean.min():.2e}, Max: {I_clean.max():.2e}")

    # Normalize
    normalizer = IntensityNormalizer()
    I_normalized = normalizer.normalize(I_clean)

    # Validate
    validation = normalizer.validate_normalization(I_normalized)
    print(f"\nAfter normalization:")
    print(f"  Total intensity: {validation['total']:.10f}")
    print(f"  Is valid: {validation['is_valid']}")
    print(f"  Min: {validation['min_value']:.2e}, Max: {validation['max_value']:.2e}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original (log scale)
    im0 = axes[0].imshow(np.log10(1 + I_clean), cmap='inferno')
    axes[0].set_title(f'Original Intensity\nTotal: {np.sum(I_clean):.2e}')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0])

    # Normalized (log scale)
    im1 = axes[1].imshow(np.log10(1 + I_normalized), cmap='inferno')
    axes[1].set_title(f'Normalized Intensity\nTotal: {np.sum(I_normalized):.10f}')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])

    # Difference (should be minimal, just scaling)
    diff = np.abs(np.log10(1 + I_clean) - np.log10(1 + I_normalized * np.sum(I_clean)))
    im2 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference (should be ~0)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    output_path = Path(__file__).parent / 'test_normalization.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nTest output saved to {output_path}")

    # Test physical meaning
    print("\n" + "=" * 60)
    print("Physical meaning demonstration:")
    print("=" * 60)

    I_sc = 1e6  # Total scattered photons
    expected_photons = I_normalized * I_sc
    print(f"If I_sc = {I_sc:.0e} photons:")
    print(f"  Expected total photons: {np.sum(expected_photons):.0f}")
    print(f"  Max expected at single pixel: {np.max(expected_photons):.2f}")
    print(f"  Min expected at single pixel: {np.min(expected_photons):.2e}")
