"""
Random Mask Applier - Simulate detector defects with random masks.

This module simulates detector defects by applying random binary masks.
This is DIFFERENT from the beam stop mask which is loaded from a .mat file.

Random masks simulate:
- Bad pixel clusters
- Dead regions
- Detector artifacts
"""

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from typing import Optional, Tuple

from .bio_config import EXP_CONFIG, RANDOM_MASK_CONFIG


class RandomMaskApplier:
    """
    Apply random masks to simulate detector defects.

    With configurable probability, applies random shapes (circles, rectangles,
    irregular blobs) that represent detector defects.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the random mask applier.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.grid_size = EXP_CONFIG['train_size']

    def apply(
        self,
        I_clean: np.ndarray,
        prob: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random detector defect mask with given probability.

        Args:
            I_clean: Input intensity pattern (585, 585).
            prob: Probability to apply mask. Uses config if None.
            seed: Optional seed to override initialization seed.

        Returns:
            Tuple of:
                - I_masked: Intensity with masked regions set to 0
                - mask_record: Boolean mask where True = masked region
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if prob is None:
            prob = RANDOM_MASK_CONFIG['apply_probability']

        # Initialize empty mask
        mask_record = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Decide whether to apply mask
        if self.rng.random() > prob:
            # No mask applied
            return I_clean.copy().astype(np.float32), mask_record

        # Generate combined random mask
        mask_record = self._generate_combined_mask()

        # Apply mask (set masked regions to 0)
        I_masked = I_clean.copy()
        I_masked[mask_record] = 0.0
        I_masked = I_masked.astype(np.float32)

        return I_masked, mask_record

    def _generate_combined_mask(self) -> np.ndarray:
        """
        Generate a combined mask from multiple random shapes.

        Returns:
            Boolean mask where True = masked region.
        """
        n_shapes = self.rng.integers(*RANDOM_MASK_CONFIG['n_shapes_range'])
        shape_types = RANDOM_MASK_CONFIG['shape_types']

        combined_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        for _ in range(n_shapes):
            shape_type = self.rng.choice(shape_types)

            if shape_type == 'circle':
                mask = self._generate_circle_mask()
            elif shape_type == 'rectangle':
                mask = self._generate_rectangle_mask()
            elif shape_type == 'irregular':
                mask = self._generate_irregular_mask()
            else:
                mask = self._generate_circle_mask()

            combined_mask = combined_mask | mask

        return combined_mask

    def _generate_circle_mask(self) -> np.ndarray:
        """
        Generate a random circular mask.

        Returns:
            Boolean mask with circular shape.
        """
        radius = self.rng.integers(*RANDOM_MASK_CONFIG['circle_radius_range'])

        # Random center (can be partially outside image)
        cy = self.rng.integers(-radius, self.grid_size + radius)
        cx = self.rng.integers(-radius, self.grid_size + radius)

        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        mask = ((y - cy) ** 2 + (x - cx) ** 2) <= radius ** 2

        return mask

    def _generate_rectangle_mask(self) -> np.ndarray:
        """
        Generate a random rectangular mask.

        Returns:
            Boolean mask with rectangular shape.
        """
        size = self.rng.integers(*RANDOM_MASK_CONFIG['rect_size_range'])

        # Random position
        y_start = self.rng.integers(-size // 2, self.grid_size - size // 2)
        x_start = self.rng.integers(-size // 2, self.grid_size - size // 2)

        # Random aspect ratio
        aspect = self.rng.uniform(0.5, 2.0)
        height = size
        width = int(size * aspect)

        # Random rotation
        angle = self.rng.uniform(0, 90)

        # Create rotated rectangle
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Simple rotation using coordinates
        center_y = y_start + height // 2
        center_x = x_start + width // 2

        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        y_coords, x_coords = np.mgrid[:self.grid_size, :self.grid_size]
        y_rel = y_coords - center_y
        x_rel = x_coords - center_x

        # Rotate coordinates
        y_rot = x_rel * sin_a + y_rel * cos_a
        x_rot = x_rel * cos_a - y_rel * sin_a

        # Check if in rectangle
        mask = (np.abs(x_rot) <= width // 2) & (np.abs(y_rot) <= height // 2)

        return mask

    def _generate_irregular_mask(self) -> np.ndarray:
        """
        Generate an irregular blob-shaped mask.

        Returns:
            Boolean mask with irregular shape.
        """
        n_points = self.rng.integers(*RANDOM_MASK_CONFIG['irregular_points_range'])

        # Random center
        center_y = self.rng.integers(50, self.grid_size - 50)
        center_x = self.rng.integers(50, self.grid_size - 50)

        # Generate random points around center
        points = []
        base_radius = self.rng.integers(10, 30)

        for _ in range(n_points):
            angle = self.rng.uniform(0, 2 * np.pi)
            r = base_radius * self.rng.uniform(0.5, 1.5)
            py = center_y + r * np.sin(angle)
            px = center_x + r * np.cos(angle)
            points.append((int(py), int(px)))

        # Create convex hull-like mask by filling the polygon
        mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        if len(points) >= 3:
            # Sort points by angle from centroid
            centroid_y = np.mean([p[0] for p in points])
            centroid_x = np.mean([p[1] for p in points])
            angles = [np.arctan2(p[0] - centroid_y, p[1] - centroid_x) for p in points]
            sorted_points = [p for _, p in sorted(zip(angles, points))]

            # Fill polygon
            from skimage.draw import polygon as draw_polygon
            try:
                rr, cc = draw_polygon(
                    [p[0] for p in sorted_points],
                    [p[1] for p in sorted_points],
                    shape=(self.grid_size, self.grid_size)
                )
                mask[rr, cc] = True
            except ImportError:
                # Fallback if skimage not available: use simple convex hull approximation
                y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
                for py, px in points:
                    dist = np.sqrt((y_coords - py) ** 2 + (x_coords - px) ** 2)
                    mask |= (dist < base_radius * 0.5)

        # Add some irregularity by dilation/erosion
        if self.rng.random() > 0.5:
            mask = binary_dilation(mask, iterations=1)
        if self.rng.random() > 0.5:
            mask = binary_erosion(mask, iterations=1)

        return mask


def apply_random_mask(
    I_clean: np.ndarray,
    prob: float = 0.5,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to apply random mask.

    Args:
        I_clean: Input intensity pattern.
        prob: Probability to apply mask.
        seed: Random seed.

    Returns:
        Tuple of (masked_intensity, mask_record).
    """
    applier = RandomMaskApplier(seed=seed)
    return applier.apply(I_clean, prob=prob)


if __name__ == "__main__":
    # Test the random mask applier
    import matplotlib.pyplot as plt
    from pathlib import Path

    from .bio_sample_generator import generate_bio_sample
    from .bio_diffraction_simulator import simulate_diffraction
    from .intensity_normalizer import normalize_intensity

    print("Testing RandomMaskApplier...")

    # Generate sample
    obj = generate_bio_sample(seed=42)
    I_clean = simulate_diffraction(obj)
    I_norm = normalize_intensity(I_clean)

    # Test with different probabilities
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    applier = RandomMaskApplier(seed=42)

    for i, ax_row in enumerate(axes):
        for j, ax in enumerate(ax_row):
            I_masked, mask = applier.apply(I_norm, prob=0.8, seed=i * 4 + j)

            if j < 2:
                # Show masked intensity
                im = ax.imshow(np.log10(1 + I_masked * 1e6), cmap='inferno')
                ax.set_title(f'Sample {i*4+j+1}\nMask applied: {mask.any()}')
            else:
                # Show mask
                ax.imshow(mask.astype(float), cmap='gray')
                ax.set_title(f'Mask {i*4+j+1}\nPixels: {mask.sum()}')

            ax.axis('off')

    plt.tight_layout()
    output_path = Path(__file__).parent / 'test_random_mask.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Test output saved to {output_path}")

    # Statistics
    print("\nMask statistics over 100 samples:")
    total_masked_pixels = 0
    samples_with_mask = 0

    for i in range(100):
        _, mask = applier.apply(I_norm, prob=0.5, seed=i)
        if mask.any():
            samples_with_mask += 1
            total_masked_pixels += mask.sum()

    print(f"  Samples with mask: {samples_with_mask}/100")
    if samples_with_mask > 0:
        print(f"  Average masked pixels: {total_masked_pixels / samples_with_mask:.0f}")
