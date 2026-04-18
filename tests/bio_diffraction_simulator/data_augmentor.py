"""
Data Augmentor - Random transformations for bio sample diversity.

This module applies random rotations, translations, and scaling to
augment the bio sample density maps.
"""

import numpy as np
from scipy.ndimage import rotate, shift, zoom
from typing import Optional

from config import EXP_CONFIG, AUGMENT_CONFIG


class DataAugmentor:
    """
    Data augmentor for bio sample density maps.

    Applies random transformations in sequence:
    1. Random rotation
    2. Random translation
    3. Random scaling
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the data augmentor.

        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        self.grid_size = EXP_CONFIG['train_size']

    def augment(
        self,
        obj: np.ndarray,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply random transformations to augment the sample.

        Args:
            obj: Input density map (585, 585).
            seed: Optional seed to override initialization seed.

        Returns:
            Augmented density map (585, 585).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Step 1: Random rotation
        obj = self._random_rotation(obj)

        # Step 2: Random translation
        obj = self._random_translation(obj)

        # Step 3: Random scaling
        obj = self._random_scaling(obj)

        # Ensure output is in valid range
        obj = np.clip(obj, 0, 1).astype(np.float32)

        return obj

    def _random_rotation(self, obj: np.ndarray) -> np.ndarray:
        """
        Apply random rotation.

        Args:
            obj: Input density map.

        Returns:
            Rotated density map.
        """
        angle = self.rng.uniform(*AUGMENT_CONFIG['rotation_range'])
        order = AUGMENT_CONFIG['rotation_order']

        # Rotate using scipy
        rotated = rotate(
            obj,
            angle,
            reshape=False,  # Keep original shape
            order=order,    # Bilinear interpolation
            mode='constant', # Fill with zeros
            cval=0.0
        )

        return rotated

    def _random_translation(self, obj: np.ndarray) -> np.ndarray:
        """
        Apply random translation.

        Args:
            obj: Input density map.

        Returns:
            Translated density map.
        """
        # Calculate translation in pixels
        translation_range = AUGMENT_CONFIG['translation_range']
        max_shift = self.grid_size * translation_range[1]

        shift_y = self.rng.uniform(-max_shift, max_shift)
        shift_x = self.rng.uniform(-max_shift, max_shift)

        # Apply shift
        translated = shift(
            obj,
            [shift_y, shift_x],
            order=1,         # Bilinear interpolation
            mode='constant', # Fill with zeros
            cval=0.0
        )

        return translated

    def _random_scaling(self, obj: np.ndarray) -> np.ndarray:
        """
        Apply random scaling.

        Args:
            obj: Input density map.

        Returns:
            Scaled density map with original size.
        """
        scale = self.rng.uniform(*AUGMENT_CONFIG['scale_range'])

        # Calculate new size
        new_size = int(self.grid_size * scale)

        # Scale the image
        if scale != 1.0:
            # Use zoom to scale
            scaled = zoom(obj, scale, order=1)

            # Create output array
            result = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

            # Calculate placement (center the scaled image)
            h, w = scaled.shape
            y_start = max(0, (self.grid_size - h) // 2)
            x_start = max(0, (self.grid_size - w) // 2)

            # Calculate source region
            src_y_start = max(0, -((self.grid_size - h) // 2))
            src_x_start = max(0, -((self.grid_size - w) // 2))

            # Calculate how much to copy
            copy_h = min(h - src_y_start, self.grid_size - y_start)
            copy_w = min(w - src_x_start, self.grid_size - x_start)

            if copy_h > 0 and copy_w > 0:
                result[y_start:y_start + copy_h, x_start:x_start + copy_w] = \
                    scaled[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w]

            return result
        else:
            return obj

    def get_transform_params(self) -> dict:
        """
        Get random transformation parameters without applying them.

        Returns:
            Dictionary with transformation parameters.
        """
        angle = self.rng.uniform(*AUGMENT_CONFIG['rotation_range'])

        translation_range = AUGMENT_CONFIG['translation_range']
        max_shift = self.grid_size * translation_range[1]
        shift_y = self.rng.uniform(-max_shift, max_shift)
        shift_x = self.rng.uniform(-max_shift, max_shift)

        scale = self.rng.uniform(*AUGMENT_CONFIG['scale_range'])

        return {
            'rotation_angle': angle,
            'translation_y': shift_y,
            'translation_x': shift_x,
            'scale': scale
        }


def augment_sample(obj: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to augment a single sample.

    Args:
        obj: Input density map (585, 585).
        seed: Random seed for reproducibility.

    Returns:
        Augmented density map (585, 585).
    """
    augmentor = DataAugmentor(seed=seed)
    return augmentor.augment(obj)


if __name__ == "__main__":
    # Test the augmentor
    import matplotlib.pyplot as plt
    from pathlib import Path

    from bio_sample_generator import generate_bio_sample

    print("Testing DataAugmentor...")

    # Generate a base sample
    base_sample = generate_bio_sample(seed=42)

    # Create augmentor
    augmentor = DataAugmentor(seed=42)

    # Test augmentation
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Original
    axes[0, 0].imshow(base_sample, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Augmented samples
    for i, ax in enumerate(axes.flat[1:]):
        augmented = augmentor.augment(base_sample, seed=i * 10)
        ax.imshow(augmented, cmap='gray')
        ax.set_title(f'Augmented {i+1}')
        ax.axis('off')

        print(f"Augmented {i+1}: range=[{augmented.min():.3f}, {augmented.max():.3f}]")

    plt.tight_layout()
    output_path = Path(__file__).parent / 'test_augmentation.png'
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nTest samples saved to {output_path}")
