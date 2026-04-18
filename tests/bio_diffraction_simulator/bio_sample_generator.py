"""
Bio Sample Generator - E. coli 2D Projection Density Map Generator

This module generates 2D projection density maps of E. coli bacteria for
diffraction simulation. It is implemented independently from pro_diverse_4.py
which generates 3D voxel data for the Condor framework.

The output is a 2D numpy array representing the projection density,
suitable for FFT-based diffraction simulation.
"""

import numpy as np
from scipy.ndimage import rotate, gaussian_filter, zoom
from typing import Optional, Tuple, List
from pathlib import Path

from config import EXP_CONFIG, BACTERIA_CONFIG


class BioSampleGenerator:
    """
    Generator for E. coli 2D projection density maps.

    Creates realistic bacteria shapes with:
    - Capsule body (rectangle + hemispherical ends)
    - Perlin noise surface irregularities
    - Internal Gaussian spots
    - Circular vacuoles (low density regions)
    - Multiple cell states (normal, dividing, curved)
    """

    def __init__(self, seed: Optional[int] = None, sample_size_px: Optional[int] = None):
        """
        Initialize the bio sample generator.

        Args:
            seed: Random seed for reproducibility.
            sample_size_px: Target max bacteria length in pixels.
                            None = use original BACTERIA_CONFIG ranges.
                            Smaller values produce smaller bacteria with wider diffraction features.
        """
        self.rng = np.random.default_rng(seed)
        self.grid_size = EXP_CONFIG['train_size']
        self.sample_size_px = sample_size_px

        # Compute scale factor relative to reference size (midpoint of default 80-200 range)
        if sample_size_px is not None:
            self.scale_factor = sample_size_px / 140.0
        else:
            self.scale_factor = 1.0

    def _get_scaled_config(self):
        """Get bacteria configuration with size scaling applied."""
        if self.sample_size_px is not None:
            s = self.sample_size_px
            sf = self.scale_factor
            return {
                'length_range_px': (s * 0.7, s),
                'diameter_range_px': (s * 0.25, s * 0.45),
                'n_gaussian_spots_range': (
                    max(1, int(3 * sf)),
                    max(2, min(int(10 * sf), 6 if s < 50 else 10))
                ),
                'n_vacuoles_range': (
                    1,
                    max(2, min(int(3 * sf), 2 if s < 50 else 3))
                ),
                'gaussian_sigma_range': (max(1.0, 3 * sf), max(2.0, 10 * sf)),
                'vacuole_radius_range': (max(2.0, 5 * sf), max(3.0, 15 * sf)),
                'constriction_width_range': (max(3.0, 10 * sf), max(4.0, 20 * sf)),
                'position_offset': max(5, 20 * sf),
                'cell_states': BACTERIA_CONFIG['cell_states'],
                'perlin_noise_scale_range': BACTERIA_CONFIG['perlin_noise_scale_range'],
                'vacuole_value_threshold': BACTERIA_CONFIG['vacuole_value_threshold'],
            }
        else:
            sf = 1.0
            return {
                'length_range_px': BACTERIA_CONFIG['length_range_px'],
                'diameter_range_px': BACTERIA_CONFIG['diameter_range_px'],
                'n_gaussian_spots_range': BACTERIA_CONFIG['n_gaussian_spots_range'],
                'n_vacuoles_range': BACTERIA_CONFIG['n_vacuoles_range'],
                'gaussian_sigma_range': (3, 10),
                'vacuole_radius_range': (5, 15),
                'constriction_width_range': (10, 20),
                'position_offset': 20,
                'cell_states': BACTERIA_CONFIG['cell_states'],
                'perlin_noise_scale_range': BACTERIA_CONFIG['perlin_noise_scale_range'],
                'vacuole_value_threshold': BACTERIA_CONFIG['vacuole_value_threshold'],
            }

    def generate(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a 2D projection density map of E. coli.

        Args:
            seed: Optional seed to override initialization seed.

        Returns:
            obj: (585, 585) float32 array with values in [0, 1]
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        cfg = self._get_scaled_config()

        # Randomly select cell state
        cell_state = self.rng.choice(cfg['cell_states'])

        # Generate based on cell state
        if cell_state == 'normal':
            obj = self._generate_normal_cell(cfg)
        elif cell_state == 'dividing':
            obj = self._generate_dividing_cell(cfg)
        elif cell_state == 'curved':
            obj = self._generate_curved_cell(cfg)
        else:
            obj = self._generate_normal_cell(cfg)

        # Add internal structures
        n_spots = self.rng.integers(*cfg['n_gaussian_spots_range'])
        obj = self._add_gaussian_spots(obj, n_spots, cfg)

        n_vacuoles = self.rng.integers(*cfg['n_vacuoles_range'])
        obj = self._add_vacuoles(obj, n_vacuoles, cfg)

        # Add surface noise
        obj = self._add_surface_noise(obj)

        # Ensure values are in [0, 1]
        obj = np.clip(obj, 0, 1).astype(np.float32)

        return obj

    def _create_capsule_2d(
        self,
        length: float,
        diameter: float,
        center: Tuple[float, float],
        angle: float
    ) -> np.ndarray:
        """
        Create a 2D capsule shape (rectangle + semicircular ends).

        Args:
            length: Total length of the capsule in pixels.
            diameter: Diameter of the capsule in pixels.
            center: (y, x) center position.
            angle: Rotation angle in degrees.

        Returns:
            2D binary mask of the capsule.
        """
        mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Create in local coordinates
        # Local coordinate system: capsule along x-axis
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]

        # Translate to center
        y_centered = y_coords - center[0]
        x_centered = x_coords - center[1]

        # Rotate coordinates (negative angle to rotate shape)
        angle_rad = np.radians(-angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_rot = x_centered * cos_a - y_centered * sin_a
        y_rot = x_centered * sin_a + y_centered * cos_a

        radius = diameter / 2
        half_length = length / 2 - radius  # Length of the rectangular part

        # Capsule: rectangle part + two semicircular caps
        if half_length > 0:
            rect_mask = (np.abs(x_rot) <= half_length) & (np.abs(y_rot) <= radius)
        else:
            rect_mask = np.zeros_like(mask, dtype=bool)

        # Left cap
        left_center_x = -half_length if half_length > 0 else 0
        left_dist = np.sqrt((x_rot - left_center_x)**2 + y_rot**2)
        left_cap = (left_dist <= radius) & (x_rot <= left_center_x)

        # Right cap
        right_center_x = half_length if half_length > 0 else 0
        right_dist = np.sqrt((x_rot - right_center_x)**2 + y_rot**2)
        right_cap = (right_dist <= radius) & (x_rot >= right_center_x)

        mask = (rect_mask | left_cap | right_cap).astype(np.float32)

        return mask

    def _generate_normal_cell(self, cfg: dict) -> np.ndarray:
        """Generate a normal single cell."""
        # Random dimensions directly in pixels
        length_px = self.rng.uniform(*cfg['length_range_px'])
        diameter_px = self.rng.uniform(*cfg['diameter_range_px'])

        # Random position (centered with some variation)
        offset = cfg['position_offset']
        center_y = self.grid_size // 2 + self.rng.uniform(-offset, offset)
        center_x = self.grid_size // 2 + self.rng.uniform(-offset, offset)

        # Random angle
        angle = self.rng.uniform(0, 360)

        # Create capsule
        obj = self._create_capsule_2d(length_px, diameter_px, (center_y, center_x), angle)

        # Add membrane density gradient (slightly higher at edges)
        obj = self._add_membrane_gradient(obj)

        return obj

    def _generate_dividing_cell(self, cfg: dict) -> np.ndarray:
        """Generate a dividing cell (two connected capsules)."""
        # Generate dimensions for both daughter cells
        length1_px = self.rng.uniform(*cfg['length_range_px']) * 0.7
        length2_px = self.rng.uniform(*cfg['length_range_px']) * 0.7
        diameter_px = self.rng.uniform(*cfg['diameter_range_px'])

        # Main angle
        main_angle = self.rng.uniform(0, 360)

        # Slight angle difference for dividing cells
        angle_diff = self.rng.uniform(-15, 15)

        # Centers - cells connected at one end
        center_y = self.grid_size // 2
        center_x = self.grid_size // 2

        # Offset for the connection point (scaled)
        offset_max = max(5, 15 * self.scale_factor)
        offset = self.rng.uniform(3, offset_max)

        # Create first cell
        obj1 = self._create_capsule_2d(
            length1_px, diameter_px,
            (center_y - offset, center_x),
            main_angle + angle_diff
        )

        # Create second cell (attached at connection point)
        obj2 = self._create_capsule_2d(
            length2_px, diameter_px,
            (center_y + offset, center_x),
            main_angle - angle_diff + 180
        )

        # Combine cells
        obj = np.maximum(obj1, obj2)

        # Add constriction at division site
        obj = self._add_division_constriction(obj, cfg)

        obj = self._add_membrane_gradient(obj)

        return obj

    def _generate_curved_cell(self, cfg: dict) -> np.ndarray:
        """Generate a curved/arc-shaped cell."""
        # Random dimensions directly in pixels
        length_px = self.rng.uniform(*cfg['length_range_px'])
        diameter_px = self.rng.uniform(*cfg['diameter_range_px'])

        # Create curved cell using arc segments
        obj = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Arc parameters
        arc_angle = self.rng.uniform(30, 90)  # degrees of arc
        arc_direction = self.rng.uniform(0, 360)  # orientation
        arc_radius = length_px / np.radians(arc_angle)  # radius of curvature

        center_y = self.grid_size // 2
        center_x = self.grid_size // 2

        # Draw arc as series of small capsules
        n_segments = max(10, int(length_px / 5))
        angles = np.linspace(0, arc_angle, n_segments)

        for i, a in enumerate(angles):
            # Position along arc
            angle_rad = np.radians(arc_direction + a - arc_angle/2)
            y_pos = center_y - arc_radius * np.sin(angle_rad)
            x_pos = center_x + arc_radius * (1 - np.cos(angle_rad))

            # Local tangent angle
            local_angle = arc_direction + a - 90

            # Small segment
            segment = self._create_capsule_2d(
                length_px / n_segments + diameter_px,
                diameter_px,
                (y_pos, x_pos),
                local_angle
            )
            obj = np.maximum(obj, segment)

        obj = self._add_membrane_gradient(obj)

        return obj

    def _add_membrane_gradient(self, obj: np.ndarray) -> np.ndarray:
        """
        Add density gradient to simulate membrane (higher density at edges).

        Args:
            obj: Input density map.

        Returns:
            Density map with membrane gradient.
        """
        # Create edge-enhanced version
        edge = gaussian_filter(obj, sigma=1)
        edge_gradient = obj - edge

        # Combine: original + edge enhancement
        result = obj + 0.2 * np.maximum(edge_gradient, 0)

        # Normalize
        if result.max() > 0:
            result = result / result.max()

        return result

    def _add_division_constriction(self, obj: np.ndarray, cfg: dict) -> np.ndarray:
        """Add constriction at division site."""
        # Find the center and add a constriction
        center = self.grid_size // 2
        constriction_width = self.rng.uniform(*cfg['constriction_width_range'])
        constriction_depth = self.rng.uniform(0.3, 0.6)

        # Create constriction mask
        y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
        constriction_region = np.abs(x_coords - center) < constriction_width / 2

        # Apply constriction
        result = obj.copy()
        result[constriction_region & (obj > 0)] *= constriction_depth

        return result

    def _add_gaussian_spots(self, obj: np.ndarray, n_spots: int, cfg: dict) -> np.ndarray:
        """
        Add internal Gaussian spots to simulate intracellular structures.

        Args:
            obj: Input density map.
            n_spots: Number of Gaussian spots to add.
            cfg: Scaled configuration dictionary.

        Returns:
            Density map with Gaussian spots.
        """
        result = obj.copy()

        # Find region where bacteria exists
        bacteria_mask = obj > 0.1
        if not bacteria_mask.any():
            return result

        # Get bacteria region bounds
        y_indices, x_indices = np.where(bacteria_mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Margin proportional to bacteria size
        margin = max(2, int(5 * self.scale_factor))

        for _ in range(n_spots):
            # Random position within bacteria region
            cy = self.rng.integers(y_min + margin, max(y_min + margin + 1, y_max - margin))
            cx = self.rng.integers(x_min + margin, max(x_min + margin + 1, x_max - margin))

            # Random size and intensity (scaled)
            sigma = self.rng.uniform(*cfg['gaussian_sigma_range'])
            intensity = self.rng.uniform(0.1, 0.3)

            # Create Gaussian spot
            y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
            spot = np.exp(-((y_coords - cy)**2 + (x_coords - cx)**2) / (2 * sigma**2))
            spot = spot * intensity

            # Only add where bacteria exists
            result[bacteria_mask] += spot[bacteria_mask]

        # Normalize to [0, 1]
        if result.max() > 0:
            result = result / result.max()

        return result

    def _add_vacuoles(self, obj: np.ndarray, n_vacuoles: int, cfg: dict) -> np.ndarray:
        """
        Add circular vacuoles (low density regions).

        Args:
            obj: Input density map.
            n_vacuoles: Number of vacuoles to add.
            cfg: Scaled configuration dictionary.

        Returns:
            Density map with vacuoles.
        """
        result = obj.copy()

        # Find region where bacteria exists
        bacteria_mask = obj > 0.1
        if not bacteria_mask.any():
            return result

        # Get bacteria region bounds
        y_indices, x_indices = np.where(bacteria_mask)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        threshold = cfg['vacuole_value_threshold']

        # Margin proportional to bacteria size
        margin = max(3, int(10 * self.scale_factor))

        for _ in range(n_vacuoles):
            # Random position within bacteria region
            cy = self.rng.integers(y_min + margin, max(y_min + margin + 1, y_max - margin))
            cx = self.rng.integers(x_min + margin, max(x_min + margin + 1, x_max - margin))

            # Random size (scaled)
            radius = self.rng.uniform(*cfg['vacuole_radius_range'])

            # Create circular vacuole
            y_coords, x_coords = np.ogrid[:self.grid_size, :self.grid_size]
            vacuole_mask = ((y_coords - cy)**2 + (x_coords - cx)**2) <= radius**2

            # Reduce density in vacuole region
            result[vacuole_mask] = np.minimum(result[vacuole_mask], threshold)

        return result

    def _add_surface_noise(self, obj: np.ndarray) -> np.ndarray:
        """
        Add Perlin-like noise to simulate irregular cell surface.

        Args:
            obj: Input density map.

        Returns:
            Density map with surface irregularities.
        """
        # Get noise scale
        noise_scale = self.rng.uniform(*BACTERIA_CONFIG['perlin_noise_scale_range'])

        # Generate noise using multiple octaves of Gaussian noise
        result = obj.copy()

        # Create multi-scale noise
        noise = np.zeros_like(obj)
        for octave in range(3):
            scale = noise_scale * (2 ** octave)
            amplitude = 1.0 / (octave + 1)

            # Low-frequency noise
            small_size = max(1, int(self.grid_size * scale))
            small_noise = self.rng.normal(0, 1, (small_size, small_size))

            # Upsample to full size
            zoom_factor = self.grid_size / small_size
            upsampled = zoom(small_noise, zoom_factor, order=1)

            # Ensure same size
            if upsampled.shape[0] > self.grid_size:
                upsampled = upsampled[:self.grid_size, :self.grid_size]
            elif upsampled.shape[0] < self.grid_size:
                padded = np.zeros((self.grid_size, self.grid_size))
                padded[:upsampled.shape[0], :upsampled.shape[1]] = upsampled
                upsampled = padded

            noise += upsampled * amplitude

        # Normalize noise
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-10)

        # Apply noise only at edges (surface)
        edge = self._detect_edges(obj)
        edge_strength = 0.15

        result = result * (1 - edge_strength * edge) + result * edge * noise * edge_strength + result * (1 - edge)

        # Smooth the result slightly
        result = gaussian_filter(result, sigma=0.5)

        # Normalize
        if result.max() > 0:
            result = result / result.max()

        return result

    def _detect_edges(self, obj: np.ndarray) -> np.ndarray:
        """Detect edges using gradient magnitude."""
        # Sobel-like edge detection
        gy = np.gradient(obj, axis=0)
        gx = np.gradient(obj, axis=1)
        edge = np.sqrt(gx**2 + gy**2)

        # Normalize
        if edge.max() > 0:
            edge = edge / edge.max()

        return edge


def generate_bio_sample(seed: Optional[int] = None, sample_size_px: Optional[int] = None) -> np.ndarray:
    """
    Convenience function to generate a single bio sample.

    Args:
        seed: Random seed for reproducibility.
        sample_size_px: Target max bacteria length in pixels. None = default.

    Returns:
        2D numpy array (585, 585) with values in [0, 1].
    """
    generator = BioSampleGenerator(seed=seed, sample_size_px=sample_size_px)
    return generator.generate()


if __name__ == "__main__":
    # Test the generator
    import matplotlib.pyplot as plt
    from pathlib import Path

    print("Testing BioSampleGenerator...")

    # Generate a few samples
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    output_dir = Path(__file__).parent

    for i, ax in enumerate(axes.flat):
        obj = generate_bio_sample(seed=i * 100)
        ax.imshow(obj, cmap='gray')
        ax.set_title(f'Sample {i}')
        ax.axis('off')

        print(f"Sample {i}: shape={obj.shape}, range=[{obj.min():.3f}, {obj.max():.3f}]")

    plt.tight_layout()
    plt.savefig(output_dir / 'test_bio_samples.png', dpi=150)
    plt.close()

    print(f"\nTest samples saved to {output_dir / 'test_bio_samples.png'}")
