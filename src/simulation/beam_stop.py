"""Beam stop simulation for X-ray diffraction patterns."""

import numpy as np
from typing import Optional, Tuple, Union


def create_beam_stop_mask(
    shape: Tuple[int, int],
    radius: int,
    center: Optional[Tuple[int, int]] = None,
    soft_edge: bool = False,
    edge_width: int = 2
) -> np.ndarray:
    """
    Create a beam stop mask.
    
    The beam stop blocks the direct beam at the center of the
    diffraction pattern, which would otherwise saturate the detector.
    
    Args:
        shape: Shape of the output mask (height, width).
        radius: Radius of the beam stop in pixels.
        center: Center of the beam stop. Uses image center if None.
        soft_edge: Whether to apply soft (Gaussian) edge.
        edge_width: Width of soft edge transition.
    
    Returns:
        Boolean mask where True indicates blocked (beam stop) region.
    """
    h, w = shape
    
    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center
    
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    if soft_edge:
        # Soft edge using sigmoid-like transition
        mask = 1.0 / (1.0 + np.exp((distance - radius) / edge_width))
        return mask > 0.5
    else:
        return distance <= radius


def apply_beam_stop(
    pattern: np.ndarray,
    radius: int,
    center: Optional[Tuple[int, int]] = None,
    fill_value: float = 0.0,
    return_mask: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply beam stop to a diffraction pattern.
    
    Args:
        pattern: 2D diffraction pattern.
        radius: Beam stop radius in pixels.
        center: Center of beam stop. Uses pattern center if None.
        fill_value: Value to fill the masked region.
        return_mask: Whether to also return the mask.
    
    Returns:
        Masked pattern, or tuple of (masked_pattern, mask) if return_mask=True.
    """
    mask = create_beam_stop_mask(pattern.shape, radius, center)
    
    result = pattern.copy()
    result[mask] = fill_value
    
    if return_mask:
        return result, mask
    return result


def apply_beam_stop_with_interpolation(
    pattern: np.ndarray,
    radius: int,
    center: Optional[Tuple[int, int]] = None,
    method: str = "radial"
) -> np.ndarray:
    """
    Apply beam stop with interpolation to fill missing data.
    
    This can be useful for visualization or as initial guess
    for iterative reconstruction.
    
    Args:
        pattern: 2D diffraction pattern.
        radius: Beam stop radius in pixels.
        center: Center of beam stop.
        method: Interpolation method ("radial", "zero", "mean").
    
    Returns:
        Pattern with beam stop region interpolated.
    """
    h, w = pattern.shape
    
    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center
    
    mask = create_beam_stop_mask(pattern.shape, radius, center)
    result = pattern.copy()
    
    if method == "zero":
        result[mask] = 0.0
        
    elif method == "mean":
        # Fill with mean of surrounding pixels
        edge_mask = create_beam_stop_mask(pattern.shape, radius + 3, center)
        edge_mask = edge_mask & ~mask
        if edge_mask.any():
            result[mask] = pattern[edge_mask].mean()
        else:
            result[mask] = 0.0
            
    elif method == "radial":
        # Radial interpolation from edge values
        y, x = np.ogrid[:h, :w]
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        
        # Get values at beam stop edge
        edge_distance = radius + 1
        edge_mask = (distance >= radius) & (distance <= radius + 2)
        
        if edge_mask.any():
            edge_value = pattern[edge_mask].mean()
            # Linear interpolation from center to edge
            result[mask] = edge_value * (distance[mask] / radius)
        else:
            result[mask] = 0.0
    
    return result


def estimate_beam_stop_radius(
    pattern: np.ndarray,
    threshold_ratio: float = 0.01
) -> int:
    """
    Estimate beam stop radius from a pattern.
    
    Looks for the central region with very low or zero values.
    
    Args:
        pattern: 2D diffraction pattern.
        threshold_ratio: Ratio of max value to consider as beam stop.
    
    Returns:
        Estimated beam stop radius in pixels.
    """
    h, w = pattern.shape
    cy, cx = h // 2, w // 2
    
    threshold = pattern.max() * threshold_ratio
    
    # Check radially outward from center
    max_radius = min(h, w) // 4
    
    for r in range(1, max_radius):
        # Sample points at this radius
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        values = []
        
        for angle in angles:
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            if 0 <= y < h and 0 <= x < w:
                values.append(pattern[y, x])
        
        if values and np.mean(values) > threshold:
            return max(1, r - 1)
    
    return 0


class BeamStopSimulator:
    """
    Beam stop simulator with various configurations.
    
    Supports different beam stop shapes and effects.
    """
    
    def __init__(
        self,
        radius: int = 5,
        shape: str = "circular",
        holder_width: int = 0,
        holder_angle: float = 0.0
    ):
        """
        Initialize beam stop simulator.
        
        Args:
            radius: Beam stop radius.
            shape: Shape type ("circular", "square").
            holder_width: Width of beam stop holder (wire).
            holder_angle: Angle of holder in radians.
        """
        self.radius = radius
        self.shape = shape
        self.holder_width = holder_width
        self.holder_angle = holder_angle
    
    def create_mask(
        self,
        shape: Tuple[int, int],
        center: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Create complete beam stop mask including holder.
        
        Args:
            shape: Output shape (height, width).
            center: Center position.
        
        Returns:
            Boolean mask.
        """
        h, w = shape
        
        if center is None:
            cy, cx = h // 2, w // 2
        else:
            cy, cx = center
        
        y, x = np.ogrid[:h, :w]
        
        # Main beam stop
        if self.shape == "circular":
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= self.radius ** 2
        elif self.shape == "square":
            mask = (np.abs(x - cx) <= self.radius) & (np.abs(y - cy) <= self.radius)
        else:
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= self.radius ** 2
        
        # Add holder (wire)
        if self.holder_width > 0:
            # Rotate coordinates
            cos_a = np.cos(self.holder_angle)
            sin_a = np.sin(self.holder_angle)
            
            x_rot = (x - cx) * cos_a + (y - cy) * sin_a
            y_rot = -(x - cx) * sin_a + (y - cy) * cos_a
            
            # Holder extends from beam stop to edge
            holder_mask = (np.abs(y_rot) <= self.holder_width / 2) & (x_rot >= 0)
            mask = mask | holder_mask
        
        return mask
    
    def apply(
        self,
        pattern: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        fill_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply beam stop to pattern.
        
        Args:
            pattern: Input diffraction pattern.
            center: Center position.
            fill_value: Fill value for masked region.
        
        Returns:
            Tuple of (masked_pattern, mask).
        """
        mask = self.create_mask(pattern.shape, center)
        result = pattern.copy()
        result[mask] = fill_value
        return result, mask
