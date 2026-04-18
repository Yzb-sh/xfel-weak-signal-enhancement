"""
Support Estimation for Phase Retrieval.

Implements methods for estimating and updating the support constraint
used in iterative phase retrieval algorithms.
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from typing import Optional


class SupportEstimator:
    """
    Support constraint estimation for phase retrieval.
    
    Provides methods for initial support estimation from autocorrelation
    and dynamic support update via shrink-wrap algorithm.
    """
    
    def __init__(self):
        """Initialize support estimator."""
        pass
    
    def from_autocorrelation(
        self,
        pattern: np.ndarray,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Estimate support from autocorrelation function.
        
        The autocorrelation of the object is the inverse FFT of the
        diffraction intensity. The support is estimated by thresholding
        the autocorrelation.
        
        Args:
            pattern: Diffraction intensity pattern (centered)
            threshold: Threshold as fraction of maximum (0-1)
            
        Returns:
            Binary support mask
        """
        # Compute autocorrelation via inverse FFT of intensity
        autocorr = np.abs(ifft2(ifftshift(pattern)))
        autocorr = fftshift(autocorr)
        
        # Normalize
        autocorr = autocorr / autocorr.max()
        
        # Threshold to get support
        # Autocorrelation is twice the size of object, so we need
        # to estimate the object support from the central region
        support = autocorr > threshold
        
        # The autocorrelation support is roughly twice the object size
        # Apply erosion to get a tighter estimate
        support = binary_erosion(support, iterations=2)
        
        # Ensure support is not empty
        if not support.any():
            # Fall back to central circular support
            size = pattern.shape[0]
            y, x = np.ogrid[:size, :size]
            center = size // 2
            radius = size // 4
            support = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        
        return support
    
    def shrink_wrap(
        self,
        density: np.ndarray,
        sigma: float = 1.0,
        threshold: float = 0.1
    ) -> np.ndarray:
        """
        Shrink-wrap dynamic support update.
        
        Applies Gaussian blur to the current density estimate and
        thresholds to update the support. This allows the support
        to adapt during reconstruction.
        
        Args:
            density: Current density estimate
            sigma: Gaussian blur sigma
            threshold: Threshold as fraction of maximum (0-1)
            
        Returns:
            Updated binary support mask
        """
        # Apply Gaussian blur
        blurred = gaussian_filter(np.abs(density), sigma=sigma)
        
        # Normalize
        if blurred.max() > 0:
            blurred = blurred / blurred.max()
        
        # Threshold
        support = blurred > threshold
        
        # Apply slight dilation to avoid cutting off edges
        support = binary_dilation(support, iterations=1)
        
        # Ensure support is not empty
        if not support.any():
            # Keep previous support or use central region
            size = density.shape[0]
            y, x = np.ogrid[:size, :size]
            center = size // 2
            radius = size // 4
            support = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        
        return support
    
    def circular_support(
        self,
        size: int,
        radius: Optional[int] = None,
        center: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Create a circular support mask.
        
        Args:
            size: Image size (assumes square)
            radius: Circle radius (default: size // 4)
            center: Circle center (default: image center)
            
        Returns:
            Binary circular support mask
        """
        if radius is None:
            radius = size // 4
        if center is None:
            center = (size // 2, size // 2)
        
        y, x = np.ogrid[:size, :size]
        support = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
        
        return support
    
    def rectangular_support(
        self,
        size: int,
        width: Optional[int] = None,
        height: Optional[int] = None,
        center: Optional[tuple] = None
    ) -> np.ndarray:
        """
        Create a rectangular support mask.
        
        Args:
            size: Image size (assumes square)
            width: Rectangle width (default: size // 2)
            height: Rectangle height (default: size // 2)
            center: Rectangle center (default: image center)
            
        Returns:
            Binary rectangular support mask
        """
        if width is None:
            width = size // 2
        if height is None:
            height = size // 2
        if center is None:
            center = (size // 2, size // 2)
        
        support = np.zeros((size, size), dtype=bool)
        
        y_start = max(0, center[0] - height // 2)
        y_end = min(size, center[0] + height // 2)
        x_start = max(0, center[1] - width // 2)
        x_end = min(size, center[1] + width // 2)
        
        support[y_start:y_end, x_start:x_end] = True
        
        return support
