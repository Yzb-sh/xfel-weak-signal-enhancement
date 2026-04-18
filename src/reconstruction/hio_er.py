"""
HIO/ER Phase Retrieval Algorithms.

Implements Hybrid Input-Output (HIO) and Error Reduction (ER) algorithms
for coherent diffraction imaging phase retrieval.
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.fft import fft2, ifft2, fftshift, ifftshift


class PhaseRetrieval:
    """
    HIO/ER phase retrieval algorithm implementation.
    
    Implements iterative projection algorithms for recovering phase
    from diffraction magnitude measurements.
    """
    
    def __init__(
        self,
        beta: float = 0.9,
        max_iter: int = 1000,
        tol: float = 1e-6
    ):
        """
        Initialize phase retrieval algorithm.
        
        Args:
            beta: HIO feedback parameter (typically 0.7-0.9)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance for error
        """
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
    
    def _fourier_constraint(
        self,
        estimate: np.ndarray,
        magnitude: np.ndarray
    ) -> np.ndarray:
        """
        Apply Fourier magnitude constraint.
        
        Replace magnitude in Fourier space while keeping phase.
        
        Args:
            estimate: Current real-space estimate
            magnitude: Measured Fourier magnitude
            
        Returns:
            Updated estimate after Fourier constraint
        """
        # Forward FFT
        f_estimate = fftshift(fft2(estimate))
        
        # Get phase from estimate
        phase = np.angle(f_estimate)
        
        # Combine measured magnitude with estimated phase
        f_constrained = magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        return np.real(ifft2(ifftshift(f_constrained)))
    
    def _support_constraint_er(
        self,
        estimate: np.ndarray,
        support: np.ndarray
    ) -> np.ndarray:
        """
        Apply support constraint (Error Reduction).
        
        Set values outside support to zero.
        
        Args:
            estimate: Current real-space estimate
            support: Binary support mask
            
        Returns:
            Updated estimate after support constraint
        """
        result = estimate.copy()
        result[~support] = 0
        # Enforce non-negativity inside support
        result[support & (result < 0)] = 0
        return result
    
    def _support_constraint_hio(
        self,
        estimate: np.ndarray,
        previous: np.ndarray,
        support: np.ndarray
    ) -> np.ndarray:
        """
        Apply support constraint (HIO).
        
        Inside support: keep positive values
        Outside support: apply feedback
        
        Args:
            estimate: Current estimate after Fourier constraint
            previous: Previous iteration estimate
            support: Binary support mask
            
        Returns:
            Updated estimate after HIO constraint
        """
        result = np.zeros_like(estimate)
        
        # Inside support: keep positive values, apply feedback to negative
        inside_positive = support & (estimate >= 0)
        inside_negative = support & (estimate < 0)
        
        result[inside_positive] = estimate[inside_positive]
        result[inside_negative] = previous[inside_negative] - self.beta * estimate[inside_negative]
        
        # Outside support: apply feedback
        outside = ~support
        result[outside] = previous[outside] - self.beta * estimate[outside]
        
        return result
    
    def _compute_error(
        self,
        estimate: np.ndarray,
        magnitude: np.ndarray
    ) -> float:
        """
        Compute Fourier space error.
        
        Args:
            estimate: Current real-space estimate
            magnitude: Measured Fourier magnitude
            
        Returns:
            Normalized error metric
        """
        f_estimate = fftshift(fft2(estimate))
        estimated_mag = np.abs(f_estimate)
        
        # Normalized error
        error = np.sqrt(np.sum((estimated_mag - magnitude) ** 2) / np.sum(magnitude ** 2))
        return error
    
    def hio(
        self,
        magnitude: np.ndarray,
        support: np.ndarray,
        initial_phase: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Hybrid Input-Output algorithm.
        
        Args:
            magnitude: Measured Fourier magnitude (centered)
            support: Binary support mask in real space
            initial_phase: Optional initial phase estimate
            
        Returns:
            Tuple of (reconstructed density, error history)
        """
        # Initialize with random phase if not provided
        if initial_phase is None:
            initial_phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        
        # Initial estimate from magnitude and phase
        f_initial = magnitude * np.exp(1j * initial_phase)
        estimate = np.real(ifft2(ifftshift(f_initial)))
        
        error_history = []
        previous = estimate.copy()
        
        for i in range(self.max_iter):
            # Fourier constraint
            estimate = self._fourier_constraint(estimate, magnitude)
            
            # HIO support constraint
            estimate = self._support_constraint_hio(estimate, previous, support)
            
            # Compute error
            error = self._compute_error(estimate, magnitude)
            error_history.append(error)
            
            # Check convergence
            if i > 0 and abs(error_history[-1] - error_history[-2]) < self.tol:
                break
            
            previous = estimate.copy()
        
        return estimate, error_history
    
    def er(
        self,
        magnitude: np.ndarray,
        support: np.ndarray,
        initial_phase: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Error Reduction algorithm.
        
        Args:
            magnitude: Measured Fourier magnitude (centered)
            support: Binary support mask in real space
            initial_phase: Optional initial phase estimate
            
        Returns:
            Tuple of (reconstructed density, error history)
        """
        # Initialize with random phase if not provided
        if initial_phase is None:
            initial_phase = np.random.uniform(-np.pi, np.pi, magnitude.shape)
        
        # Initial estimate from magnitude and phase
        f_initial = magnitude * np.exp(1j * initial_phase)
        estimate = np.real(ifft2(ifftshift(f_initial)))
        
        error_history = []
        
        for i in range(self.max_iter):
            # Fourier constraint
            estimate = self._fourier_constraint(estimate, magnitude)
            
            # ER support constraint
            estimate = self._support_constraint_er(estimate, support)
            
            # Compute error
            error = self._compute_error(estimate, magnitude)
            error_history.append(error)
            
            # Check convergence
            if i > 0 and abs(error_history[-1] - error_history[-2]) < self.tol:
                break
        
        return estimate, error_history
    
    def hybrid(
        self,
        magnitude: np.ndarray,
        support: np.ndarray,
        hio_iter: int = 800,
        er_iter: int = 200,
        initial_phase: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        HIO + ER hybrid strategy.
        
        Run HIO for initial iterations, then switch to ER for refinement.
        
        Args:
            magnitude: Measured Fourier magnitude (centered)
            support: Binary support mask in real space
            hio_iter: Number of HIO iterations
            er_iter: Number of ER iterations
            initial_phase: Optional initial phase estimate
            
        Returns:
            Tuple of (reconstructed density, combined error history)
        """
        # Store original max_iter
        original_max_iter = self.max_iter
        
        # Phase 1: HIO
        self.max_iter = hio_iter
        estimate, hio_errors = self.hio(magnitude, support, initial_phase)
        
        # Get final phase from HIO result
        f_estimate = fftshift(fft2(estimate))
        hio_phase = np.angle(f_estimate)
        
        # Phase 2: ER refinement
        self.max_iter = er_iter
        estimate, er_errors = self.er(magnitude, support, hio_phase)
        
        # Restore original max_iter
        self.max_iter = original_max_iter
        
        # Combine error histories
        error_history = hio_errors + er_errors
        
        return estimate, error_history
