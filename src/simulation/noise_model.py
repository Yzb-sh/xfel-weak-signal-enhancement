"""Analytic noise models for X-ray diffraction patterns."""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class NoiseParameters:
    """Parameters for noise generation."""
    exposure_level: float = 100.0  # Total scattered photon count (I_sc)
    readout_noise: float = 2.0     # Gaussian readout noise std (ADU)
    dark_current: float = 0.1      # Dark current (counts/pixel)
    quantum_efficiency: float = 0.9  # Detector quantum efficiency


class AnalyticNoiseModel:
    """
    Analytic noise model for X-ray diffraction.
    
    Implements physically-motivated noise:
    - Poisson noise (photon shot noise)
    - Gaussian noise (detector readout noise)
    - Dark current
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize noise model.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
    
    def add_poisson_noise(
        self,
        intensity: np.ndarray,
        exposure_level: float = 100.0
    ) -> np.ndarray:
        """
        Add Poisson (shot) noise to intensity pattern.
        
        Models photon counting statistics where variance equals mean.
        
        Args:
            intensity: Normalized intensity pattern (sum=1, photon probability distribution).
            exposure_level: Total scattered photon count (I_sc).

        Returns:
            Noisy intensity (photon counts per pixel).
        """
        # Expected photon count at each pixel: λ_i = P_i × I_sc
        photon_counts = intensity * exposure_level

        # Ensure non-negative
        photon_counts = np.maximum(photon_counts, 0)

        # Sample from Poisson distribution: N_i ~ Poisson(λ_i)
        noisy = self.rng.poisson(photon_counts).astype(np.float64)

        return noisy
    
    def add_gaussian_noise(
        self,
        intensity: np.ndarray,
        std: float = 2.0
    ) -> np.ndarray:
        """
        Add Gaussian (readout) noise to intensity pattern.
        
        Models detector readout electronics noise.
        
        Args:
            intensity: Intensity pattern.
            std: Standard deviation of Gaussian noise.
        
        Returns:
            Noisy intensity with added Gaussian noise.
        """
        noise = self.rng.normal(0, std, intensity.shape)
        noisy = intensity + noise
        return noisy
    
    def add_poisson_gaussian(
        self,
        intensity: np.ndarray,
        exposure_level: float = 100.0,
        readout_noise: float = 2.0,
        clip_negative: bool = True
    ) -> np.ndarray:
        """
        Add combined Poisson + Gaussian noise.
        
        This is the standard noise model for X-ray detectors:
        1. Poisson noise from photon counting
        2. Gaussian noise from detector readout
        
        Args:
            intensity: Normalized intensity pattern (sum=1, photon probability distribution).
            exposure_level: Total scattered photon count (I_sc).
            readout_noise: Standard deviation of readout noise (ADU).
            clip_negative: Whether to clip negative values to zero.

        Returns:
            Noisy intensity pattern (photon counts + readout noise).
        """
        # Apply Poisson noise first
        noisy = self.add_poisson_noise(intensity, exposure_level)
        
        # Add Gaussian readout noise
        noisy = self.add_gaussian_noise(noisy, readout_noise)
        
        # Clip negative values (physical constraint)
        if clip_negative:
            noisy = np.maximum(noisy, 0)
        
        return noisy

    def add_full_noise(
        self,
        intensity: np.ndarray,
        params: Optional[NoiseParameters] = None
    ) -> np.ndarray:
        """
        Add full detector noise model.
        
        Includes:
        - Quantum efficiency
        - Dark current
        - Poisson shot noise
        - Gaussian readout noise
        
        Args:
            intensity: Normalized intensity pattern (sum=1, photon probability distribution).
            params: Noise parameters. Uses defaults if None.
        
        Returns:
            Noisy intensity pattern.
        """
        params = params or NoiseParameters()
        
        # Apply quantum efficiency
        detected = intensity * params.quantum_efficiency
        
        # Scale to photon counts and add dark current
        photon_counts = detected * params.exposure_level + params.dark_current
        
        # Poisson sampling
        noisy = self.rng.poisson(np.maximum(photon_counts, 0)).astype(np.float64)
        
        # Add readout noise
        noisy = noisy + self.rng.normal(0, params.readout_noise, intensity.shape)
        
        # Clip negative values
        noisy = np.maximum(noisy, 0)
        
        return noisy
    
    def estimate_snr(
        self,
        intensity: np.ndarray,
        exposure_level: float = 100.0,
        readout_noise: float = 2.0
    ) -> np.ndarray:
        """
        Estimate signal-to-noise ratio for each pixel.
        
        SNR = signal / sqrt(signal + readout_noise^2)
        
        Args:
            intensity: Normalized intensity pattern (sum=1, photon probability distribution).
            exposure_level: Total scattered photon count (I_sc).
            readout_noise: Standard deviation of readout noise (ADU).
        
        Returns:
            SNR map.
        """
        signal = intensity * exposure_level
        variance = signal + readout_noise ** 2
        snr = signal / np.sqrt(np.maximum(variance, 1e-10))
        return snr
    
    def generate_noise_levels(
        self,
        intensity: np.ndarray,
        exposure_levels: list,
        readout_noise: float = 2.0
    ) -> list:
        """
        Generate multiple noise realizations at different exposure levels.
        
        Useful for curriculum learning with increasing difficulty.
        
        Args:
            intensity: Normalized intensity pattern (sum=1, photon probability distribution).
            exposure_levels: List of total photon counts (I_sc values).
            readout_noise: Readout noise std.
        
        Returns:
            List of noisy patterns at each exposure level.
        """
        return [
            self.add_poisson_gaussian(intensity, exp, readout_noise)
            for exp in exposure_levels
        ]


def compute_noise_statistics(
    clean: np.ndarray,
    noisy: np.ndarray,
    exposure_level: float
) -> dict:
    """
    Compute noise statistics for validation.
    
    Args:
        clean: Clean intensity pattern.
        noisy: Noisy intensity pattern.
        exposure_level: Exposure level used.
    
    Returns:
        Dictionary with noise statistics.
    """
    expected_mean = clean * exposure_level
    actual_mean = noisy
    
    # For Poisson, variance should equal mean
    residual = noisy - expected_mean
    
    stats = {
        "mean_error": np.mean(np.abs(actual_mean - expected_mean)),
        "relative_error": np.mean(np.abs(residual) / np.maximum(expected_mean, 1)),
        "max_intensity": noisy.max(),
        "min_intensity": noisy.min(),
        "mean_intensity": noisy.mean(),
        "std_intensity": noisy.std(),
    }
    
    return stats
