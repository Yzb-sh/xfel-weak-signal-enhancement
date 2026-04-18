"""
Evaluation Metrics for DeepPhase-X.

Implements metrics for evaluating diffraction pattern quality,
phase retrieval performance, and noise generation quality.
"""

import numpy as np
from scipy import stats
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
from typing import Tuple, List, Optional, Callable
import warnings


class DiffractionMetrics:
    """Metrics for evaluating diffraction pattern quality."""
    
    @staticmethod
    def r_factor(pred: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> float:
        """
        Calculate R-factor (crystallographic reliability factor).
        
        R = Σ|F_obs - F_calc| / Σ|F_obs|
        
        For intensity: R = Σ|√I_obs - √I_calc| / Σ|√I_obs|
        
        Args:
            pred: Predicted diffraction intensity
            target: Target diffraction intensity
            eps: Small value to avoid division by zero
            
        Returns:
            R-factor value in range [0, 2]
        """
        # Work with amplitudes (sqrt of intensity)
        pred_amp = np.sqrt(np.maximum(pred, 0))
        target_amp = np.sqrt(np.maximum(target, 0))
        
        numerator = np.sum(np.abs(target_amp - pred_amp))
        denominator = np.sum(np.abs(target_amp)) + eps
        
        r = numerator / denominator
        
        # Clamp to valid range [0, 2]
        return float(np.clip(r, 0, 2))
    
    @staticmethod
    def psnr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        PSNR = 10 * log10(MAX^2 / MSE)
        
        Args:
            pred: Predicted diffraction pattern
            target: Target diffraction pattern
            eps: Small value to avoid log(0)
            
        Returns:
            PSNR value in dB
        """
        mse = np.mean((pred - target) ** 2)
        if mse < eps:
            return float('inf')
        
        max_val = np.max(target)
        if max_val < eps:
            max_val = 1.0
        
        psnr = 10 * np.log10(max_val ** 2 / (mse + eps))
        return float(psnr)
    
    @staticmethod
    def ssim(
        pred: np.ndarray,
        target: np.ndarray,
        k1: float = 0.01,
        k2: float = 0.03,
        win_size: int = 7
    ) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            pred: Predicted diffraction pattern
            target: Target diffraction pattern
            k1: Constant for luminance (default: 0.01)
            k2: Constant for contrast (default: 0.03)
            win_size: Window size for local statistics
            
        Returns:
            SSIM value in range [-1, 1], higher is better
        """
        # Dynamic range
        L = max(target.max() - target.min(), 1e-10)
        c1 = (k1 * L) ** 2
        c2 = (k2 * L) ** 2
        
        # Local means using Gaussian filter
        sigma = win_size / 6.0
        mu_pred = gaussian_filter(pred.astype(np.float64), sigma)
        mu_target = gaussian_filter(target.astype(np.float64), sigma)
        
        # Local variances and covariance
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = gaussian_filter(pred.astype(np.float64) ** 2, sigma) - mu_pred_sq
        sigma_target_sq = gaussian_filter(target.astype(np.float64) ** 2, sigma) - mu_target_sq
        sigma_pred_target = gaussian_filter(
            pred.astype(np.float64) * target.astype(np.float64), sigma
        ) - mu_pred_target
        
        # SSIM formula
        numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
        denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        
        ssim_map = numerator / (denominator + 1e-10)
        
        return float(np.mean(ssim_map))
    
    @staticmethod
    def normalized_mse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-10) -> float:
        """
        Calculate normalized mean squared error.
        
        NMSE = MSE / Var(target)
        
        Args:
            pred: Predicted diffraction pattern
            target: Target diffraction pattern
            eps: Small value to avoid division by zero
            
        Returns:
            Normalized MSE value
        """
        mse = np.mean((pred - target) ** 2)
        var = np.var(target) + eps
        return float(mse / var)


class PhaseRetrievalMetrics:
    """Metrics for evaluating phase retrieval quality."""
    
    @staticmethod
    def prtf(
        reconstructions: List[np.ndarray],
        resolution_bins: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Phase Retrieval Transfer Function.
        
        PRTF measures the consistency of phase across multiple
        independent reconstructions as a function of resolution.
        
        PRTF(q) = |<F(q)>| / <|F(q)|>
        
        Args:
            reconstructions: List of independent reconstructions
            resolution_bins: Optional resolution bin edges
            
        Returns:
            Tuple of (resolution bins, PRTF values in [0, 1])
        """
        if len(reconstructions) < 2:
            raise ValueError("Need at least 2 reconstructions for PRTF")
        
        # Get shape
        shape = reconstructions[0].shape
        size = shape[0]
        
        # Compute FFT of all reconstructions
        ffts = [fftshift(fft2(r)) for r in reconstructions]
        
        # Create radial bins
        y, x = np.ogrid[:size, :size]
        center = size // 2
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        
        if resolution_bins is None:
            max_r = size // 2
            resolution_bins = np.linspace(0, max_r, min(size // 4, 50))
        
        # Compute PRTF for each bin
        prtf_values = []
        bin_centers = []
        
        for i in range(len(resolution_bins) - 1):
            r_min, r_max = resolution_bins[i], resolution_bins[i + 1]
            mask = (r >= r_min) & (r < r_max)
            
            if not mask.any():
                continue
            
            # Average complex amplitude
            avg_complex = np.mean([f[mask] for f in ffts], axis=0)
            avg_magnitude = np.abs(avg_complex)
            
            # Average of magnitudes
            avg_of_magnitudes = np.mean([np.abs(f[mask]) for f in ffts], axis=0)
            
            # PRTF = |<F>| / <|F|>
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prtf = np.mean(avg_magnitude) / (np.mean(avg_of_magnitudes) + 1e-10)
            
            # Clamp to [0, 1]
            prtf = np.clip(prtf, 0, 1)
            
            prtf_values.append(prtf)
            bin_centers.append((r_min + r_max) / 2)
        
        return np.array(bin_centers), np.array(prtf_values)
    
    @staticmethod
    def fsc(
        recon1: np.ndarray,
        recon2: np.ndarray,
        resolution_bins: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Fourier Shell Correlation (2D version: Fourier Ring Correlation).
        
        FSC measures the correlation between two independent reconstructions
        as a function of spatial frequency.
        
        FSC(q) = Re(Σ F1(q) * F2*(q)) / sqrt(Σ|F1(q)|² * Σ|F2(q)|²)
        
        Args:
            recon1: First reconstruction
            recon2: Second reconstruction
            resolution_bins: Optional resolution bin edges
            
        Returns:
            Tuple of (resolution bins, FSC values in [-1, 1])
        """
        # Compute FFTs
        f1 = fftshift(fft2(recon1))
        f2 = fftshift(fft2(recon2))
        
        size = recon1.shape[0]
        
        # Create radial bins
        y, x = np.ogrid[:size, :size]
        center = size // 2
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
        
        if resolution_bins is None:
            max_r = size // 2
            resolution_bins = np.linspace(0, max_r, min(size // 4, 50))
        
        # Compute FSC for each bin
        fsc_values = []
        bin_centers = []
        
        for i in range(len(resolution_bins) - 1):
            r_min, r_max = resolution_bins[i], resolution_bins[i + 1]
            mask = (r >= r_min) & (r < r_max)
            
            if not mask.any():
                continue
            
            # Cross-correlation
            cross = np.sum(f1[mask] * np.conj(f2[mask]))
            
            # Auto-correlations
            auto1 = np.sum(np.abs(f1[mask]) ** 2)
            auto2 = np.sum(np.abs(f2[mask]) ** 2)
            
            # FSC
            denominator = np.sqrt(auto1 * auto2)
            if denominator > 1e-10:
                fsc = np.real(cross) / denominator
            else:
                fsc = 0.0
            
            # Clamp to [-1, 1]
            fsc = np.clip(fsc, -1, 1)
            
            fsc_values.append(fsc)
            bin_centers.append((r_min + r_max) / 2)
        
        return np.array(bin_centers), np.array(fsc_values)
    
    @staticmethod
    def reconstruction_error(
        reconstructed: np.ndarray,
        original: np.ndarray
    ) -> float:
        """
        Calculate normalized reconstruction error.
        
        Args:
            reconstructed: Reconstructed density
            original: Original density
            
        Returns:
            Normalized error value
        """
        # Normalize both to [0, 1]
        recon_norm = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-10)
        orig_norm = (original - original.min()) / (original.max() - original.min() + 1e-10)
        
        error = np.sqrt(np.sum((recon_norm - orig_norm) ** 2) / np.sum(orig_norm ** 2 + 1e-10))
        return float(error)


class NoiseMetrics:
    """Metrics for evaluating noise generation quality."""
    
    @staticmethod
    def fid(
        generated: np.ndarray,
        real: np.ndarray,
        feature_extractor: Optional[Callable] = None
    ) -> float:
        """
        Calculate Fréchet Inception Distance (simplified version).
        
        For diffraction patterns, we use statistical moments instead of
        deep features when no feature extractor is provided.
        
        This is a simplified FID that uses mean squared difference
        of statistics rather than full covariance computation.
        
        Args:
            generated: Generated samples (N, H, W) or (H, W)
            real: Real samples (N, H, W) or (H, W)
            feature_extractor: Optional feature extraction function
            
        Returns:
            FID-like score (lower is better)
        """
        # Ensure 3D arrays
        if generated.ndim == 2:
            generated = generated[np.newaxis, ...]
        if real.ndim == 2:
            real = real[np.newaxis, ...]
        
        if feature_extractor is not None:
            # Use provided feature extractor
            gen_features = feature_extractor(generated)
            real_features = feature_extractor(real)
        else:
            # Use simple statistics as features
            # Compute per-sample statistics to avoid large covariance matrices
            gen_stats = []
            real_stats = []
            
            for g in generated:
                gen_stats.append([
                    np.mean(g),
                    np.std(g),
                    np.median(g),
                    np.percentile(g, 25),
                    np.percentile(g, 75),
                ])
            
            for r in real:
                real_stats.append([
                    np.mean(r),
                    np.std(r),
                    np.median(r),
                    np.percentile(r, 25),
                    np.percentile(r, 75),
                ])
            
            gen_features = np.array(gen_stats)
            real_features = np.array(real_stats)
        
        # Compute mean statistics
        mu_gen = np.mean(gen_features, axis=0)
        mu_real = np.mean(real_features, axis=0)
        
        # Mean difference term (simplified FID)
        diff = mu_gen - mu_real
        mean_term = np.sum(diff ** 2)
        
        # Add variance difference term
        var_gen = np.var(gen_features, axis=0)
        var_real = np.var(real_features, axis=0)
        var_term = np.sum((np.sqrt(var_gen + 1e-10) - np.sqrt(var_real + 1e-10)) ** 2)
        
        fid = mean_term + var_term
        
        return float(max(fid, 0))
    
    @staticmethod
    def radial_psd(pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate radial Power Spectral Density.
        
        Args:
            pattern: Diffraction pattern (2D array)
            
        Returns:
            Tuple of (radii, PSD values)
        """
        # Compute 2D FFT and power spectrum
        f = fftshift(fft2(pattern))
        psd_2d = np.abs(f) ** 2
        
        size = pattern.shape[0]
        center = size // 2
        
        # Create radial coordinate
        y, x = np.ogrid[:size, :size]
        r = np.sqrt((x - center) ** 2 + (y - center) ** 2).astype(int)
        
        # Compute radial average
        max_r = min(center, size - center)
        radii = np.arange(0, max_r)
        psd_radial = np.zeros(max_r)
        
        for i in radii:
            mask = r == i
            if mask.any():
                psd_radial[i] = np.mean(psd_2d[mask])
        
        # Ensure non-negative (should already be, but for safety)
        psd_radial = np.maximum(psd_radial, 0)
        
        return radii.astype(float), psd_radial
    
    @staticmethod
    def ks_test(
        generated: np.ndarray,
        real: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Tests whether two samples come from the same distribution.
        
        Args:
            generated: Generated sample
            real: Real sample
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        result = stats.ks_2samp(generated.flatten(), real.flatten())
        return float(result.statistic), float(result.pvalue)
    
    @staticmethod
    def autocorrelation_error(
        denoised: np.ndarray,
        original: np.ndarray
    ) -> float:
        """
        Calculate autocorrelation error.
        
        Compares the autocorrelation functions of denoised and original
        patterns to assess preservation of structural information.
        
        Args:
            denoised: Denoised diffraction pattern
            original: Original clean diffraction pattern
            
        Returns:
            Normalized autocorrelation error
        """
        # Compute autocorrelations via FFT
        acf_denoised = np.abs(fftshift(fft2(denoised))) ** 2
        acf_original = np.abs(fftshift(fft2(original))) ** 2
        
        # Normalize
        acf_denoised = acf_denoised / (acf_denoised.max() + 1e-10)
        acf_original = acf_original / (acf_original.max() + 1e-10)
        
        # Compute normalized error
        error = np.sqrt(np.sum((acf_denoised - acf_original) ** 2) / np.sum(acf_original ** 2 + 1e-10))
        
        return float(error)
    
    @staticmethod
    def noise_level_estimate(pattern: np.ndarray, method: str = "mad") -> float:
        """
        Estimate noise level in a pattern.
        
        Args:
            pattern: Input pattern
            method: Estimation method ("mad" for median absolute deviation,
                   "std" for standard deviation of high-frequency components)
            
        Returns:
            Estimated noise standard deviation
        """
        if method == "mad":
            # Median Absolute Deviation (robust estimator)
            median = np.median(pattern)
            mad = np.median(np.abs(pattern - median))
            # Scale factor for Gaussian distribution
            sigma = 1.4826 * mad
        elif method == "std":
            # High-frequency component std
            # Apply high-pass filter
            from scipy.ndimage import laplace
            high_freq = laplace(pattern)
            sigma = np.std(high_freq) / np.sqrt(2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return float(sigma)
