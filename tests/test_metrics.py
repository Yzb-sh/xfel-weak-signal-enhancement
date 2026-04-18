"""
Property-based tests for evaluation metrics module.

Tests R-factor range, PRTF range, and radial PSD symmetry.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.fft import fft2, fftshift

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import DiffractionMetrics, PhaseRetrievalMetrics, NoiseMetrics


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_pattern():
    """Generate a sample diffraction pattern."""
    size = 64
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    pattern = np.exp(-r ** 2 / (2 * 10 ** 2)) * 1000
    return pattern


@pytest.fixture
def sample_reconstructions():
    """Generate sample reconstructions for PRTF testing."""
    size = 64
    rng = np.random.default_rng(42)
    
    # Base reconstruction
    base = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = 10
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    base[mask] = 1.0
    
    # Add small variations
    reconstructions = []
    for i in range(5):
        noise = rng.normal(0, 0.1, base.shape)
        reconstructions.append(base + noise)
    
    return reconstructions


# =============================================================================
# Property 11: R-factor Range Constraint
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64, 128]),
    scale=st.floats(min_value=0.1, max_value=10.0),
)
def test_r_factor_range(size, scale):
    """
    **Feature: deepphase-x, Property 11: R-factor 范围约束**
    **Validates: Requirements 6.1**
    
    For any predicted and target diffraction intensities,
    R-factor should be in range [0, 2].
    
    R = Σ|√I_obs - √I_calc| / Σ|√I_obs|
    
    Theoretical bounds:
    - R = 0 when pred == target
    - R ≤ 2 (worst case when pred and target are completely different)
    """
    rng = np.random.default_rng()
    
    # Generate random patterns
    pred = rng.random((size, size)) * scale
    target = rng.random((size, size)) * scale
    
    # Ensure non-negative
    pred = np.maximum(pred, 0)
    target = np.maximum(target, 0)
    
    r = DiffractionMetrics.r_factor(pred, target)
    
    assert 0 <= r <= 2, f"R-factor {r:.4f} out of range [0, 2]"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    scale=st.floats(min_value=1.0, max_value=100.0),
)
def test_r_factor_zero_for_identical(size, scale):
    """
    **Feature: deepphase-x, Property 11: R-factor 相同数据为零**
    **Validates: Requirements 6.1**
    
    When prediction equals target, R-factor should be 0.
    """
    rng = np.random.default_rng()
    target = rng.random((size, size)) * scale
    target = np.maximum(target, 0)
    
    r = DiffractionMetrics.r_factor(target, target)
    
    assert np.isclose(r, 0, atol=1e-6), f"R-factor for identical data should be 0, got {r:.6f}"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    noise_level=st.floats(min_value=0.01, max_value=0.5),
)
def test_r_factor_increases_with_noise(size, noise_level):
    """
    **Feature: deepphase-x, Property 11: R-factor 随噪声增加**
    **Validates: Requirements 6.1**
    
    R-factor should increase as the difference between pred and target increases.
    """
    rng = np.random.default_rng()
    
    # Create target
    target = rng.random((size, size)) * 100
    target = np.maximum(target, 1)  # Avoid zeros
    
    # Create predictions with different noise levels
    noise_small = rng.normal(0, noise_level * 0.5, target.shape)
    noise_large = rng.normal(0, noise_level * 2.0, target.shape)
    
    pred_small_noise = np.maximum(target + noise_small * target.mean(), 0)
    pred_large_noise = np.maximum(target + noise_large * target.mean(), 0)
    
    r_small = DiffractionMetrics.r_factor(pred_small_noise, target)
    r_large = DiffractionMetrics.r_factor(pred_large_noise, target)
    
    # Larger noise should generally give larger R-factor
    # (with some tolerance for random variation)
    assert r_small < r_large + 0.3, \
        f"R-factor should increase with noise: small={r_small:.4f}, large={r_large:.4f}"


# =============================================================================
# Property 12: PRTF Range Constraint
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    n_recons=st.integers(min_value=2, max_value=5),
)
def test_prtf_range(size, n_recons):
    """
    **Feature: deepphase-x, Property 12: PRTF 范围约束**
    **Validates: Requirements 6.2**
    
    For any set of phase retrieval reconstructions,
    PRTF values should be in range [0, 1].
    
    PRTF(q) = |<F(q)>| / <|F(q)|>
    
    By Cauchy-Schwarz inequality, this ratio is always ≤ 1.
    """
    rng = np.random.default_rng()
    
    # Generate base reconstruction
    base = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    base[mask] = 1.0
    
    # Generate variations
    reconstructions = []
    for _ in range(n_recons):
        noise = rng.normal(0, 0.2, base.shape)
        reconstructions.append(base + noise)
    
    # Compute PRTF
    radii, prtf_values = PhaseRetrievalMetrics.prtf(reconstructions)
    
    # All PRTF values should be in [0, 1]
    assert all(0 <= v <= 1 for v in prtf_values), \
        f"PRTF values out of range: min={prtf_values.min():.4f}, max={prtf_values.max():.4f}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_prtf_perfect_consistency(size):
    """
    **Feature: deepphase-x, Property 12: PRTF 完美一致性**
    **Validates: Requirements 6.2**
    
    When all reconstructions are identical, PRTF should be 1.0.
    """
    # Create identical reconstructions
    base = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    base[mask] = 1.0
    
    reconstructions = [base.copy() for _ in range(3)]
    
    # Compute PRTF
    radii, prtf_values = PhaseRetrievalMetrics.prtf(reconstructions)
    
    # All values should be close to 1.0
    assert all(v > 0.99 for v in prtf_values), \
        f"PRTF for identical reconstructions should be ~1.0, got min={prtf_values.min():.4f}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_fsc_range(size):
    """
    **Feature: deepphase-x, Property 12: FSC 范围约束**
    **Validates: Requirements 6.3**
    
    Fourier Shell Correlation should be in range [-1, 1].
    """
    rng = np.random.default_rng()
    
    # Generate two reconstructions
    base = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    base[mask] = 1.0
    
    recon1 = base + rng.normal(0, 0.1, base.shape)
    recon2 = base + rng.normal(0, 0.1, base.shape)
    
    # Compute FSC
    radii, fsc_values = PhaseRetrievalMetrics.fsc(recon1, recon2)
    
    # All FSC values should be in [-1, 1]
    assert all(-1 <= v <= 1 for v in fsc_values), \
        f"FSC values out of range: min={fsc_values.min():.4f}, max={fsc_values.max():.4f}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_fsc_identical_reconstructions(size):
    """
    **Feature: deepphase-x, Property 12: FSC 相同重构**
    **Validates: Requirements 6.3**
    
    FSC of identical reconstructions should be 1.0.
    """
    # Create reconstruction
    base = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    base[mask] = 1.0
    
    # FSC with itself
    radii, fsc_values = PhaseRetrievalMetrics.fsc(base, base)
    
    # All values should be 1.0
    assert all(v > 0.99 for v in fsc_values), \
        f"FSC for identical reconstructions should be 1.0, got min={fsc_values.min():.4f}"


# =============================================================================
# Property 13: Radial PSD Symmetry
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64, 128]),
)
def test_radial_psd_non_negative(size):
    """
    **Feature: deepphase-x, Property 13: 径向 PSD 非负性**
    **Validates: Requirements 6.6**
    
    For any diffraction pattern, radial PSD values should be non-negative.
    
    PSD = |F(k)|² ≥ 0 by definition.
    """
    rng = np.random.default_rng()
    
    # Generate random pattern
    pattern = rng.random((size, size)) * 100
    
    # Compute radial PSD
    radii, psd = NoiseMetrics.radial_psd(pattern)
    
    # All PSD values should be non-negative
    assert all(p >= 0 for p in psd), \
        f"PSD values should be non-negative, got min={psd.min():.6f}"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_radial_psd_centrosymmetric_pattern(size):
    """
    **Feature: deepphase-x, Property 13: 径向 PSD 中心对称图案**
    **Validates: Requirements 6.6**
    
    For a centrosymmetric pattern, the radial PSD should be well-defined
    and smooth (no artifacts from asymmetry).
    """
    # Create centrosymmetric pattern (Gaussian)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    r = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    pattern = np.exp(-r ** 2 / (2 * 10 ** 2)) * 1000
    
    # Compute radial PSD
    radii, psd = NoiseMetrics.radial_psd(pattern)
    
    # PSD should be non-negative
    assert all(p >= 0 for p in psd)
    
    # PSD should be monotonically related to radius for Gaussian
    # (generally decreasing for a Gaussian pattern)
    # Just check it's well-behaved (no NaN or Inf)
    assert not np.any(np.isnan(psd))
    assert not np.any(np.isinf(psd))


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    scale=st.floats(min_value=0.1, max_value=100.0),
)
def test_radial_psd_scaling(size, scale):
    """
    **Feature: deepphase-x, Property 13: 径向 PSD 缩放**
    **Validates: Requirements 6.6**
    
    Scaling the pattern by a factor should scale PSD by factor².
    """
    rng = np.random.default_rng()
    
    # Generate pattern
    pattern = rng.random((size, size)) * 10
    
    # Compute PSD for original and scaled
    radii1, psd1 = NoiseMetrics.radial_psd(pattern)
    radii2, psd2 = NoiseMetrics.radial_psd(pattern * scale)
    
    # PSD should scale by scale²
    expected_ratio = scale ** 2
    actual_ratio = psd2 / (psd1 + 1e-10)
    
    # Check ratio is approximately correct (with tolerance)
    mean_ratio = np.mean(actual_ratio[psd1 > 1e-6])
    assert np.isclose(mean_ratio, expected_ratio, rtol=0.1), \
        f"PSD scaling incorrect: expected {expected_ratio:.2f}, got {mean_ratio:.2f}"


# =============================================================================
# Additional Metric Tests
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_psnr_identical_images(size):
    """
    Test PSNR is infinity for identical images.
    """
    rng = np.random.default_rng()
    pattern = rng.random((size, size)) * 100
    
    psnr = DiffractionMetrics.psnr(pattern, pattern)
    
    assert psnr == float('inf'), f"PSNR for identical images should be inf, got {psnr}"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_ssim_range(size):
    """
    Test SSIM is in valid range [-1, 1].
    """
    rng = np.random.default_rng()
    
    pred = rng.random((size, size)) * 100
    target = rng.random((size, size)) * 100
    
    ssim = DiffractionMetrics.ssim(pred, target)
    
    assert -1 <= ssim <= 1, f"SSIM {ssim:.4f} out of range [-1, 1]"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_ssim_identical_images(size):
    """
    Test SSIM is 1.0 for identical images.
    """
    rng = np.random.default_rng()
    pattern = rng.random((size, size)) * 100
    
    ssim = DiffractionMetrics.ssim(pattern, pattern)
    
    assert ssim > 0.99, f"SSIM for identical images should be ~1.0, got {ssim:.4f}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_ks_test_identical_distributions(size):
    """
    Test KS test for identical distributions.
    """
    rng = np.random.default_rng()
    
    # Same distribution
    data = rng.normal(0, 1, (size, size))
    
    statistic, pvalue = NoiseMetrics.ks_test(data, data)
    
    # Identical data should have statistic = 0
    assert statistic == 0, f"KS statistic for identical data should be 0, got {statistic}"
    assert pvalue == 1.0, f"KS p-value for identical data should be 1.0, got {pvalue}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_ks_test_different_distributions(size):
    """
    Test KS test detects different distributions.
    """
    rng = np.random.default_rng()
    
    # Different distributions
    data1 = rng.normal(0, 1, (size, size))
    data2 = rng.normal(5, 1, (size, size))  # Different mean
    
    statistic, pvalue = NoiseMetrics.ks_test(data1, data2)
    
    # Should detect difference (low p-value)
    assert pvalue < 0.05, f"KS test should detect different distributions, p-value={pvalue}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_autocorrelation_error_identical(size):
    """
    Test autocorrelation error is 0 for identical patterns.
    """
    rng = np.random.default_rng()
    pattern = rng.random((size, size)) * 100
    
    error = NoiseMetrics.autocorrelation_error(pattern, pattern)
    
    assert np.isclose(error, 0, atol=1e-6), \
        f"Autocorrelation error for identical patterns should be 0, got {error}"


def test_fid_identical_samples():
    """
    Test FID is 0 for identical samples.
    """
    rng = np.random.default_rng(42)
    # Use small samples to avoid memory issues
    samples = rng.random((10, 16, 16))
    
    fid = NoiseMetrics.fid(samples, samples)
    
    assert np.isclose(fid, 0, atol=0.1), \
        f"FID for identical samples should be ~0, got {fid}"


def test_normalized_mse():
    """Test normalized MSE calculation."""
    rng = np.random.default_rng()
    
    target = rng.random((64, 64)) * 100
    pred = target + rng.normal(0, 1, target.shape)
    
    nmse = DiffractionMetrics.normalized_mse(pred, target)
    
    # NMSE should be positive
    assert nmse >= 0
    
    # NMSE for identical should be 0
    nmse_identical = DiffractionMetrics.normalized_mse(target, target)
    assert np.isclose(nmse_identical, 0, atol=1e-10)


def test_reconstruction_error():
    """Test reconstruction error calculation."""
    size = 64
    
    # Create original
    original = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = 10
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    original[mask] = 1.0
    
    # Identical reconstruction
    error_identical = PhaseRetrievalMetrics.reconstruction_error(original, original)
    assert np.isclose(error_identical, 0, atol=1e-6)
    
    # Different reconstruction
    different = np.zeros((size, size))
    different[mask] = 0.5  # Half intensity
    error_different = PhaseRetrievalMetrics.reconstruction_error(different, original)
    assert error_different > 0


def test_noise_level_estimate():
    """Test noise level estimation."""
    rng = np.random.default_rng(42)
    
    # Create pattern with known noise level
    clean = np.ones((64, 64)) * 100
    noise_std = 5.0
    noisy = clean + rng.normal(0, noise_std, clean.shape)
    
    # Estimate noise level
    estimated_mad = NoiseMetrics.noise_level_estimate(noisy, method="mad")
    
    # Should be close to true noise level (within factor of 2)
    assert 0.5 * noise_std < estimated_mad < 2.0 * noise_std, \
        f"Noise estimate {estimated_mad:.2f} far from true {noise_std:.2f}"
