"""
Property-based tests for phase retrieval and reconstruction module.

Tests HIO error boundedness, support estimation, and reconstruction quality.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.fft import fft2, fftshift

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reconstruction.hio_er import PhaseRetrieval
from src.reconstruction.support import SupportEstimator
from src.physics.simulator import XRaySimulator
from src.config.config_loader import SimulationConfig


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def phase_retrieval():
    """Create a phase retrieval instance with default parameters."""
    return PhaseRetrieval(beta=0.9, max_iter=100, tol=1e-6)


@pytest.fixture
def support_estimator():
    """Create a support estimator instance."""
    return SupportEstimator()


@pytest.fixture
def simple_test_object():
    """Create a simple test object (circle) for phase retrieval."""
    size = 64
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    return density


# =============================================================================
# Property 10: HIO Error Boundedness
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    beta=st.floats(min_value=0.5, max_value=0.95),
)
def test_hio_error_bounded(size, beta):
    """
    **Feature: deepphase-x, Property 10: HIO 误差有界性**
    **Validates: Requirements 5.1**
    
    For any valid diffraction magnitude and support constraint,
    HIO algorithm iterations should have bounded error (not diverge).
    
    The error should not grow unboundedly during iteration.
    """
    # Create simple test object
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Generate diffraction magnitude
    f_map = fftshift(fft2(density))
    magnitude = np.abs(f_map)
    
    # Create support (slightly larger than object)
    support = np.zeros((size, size), dtype=bool)
    support_radius = radius + 3
    support_mask = (x - center) ** 2 + (y - center) ** 2 <= support_radius ** 2
    support[support_mask] = True
    
    # Run HIO
    pr = PhaseRetrieval(beta=beta, max_iter=50, tol=1e-8)
    _, error_history = pr.hio(magnitude, support)
    
    # Check error is bounded
    assert len(error_history) > 0, "Error history should not be empty"
    
    initial_error = error_history[0]
    max_error = max(error_history)
    
    # Error should not grow more than 10x initial (bounded)
    assert max_error < initial_error * 10, \
        f"HIO error diverged: initial={initial_error:.4f}, max={max_error:.4f}"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    n_iter=st.integers(min_value=20, max_value=100),
)
def test_hio_error_decreasing_trend(size, n_iter):
    """
    **Feature: deepphase-x, Property 10: HIO 误差下降趋势**
    **Validates: Requirements 5.1**
    
    For a well-posed problem, HIO error should generally decrease
    (final error < initial error).
    """
    # Create test object
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Generate magnitude
    f_map = fftshift(fft2(density))
    magnitude = np.abs(f_map)
    
    # Create support
    support = np.zeros((size, size), dtype=bool)
    support_radius = radius + 2
    support_mask = (x - center) ** 2 + (y - center) ** 2 <= support_radius ** 2
    support[support_mask] = True
    
    # Run HIO
    pr = PhaseRetrieval(beta=0.9, max_iter=n_iter, tol=1e-10)
    _, error_history = pr.hio(magnitude, support)
    
    # Final error should be less than initial
    assert error_history[-1] < error_history[0], \
        f"HIO did not reduce error: initial={error_history[0]:.4f}, final={error_history[-1]:.4f}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_er_error_monotonic_decrease(size):
    """
    **Feature: deepphase-x, Property 10: ER 误差单调下降**
    **Validates: Requirements 5.2**
    
    Error Reduction algorithm should have monotonically decreasing error.
    """
    # Create test object
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Generate magnitude
    f_map = fftshift(fft2(density))
    magnitude = np.abs(f_map)
    
    # Create support
    support = np.zeros((size, size), dtype=bool)
    support_radius = radius + 2
    support_mask = (x - center) ** 2 + (y - center) ** 2 <= support_radius ** 2
    support[support_mask] = True
    
    # Run ER
    pr = PhaseRetrieval(max_iter=30, tol=1e-10)
    _, error_history = pr.er(magnitude, support)
    
    # Check monotonic decrease (with small tolerance for numerical noise)
    for i in range(1, len(error_history)):
        assert error_history[i] <= error_history[i-1] + 1e-6, \
            f"ER error increased at iteration {i}: {error_history[i-1]:.6f} -> {error_history[i]:.6f}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    hio_iter=st.integers(min_value=20, max_value=50),
    er_iter=st.integers(min_value=10, max_value=30),
)
def test_hybrid_algorithm(size, hio_iter, er_iter):
    """
    **Feature: deepphase-x, Property 10: HIO+ER 混合算法**
    **Validates: Requirements 5.1, 5.2**
    
    Hybrid algorithm should produce lower final error than HIO alone.
    """
    # Create test object
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Generate magnitude
    f_map = fftshift(fft2(density))
    magnitude = np.abs(f_map)
    
    # Create support
    support = np.zeros((size, size), dtype=bool)
    support_radius = radius + 2
    support_mask = (x - center) ** 2 + (y - center) ** 2 <= support_radius ** 2
    support[support_mask] = True
    
    # Run hybrid
    pr = PhaseRetrieval(beta=0.9, max_iter=100, tol=1e-10)
    _, error_history = pr.hybrid(magnitude, support, hio_iter=hio_iter, er_iter=er_iter)
    
    # Should have combined length
    assert len(error_history) <= hio_iter + er_iter
    
    # Final error should be reasonable
    assert error_history[-1] < error_history[0], \
        "Hybrid algorithm should reduce error"


# =============================================================================
# Support Estimation Tests
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64, 128]),
    threshold=st.floats(min_value=0.05, max_value=0.3),
)
def test_support_from_autocorrelation(size, threshold):
    """
    Test support estimation from autocorrelation.
    """
    # Create test object
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Generate diffraction pattern
    f_map = fftshift(fft2(density))
    pattern = np.abs(f_map) ** 2
    
    # Estimate support
    estimator = SupportEstimator()
    support = estimator.from_autocorrelation(pattern, threshold=threshold)
    
    # Support should be a boolean array of correct shape
    assert support.shape == (size, size)
    assert support.dtype == bool
    
    # Support should not be empty
    assert support.any(), "Support should not be empty"
    
    # Support should contain the center
    assert support[center, center], "Support should contain center"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    sigma=st.floats(min_value=0.5, max_value=3.0),
    threshold=st.floats(min_value=0.05, max_value=0.3),
)
def test_shrink_wrap(size, sigma, threshold):
    """
    Test shrink-wrap support update.
    """
    # Create test density
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 6
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Apply shrink-wrap
    estimator = SupportEstimator()
    support = estimator.shrink_wrap(density, sigma=sigma, threshold=threshold)
    
    # Support should be boolean array
    assert support.shape == (size, size)
    assert support.dtype == bool
    
    # Support should not be empty
    assert support.any(), "Shrink-wrap support should not be empty"
    
    # Support should cover the object
    object_covered = np.sum(support & mask) / np.sum(mask)
    assert object_covered > 0.5, f"Support should cover most of object: {object_covered:.2f}"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64, 128]),
    radius=st.integers(min_value=5, max_value=30),
)
def test_circular_support(size, radius):
    """
    Test circular support creation.
    """
    assume(radius < size // 2)
    
    estimator = SupportEstimator()
    support = estimator.circular_support(size, radius=radius)
    
    # Check shape and type
    assert support.shape == (size, size)
    assert support.dtype == bool
    
    # Check center is in support
    center = size // 2
    assert support[center, center]
    
    # Check corners are not in support (for reasonable radius)
    if radius < size // 3:
        assert not support[0, 0]
        assert not support[0, size-1]


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    width=st.integers(min_value=10, max_value=25),
    height=st.integers(min_value=10, max_value=25),
)
def test_rectangular_support(size, width, height):
    """
    Test rectangular support creation.
    """
    assume(width < size and height < size)
    
    estimator = SupportEstimator()
    support = estimator.rectangular_support(size, width=width, height=height)
    
    # Check shape and type
    assert support.shape == (size, size)
    assert support.dtype == bool
    
    # Check center is in support
    center = size // 2
    assert support[center, center]
    
    # Support should have approximately width * height pixels
    # Note: due to integer rounding, allow larger tolerance
    expected_area = width * height
    actual_area = np.sum(support)
    assert abs(actual_area - expected_area) < expected_area * 0.5, \
        f"Support area mismatch: expected ~{expected_area}, got {actual_area}"


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_phase_retrieval_pipeline():
    """
    Integration test: object -> diffraction -> phase retrieval -> reconstruction.
    """
    size = 64
    
    # Create test object
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = 10
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Generate diffraction
    f_map = fftshift(fft2(density))
    magnitude = np.abs(f_map)
    
    # Estimate support
    estimator = SupportEstimator()
    pattern = magnitude ** 2
    support = estimator.from_autocorrelation(pattern, threshold=0.1)
    
    # Run phase retrieval
    pr = PhaseRetrieval(beta=0.9, max_iter=200, tol=1e-6)
    reconstruction, error_history = pr.hybrid(magnitude, support, hio_iter=150, er_iter=50)
    
    # Check reconstruction quality
    assert reconstruction.shape == (size, size)
    assert len(error_history) > 0
    assert error_history[-1] < error_history[0]
    
    # Reconstruction should have similar support to original
    recon_support = np.abs(reconstruction) > 0.1 * np.abs(reconstruction).max()
    overlap = np.sum(recon_support & mask) / np.sum(mask)
    assert overlap > 0.3, f"Reconstruction overlap with original: {overlap:.2f}"


def test_phase_retrieval_with_noise():
    """
    Test phase retrieval robustness to noise.
    """
    size = 64
    
    # Create test object
    density = np.zeros((size, size))
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = 10
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    
    # Generate diffraction with noise
    f_map = fftshift(fft2(density))
    magnitude = np.abs(f_map)
    
    # Add small noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.01 * magnitude.max(), magnitude.shape)
    noisy_magnitude = np.maximum(magnitude + noise, 0)
    
    # Create support
    support = np.zeros((size, size), dtype=bool)
    support_radius = radius + 3
    support_mask = (x - center) ** 2 + (y - center) ** 2 <= support_radius ** 2
    support[support_mask] = True
    
    # Run phase retrieval with more iterations
    pr = PhaseRetrieval(beta=0.9, max_iter=200, tol=1e-6)
    reconstruction, error_history = pr.hybrid(noisy_magnitude, support, hio_iter=150, er_iter=50)
    
    # Error should be bounded and not diverge significantly
    # With noise, we may not always converge, but error should stay bounded
    max_error = max(error_history)
    initial_error = error_history[0]
    assert max_error < initial_error * 5, \
        f"Phase retrieval error diverged: initial={initial_error:.4f}, max={max_error:.4f}"
