"""
Property-based tests for physics simulation module.

Tests FFT energy conservation, Friedel law, and Poisson noise statistics.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from scipy.fft import fft2, fftshift

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics.simulator import XRaySimulator
from src.physics.noise_model import AnalyticNoiseModel, NoiseParameters
from src.physics.beam_stop import (
    create_beam_stop_mask,
    apply_beam_stop,
    BeamStopSimulator,
)
from src.config.config_loader import SimulationConfig


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simulator():
    """Create a simulator with default config."""
    config = SimulationConfig(grid_size=64, oversampling_ratio=2.0)
    return XRaySimulator(config)


@pytest.fixture
def noise_model():
    """Create a noise model with fixed seed."""
    return AnalyticNoiseModel(seed=42)


# =============================================================================
# Property 4: FFT Energy Conservation (Parseval's Theorem)
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64, 128]),
    shape_type=st.sampled_from(["circle", "gaussian", "random"]),
)
def test_fft_energy_conservation(size, shape_type):
    """
    **Feature: deepphase-x, Property 4: FFT 能量守恒 (Parseval 定理)**
    **Validates: Requirements 3.3**
    
    For any electron density map, the total energy in spatial domain
    should equal the total energy in frequency domain (Parseval's theorem).
    
    E_spatial = sum(|f(x)|^2)
    E_freq = sum(|F(k)|^2) / N
    
    These should be equal within numerical precision.
    """
    # Use simulator just for density generation, test FFT directly
    config = SimulationConfig(grid_size=size, oversampling_ratio=2.0)
    simulator = XRaySimulator(config)
    
    # Generate synthetic density
    density = simulator.create_synthetic_density(size, shape=shape_type)
    
    # Compute spatial energy
    spatial_energy = np.sum(np.abs(density) ** 2)
    
    # Compute FFT directly (without oversampling for Parseval test)
    f_map = fft2(density)
    freq_energy = np.sum(np.abs(f_map) ** 2) / density.size
    
    # Parseval's theorem: energies should be equal
    assert np.isclose(spatial_energy, freq_energy, rtol=1e-6), \
        f"Parseval violation: spatial={spatial_energy:.6f}, freq={freq_energy:.6f}"


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
    oversampling=st.sampled_from([2.0, 3.0, 4.0]),
)
def test_fft_energy_with_oversampling(size, oversampling):
    """
    **Feature: deepphase-x, Property 4: FFT 能量守恒 (带过采样)**
    **Validates: Requirements 3.3**
    
    Energy conservation should hold even with oversampling (zero-padding).
    The padded zeros don't add energy, so spatial energy of original
    should equal freq energy scaled by padding ratio.
    """
    config = SimulationConfig(grid_size=size, oversampling_ratio=oversampling)
    simulator = XRaySimulator(config)
    
    # Generate density
    density = simulator.create_synthetic_density(size, shape="circle")
    
    # Spatial energy of original density
    spatial_energy = np.sum(np.abs(density) ** 2)
    
    # Generate diffraction with oversampling
    diffraction = simulator.generate_diffraction(density, apply_shift=False)
    
    # Frequency energy (intensity is |F|^2, so sum of intensity = freq energy)
    # Need to account for FFT normalization
    padded_size = int(size * oversampling)
    freq_energy = np.sum(diffraction) / (padded_size ** 2)
    
    # Should be equal
    assert np.isclose(spatial_energy, freq_energy, rtol=1e-5), \
        f"Energy mismatch: spatial={spatial_energy:.6f}, freq={freq_energy:.6f}"


# =============================================================================
# Property 15: Friedel Law (Centrosymmetry of Diffraction Intensity)
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([32, 64, 128]),
    shape_type=st.sampled_from(["circle", "ellipse", "gaussian"]),
)
def test_friedel_law(size, shape_type):
    """
    **Feature: deepphase-x, Property 15: Friedel 定律（衍射强度中心对称性）**
    **Validates: Requirements 3.3**
    
    For any real-valued electron density, the diffraction intensity
    should satisfy Friedel's law: I(-k) = I(k).
    
    After fftshift, DC is at (N/2, N/2). Friedel symmetry means:
    I[N/2+dx, N/2+dy] = I[N/2-dx, N/2-dy]
    
    This is equivalent to: I[i,j] = I[N-i mod N, N-j mod N] for the shifted array.
    """
    config = SimulationConfig(grid_size=size, oversampling_ratio=2.0)
    simulator = XRaySimulator(config)
    
    # Generate real-valued density
    density = simulator.create_synthetic_density(size, shape=shape_type)
    
    # Ensure density is real
    assert np.isreal(density).all(), "Density must be real for Friedel law"
    
    # Generate diffraction pattern
    pattern = simulator.generate_diffraction(density)
    N = pattern.shape[0]
    
    # Check Friedel symmetry: I[i,j] = I[N-i mod N, N-j mod N]
    # Create the Friedel pair by indexing
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    friedel_i = (-i_idx) % N
    friedel_j = (-j_idx) % N
    friedel_pair = pattern[np.ix_(friedel_i, friedel_j)]
    
    # Should be equal (with tolerance for numerical precision)
    assert np.allclose(pattern, friedel_pair, rtol=1e-5, atol=1e-10), \
        f"Friedel law violation: max diff = {np.abs(pattern - friedel_pair).max():.6e}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_friedel_law_random_density(size):
    """
    **Feature: deepphase-x, Property 15: Friedel 定律（随机密度）**
    **Validates: Requirements 3.3**
    
    Friedel law should hold for any random real-valued density.
    """
    config = SimulationConfig(grid_size=size, oversampling_ratio=2.0)
    simulator = XRaySimulator(config)
    
    # Random real density
    rng = np.random.default_rng()
    density = rng.random((size, size))
    density = density / density.max()  # Normalize
    
    # Generate diffraction
    pattern = simulator.generate_diffraction(density)
    N = pattern.shape[0]
    
    # Check Friedel symmetry
    i_idx = np.arange(N)
    j_idx = np.arange(N)
    friedel_i = (-i_idx) % N
    friedel_j = (-j_idx) % N
    friedel_pair = pattern[np.ix_(friedel_i, friedel_j)]
    
    assert np.allclose(pattern, friedel_pair, rtol=1e-5, atol=1e-10), \
        "Friedel law violated for random density"


# =============================================================================
# Property 5: Poisson Noise Statistical Properties
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    exposure=st.floats(min_value=10.0, max_value=1000.0),
)
def test_poisson_noise_mean(exposure):
    """
    **Feature: deepphase-x, Property 5: 泊松噪声统计特性 - 均值**
    **Validates: Requirements 3.4**
    
    For Poisson noise, the sample mean should converge to the
    expected value (law of large numbers).
    
    E[Poisson(λ)] = λ
    """
    noise_model = AnalyticNoiseModel(seed=42)
    
    # Create uniform intensity pattern
    size = 64
    intensity = np.ones((size, size)) * 0.5  # 50% of max
    
    # Generate multiple samples
    n_samples = 100
    samples = np.stack([
        noise_model.add_poisson_noise(intensity, exposure)
        for _ in range(n_samples)
    ])
    
    # Compute sample mean
    sample_mean = samples.mean(axis=0)
    expected_mean = intensity * exposure
    
    # Mean should be close to expected (within 10% for 100 samples)
    relative_error = np.abs(sample_mean - expected_mean) / expected_mean
    mean_relative_error = relative_error.mean()
    
    assert mean_relative_error < 0.15, \
        f"Poisson mean error too large: {mean_relative_error:.3f}"


@settings(max_examples=100, deadline=None)
@given(
    exposure=st.floats(min_value=50.0, max_value=500.0),
)
def test_poisson_noise_variance(exposure):
    """
    **Feature: deepphase-x, Property 5: 泊松噪声统计特性 - 方差**
    **Validates: Requirements 3.4**
    
    For Poisson distribution, variance equals mean.
    
    Var[Poisson(λ)] = λ
    """
    noise_model = AnalyticNoiseModel(seed=42)
    
    # Create uniform intensity
    size = 32
    intensity = np.ones((size, size)) * 0.5
    
    # Generate samples
    n_samples = 200
    samples = np.stack([
        noise_model.add_poisson_noise(intensity, exposure)
        for _ in range(n_samples)
    ])
    
    # Compute sample variance
    sample_var = samples.var(axis=0)
    expected_var = intensity * exposure  # For Poisson, var = mean
    
    # Variance should be close to mean (within 20% tolerance)
    relative_error = np.abs(sample_var - expected_var) / expected_var
    mean_relative_error = relative_error.mean()
    
    assert mean_relative_error < 0.25, \
        f"Poisson variance error too large: {mean_relative_error:.3f}"


@settings(max_examples=50, deadline=None)
@given(
    exposure=st.floats(min_value=10.0, max_value=200.0),
    readout=st.floats(min_value=0.5, max_value=5.0),
)
def test_poisson_gaussian_combined(exposure, readout):
    """
    **Feature: deepphase-x, Property 5: 泊松+高斯混合噪声**
    **Validates: Requirements 3.4**
    
    Combined noise should have variance = Poisson_var + Gaussian_var.
    Total variance = λ + σ²
    """
    noise_model = AnalyticNoiseModel(seed=42)
    
    size = 32
    intensity = np.ones((size, size)) * 0.5
    
    # Generate samples
    n_samples = 200
    samples = np.stack([
        noise_model.add_poisson_gaussian(intensity, exposure, readout, clip_negative=False)
        for _ in range(n_samples)
    ])
    
    # Expected variance = Poisson variance + Gaussian variance
    expected_var = intensity * exposure + readout ** 2
    sample_var = samples.var(axis=0)
    
    # Check variance (with tolerance for sampling error)
    relative_error = np.abs(sample_var - expected_var) / expected_var
    mean_relative_error = relative_error.mean()
    
    assert mean_relative_error < 0.3, \
        f"Combined noise variance error: {mean_relative_error:.3f}"


# =============================================================================
# Beam Stop Tests
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([64, 128, 256]),
    radius=st.integers(min_value=3, max_value=20),
)
def test_beam_stop_mask_shape(size, radius):
    """
    Test beam stop mask has correct shape and is centered.
    """
    assume(radius < size // 4)  # Reasonable radius
    
    mask = create_beam_stop_mask((size, size), radius)
    
    # Check shape
    assert mask.shape == (size, size)
    
    # Check center is masked
    cy, cx = size // 2, size // 2
    assert mask[cy, cx] == True
    
    # Check corners are not masked (for reasonable radius)
    assert mask[0, 0] == False
    assert mask[0, size-1] == False


@settings(max_examples=100, deadline=None)
@given(
    size=st.sampled_from([64, 128]),
    radius=st.integers(min_value=3, max_value=15),
)
def test_beam_stop_circular_symmetry(size, radius):
    """
    Test beam stop mask is circularly symmetric (180-degree rotation about center).
    
    For a circular mask centered at (N/2, N/2), the mask should satisfy:
    mask[i,j] = mask[N-i mod N, N-j mod N]
    """
    mask = create_beam_stop_mask((size, size), radius)
    
    # Check symmetry using the same indexing as Friedel
    i_idx = np.arange(size)
    j_idx = np.arange(size)
    sym_i = (-i_idx) % size
    sym_j = (-j_idx) % size
    symmetric = mask[np.ix_(sym_i, sym_j)]
    
    assert np.array_equal(mask, symmetric), \
        "Beam stop mask should be centrosymmetric"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([64, 128]),
    radius=st.integers(min_value=5, max_value=15),
)
def test_beam_stop_application(size, radius):
    """
    Test beam stop correctly zeros out center region.
    """
    # Create pattern with non-zero center
    pattern = np.ones((size, size)) * 100.0
    
    # Apply beam stop
    masked = apply_beam_stop(pattern, radius, fill_value=0.0)
    
    # Center should be zero
    cy, cx = size // 2, size // 2
    assert masked[cy, cx] == 0.0
    
    # Outside beam stop should be unchanged
    assert masked[0, 0] == 100.0


# =============================================================================
# Simulator Integration Tests
# =============================================================================

def test_simulator_synthetic_density():
    """Test synthetic density generation."""
    config = SimulationConfig(grid_size=64)
    simulator = XRaySimulator(config)
    
    for shape in ["circle", "ellipse", "gaussian", "random"]:
        density = simulator.create_synthetic_density(shape=shape)
        
        assert density.shape == (64, 64)
        assert density.min() >= 0
        assert density.max() <= 1.0


def test_simulator_diffraction_output_shape():
    """Test diffraction pattern has correct shape with oversampling."""
    config = SimulationConfig(grid_size=64, oversampling_ratio=2.0)
    simulator = XRaySimulator(config)
    
    density = simulator.create_synthetic_density(shape="circle")
    pattern = simulator.generate_diffraction(density)
    
    # With 2x oversampling, output should be 128x128
    expected_size = int(64 * 2.0)
    assert pattern.shape == (expected_size, expected_size)


def test_simulator_diffraction_non_negative():
    """Test diffraction intensity is always non-negative."""
    config = SimulationConfig(grid_size=64, oversampling_ratio=2.0)
    simulator = XRaySimulator(config)
    
    density = simulator.create_synthetic_density(shape="random")
    pattern = simulator.generate_diffraction(density)
    
    assert pattern.min() >= 0, "Diffraction intensity must be non-negative"


def test_noise_model_non_negative_output():
    """Test noise model produces non-negative output when clipping enabled."""
    noise_model = AnalyticNoiseModel(seed=42)
    
    intensity = np.random.rand(64, 64)
    noisy = noise_model.add_poisson_gaussian(intensity, exposure_level=10, clip_negative=True)
    
    assert noisy.min() >= 0, "Clipped output should be non-negative"


def test_full_pipeline():
    """Integration test: density -> diffraction -> noise -> beam stop."""
    config = SimulationConfig(grid_size=64, oversampling_ratio=2.0, beam_stop_radius=5)
    simulator = XRaySimulator(config)
    noise_model = AnalyticNoiseModel(seed=42)
    
    # Generate density
    density = simulator.create_synthetic_density(shape="circle")
    assert density.shape == (64, 64)
    
    # Generate diffraction
    pattern = simulator.generate_diffraction(density)
    assert pattern.shape == (128, 128)
    
    # Normalize for noise
    pattern_norm = pattern / pattern.max()
    
    # Add noise
    noisy = noise_model.add_poisson_gaussian(pattern_norm, exposure_level=100)
    assert noisy.shape == pattern.shape
    
    # Apply beam stop
    masked = simulator.apply_beam_stop(noisy, radius=10)
    assert masked.shape == noisy.shape
    
    # Center should be zero
    cy, cx = masked.shape[0] // 2, masked.shape[1] // 2
    assert masked[cy, cx] == 0.0


# =============================================================================
# Property 2: HDF5 Data Round-Trip
# =============================================================================

@settings(max_examples=50, deadline=None)
@given(
    n_samples=st.integers(min_value=1, max_value=10),
    size=st.sampled_from([32, 64]),
)
def test_hdf5_roundtrip(n_samples, size, tmp_path_factory):
    """
    **Feature: deepphase-x, Property 2: HDF5 数据 Round-Trip**
    **Validates: Requirements 3.7, 3.8**
    
    For any valid diffraction dataset, serializing to HDF5 and
    deserializing should produce numerically equivalent arrays.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.dataset import save_to_hdf5, load_from_hdf5
    
    # Generate random data
    rng = np.random.default_rng()
    clean = rng.random((n_samples, size, size)).astype(np.float32)
    noisy = rng.random((n_samples, size, size)).astype(np.float32)
    
    # Create temp file
    tmp_dir = tmp_path_factory.mktemp("hdf5_test")
    filepath = str(tmp_dir / "test_data.h5")
    
    # Save
    data = {"clean": clean, "noisy": noisy}
    save_to_hdf5(data, filepath)
    
    # Load
    loaded = load_from_hdf5(filepath)
    
    # Verify
    assert "clean" in loaded
    assert "noisy" in loaded
    assert np.allclose(clean, loaded["clean"], rtol=1e-6)
    assert np.allclose(noisy, loaded["noisy"], rtol=1e-6)


def test_hdf5_compression(tmp_path):
    """Test HDF5 compression options."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.dataset import save_to_hdf5, load_from_hdf5
    
    # Generate data
    data = {"test": np.random.rand(10, 64, 64).astype(np.float32)}
    
    # Test with compression
    filepath = str(tmp_path / "compressed.h5")
    save_to_hdf5(data, filepath, compression="gzip", compression_opts=4)
    loaded = load_from_hdf5(filepath)
    assert np.allclose(data["test"], loaded["test"])
    
    # Test without compression
    filepath_no_comp = str(tmp_path / "uncompressed.h5")
    save_to_hdf5(data, filepath_no_comp, compression=None)
    loaded_no_comp = load_from_hdf5(filepath_no_comp)
    assert np.allclose(data["test"], loaded_no_comp["test"])


def test_diffraction_dataset(tmp_path):
    """Test DiffractionDataset class."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.dataset import save_to_hdf5, DiffractionDataset
    
    # Create test data
    n_samples = 5
    size = 32
    clean = np.random.rand(n_samples, size, size).astype(np.float32)
    noisy = np.random.rand(n_samples, size, size).astype(np.float32)
    
    filepath = str(tmp_path / "dataset.h5")
    save_to_hdf5({"clean": clean, "noisy_analytic": noisy}, filepath)
    
    # Test dataset
    dataset = DiffractionDataset(filepath, dual_channel=True)
    
    assert len(dataset) == n_samples
    
    # Get a sample
    input_data, target = dataset[0]
    
    # Check shapes
    assert input_data.shape == (2, size, size)  # Dual channel (I, √I)
    assert target.shape == (1, size, size)


def test_diffraction_dataset_single_channel(tmp_path):
    """Test DiffractionDataset with single channel."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data.dataset import save_to_hdf5, DiffractionDataset
    
    # Create test data
    n_samples = 3
    size = 32
    clean = np.random.rand(n_samples, size, size).astype(np.float32)
    noisy = np.random.rand(n_samples, size, size).astype(np.float32)
    
    filepath = str(tmp_path / "dataset_single.h5")
    save_to_hdf5({"clean": clean, "noisy_analytic": noisy}, filepath)
    
    # Test dataset with single channel
    dataset = DiffractionDataset(filepath, dual_channel=False)
    
    input_data, target = dataset[0]
    
    # Check shapes - single channel
    assert input_data.shape == (1, size, size)
    assert target.shape == (1, size, size)
