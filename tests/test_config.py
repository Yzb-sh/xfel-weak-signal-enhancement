"""
Property-based tests for configuration module.

**Feature: deepphase-x, Property 1: 配置序列化 Round-Trip**
**Validates: Requirements 1.7, 1.8**
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config_loader import (
    SimulationConfig,
    ModelConfig,
    GANConfig,
    TrainingConfig,
    CurriculumConfig,
    ConfigLoader,
    ConfigParseError,
    ConfigValidationError,
)


# =============================================================================
# Hypothesis Strategies for generating valid configurations
# =============================================================================

@st.composite
def simulation_config_strategy(draw):
    """Generate valid SimulationConfig instances."""
    return SimulationConfig(
        wavelength=draw(st.floats(min_value=1e-12, max_value=1e-8, allow_nan=False, allow_infinity=False)),
        pixel_size=draw(st.floats(min_value=1e-7, max_value=1e-3, allow_nan=False, allow_infinity=False)),
        detector_dist=draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)),
        grid_size=draw(st.integers(min_value=32, max_value=512)),
        resolution=draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)),
        oversampling_ratio=draw(st.floats(min_value=2.0, max_value=4.0, allow_nan=False, allow_infinity=False)),
        beam_stop_radius=draw(st.integers(min_value=0, max_value=20)),
    )


@st.composite
def model_config_strategy(draw):
    """Generate valid ModelConfig instances."""
    return ModelConfig(
        in_channels=draw(st.integers(min_value=1, max_value=4)),
        out_channels=draw(st.integers(min_value=1, max_value=4)),
        base_filters=draw(st.sampled_from([32, 64, 128])),
        depth=draw(st.integers(min_value=2, max_value=6)),
        use_fourier_conv=draw(st.booleans()),
        dropout=draw(st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def gan_config_strategy(draw):
    """Generate valid GANConfig instances."""
    return GANConfig(
        mode=draw(st.sampled_from(["cyclegan", "wgan-gp"])),
        lambda_cycle=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        lambda_identity=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        lambda_physics=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        patch_size=draw(st.sampled_from([32, 64, 128])),
        spectral_disc=draw(st.booleans()),
        signal_threshold=draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)),
        generator_filters=draw(st.sampled_from([32, 64, 128])),
        discriminator_filters=draw(st.sampled_from([32, 64, 128])),
        n_residual_blocks=draw(st.integers(min_value=3, max_value=12)),
    )


@st.composite
def training_config_strategy(draw):
    """Generate valid TrainingConfig instances."""
    return TrainingConfig(
        batch_size=draw(st.integers(min_value=1, max_value=64)),
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False, allow_infinity=False)),
        epochs=draw(st.integers(min_value=1, max_value=500)),
        loss_type=draw(st.sampled_from(["pnll", "mse"])),
        freq_weight=draw(st.booleans()),
        high_freq_weight=draw(st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        use_amp=draw(st.booleans()),
        distributed=draw(st.booleans()),
        num_workers=draw(st.integers(min_value=0, max_value=8)),
        save_interval=draw(st.integers(min_value=1, max_value=50)),
        curriculum=CurriculumConfig(
            phase1_epochs=draw(st.integers(min_value=1, max_value=100)),
            phase2_epochs=draw(st.integers(min_value=1, max_value=100)),
        ),
    )


# =============================================================================
# Property Tests: Round-Trip Serialization
# =============================================================================

class TestSimulationConfigRoundTrip:
    """
    **Feature: deepphase-x, Property 1: 配置序列化 Round-Trip**
    **Validates: Requirements 1.7, 1.8**
    
    For any valid SimulationConfig, serializing to YAML and deserializing
    should produce an equivalent configuration.
    """
    
    @settings(max_examples=100)
    @given(simulation_config_strategy())
    def test_yaml_roundtrip(self, config: SimulationConfig):
        """Test YAML serialization round-trip for SimulationConfig."""
        yaml_str = config.to_yaml()
        restored = SimulationConfig.from_yaml(yaml_str)
        
        assert config.wavelength == pytest.approx(restored.wavelength, rel=1e-9)
        assert config.pixel_size == pytest.approx(restored.pixel_size, rel=1e-9)
        assert config.detector_dist == pytest.approx(restored.detector_dist, rel=1e-9)
        assert config.grid_size == restored.grid_size
        assert config.resolution == pytest.approx(restored.resolution, rel=1e-9)
        assert config.oversampling_ratio == pytest.approx(restored.oversampling_ratio, rel=1e-9)
        assert config.beam_stop_radius == restored.beam_stop_radius
    
    @settings(max_examples=100)
    @given(simulation_config_strategy())
    def test_dict_roundtrip(self, config: SimulationConfig):
        """Test dictionary serialization round-trip for SimulationConfig."""
        d = config.to_dict()
        restored = SimulationConfig.from_dict(d)
        
        assert config.wavelength == pytest.approx(restored.wavelength, rel=1e-9)
        assert config.grid_size == restored.grid_size


class TestModelConfigRoundTrip:
    """
    **Feature: deepphase-x, Property 1: 配置序列化 Round-Trip**
    **Validates: Requirements 1.7, 1.8**
    """
    
    @settings(max_examples=100)
    @given(model_config_strategy())
    def test_yaml_roundtrip(self, config: ModelConfig):
        """Test YAML serialization round-trip for ModelConfig."""
        yaml_str = config.to_yaml()
        restored = ModelConfig.from_yaml(yaml_str)
        
        assert config.in_channels == restored.in_channels
        assert config.out_channels == restored.out_channels
        assert config.base_filters == restored.base_filters
        assert config.depth == restored.depth
        assert config.use_fourier_conv == restored.use_fourier_conv
        assert config.dropout == pytest.approx(restored.dropout, rel=1e-9)


class TestGANConfigRoundTrip:
    """
    **Feature: deepphase-x, Property 1: 配置序列化 Round-Trip**
    **Validates: Requirements 1.7, 1.8**
    """
    
    @settings(max_examples=100)
    @given(gan_config_strategy())
    def test_yaml_roundtrip(self, config: GANConfig):
        """Test YAML serialization round-trip for GANConfig."""
        yaml_str = config.to_yaml()
        restored = GANConfig.from_yaml(yaml_str)
        
        assert config.mode == restored.mode
        assert config.lambda_cycle == pytest.approx(restored.lambda_cycle, rel=1e-9)
        assert config.lambda_identity == pytest.approx(restored.lambda_identity, rel=1e-9)
        assert config.lambda_physics == pytest.approx(restored.lambda_physics, rel=1e-9)
        assert config.patch_size == restored.patch_size
        assert config.spectral_disc == restored.spectral_disc
        assert config.signal_threshold == pytest.approx(restored.signal_threshold, rel=1e-9)


class TestTrainingConfigRoundTrip:
    """
    **Feature: deepphase-x, Property 1: 配置序列化 Round-Trip**
    **Validates: Requirements 1.7, 1.8**
    """
    
    @settings(max_examples=100)
    @given(training_config_strategy())
    def test_yaml_roundtrip(self, config: TrainingConfig):
        """Test YAML serialization round-trip for TrainingConfig."""
        yaml_str = config.to_yaml()
        restored = TrainingConfig.from_yaml(yaml_str)
        
        assert config.batch_size == restored.batch_size
        assert config.learning_rate == pytest.approx(restored.learning_rate, rel=1e-9)
        assert config.epochs == restored.epochs
        assert config.loss_type == restored.loss_type
        assert config.freq_weight == restored.freq_weight
        assert config.use_amp == restored.use_amp
        assert config.distributed == restored.distributed


# =============================================================================
# Unit Tests: Validation
# =============================================================================

class TestConfigValidation:
    """Test configuration validation."""
    
    def test_simulation_config_invalid_wavelength(self):
        """Test that negative wavelength raises error."""
        with pytest.raises(ConfigValidationError):
            SimulationConfig(wavelength=-1e-10)
    
    def test_simulation_config_invalid_oversampling(self):
        """Test that oversampling < 2 raises error."""
        with pytest.raises(ConfigValidationError):
            SimulationConfig(oversampling_ratio=1.5)
    
    def test_model_config_invalid_channels(self):
        """Test that non-positive channels raises error."""
        with pytest.raises(ConfigValidationError):
            ModelConfig(in_channels=0)
    
    def test_model_config_invalid_dropout(self):
        """Test that dropout >= 1 raises error."""
        with pytest.raises(ConfigValidationError):
            ModelConfig(dropout=1.0)
    
    def test_gan_config_invalid_mode(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ConfigValidationError):
            GANConfig(mode="invalid")
    
    def test_training_config_invalid_loss_type(self):
        """Test that invalid loss type raises error."""
        with pytest.raises(ConfigValidationError):
            TrainingConfig(loss_type="invalid")


# =============================================================================
# Integration Tests: ConfigLoader
# =============================================================================

class TestConfigLoader:
    """Test ConfigLoader functionality."""
    
    def test_save_and_load_single(self):
        """Test saving and loading a single config file."""
        config = SimulationConfig(grid_size=256)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_config.yaml"
            ConfigLoader.save(config, str(filepath))
            
            loaded = ConfigLoader.load_single(str(filepath), "simulation")
            
            assert loaded.grid_size == 256
    
    def test_save_and_load_all(self):
        """Test saving and loading all config files."""
        configs = {
            "simulation": SimulationConfig(grid_size=256),
            "model": ModelConfig(base_filters=128),
            "gan": GANConfig(mode="wgan-gp"),
            "training": TrainingConfig(batch_size=32),
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ConfigLoader.save_all(configs, tmpdir)
            loaded = ConfigLoader.load(tmpdir)
            
            assert loaded["simulation"].grid_size == 256
            assert loaded["model"].base_filters == 128
            assert loaded["gan"].mode == "wgan-gp"
            assert loaded["training"].batch_size == 32
    
    def test_load_with_defaults(self):
        """Test loading from empty directory uses defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = ConfigLoader.load(tmpdir)
            
            # Should have all configs with default values
            assert "simulation" in configs
            assert "model" in configs
            assert "gan" in configs
            assert "training" in configs
            
            # Check default values
            assert configs["simulation"].grid_size == 128
            assert configs["model"].in_channels == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
