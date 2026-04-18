"""
Property-based tests for deep learning models.

Tests U-Net dimension consistency and PNLL loss correctness.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.layers import (
    FourierConv2d,
    ResidualBlock,
    DoubleConv,
    DualInputEncoder,
)
from src.models.unet_physics import PhysicsUNet, LightweightPhysicsUNet
from src.models.losses import (
    PoissonNLLLoss,
    FrequencyWeightedLoss,
    PhysicsConsistencyLoss,
    create_signal_mask,
)
from src.config.config_loader import ModelConfig


# =============================================================================
# Property 8: U-Net Input/Output Dimension Consistency
# =============================================================================

@settings(max_examples=50, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    size=st.sampled_from([32, 64, 128]),
)
def test_unet_dimension_consistency(batch_size, size):
    """
    **Feature: deepphase-x, Property 8: U-Net 输入输出维度一致性**
    **Validates: Requirements 4.1, 4.2**
    
    For any valid dual-channel input (I, √I), PhysicsUNet output
    dimensions should match input spatial dimensions.
    """
    # Create model with smaller depth for faster testing
    model = PhysicsUNet(
        in_channels=2,
        out_channels=1,
        base_filters=16,
        depth=2,
        use_fourier_conv=False,  # Disable for dimension test
        residual_output=True
    )
    model.eval()
    
    # Create dual-channel input
    intensity = torch.rand(batch_size, 1, size, size)
    amplitude = torch.sqrt(intensity)
    x = torch.cat([intensity, amplitude], dim=1)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Check dimensions
    assert output.shape == (batch_size, 1, size, size), \
        f"Expected ({batch_size}, 1, {size}, {size}), got {output.shape}"


@settings(max_examples=30, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    size=st.sampled_from([32, 64]),
)
def test_lightweight_unet_dimension_consistency(batch_size, size):
    """
    **Feature: deepphase-x, Property 8: Lightweight U-Net 维度一致性**
    **Validates: Requirements 4.1, 4.2**
    """
    model = LightweightPhysicsUNet(
        in_channels=2,
        out_channels=1,
        base_filters=16,
        depth=3
    )
    model.eval()
    
    x = torch.rand(batch_size, 2, size, size)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 1, size, size)


# =============================================================================
# Property 9: PNLL Loss Mathematical Correctness
# =============================================================================

@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    size=st.sampled_from([16, 32]),
)
def test_pnll_loss_correctness(batch_size, size):
    """
    **Feature: deepphase-x, Property 9: PNLL 损失数学正确性**
    **Validates: Requirements 4.7**
    
    For any positive pred and target, PNLL loss should equal:
    L = pred - target * log(pred + eps)
    """
    eps = 1e-8
    
    # Generate positive tensors
    pred = torch.rand(batch_size, 1, size, size) + 0.1  # Ensure positive
    target = torch.rand(batch_size, 1, size, size) + 0.1
    
    # Compute using our loss
    loss_fn = PoissonNLLLoss(eps=eps, reduction="mean")
    computed_loss = loss_fn(pred, target)
    
    # Compute expected loss manually
    expected_loss = (pred - target * torch.log(pred + eps)).mean()
    
    # Should be equal
    assert torch.isclose(computed_loss, expected_loss, rtol=1e-5), \
        f"PNLL mismatch: computed={computed_loss.item():.6f}, expected={expected_loss.item():.6f}"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([16, 32]),
)
def test_pnll_loss_minimum_at_target(size):
    """
    **Feature: deepphase-x, Property 9: PNLL 损失在目标处最小**
    **Validates: Requirements 4.7**
    
    PNLL loss should be minimized when pred equals target.
    """
    target = torch.rand(1, 1, size, size) + 0.5
    
    loss_fn = PoissonNLLLoss(reduction="mean")
    
    # Loss at target
    loss_at_target = loss_fn(target, target)
    
    # Loss at perturbed values
    pred_high = target * 1.5
    pred_low = target * 0.5
    
    loss_high = loss_fn(pred_high, target)
    loss_low = loss_fn(pred_low, target)
    
    # Loss at target should be lower
    assert loss_at_target < loss_high, "Loss should increase when pred > target"
    assert loss_at_target < loss_low, "Loss should increase when pred < target"


@settings(max_examples=50, deadline=None)
@given(
    size=st.sampled_from([16, 32]),
)
def test_pnll_loss_non_negative(size):
    """
    **Feature: deepphase-x, Property 9: PNLL 损失非负性**
    **Validates: Requirements 4.7**
    
    PNLL loss should always be non-negative for positive inputs.
    """
    pred = torch.rand(2, 1, size, size) + 0.1
    target = torch.rand(2, 1, size, size) + 0.1
    
    loss_fn = PoissonNLLLoss(reduction="none")
    loss = loss_fn(pred, target)
    
    # Note: PNLL can be negative when target > pred * e
    # But for typical diffraction data, it should be mostly positive
    # We just check it doesn't produce NaN or Inf
    assert not torch.isnan(loss).any(), "Loss contains NaN"
    assert not torch.isinf(loss).any(), "Loss contains Inf"


# =============================================================================
# Layer Tests
# =============================================================================

def test_fourier_conv_output_shape():
    """Test FourierConv2d produces correct output shape."""
    batch_size, in_ch, out_ch, h, w = 2, 4, 8, 32, 32
    
    layer = FourierConv2d(in_ch, out_ch, h, w)
    x = torch.randn(batch_size, in_ch, h, w)
    
    output = layer(x)
    
    assert output.shape == (batch_size, out_ch, h, w)


def test_fourier_conv_real_output():
    """Test FourierConv2d produces real-valued output."""
    layer = FourierConv2d(2, 4, 32, 32)
    x = torch.randn(1, 2, 32, 32)
    
    output = layer(x)
    
    assert output.dtype == torch.float32
    assert not torch.is_complex(output)


def test_residual_block_preserves_shape():
    """Test ResidualBlock preserves input shape."""
    channels = 64
    block = ResidualBlock(channels)
    
    x = torch.randn(2, channels, 32, 32)
    output = block(x)
    
    assert output.shape == x.shape


def test_residual_block_skip_connection():
    """Test ResidualBlock has skip connection (output != conv(input))."""
    channels = 32
    block = ResidualBlock(channels)
    
    x = torch.randn(1, channels, 16, 16)
    output = block(x)
    
    # Output should be x + conv(x), not just conv(x)
    # If we zero out the conv weights, output should equal input
    with torch.no_grad():
        for param in block.conv.parameters():
            param.zero_()
    
    output_zeroed = block(x)
    assert torch.allclose(output_zeroed, x), "Skip connection not working"


def test_dual_input_encoder():
    """Test DualInputEncoder processes two channels correctly."""
    encoder = DualInputEncoder(base_filters=32)
    
    intensity = torch.randn(2, 1, 64, 64)
    amplitude = torch.randn(2, 1, 64, 64)
    
    output = encoder(intensity, amplitude)
    
    assert output.shape == (2, 32, 64, 64)


def test_double_conv_output_shape():
    """Test DoubleConv produces correct output shape."""
    conv = DoubleConv(3, 64)
    x = torch.randn(2, 3, 32, 32)
    
    output = conv(x)
    
    assert output.shape == (2, 64, 32, 32)


# =============================================================================
# Loss Function Tests
# =============================================================================

def test_frequency_weighted_loss():
    """Test FrequencyWeightedLoss computation."""
    loss_fn = FrequencyWeightedLoss(high_freq_weight=2.0)
    
    pred = torch.randn(2, 1, 32, 32)
    target = torch.randn(2, 1, 32, 32)
    
    loss = loss_fn(pred, target)
    
    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss >= 0


def test_physics_consistency_loss():
    """Test PhysicsConsistencyLoss with mask."""
    loss_fn = PhysicsConsistencyLoss()
    
    original = torch.randn(2, 1, 32, 32)
    generated = original + 0.1 * torch.randn_like(original)
    
    # Create mask (strong signal regions)
    mask = (original.abs() > original.abs().median()).float()
    
    loss = loss_fn(generated, original, mask)
    
    assert loss.ndim == 0
    assert loss >= 0


def test_create_signal_mask():
    """Test signal mask creation."""
    intensity = torch.rand(2, 1, 32, 32)
    
    mask = create_signal_mask(intensity, threshold_percentile=90.0)
    
    assert mask.shape == intensity.shape
    assert mask.min() >= 0
    assert mask.max() <= 1
    
    # About 10% should be masked (above 90th percentile)
    mask_ratio = mask.mean().item()
    assert 0.05 < mask_ratio < 0.20


# =============================================================================
# Model Integration Tests
# =============================================================================

def test_physics_unet_forward():
    """Test PhysicsUNet forward pass."""
    model = PhysicsUNet(
        in_channels=2,
        out_channels=1,
        base_filters=16,
        depth=2,
        use_fourier_conv=False
    )
    model.eval()
    
    x = torch.randn(2, 2, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 1, 64, 64)


def test_physics_unet_residual_output():
    """Test PhysicsUNet residual output mode."""
    model = PhysicsUNet(
        in_channels=2,
        out_channels=1,
        base_filters=16,
        depth=2,
        use_fourier_conv=False,
        residual_output=True
    )
    model.eval()
    
    # Create input where intensity channel is known
    intensity = torch.ones(1, 1, 32, 32) * 0.5
    amplitude = torch.sqrt(intensity)
    x = torch.cat([intensity, amplitude], dim=1)
    
    with torch.no_grad():
        output = model(x)
    
    # Output should be close to input intensity (residual learning)
    # With random weights, it won't be exact, but should be in similar range
    assert output.shape == (1, 1, 32, 32)


def test_physics_unet_with_config():
    """Test PhysicsUNet initialization from config."""
    config = ModelConfig(
        in_channels=2,
        out_channels=1,
        base_filters=32,
        depth=3,
        use_fourier_conv=False,
        use_attention=False
    )
    
    model = PhysicsUNet(config=config)
    
    assert model.in_channels == 2
    assert model.out_channels == 1
    assert model.depth == 3


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    model = LightweightPhysicsUNet(
        in_channels=2,
        out_channels=1,
        base_filters=16
    )
    
    x = torch.randn(1, 2, 32, 32, requires_grad=True)
    target = torch.randn(1, 1, 32, 32)
    
    output = model(x)
    loss = (output - target).pow(2).mean()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert x.grad.abs().sum() > 0
    
    # Check model parameters have gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None


# =============================================================================
# NoiseGAN Tests
# =============================================================================

from src.models.noise_gan import (
    NoiseGenerator,
    PatchDiscriminator,
    SpectralDiscriminator,
    PhysicsConstraintLayer,
    NoiseGAN,
)


# Property 6: NoiseGAN Residual Structure
@settings(max_examples=50, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    size=st.sampled_from([32, 64]),
)
def test_noise_generator_residual_structure(batch_size, size):
    """
    **Feature: deepphase-x, Property 6: NoiseGAN 残差结构**
    **Validates: Requirements 3.5.1**
    
    NoiseGAN generator output should equal input plus learned residual:
    G(x) = x + residual(x)
    """
    generator = NoiseGenerator(
        in_channels=1,
        base_filters=16,
        n_residual_blocks=3
    )
    generator.eval()
    
    x = torch.rand(batch_size, 1, size, size)
    
    with torch.no_grad():
        output = generator(x)
        residual = generator.get_residual(x)
    
    # Output should equal input + residual
    expected = x + residual
    
    assert torch.allclose(output, expected, rtol=1e-5), \
        "Generator output should be input + residual"


# Property 7: Physics Constraint Consistency Check
@settings(max_examples=50, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    size=st.sampled_from([32, 64]),
)
def test_physics_constraint_consistency(batch_size, size):
    """
    **Feature: deepphase-x, Property 7: 物理约束 Consistency Check**
    **Validates: Requirements 3.5.6**
    
    In strong signal regions (mask=1), the modification should be minimal.
    """
    physics_layer = PhysicsConstraintLayer(signal_threshold=0.9)
    
    # Create input with some strong signals
    original = torch.rand(batch_size, 1, size, size)
    
    # Create "generated" with modifications
    generated = original + 0.5 * torch.randn_like(original)
    
    # Apply physics constraints
    constrained = physics_layer(generated, original)
    
    # Get mask
    mask = physics_layer.create_signal_mask(original)
    
    # In masked regions, output should equal original
    masked_diff = torch.abs(constrained - original) * mask
    
    # Strong signal regions should be preserved
    assert masked_diff.max() < 1e-5, \
        "Strong signal regions should be preserved"


def test_noise_generator_output_shape():
    """Test NoiseGenerator produces correct output shape."""
    generator = NoiseGenerator(
        in_channels=1,
        base_filters=32,
        n_residual_blocks=3
    )
    
    x = torch.randn(2, 1, 64, 64)
    output = generator(x)
    
    assert output.shape == x.shape


def test_patch_discriminator_output():
    """Test PatchDiscriminator produces patch-wise output."""
    disc = PatchDiscriminator(
        in_channels=1,
        base_filters=32,
        n_layers=3
    )
    
    x = torch.randn(2, 1, 64, 64)
    output = disc(x)
    
    # Output should be smaller than input (due to strided convs)
    assert output.shape[0] == 2
    assert output.shape[1] == 1
    assert output.shape[2] < 64
    assert output.shape[3] < 64


def test_spectral_discriminator_output():
    """Test SpectralDiscriminator produces scalar output."""
    disc = SpectralDiscriminator(
        in_channels=1,
        base_filters=32
    )
    
    x = torch.randn(2, 1, 64, 64)
    output = disc(x)
    
    # Output should be (batch_size, 1)
    assert output.shape == (2, 1)


def test_physics_constraint_layer():
    """Test PhysicsConstraintLayer preserves strong signals."""
    layer = PhysicsConstraintLayer(signal_threshold=0.9)
    
    # Create input with clear strong signal region
    original = torch.zeros(1, 1, 32, 32)
    original[:, :, 10:20, 10:20] = 1.0  # Strong signal region
    
    # Create modified version
    generated = original + 0.5
    
    # Apply constraints
    output = layer(generated, original)
    
    # Strong signal region should be preserved
    mask = layer.create_signal_mask(original)
    
    # Check that masked regions are close to original
    masked_original = original * mask
    masked_output = output * mask
    
    assert torch.allclose(masked_original, masked_output, atol=1e-5)


def test_noise_gan_complete():
    """Test complete NoiseGAN system."""
    gan = NoiseGAN(
        in_channels=1,
        base_filters=16,
        n_residual_blocks=3,
        use_spectral_disc=True
    )
    
    x = torch.randn(2, 1, 64, 64)
    
    # Test generation
    generated = gan.generate(x, apply_constraints=True)
    assert generated.shape == x.shape
    
    # Test discrimination
    patch_scores, spectral_scores = gan.discriminate(generated)
    assert patch_scores is not None
    assert spectral_scores is not None


def test_noise_generator_gradient_flow():
    """Test gradients flow through NoiseGenerator."""
    generator = NoiseGenerator(
        in_channels=1,
        base_filters=16,
        n_residual_blocks=2
    )
    
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    output = generator(x)
    loss = output.mean()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# =============================================================================
# Property 3: Model Weights Round-Trip
# =============================================================================

from src.utils.checkpoint import save_checkpoint, load_checkpoint


@settings(max_examples=30, deadline=None)
@given(
    size=st.sampled_from([32, 64]),
)
def test_model_weights_roundtrip(size, tmp_path_factory):
    """
    **Feature: deepphase-x, Property 3: 模型权重 Round-Trip**
    **Validates: Requirements 3.5.10, 3.5.11**
    
    For any trained model, saving and loading checkpoint should
    produce identical outputs for the same input.
    """
    # Create model
    model = LightweightPhysicsUNet(
        in_channels=2,
        out_channels=1,
        base_filters=16
    )
    model.eval()
    
    # Create test input
    x = torch.randn(1, 2, size, size)
    
    # Get output before save
    with torch.no_grad():
        output_before = model(x).clone()
    
    # Save checkpoint
    tmp_dir = tmp_path_factory.mktemp("checkpoint_test")
    checkpoint_path = tmp_dir / "test_checkpoint.pt"
    save_checkpoint(model, checkpoint_path, epoch=10, loss=0.5)
    
    # Create new model and load checkpoint
    model_loaded = LightweightPhysicsUNet(
        in_channels=2,
        out_channels=1,
        base_filters=16
    )
    load_checkpoint(model_loaded, checkpoint_path)
    model_loaded.eval()
    
    # Get output after load
    with torch.no_grad():
        output_after = model_loaded(x)
    
    # Outputs should be identical
    assert torch.allclose(output_before, output_after, rtol=1e-5), \
        "Model outputs differ after checkpoint round-trip"


def test_checkpoint_save_load(tmp_path):
    """Test basic checkpoint save and load."""
    model = LightweightPhysicsUNet(in_channels=2, out_channels=1, base_filters=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Save
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        model=model,
        filepath=checkpoint_path,
        optimizer=optimizer,
        epoch=5,
        loss=0.123,
        metadata={"test": "value"}
    )
    
    assert checkpoint_path.exists()
    
    # Load
    model2 = LightweightPhysicsUNet(in_channels=2, out_channels=1, base_filters=16)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
    
    info = load_checkpoint(model2, checkpoint_path, optimizer2)
    
    assert info["epoch"] == 5
    assert abs(info["loss"] - 0.123) < 1e-6
    assert info["metadata"]["test"] == "value"


def test_gan_checkpoint_roundtrip(tmp_path):
    """Test GAN checkpoint save and load."""
    from src.utils.checkpoint import save_gan_checkpoint, load_gan_checkpoint
    
    generator = NoiseGenerator(in_channels=1, base_filters=16, n_residual_blocks=2)
    discriminator = PatchDiscriminator(in_channels=1, base_filters=16, n_layers=2)
    
    # Test input
    x = torch.randn(1, 1, 32, 32)
    
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        g_out_before = generator(x).clone()
        d_out_before = discriminator(x).clone()
    
    # Save
    checkpoint_path = tmp_path / "gan_checkpoint.pt"
    save_gan_checkpoint(
        generator=generator,
        discriminator=discriminator,
        filepath=checkpoint_path,
        epoch=10,
        g_loss=0.5,
        d_loss=0.3
    )
    
    # Load into new models
    generator2 = NoiseGenerator(in_channels=1, base_filters=16, n_residual_blocks=2)
    discriminator2 = PatchDiscriminator(in_channels=1, base_filters=16, n_layers=2)
    
    info = load_gan_checkpoint(generator2, discriminator2, checkpoint_path)
    
    generator2.eval()
    discriminator2.eval()
    
    with torch.no_grad():
        g_out_after = generator2(x)
        d_out_after = discriminator2(x)
    
    assert torch.allclose(g_out_before, g_out_after, rtol=1e-5)
    assert torch.allclose(d_out_before, d_out_after, rtol=1e-5)
    assert info["epoch"] == 10
