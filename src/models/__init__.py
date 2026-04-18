"""Deep learning models for DeepPhase-X."""

from .layers import (
    FourierConv2d,
    ResidualBlock,
    DoubleConv,
    DownBlock,
    UpBlock,
    DualInputEncoder,
    AttentionGate,
)

from .unet_physics import (
    PhysicsUNet,
    LightweightPhysicsUNet,
)

from .noise_gan import (
    NoiseGenerator,
    PatchDiscriminator,
    SpectralDiscriminator,
    PhysicsConstraintLayer,
    NoiseGAN,
)

__all__ = [
    # Layers
    "FourierConv2d",
    "ResidualBlock",
    "DoubleConv",
    "DownBlock",
    "UpBlock",
    "DualInputEncoder",
    "AttentionGate",
    # Models
    "PhysicsUNet",
    "LightweightPhysicsUNet",
    # NoiseGAN
    "NoiseGenerator",
    "PatchDiscriminator",
    "SpectralDiscriminator",
    "PhysicsConstraintLayer",
    "NoiseGAN",
]
