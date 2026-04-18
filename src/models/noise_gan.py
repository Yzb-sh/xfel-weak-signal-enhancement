"""
NoiseGAN module for Sim-to-Real noise transfer.

Implements:
- NoiseGenerator: Residual-based noise generator (Sim -> Real)
- PatchDiscriminator: PatchGAN discriminator
- SpectralDiscriminator: Frequency-domain discriminator
- PhysicsConstraintLayer: Enforces physical constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.layers import ResidualBlock, DoubleConv


class NoiseGenerator(nn.Module):
    """
    Noise Generator for Sim-to-Real transfer.
    
    Based on ResNet architecture with residual learning:
    output = input + learned_noise_residual
    
    This ensures the generator only learns to add realistic noise
    while preserving the underlying signal structure.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        n_residual_blocks: int = 9
    ):
        """
        Initialize NoiseGenerator.
        
        Args:
            in_channels: Number of input channels.
            base_filters: Base number of filters.
            n_residual_blocks: Number of residual blocks.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.n_residual_blocks = n_residual_blocks
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, 7, padding=3),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_filters * 4)
            for _ in range(n_residual_blocks)
        ])
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution (produces noise residual)
        self.output = nn.Sequential(
            nn.Conv2d(base_filters, in_channels, 7, padding=3),
            nn.Tanh()  # Bounded output for stable training
        )
        
        # Noise scale factor (learnable)
        self.noise_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def get_residual(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the learned noise residual.
        
        Args:
            x: Input tensor.
        
        Returns:
            Noise residual tensor.
        """
        # Encoder
        h = self.initial(x)
        h = self.down1(h)
        h = self.down2(h)
        
        # Residual blocks
        h = self.residual_blocks(h)
        
        # Decoder
        h = self.up1(h)
        h = self.up2(h)
        
        # Output noise residual
        residual = self.output(h) * self.noise_scale
        
        return residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = input + noise_residual.
        
        Args:
            x: Input tensor (simulated clean/noisy pattern).
        
        Returns:
            Output tensor with added realistic noise.
        """
        residual = self.get_residual(x)
        return x + residual


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    
    Classifies N×N patches as real or fake, which helps
    capture high-frequency details and local texture.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        n_layers: int = 3
    ):
        """
        Initialize PatchDiscriminator.
        
        Args:
            in_channels: Number of input channels.
            base_filters: Base number of filters.
            n_layers: Number of discriminator layers.
        """
        super().__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Conv2d(in_channels, base_filters, 4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(
                base_filters * nf_mult_prev,
                base_filters * nf_mult,
                4, stride=2, padding=1
            ))
            layers.append(nn.InstanceNorm2d(base_filters * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(
            base_filters * nf_mult_prev,
            base_filters * nf_mult,
            4, stride=1, padding=1
        ))
        layers.append(nn.InstanceNorm2d(base_filters * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer (1 channel for real/fake)
        layers.append(nn.Conv2d(base_filters * nf_mult, 1, 4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
        
        Returns:
            Patch-wise real/fake scores.
        """
        return self.model(x)


class SpectralDiscriminator(nn.Module):
    """
    Spectral Discriminator.
    
    Operates in frequency domain to ensure power spectral density
    consistency between generated and real noise.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64
    ):
        """
        Initialize SpectralDiscriminator.
        
        Args:
            in_channels: Number of input channels.
            base_filters: Base number of filters.
        """
        super().__init__()
        
        # Process power spectrum (2 channels: real and imaginary parts)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels * 2, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters, base_filters * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters * 2, base_filters * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_filters * 4, base_filters * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_filters * 8, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor in spatial domain.
        
        Returns:
            Real/fake score based on spectral analysis.
        """
        # Compute FFT
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft)
        
        # Stack real and imaginary parts
        x_spec = torch.cat([x_fft.real, x_fft.imag], dim=1)
        
        # Process through conv layers
        features = self.conv_layers(x_spec)
        
        # Final classification
        return self.fc(features)


class PhysicsConstraintLayer(nn.Module):
    """
    Physics Constraint Layer (non-trainable).
    
    Enforces physical constraints on generated noise:
    - Strong signal mask protection
    - Detector saturation truncation
    - Non-negativity constraint
    """
    
    def __init__(
        self,
        signal_threshold: float = 0.9,
        saturation_value: float = 1.0
    ):
        """
        Initialize PhysicsConstraintLayer.
        
        Args:
            signal_threshold: Threshold for strong signal regions (percentile).
            saturation_value: Maximum allowed value (detector saturation).
        """
        super().__init__()
        self.signal_threshold = signal_threshold
        self.saturation_value = saturation_value
    
    def create_signal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create mask for strong signal regions.
        
        Args:
            x: Input intensity tensor.
        
        Returns:
            Binary mask (1 = strong signal, 0 = weak signal).
        """
        # Compute threshold per sample
        flat = x.view(x.size(0), -1)
        threshold = torch.quantile(flat, self.signal_threshold, dim=1)
        threshold = threshold.view(-1, 1, 1, 1)
        
        return (x > threshold).float()
    
    def forward(
        self,
        generated: torch.Tensor,
        original: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply physics constraints.
        
        Args:
            generated: Generator output.
            original: Original input.
            mask: Optional pre-computed signal mask.
        
        Returns:
            Constrained output.
        """
        if mask is None:
            mask = self.create_signal_mask(original)
        
        # Preserve strong signal regions
        output = mask * original + (1 - mask) * generated
        
        # Apply saturation
        output = torch.clamp(output, 0, self.saturation_value)
        
        return output
    
    def compute_consistency_loss(
        self,
        generated: torch.Tensor,
        original: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute physics consistency loss.
        
        L_consistency = || M ⊙ (G(x) - x) ||₁
        
        Args:
            generated: Generator output.
            original: Original input.
            mask: Signal mask.
        
        Returns:
            Consistency loss value.
        """
        if mask is None:
            mask = self.create_signal_mask(original)
        
        modification = torch.abs(generated - original)
        masked_modification = modification * mask
        
        # Normalize by mask area
        mask_sum = mask.sum() + 1e-8
        return masked_modification.sum() / mask_sum


class NoiseGAN(nn.Module):
    """
    Complete NoiseGAN system.
    
    Combines generator, discriminators, and physics constraints
    for Sim-to-Real noise transfer.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        n_residual_blocks: int = 9,
        use_spectral_disc: bool = True,
        signal_threshold: float = 0.9
    ):
        """
        Initialize NoiseGAN.
        
        Args:
            in_channels: Number of input channels.
            base_filters: Base number of filters.
            n_residual_blocks: Number of residual blocks in generator.
            use_spectral_disc: Whether to use spectral discriminator.
            signal_threshold: Threshold for physics constraints.
        """
        super().__init__()
        
        # Generator (Sim -> Real)
        self.generator = NoiseGenerator(
            in_channels=in_channels,
            base_filters=base_filters,
            n_residual_blocks=n_residual_blocks
        )
        
        # Discriminators
        self.patch_disc = PatchDiscriminator(
            in_channels=in_channels,
            base_filters=base_filters
        )
        
        self.spectral_disc = None
        if use_spectral_disc:
            self.spectral_disc = SpectralDiscriminator(
                in_channels=in_channels,
                base_filters=base_filters
            )
        
        # Physics constraints
        self.physics_layer = PhysicsConstraintLayer(
            signal_threshold=signal_threshold
        )
    
    def generate(
        self,
        x: torch.Tensor,
        apply_constraints: bool = True
    ) -> torch.Tensor:
        """
        Generate realistic noise.
        
        Args:
            x: Input simulated pattern.
            apply_constraints: Whether to apply physics constraints.
        
        Returns:
            Pattern with realistic noise.
        """
        generated = self.generator(x)
        
        if apply_constraints:
            generated = self.physics_layer(generated, x)
        
        return generated
    
    def discriminate(
        self,
        x: torch.Tensor,
        use_spectral: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Discriminate real vs fake.
        
        Args:
            x: Input pattern.
            use_spectral: Whether to use spectral discriminator.
        
        Returns:
            Tuple of (patch_scores, spectral_scores).
        """
        patch_scores = self.patch_disc(x)
        
        spectral_scores = None
        if use_spectral and self.spectral_disc is not None:
            spectral_scores = self.spectral_disc(x)
        
        return patch_scores, spectral_scores
