"""
Physics-guided U-Net for diffraction pattern denoising.

Features:
- Dual-channel input (I, √I)
- FourierConv layers for global context
- Skip connections with attention gates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config.config_loader import ModelConfig
from src.models.layers import (
    FourierConv2d,
    ResidualBlock,
    DoubleConv,
    DownBlock,
    UpBlock,
    DualInputEncoder,
    AttentionGate,
)


class PhysicsUNet(nn.Module):
    """
    Physics-guided U-Net for diffraction pattern denoising.
    
    Architecture:
    - Dual-channel input encoder (I, √I)
    - U-Net encoder-decoder with skip connections
    - Optional FourierConv layers for global context
    - Residual learning (output = input + learned_residual)
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        in_channels: int = 2,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 4,
        use_fourier_conv: bool = True,
        use_attention: bool = False,
        bilinear: bool = True,
        residual_output: bool = True
    ):
        """
        Initialize PhysicsUNet.
        
        Args:
            config: Model configuration (overrides other args if provided).
            in_channels: Number of input channels (2 for dual-channel).
            out_channels: Number of output channels.
            base_filters: Base number of filters.
            depth: Depth of U-Net (number of downsampling levels).
            use_fourier_conv: Whether to use FourierConv layers.
            use_attention: Whether to use attention gates.
            bilinear: Use bilinear upsampling (vs transposed conv).
            residual_output: Output residual (input + learned).
        """
        super().__init__()
        
        # Use config if provided
        if config is not None:
            in_channels = config.in_channels
            out_channels = config.out_channels
            base_filters = config.base_filters
            depth = config.depth
            use_fourier_conv = config.use_fourier_conv
            use_attention = config.use_attention
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.use_fourier_conv = use_fourier_conv
        self.use_attention = use_attention
        self.residual_output = residual_output
        
        # Input processing
        if in_channels == 2:
            # Dual-channel input (I, √I)
            self.input_conv = DualInputEncoder(base_filters)
        else:
            self.input_conv = DoubleConv(in_channels, base_filters)
        
        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Track encoder output channels for skip connections
        self.encoder_channels = [base_filters]  # Start with input_conv output
        
        channels = base_filters
        for i in range(depth):
            out_ch = channels * 2
            self.encoders.append(DoubleConv(channels, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            self.encoder_channels.append(out_ch)
            channels = out_ch
        
        # Bottleneck
        bottleneck_ch = channels * 2
        self.bottleneck = DoubleConv(channels, bottleneck_ch)

        # Decoder path
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if use_attention else None
        
        channels = bottleneck_ch
        for i in range(depth):
            # Skip connection comes from encoder (in reverse order)
            skip_ch = self.encoder_channels[-(i + 1)]
            out_ch = skip_ch  # Output same as skip for clean architecture
            
            if bilinear:
                self.upsamples.append(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                )
                # After upsample: channels, after concat with skip: channels + skip_ch
                self.decoders.append(DoubleConv(channels + skip_ch, out_ch))
            else:
                self.upsamples.append(
                    nn.ConvTranspose2d(channels, channels // 2, 2, stride=2)
                )
                self.decoders.append(DoubleConv(channels // 2 + skip_ch, out_ch))
            
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(channels, skip_ch, skip_ch // 2 if skip_ch > 1 else 1)
                )
            
            channels = out_ch
        
        # Output convolution - channels should now be base_filters
        self.output_conv = nn.Conv2d(channels, out_channels, 1)
        
        # Optional FourierConv for global context (applied at bottleneck)
        self.fourier_conv = None
        if use_fourier_conv:
            # Will be initialized on first forward pass with actual size
            self._fourier_initialized = False
            self._bottleneck_channels = bottleneck_ch
    
    def _init_fourier_conv(self, h: int, w: int):
        """Initialize FourierConv with actual spatial dimensions."""
        if not self._fourier_initialized and self.use_fourier_conv:
            # Bottleneck spatial size after depth downsamples
            bottleneck_h = h // (2 ** self.depth)
            bottleneck_w = w // (2 ** self.depth)
            
            self.fourier_conv = FourierConv2d(
                self._bottleneck_channels,
                self._bottleneck_channels,
                bottleneck_h,
                bottleneck_w
            ).to(next(self.parameters()).device)
            
            self._fourier_initialized = True
    
    def forward(
        self,
        x: torch.Tensor,
        intensity: Optional[torch.Tensor] = None,
        amplitude: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor. If in_channels=2, shape (B, 2, H, W).
               Otherwise, use intensity and amplitude args.
            intensity: (B, 1, H, W) intensity channel (optional).
            amplitude: (B, 1, H, W) amplitude channel (optional).
        
        Returns:
            Denoised output (B, out_channels, H, W).
        """
        # Handle dual-channel input
        if self.in_channels == 2:
            if intensity is not None and amplitude is not None:
                x = torch.cat([intensity, amplitude], dim=1)
            # x should be (B, 2, H, W)
            intensity_input = x[:, 0:1, :, :]  # For residual
        else:
            intensity_input = x[:, 0:1, :, :] if x.shape[1] > 1 else x
        
        B, C, H, W = x.shape
        
        # Initialize FourierConv if needed
        if self.use_fourier_conv and not self._fourier_initialized:
            self._init_fourier_conv(H, W)
        
        # Input processing
        if self.in_channels == 2 and hasattr(self.input_conv, 'intensity_conv'):
            x0 = self.input_conv(x[:, 0:1, :, :], x[:, 1:2, :, :])
        else:
            x0 = self.input_conv(x)
        
        # Encoder path with skip connections
        skips = [x0]
        x_enc = x0
        
        for i, (encoder, pool) in enumerate(zip(self.encoders, self.pools)):
            x_enc = encoder(x_enc)
            skips.append(x_enc)
            x_enc = pool(x_enc)
        
        # Bottleneck
        x_dec = self.bottleneck(x_enc)
        
        # Apply FourierConv at bottleneck for global context
        if self.fourier_conv is not None:
            x_dec = x_dec + self.fourier_conv(x_dec)
        
        # Decoder path with skip connections
        for i, (upsample, decoder) in enumerate(zip(self.upsamples, self.decoders)):
            x_dec = upsample(x_dec)
            
            # Get corresponding skip connection
            skip = skips[-(i + 1)]
            
            # Apply attention gate if enabled
            if self.use_attention and self.attention_gates is not None:
                skip = self.attention_gates[i](x_dec, skip)
            
            # Handle size mismatch
            if x_dec.shape[2:] != skip.shape[2:]:
                x_dec = F.interpolate(x_dec, size=skip.shape[2:], mode='bilinear', align_corners=True)
            
            # Concatenate and decode
            x_dec = torch.cat([x_dec, skip], dim=1)
            x_dec = decoder(x_dec)
        
        # Output
        out = self.output_conv(x_dec)
        
        # Residual learning: output = input + learned_residual
        if self.residual_output:
            out = intensity_input + out
        
        return out


class LightweightPhysicsUNet(nn.Module):
    """
    Lightweight version of PhysicsUNet for faster training/inference.
    
    Uses fewer filters and shallower depth.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int = 3
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Simple encoder
        self.enc1 = DoubleConv(in_channels, base_filters)
        self.enc2 = DoubleConv(base_filters, base_filters * 2)
        self.enc3 = DoubleConv(base_filters * 2, base_filters * 4)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(base_filters * 4, base_filters * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = DoubleConv(base_filters * 2, base_filters)
        
        self.output = nn.Conv2d(base_filters, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        # Residual output
        out = self.output(d1)
        return x[:, 0:1, :, :] + out  # Add to intensity channel
