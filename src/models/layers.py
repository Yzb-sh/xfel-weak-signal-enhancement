"""
Custom neural network layers for DeepPhase-X.

Includes FourierConv2d for frequency-domain convolution and ResidualBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FourierConv2d(nn.Module):
    """
    Fourier Convolution Layer (Global Filter).
    
    Performs convolution in the frequency domain using element-wise
    multiplication, which captures global structural information.
    
    This is equivalent to a global convolution in spatial domain.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        bias: bool = True
    ):
        """
        Initialize FourierConv2d layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            height: Expected input height.
            width: Expected input width.
            bias: Whether to include bias term.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        
        # Complex weight for frequency domain multiplication
        # Shape: (out_channels, in_channels, height, width//2+1, 2)
        # The last dimension stores real and imaginary parts
        self.complex_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, height, width // 2 + 1, 2) * 0.02
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        B, C, H, W = x.shape
        
        # 1. FFT: (B, C, H, W) -> (B, C, H, W//2+1) complex
        x_fft = torch.fft.rfft2(x, norm="ortho")
        
        # 2. Frequency domain multiplication (convolution)
        # Convert weight to complex tensor
        weight = torch.view_as_complex(self.complex_weight)
        
        # Reshape for batch matrix multiplication
        # x_fft: (B, in_channels, H, W//2+1)
        # weight: (out_channels, in_channels, H, W//2+1)
        # Output: (B, out_channels, H, W//2+1)
        
        # Use einsum for efficient computation
        out_fft = torch.einsum('bihw,oihw->bohw', x_fft, weight)
        
        # 3. IFFT: (B, out_channels, H, W//2+1) -> (B, out_channels, H, W)
        out = torch.fft.irfft2(out_fft, s=(H, W), norm="ortho")
        
        # 4. Add bias
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        
        return out


class ResidualBlock(nn.Module):
    """
    Residual Block with Instance Normalization.
    
    Uses two 3x3 convolutions with skip connection.
    """
    
    def __init__(
        self,
        channels: int,
        use_instance_norm: bool = True,
        activation: str = "relu"
    ):
        """
        Initialize ResidualBlock.
        
        Args:
            channels: Number of input/output channels.
            use_instance_norm: Whether to use instance normalization.
            activation: Activation function ("relu", "leaky_relu", "gelu").
        """
        super().__init__()
        
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        if use_instance_norm:
            layers.append(nn.InstanceNorm2d(channels))
        layers.append(self._get_activation(activation))
        
        # Second conv
        layers.append(nn.Conv2d(channels, channels, 3, padding=1))
        if use_instance_norm:
            layers.append(nn.InstanceNorm2d(channels))
        
        self.conv = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif name == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        return x + self.conv(x)


class DoubleConv(nn.Module):
    """
    Double convolution block for U-Net.
    
    Conv -> Norm -> Act -> Conv -> Norm -> Act
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        use_batch_norm: bool = True
    ):
        """
        Initialize DoubleConv block.
        
        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            mid_channels: Middle channels (defaults to out_channels).
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels
        
        if use_batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True
    ):
        """
        Initialize UpBlock.
        
        Args:
            in_channels: Input channels (from previous layer).
            out_channels: Output channels.
            bilinear: Use bilinear upsampling (vs transposed conv).
        """
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.
        
        Args:
            x1: Input from previous layer.
            x2: Skip connection from encoder.
        
        Returns:
            Output tensor.
        """
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DualInputEncoder(nn.Module):
    """
    Dual-channel input encoder.
    
    Processes intensity I and amplitude √I separately before merging.
    """
    
    def __init__(self, base_filters: int = 64):
        """
        Initialize DualInputEncoder.
        
        Args:
            base_filters: Base number of filters.
        """
        super().__init__()
        
        # Separate initial processing for I and √I
        self.intensity_conv = nn.Sequential(
            nn.Conv2d(1, base_filters // 2, 3, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True)
        )
        
        self.amplitude_conv = nn.Sequential(
            nn.Conv2d(1, base_filters // 2, 3, padding=1),
            nn.BatchNorm2d(base_filters // 2),
            nn.ReLU(inplace=True)
        )
        
        # Merge and process
        self.merge_conv = DoubleConv(base_filters, base_filters)
    
    def forward(
        self,
        intensity: torch.Tensor,
        amplitude: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            intensity: (B, 1, H, W) intensity channel.
            amplitude: (B, 1, H, W) amplitude channel.
        
        Returns:
            Merged features (B, base_filters, H, W).
        """
        i_feat = self.intensity_conv(intensity)
        a_feat = self.amplitude_conv(amplitude)
        
        # Concatenate and merge
        merged = torch.cat([i_feat, a_feat], dim=1)
        return self.merge_conv(merged)


class AttentionGate(nn.Module):
    """
    Attention Gate for U-Net.
    
    Focuses on relevant features from skip connections.
    """
    
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        """
        Initialize AttentionGate.
        
        Args:
            gate_channels: Channels from gating signal.
            skip_channels: Channels from skip connection.
            inter_channels: Intermediate channels.
        """
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            g: Gating signal from decoder.
            x: Skip connection from encoder.
        
        Returns:
            Attention-weighted skip connection.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi
