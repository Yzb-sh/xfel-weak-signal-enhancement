"""
Loss functions for DeepPhase-X.

Includes:
- PoissonNLLLoss: Poisson negative log-likelihood for photon counting
- FrequencyWeightedLoss: Emphasizes high-frequency details
- CycleConsistencyLoss: For CycleGAN training
- PhysicsConsistencyLoss: Preserves strong signal regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss.
    
    Appropriate for photon counting statistics where variance equals mean.
    
    L = pred - target * log(pred + eps)
    
    This is derived from the Poisson distribution:
    P(k|λ) = λ^k * e^(-λ) / k!
    -log P = λ - k*log(λ) + log(k!)
    
    Ignoring the constant term log(k!), we get: L = λ - k*log(λ)
    where λ = pred (predicted intensity) and k = target (observed counts).
    """
    
    def __init__(self, eps: float = 1e-8, reduction: str = "mean"):
        """
        Initialize PoissonNLLLoss.
        
        Args:
            eps: Small constant for numerical stability.
            reduction: Reduction method ("mean", "sum", "none").
        """
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Poisson NLL loss.
        
        Args:
            pred: Predicted intensity (must be positive).
            target: Target intensity (observed counts).
        
        Returns:
            Loss value.
        """
        # Ensure pred is positive
        pred = torch.clamp(pred, min=self.eps)
        
        # Poisson NLL: pred - target * log(pred)
        loss = pred - target * torch.log(pred + self.eps)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FrequencyWeightedLoss(nn.Module):
    """
    Frequency-Weighted Loss.
    
    Applies higher weights to high-frequency components in Fourier space.
    This counteracts the tendency of neural networks to learn low-frequency
    features first (spectral bias) and helps preserve fine details.
    """
    
    def __init__(
        self,
        high_freq_weight: float = 2.0,
        base_loss: str = "l1",
        reduction: str = "mean"
    ):
        """
        Initialize FrequencyWeightedLoss.
        
        Args:
            high_freq_weight: Weight multiplier for high frequencies.
            base_loss: Base loss type ("l1", "l2", "pnll").
            reduction: Reduction method.
        """
        super().__init__()
        self.high_freq_weight = high_freq_weight
        self.base_loss = base_loss
        self.reduction = reduction
        self._weight_cache = {}
    
    def _get_frequency_weights(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate radial frequency weights.
        
        Higher frequencies get higher weights.
        """
        cache_key = (height, width, str(device))
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]
        
        # Create frequency grid
        fy = torch.fft.fftfreq(height, device=device)
        fx = torch.fft.fftfreq(width, device=device)
        fy, fx = torch.meshgrid(fy, fx, indexing='ij')
        
        # Radial frequency (normalized to [0, 1])
        freq_radius = torch.sqrt(fx**2 + fy**2)
        freq_radius = freq_radius / freq_radius.max()
        
        # Weight: 1 at low freq, high_freq_weight at high freq
        weights = 1.0 + (self.high_freq_weight - 1.0) * freq_radius
        
        self._weight_cache[cache_key] = weights
        return weights
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency-weighted loss.
        
        Args:
            pred: Predicted tensor (B, C, H, W).
            target: Target tensor (B, C, H, W).
        
        Returns:
            Weighted loss value.
        """
        B, C, H, W = pred.shape
        
        # Transform to frequency domain
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Get frequency weights
        weights = self._get_frequency_weights(H, W, pred.device)
        weights = weights.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Compute weighted difference in frequency domain
        diff = pred_fft - target_fft
        
        if self.base_loss == "l1":
            loss = torch.abs(diff) * weights
        elif self.base_loss == "l2":
            loss = (diff.real**2 + diff.imag**2) * weights
        else:
            loss = torch.abs(diff) * weights
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components.
    
    L = α * L_pnll + β * L_freq + γ * L_l1
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1,
        high_freq_weight: float = 2.0
    ):
        """
        Initialize CombinedLoss.
        
        Args:
            alpha: Weight for Poisson NLL loss.
            beta: Weight for frequency-weighted loss.
            gamma: Weight for L1 loss.
            high_freq_weight: High frequency weight for freq loss.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.pnll = PoissonNLLLoss()
        self.freq = FrequencyWeightedLoss(high_freq_weight=high_freq_weight)
        self.l1 = nn.L1Loss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Returns:
            Tuple of (total_loss, loss_dict).
        """
        loss_pnll = self.pnll(pred, target)
        loss_freq = self.freq(pred, target)
        loss_l1 = self.l1(pred, target)
        
        total = self.alpha * loss_pnll + self.beta * loss_freq + self.gamma * loss_l1
        
        loss_dict = {
            "pnll": loss_pnll.item(),
            "freq": loss_freq.item(),
            "l1": loss_l1.item(),
            "total": total.item()
        }
        
        return total, loss_dict


class CycleConsistencyLoss(nn.Module):
    """
    Cycle Consistency Loss for CycleGAN.
    
    Ensures that translating from domain A to B and back to A
    recovers the original input: ||G_BA(G_AB(x)) - x||
    """
    
    def __init__(self, loss_type: str = "l1"):
        """
        Initialize CycleConsistencyLoss.
        
        Args:
            loss_type: Base loss type ("l1" or "l2").
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cycle consistency loss.
        
        Args:
            reconstructed: Reconstructed tensor after full cycle.
            original: Original input tensor.
        
        Returns:
            Cycle consistency loss.
        """
        return self.loss_fn(reconstructed, original)


class IdentityLoss(nn.Module):
    """
    Identity Loss for CycleGAN.
    
    Encourages generator to be identity when input is from target domain:
    ||G_AB(y) - y|| where y is from domain B
    """
    
    def __init__(self, loss_type: str = "l1"):
        super().__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(
        self,
        generated: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute identity loss.
        
        Args:
            generated: Generator output when input is from target domain.
            original: Original input from target domain.
        
        Returns:
            Identity loss.
        """
        return self.loss_fn(generated, original)


class PhysicsConsistencyLoss(nn.Module):
    """
    Physics Consistency Loss.
    
    Penalizes modifications to strong signal regions where the
    signal-to-noise ratio is already high.
    
    L_consistency = || M ⊙ (G(x) - x) ||₁
    
    where M is a mask indicating strong signal regions.
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize PhysicsConsistencyLoss.
        
        Args:
            reduction: Reduction method.
        """
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        generated: torch.Tensor,
        original: torch.Tensor,
        signal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics consistency loss.
        
        Args:
            generated: Generator output.
            original: Original input.
            signal_mask: Binary mask (1 = strong signal, 0 = weak signal).
        
        Returns:
            Physics consistency loss.
        """
        # Compute modification in masked regions
        modification = torch.abs(generated - original)
        masked_modification = modification * signal_mask
        
        if self.reduction == "mean":
            # Normalize by mask area
            mask_sum = signal_mask.sum() + 1e-8
            return masked_modification.sum() / mask_sum
        elif self.reduction == "sum":
            return masked_modification.sum()
        else:
            return masked_modification


class WassersteinLoss(nn.Module):
    """
    Wasserstein Loss for WGAN.
    
    L_D = E[D(fake)] - E[D(real)]
    L_G = -E[D(fake)]
    """
    
    def __init__(self):
        super().__init__()
    
    def discriminator_loss(
        self,
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor
    ) -> torch.Tensor:
        """Discriminator loss: maximize E[D(real)] - E[D(fake)]."""
        return fake_scores.mean() - real_scores.mean()
    
    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """Generator loss: maximize E[D(fake)]."""
        return -fake_scores.mean()


class GradientPenalty(nn.Module):
    """
    Gradient Penalty for WGAN-GP.
    
    Enforces Lipschitz constraint by penalizing gradients that deviate from 1.
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        """
        Initialize GradientPenalty.
        
        Args:
            lambda_gp: Gradient penalty coefficient.
        """
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(
        self,
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            discriminator: Discriminator network.
            real: Real samples.
            fake: Fake samples.
        
        Returns:
            Gradient penalty loss.
        """
        batch_size = real.size(0)
        device = real.device
        
        # Random interpolation coefficient
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Interpolated samples
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Discriminator output for interpolated samples
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Gradient penalty: (||grad|| - 1)^2
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * penalty


def create_signal_mask(
    intensity: torch.Tensor,
    threshold_percentile: float = 90.0
) -> torch.Tensor:
    """
    Create a mask for strong signal regions.
    
    Args:
        intensity: Input intensity tensor.
        threshold_percentile: Percentile threshold for strong signals.
    
    Returns:
        Binary mask tensor.
    """
    # Compute threshold
    flat = intensity.view(intensity.size(0), -1)
    threshold = torch.quantile(flat, threshold_percentile / 100.0, dim=1)
    threshold = threshold.view(-1, 1, 1, 1)
    
    # Create mask
    mask = (intensity > threshold).float()
    
    return mask
