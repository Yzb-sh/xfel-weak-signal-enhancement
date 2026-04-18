"""
Model checkpoint utilities for DeepPhase-X.

Provides functions for saving and loading model checkpoints,
including generator, discriminator weights and optimizer states.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime


def save_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save.
        filepath: Path to save checkpoint.
        optimizer: Optional optimizer to save.
        epoch: Current epoch number.
        loss: Current loss value.
        metadata: Optional additional metadata.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into.
        filepath: Path to checkpoint file.
        optimizer: Optional optimizer to load state into.
        device: Device to load tensors to.
        strict: Whether to strictly enforce state dict matching.
    
    Returns:
        Dictionary with checkpoint info (epoch, loss, metadata).
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "timestamp": checkpoint.get("timestamp", ""),
        "metadata": checkpoint.get("metadata", {}),
    }


def save_gan_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    filepath: Union[str, Path],
    g_optimizer: Optional[torch.optim.Optimizer] = None,
    d_optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    g_loss: float = 0.0,
    d_loss: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save GAN checkpoint (generator + discriminator).
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        filepath: Path to save checkpoint.
        g_optimizer: Generator optimizer.
        d_optimizer: Discriminator optimizer.
        epoch: Current epoch.
        g_loss: Generator loss.
        d_loss: Discriminator loss.
        metadata: Additional metadata.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "epoch": epoch,
        "g_loss": g_loss,
        "d_loss": d_loss,
        "timestamp": datetime.now().isoformat(),
    }
    
    if g_optimizer is not None:
        checkpoint["g_optimizer_state_dict"] = g_optimizer.state_dict()
    
    if d_optimizer is not None:
        checkpoint["d_optimizer_state_dict"] = d_optimizer.state_dict()
    
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    torch.save(checkpoint, filepath)


def load_gan_checkpoint(
    generator: nn.Module,
    discriminator: nn.Module,
    filepath: Union[str, Path],
    g_optimizer: Optional[torch.optim.Optimizer] = None,
    d_optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load GAN checkpoint.
    
    Args:
        generator: Generator model.
        discriminator: Discriminator model.
        filepath: Path to checkpoint.
        g_optimizer: Generator optimizer.
        d_optimizer: Discriminator optimizer.
        device: Device to load to.
        strict: Strict state dict matching.
    
    Returns:
        Checkpoint info dictionary.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    generator.load_state_dict(checkpoint["generator_state_dict"], strict=strict)
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=strict)
    
    if g_optimizer is not None and "g_optimizer_state_dict" in checkpoint:
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
    
    if d_optimizer is not None and "d_optimizer_state_dict" in checkpoint:
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
    
    return {
        "epoch": checkpoint.get("epoch", 0),
        "g_loss": checkpoint.get("g_loss", 0.0),
        "d_loss": checkpoint.get("d_loss", 0.0),
        "timestamp": checkpoint.get("timestamp", ""),
        "metadata": checkpoint.get("metadata", {}),
    }


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup.
    
    Keeps only the N best checkpoints based on a metric.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        metric_name: str = "loss",
        mode: str = "min"
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            max_checkpoints: Maximum number of checkpoints to keep.
            metric_name: Name of metric to track.
            mode: "min" or "max" - whether lower or higher is better.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.mode = mode
        self.checkpoints = []  # List of (metric, filepath) tuples
    
    def save(
        self,
        model: nn.Module,
        metric_value: float,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ) -> Optional[Path]:
        """
        Save checkpoint if it's among the best.
        
        Args:
            model: Model to save.
            metric_value: Current metric value.
            epoch: Current epoch.
            optimizer: Optional optimizer.
            **kwargs: Additional metadata.
        
        Returns:
            Path to saved checkpoint, or None if not saved.
        """
        # Check if this checkpoint should be saved
        should_save = len(self.checkpoints) < self.max_checkpoints
        
        if not should_save and self.checkpoints:
            # Check if better than worst checkpoint
            worst_metric, _ = self.checkpoints[-1]
            if self.mode == "min":
                should_save = metric_value < worst_metric
            else:
                should_save = metric_value > worst_metric
        
        if not should_save:
            return None
        
        # Save checkpoint
        filename = f"checkpoint_epoch{epoch:04d}_{self.metric_name}{metric_value:.4f}.pt"
        filepath = self.checkpoint_dir / filename
        
        save_checkpoint(
            model=model,
            filepath=filepath,
            optimizer=optimizer,
            epoch=epoch,
            loss=metric_value,
            metadata=kwargs
        )
        
        # Update checkpoint list
        self.checkpoints.append((metric_value, filepath))
        
        # Sort by metric
        reverse = self.mode == "max"
        self.checkpoints.sort(key=lambda x: x[0], reverse=reverse)
        
        # Remove worst checkpoints if over limit
        while len(self.checkpoints) > self.max_checkpoints:
            _, old_path = self.checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
        
        return filepath
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        if not self.checkpoints:
            return None
        return self.checkpoints[0][1]
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: p.stat().st_mtime)
