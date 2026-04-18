"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
import torch
from pathlib import Path

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root):
    """Return the configs directory."""
    return project_root / "configs"


@pytest.fixture
def data_dir(project_root):
    """Return the data directory."""
    return project_root / "data"


@pytest.fixture
def sample_density():
    """Generate a sample electron density map for testing."""
    size = 64
    density = np.zeros((size, size))
    # Create a simple circular object
    y, x = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 4
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    density[mask] = 1.0
    return density


@pytest.fixture
def sample_diffraction(sample_density):
    """Generate a sample diffraction pattern from density."""
    from scipy.fft import fft2, fftshift
    f_map = fft2(sample_density)
    f_shift = fftshift(f_map)
    intensity = np.abs(f_shift) ** 2
    return intensity


@pytest.fixture
def sample_tensor():
    """Generate a sample PyTorch tensor for model testing."""
    return torch.randn(2, 1, 128, 128)
