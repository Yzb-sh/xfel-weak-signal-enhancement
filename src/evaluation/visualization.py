"""
Visualization Module for DeepPhase-X.

Provides publication-quality plotting functions for diffraction patterns,
electron density maps, and evaluation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle
from typing import Optional, List, Tuple, Union
from pathlib import Path


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
    })


def plot_diffraction_pattern(
    pattern: np.ndarray,
    title: str = "Diffraction Pattern",
    log_scale: bool = True,
    resolution_rings: Optional[List[float]] = None,
    pixel_size: Optional[float] = None,
    wavelength: Optional[float] = None,
    detector_dist: Optional[float] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a diffraction pattern with optional resolution rings.
    
    Args:
        pattern: 2D diffraction intensity array
        title: Plot title
        log_scale: Use logarithmic color scale
        resolution_rings: List of resolution values (Å) to mark
        pixel_size: Detector pixel size (m)
        wavelength: X-ray wavelength (m)
        detector_dist: Sample-detector distance (m)
        cmap: Colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        ax: Existing axes to plot on
        colorbar: Whether to show colorbar
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    
    # Handle negative values for log scale
    plot_data = pattern.copy()
    if log_scale:
        plot_data = np.maximum(plot_data, 1e-10)
        norm = LogNorm(vmin=vmin or plot_data[plot_data > 0].min(),
                       vmax=vmax or plot_data.max())
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    
    im = ax.imshow(plot_data, cmap=cmap, norm=norm, origin='lower')
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Intensity' + (' (log)' if log_scale else ''))
    
    # Add resolution rings if parameters provided
    if resolution_rings and pixel_size and wavelength and detector_dist:
        center = np.array(pattern.shape) / 2
        for res in resolution_rings:
            # Convert resolution to pixel radius
            # q = 2π/d, q = 2π sin(θ)/λ, r = D tan(2θ)
            theta = np.arcsin(wavelength / (2 * res * 1e-10))
            r_meters = detector_dist * np.tan(2 * theta)
            r_pixels = r_meters / pixel_size
            
            circle = Circle(center[::-1], r_pixels, fill=False, 
                          color='white', linestyle='--', linewidth=0.8)
            ax.add_patch(circle)
            ax.annotate(f'{res:.1f}Å', xy=(center[1] + r_pixels * 0.7, center[0]),
                       color='white', fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Pixel')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_electron_density(
    density: np.ndarray,
    title: str = "Electron Density",
    cmap: str = "gray",
    scale_bar: Optional[float] = None,
    resolution: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot an electron density map.
    
    Args:
        density: 2D electron density array
        title: Plot title
        cmap: Colormap name
        scale_bar: Scale bar length in Angstroms
        resolution: Pixel resolution (Å/pixel)
        ax: Existing axes to plot on
        colorbar: Whether to show colorbar
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    
    im = ax.imshow(density, cmap=cmap, origin='lower')
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Electron Density (a.u.)')
    
    # Add scale bar
    if scale_bar and resolution:
        bar_pixels = scale_bar / resolution
        bar_y = density.shape[0] * 0.05
        bar_x_start = density.shape[1] * 0.05
        ax.plot([bar_x_start, bar_x_start + bar_pixels], [bar_y, bar_y],
               'w-', linewidth=3)
        ax.text(bar_x_start + bar_pixels / 2, bar_y + density.shape[0] * 0.03,
               f'{scale_bar:.0f} Å', color='white', ha='center', fontsize=9)
    
    ax.set_title(title)
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Pixel')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_prtf_fsc_curves(
    curves: List[Tuple[np.ndarray, np.ndarray, str]],
    metric_type: str = "PRTF",
    threshold: float = 0.5,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot PRTF or FSC curves with resolution threshold.
    
    Args:
        curves: List of (resolution, values, label) tuples
        metric_type: "PRTF" or "FSC"
        threshold: Resolution threshold line value
        title: Plot title
        ax: Existing axes to plot on
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves)))
    
    for (resolution, values, label), color in zip(curves, colors):
        ax.plot(resolution, values, '-', label=label, color=color, linewidth=1.5)
    
    # Add threshold line
    ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=1,
               label=f'Threshold ({threshold})')
    
    ax.set_xlabel('Spatial Frequency (1/pixel)')
    ax.set_ylabel(metric_type)
    ax.set_title(title or f'{metric_type} Curve')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_radial_psd_comparison(
    psd_curves: List[Tuple[np.ndarray, np.ndarray, str]],
    title: str = "Radial Power Spectral Density",
    log_scale: bool = True,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot radial PSD comparison between multiple patterns.
    
    Args:
        psd_curves: List of (radii, psd, label) tuples
        title: Plot title
        log_scale: Use logarithmic y-axis
        ax: Existing axes to plot on
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(psd_curves)))
    
    for (radii, psd, label), color in zip(psd_curves, colors):
        if log_scale:
            ax.semilogy(radii, psd + 1e-10, '-', label=label, color=color, linewidth=1.5)
        else:
            ax.plot(radii, psd, '-', label=label, color=color, linewidth=1.5)
    
    ax.set_xlabel('Spatial Frequency (1/pixel)')
    ax.set_ylabel('Power Spectral Density' + (' (log)' if log_scale else ''))
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: dict,
    metrics: List[str] = None,
    title: str = "Training History",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss and metrics history.
    
    Args:
        history: Dictionary with metric names as keys and lists of values
        metrics: List of metric names to plot (default: all)
        title: Plot title
        ax: Existing axes to plot on
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    if metrics is None:
        metrics = list(history.keys())
    
    n_metrics = len(metrics)
    if ax is None:
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
    
    for ax, metric in zip(axes, metrics):
        values = history.get(metric, [])
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, '-', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_comparison_grid(
    images: List[Tuple[np.ndarray, str]],
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    log_scale: bool = False,
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a grid of images for comparison.
    
    Args:
        images: List of (image, title) tuples
        ncols: Number of columns
        figsize: Figure size
        cmap: Colormap
        log_scale: Use log scale for all images
        suptitle: Super title for the figure
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for idx, (img, title) in enumerate(images):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        if log_scale:
            img_plot = np.maximum(img, 1e-10)
            im = ax.imshow(img_plot, cmap=cmap, norm=LogNorm(), origin='lower')
        else:
            im = ax.imshow(img, cmap=cmap, origin='lower')
        
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for idx in range(n_images, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].axis('off')
    
    if suptitle:
        plt.suptitle(suptitle, y=1.02, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    formats: List[str] = None,
    dpi: int = 300
) -> None:
    """
    Save figure in multiple formats.
    
    Args:
        fig: matplotlib Figure object
        path: Base path (without extension)
        formats: List of formats (default: ['pdf', 'png'])
        dpi: Resolution for raster formats
    """
    if formats is None:
        formats = ['pdf', 'png']
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        save_path = path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches='tight')


def create_summary_figure(
    clean_pattern: np.ndarray,
    noisy_pattern: np.ndarray,
    denoised_pattern: np.ndarray,
    reconstructed_density: Optional[np.ndarray] = None,
    original_density: Optional[np.ndarray] = None,
    metrics: Optional[dict] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive summary figure for a denoising result.
    
    Args:
        clean_pattern: Clean diffraction pattern
        noisy_pattern: Noisy input pattern
        denoised_pattern: Denoised output pattern
        reconstructed_density: Reconstructed electron density
        original_density: Original electron density
        metrics: Dictionary of evaluation metrics
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure object
    """
    has_density = reconstructed_density is not None
    ncols = 5 if has_density else 3
    
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    
    # Diffraction patterns
    for ax, (img, title) in zip(axes[:3], [
        (clean_pattern, 'Clean'),
        (noisy_pattern, 'Noisy'),
        (denoised_pattern, 'Denoised')
    ]):
        img_plot = np.maximum(img, 1e-10)
        im = ax.imshow(img_plot, cmap='viridis', norm=LogNorm(), origin='lower')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Density maps
    if has_density:
        for ax, (img, title) in zip(axes[3:], [
            (original_density, 'Original Density'),
            (reconstructed_density, 'Reconstructed')
        ]):
            if img is not None:
                im = ax.imshow(img, cmap='gray', origin='lower')
                ax.set_title(title)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add metrics text
    if metrics:
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        fig.text(0.02, 0.02, metrics_text, fontsize=9, 
                verticalalignment='bottom', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
