# -*- coding: utf-8 -*-
"""
组会汇报 - 数据模拟模块全部图片生成脚本
生成所有汇报所需的PNG图片

Usage:
    cd tests/bio_diffraction_simulator
    python 数据模拟-组会汇报/generate_figures.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# Add paths
SCRIPT_DIR = Path(__file__).parent
SIMULATOR_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SIMULATOR_DIR.parent.parent
sys.path.insert(0, str(SIMULATOR_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Output directory
OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed parameters
SAMPLE_SIZE_PX = 70
I_SC_RANGE = [1e7, 1e9]

# Try importing project modules
try:
    from config import EXP_CONFIG, BACTERIA_CONFIG, AUGMENT_CONFIG, RANDOM_MASK_CONFIG, compute_physical_parameters
    from bio_sample_generator import BioSampleGenerator
    from data_augmentor import DataAugmentor
    from diffraction_simulator import DiffractionSimulator
    from intensity_normalizer import IntensityNormalizer
    from random_mask_applier import RandomMaskApplier
    from noise_beamstop_applier import NoiseAndBeamstopApplier
    from src.physics.noise_model import AnalyticNoiseModel
    HAS_MODULES = True
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    HAS_MODULES = False

# Try loading experimental data
EXP_DATA_DIR = SIMULATOR_DIR / '实验数据' / '实验数据1'
HAS_EXP_DATA = (EXP_DATA_DIR / 'Ecoli_AB_20min_7897_background_subtraction_result.mat').exists()

# matplotlib style setup - clean professional look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 11,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})


def fig_01_pipeline_overview():
    """图1: Pipeline总览流程图 - 改进版"""
    print("Generating fig_01_pipeline_overview.png ...")

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Title
    ax.text(7, 5.2, 'Data Simulation Pipeline Overview', fontsize=15, fontweight='bold',
            ha='center', va='top')

    # Define boxes in a single row for cleaner look
    # Pipeline order: Normalize → Noise → Defects → Beamstop
    boxes = [
        {'x': 1.0, 'label': 'Step 1', 'desc': 'BioSample\nGenerator', 'detail': 'E.coli 2D\ndensity [0,1]', 'color': '#4ECDC4'},
        {'x': 3.0, 'label': 'Step 2', 'desc': 'Data\nAugmentor', 'detail': 'rotation\nshift scale', 'color': '#45B7D1'},
        {'x': 5.0, 'label': 'Step 3', 'desc': 'Diffraction\nSimulator', 'detail': 'FFT\nI=|F|^2', 'color': '#96CEB4'},
        {'x': 7.0, 'label': 'Step 4', 'desc': 'Intensity\nNormalizer', 'detail': 'sum(I)=1\nPoisson prep', 'color': '#FFEAA7'},
        {'x': 9.0, 'label': 'Step 5', 'desc': 'Poisson+Gauss\nNoise', 'detail': 'shot noise\nread noise', 'color': '#FF6B6B'},
        {'x': 11.0, 'label': 'Step 6', 'desc': 'Random\nMask', 'detail': 'detector\ndefects 50%', 'color': '#DDA0DD'},
        {'x': 13.0, 'label': 'Output', 'desc': 'Beamstop\n+ HDF5', 'detail': 'beamstop mask\ninput/target', 'color': '#FDCB6E'},
    ]

    for i, box in enumerate(boxes):
        x = box['x']
        # Box
        rect = FancyBboxPatch((x - 0.8, 1.0), 1.6, 3.2,
                               boxstyle="round,pad=0.08", facecolor=box['color'],
                               edgecolor='#333333', linewidth=1.2, alpha=0.85)
        ax.add_patch(rect)
        # Step label
        ax.text(x, 3.7, box['label'], fontsize=9, fontweight='bold',
                ha='center', va='center', color='#222222')
        # Main description
        ax.text(x, 2.9, box['desc'], fontsize=8, fontweight='bold',
                ha='center', va='center', color='#111111')
        # Detail
        ax.text(x, 1.9, box['detail'], fontsize=6.5,
                ha='center', va='center', color='#444444')

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x_start = boxes[i]['x'] + 0.8
        x_end = boxes[i + 1]['x'] - 0.8
        y = 2.6
        ax.annotate('', xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='#555555'))

    # Bottom annotations
    ax.text(7, 0.3,
            'Shanghai SXFEL  |  585x585 Grid  |  I_sc = [1e7, 1e9]  |  Noise Calibrated from Experimental Data',
            fontsize=8, ha='center', va='center', style='italic', color='#666666')

    plt.savefig(OUT_DIR / 'fig_01_pipeline_overview.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_02_experimental_data():
    """图2: 实验数据概览与噪声分析 - 全局背景分析 + log-log Var vs Mean"""
    print("Generating fig_02_experimental_data.png ...")

    if not HAS_EXP_DATA:
        print("  -> SKIPPED (no experimental data)")
        return

    data_file = EXP_DATA_DIR / 'Ecoli_AB_20min_7897_background_subtraction_result.mat'
    mask_file = EXP_DATA_DIR / 'Ecoli_AB_20min_7897_20260322_140309_missing-latest.mat'

    data = loadmat(str(data_file))['data_sb'].astype(np.float64)
    mask = loadmat(str(mask_file))['mask'].astype(np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel (a): Full diffraction pattern (log scale)
    I_log = np.log10(1 + np.abs(data))
    im = axes[0, 0].imshow(I_log, cmap='inferno', vmin=np.percentile(I_log, 1), vmax=np.percentile(I_log, 99.9))
    axes[0, 0].set_title('(a) Experimental Diffraction Pattern\n(log scale, 4096 x 4096)')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], label='log10(1+I)', shrink=0.8)

    # Panel (b): Beamstop mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('(b) Beamstop Mask\n(blocked: {:,} pixels)'.format(int(np.sum(mask))))
    axes[0, 1].axis('off')

    # Panel (c): Valid region only
    # NOTE: colorbar range is narrower than (a) because beamstop blocks the strongest center signal
    data_valid = data.copy()
    data_valid[mask > 0.5] = 0
    I_valid_log = np.log10(1 + np.abs(data_valid))
    im = axes[0, 2].imshow(I_valid_log, cmap='inferno', vmin=0, vmax=np.percentile(I_valid_log[I_valid_log > 0], 99))
    axes[0, 2].set_title('(c) Valid Region Only\n(colorbar < (a): beamstop blocks peak)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], label='log10(1+I)', shrink=0.8)

    # Panel (d): Global background analysis (4 edges, 100px border)
    h, w = data.shape
    border = 100
    edge_patches = [
        data[:border, :][mask[:border, :] == 0],                                      # top edge
        data[h-border:, :][mask[h-border:, :] == 0],                                  # bottom edge
        data[border:h-border, :border][mask[border:h-border, :border] == 0],           # left edge (no corners)
        data[border:h-border, w-border:][mask[border:h-border, w-border:] == 0],       # right edge (no corners)
    ]
    bg_valid = np.concatenate(edge_patches)

    axes[1, 0].hist(bg_valid, bins=100, color='steelblue', alpha=0.7, edgecolor='none')
    axes[1, 0].axvline(np.mean(bg_valid), color='red', linestyle='--', linewidth=2,
                       label='Mean = {:.2f}'.format(np.mean(bg_valid)))
    axes[1, 0].axvline(np.mean(bg_valid) + np.std(bg_valid), color='orange', linestyle='--',
                       linewidth=1.5, label='Std = {:.2f}'.format(np.std(bg_valid)))
    axes[1, 0].axvline(np.mean(bg_valid) - np.std(bg_valid), color='orange', linestyle='--', linewidth=1.5)
    axes[1, 0].set_title('(d) Global Edge Background\n(4 borders, 100px width, bg-subtracted)')
    axes[1, 0].set_xlabel('Intensity (ADU)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()

    # Panel (e): Variance vs Mean analysis (log-log scale)
    patch_size = 64
    means = []
    vars_ = []
    for iy in range(0, h - patch_size, patch_size):
        for ix in range(0, w - patch_size, patch_size):
            patch = data[iy:iy+patch_size, ix:ix+patch_size]
            patch_mask = mask[iy:iy+patch_size, ix:ix+patch_size]
            valid_pixels = patch[patch_mask == 0]
            if len(valid_pixels) > patch_size * patch_size * 0.8:
                m = np.mean(valid_pixels)
                v = np.var(valid_pixels)
                if m > 0 and v > 0:
                    means.append(m)
                    vars_.append(v)

    means = np.array(means)
    vars_ = np.array(vars_)

    # Separate background patches (low mean) from signal patches (high mean)
    bg_threshold = np.percentile(means, 20)
    is_bg = means < bg_threshold

    axes[1, 1].scatter(means[is_bg], vars_[is_bg], s=3, alpha=0.3, color='steelblue', label='Background patches')
    axes[1, 1].scatter(means[~is_bg], vars_[~is_bg], s=3, alpha=0.3, color='coral', label='Signal patches')

    # Theoretical lines in log-log space
    x_min = max(means.min() * 0.5, 0.01)
    x_max = means.max() * 2
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    axes[1, 1].plot(x_line, x_line, 'r--', linewidth=2, label='Var = Mean (Poisson)')
    axes[1, 1].plot(x_line, 2*x_line, 'g--', linewidth=2, label='Var = 2$\\times$Mean (bg subtracted)')

    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    axes[1, 1].set_xlabel('Local Mean (ADU)')
    axes[1, 1].set_ylabel('Local Variance (ADU$^2$)')
    axes[1, 1].set_title('(e) Noise Characterization\n(Var vs Mean, log-log scale)')
    axes[1, 1].legend(fontsize=8)

    # Panel (f): Summary statistics table
    axes[1, 2].axis('off')
    bg_mean = np.mean(bg_valid)
    bg_std = np.std(bg_valid)
    bg_var = np.var(bg_valid)
    cv = bg_std / abs(bg_mean) if bg_mean != 0 else 0
    total_signal = np.sum(data[mask == 0])
    original_bg_photons = bg_var / 2

    stats_text = (
        f"Noise Analysis Results\n"
        f"{'='*40}\n\n"
        f"Image Size: {h} x {w}\n"
        f"Valid Pixels: {int(np.sum(mask == 0)):,}\n"
        f"Beamstop Pixels: {int(np.sum(mask)):,}\n\n"
        f"Global Edge Background:\n"
        f"  Mean:     {bg_mean:.2f} ADU\n"
        f"  Std:      {bg_std:.2f} ADU\n"
        f"  Variance: {bg_var:.2f} ADU^2\n"
        f"  CV:       {cv:.4f}\n\n"
        f"Original BG Photons: {original_bg_photons:.1f}\n"
        f"Total Signal: {total_signal:.2e}\n\n"
        f"{'='*40}\n"
        f"Recommended I_sc: [1e7, 1e9]"
    )
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 2].set_title('(f) Summary Statistics')

    fig.suptitle('Experimental Data Noise Analysis (E. coli, 20min exposure)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(OUT_DIR / 'fig_02_experimental_data.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_03_noise_calibration():
    """图3: I_sc参数选择与噪声标定 - 2D patch + 扩展SNR范围"""
    print("Generating fig_03_noise_calibration.png ...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel (a): I_sc vs noise level
    I_sc_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    bg_frac = 1e-7  # typical normalized background intensity

    bg_photons = [I_sc * bg_frac for I_sc in I_sc_values]
    bg_sigmas = [np.sqrt(p) for p in bg_photons]

    ax = axes[0]
    ax.semilogx(I_sc_values, bg_sigmas, 'bo-', markersize=8, linewidth=2, label='Background sigma')
    ax.axhline(y=10.9, color='red', linestyle='--', linewidth=2, label='Experimental BG sigma = 10.9')
    ax.axvspan(1e7, 1e9, alpha=0.15, color='green', label='Target range [1e7, 1e9]')
    ax.set_xlabel('I_sc (Total Photon Count)')
    ax.set_ylabel('Background sigma (ADU)')
    ax.set_title('(a) Noise Level vs. I_sc')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (b): 2D diffraction patches at different I_sc levels
    ax = axes[1]
    ax.set_title('(b) Noise at Different I_sc Levels\n(2D patches, same sample)')
    ax.axis('off')

    I_sc_demo = [1e7, 5e7, 1e8, 1e9]
    colors_demo = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    labels_demo = ['I_sc=1e7 (extreme)', 'I_sc=5e7 (high)', 'I_sc=1e8 (moderate)', 'I_sc=1e9 (low)']

    if HAS_MODULES:
        # Generate one sample and show at different noise levels
        gen = BioSampleGenerator(seed=42, sample_size_px=SAMPLE_SIZE_PX)
        obj = gen.generate()
        sim = DiffractionSimulator()
        I_clean = sim.simulate(obj)
        norm = IntensityNormalizer()
        I_norm = norm.normalize(I_clean)

        # Crop a 145x145 center region for display
        crop = slice(220, 365)

        for idx, I_sc_val in enumerate(I_sc_demo):
            row, col = divmod(idx, 2)
            sub_ax = ax.inset_axes([col*0.48 + 0.02, (1-row)*0.48 + 0.02, 0.44, 0.44])
            rng_local = np.random.default_rng(42)
            I_expected = I_norm * I_sc_val
            I_noisy = rng_local.poisson(np.maximum(I_expected, 0).astype(np.float64)).astype(np.float64)
            I_noisy = I_noisy + rng_local.normal(0, 5.0, I_noisy.shape)
            I_noisy = np.maximum(I_noisy, 0)
            sub_ax.imshow(np.log10(1 + I_noisy[crop, crop]), cmap='inferno')
            sub_ax.set_title('I_sc={:.0e}'.format(I_sc_val), fontsize=8)
            sub_ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'Requires project modules', ha='center', va='center',
                transform=ax.transAxes)

    # Panel (c): SNR analysis with extended x-axis range
    ax = axes[2]
    # Extend range to cover background pixels (I_norm ~ 1e-7 to 1e-9)
    signal_levels = np.logspace(-9, 0, 500)
    for I_sc, color, label in zip(I_sc_demo, colors_demo, labels_demo):
        signal = signal_levels * I_sc
        noise = np.sqrt(signal + 5.0**2)
        snr = signal / noise
        ax.semilogy(signal_levels, snr, color=color, linewidth=2, label=label)

    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1, label='SNR=1 (detection limit)')
    ax.set_xlabel('Relative Intensity (I_norm)')
    ax.set_ylabel('SNR')
    ax.set_title('(c) Signal-to-Noise Ratio\n(x-axis covers background to peak)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1e-3, 1e7)

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'fig_03_noise_calibration.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_04_cell_states():
    """图4: 三种细胞形态 - 每行强制对应一种形态"""
    print("Generating fig_04_cell_states.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    rng = np.random.default_rng(42)

    for row, state in enumerate(['normal', 'dividing', 'curved']):
        for col in range(4):
            seed = rng.integers(0, 100000)
            gen = BioSampleGenerator(seed=int(seed), sample_size_px=SAMPLE_SIZE_PX)
            gen.rng = np.random.default_rng(int(seed))
            cfg = gen._get_scaled_config()

            # Force specific cell state by calling the corresponding generator directly
            if state == 'normal':
                obj = gen._generate_normal_cell(cfg)
            elif state == 'dividing':
                obj = gen._generate_dividing_cell(cfg)
            else:
                obj = gen._generate_curved_cell(cfg)

            # Manually add internal structures (same sequence as generate())
            n_spots = gen.rng.integers(*cfg['n_gaussian_spots_range'])
            obj = gen._add_gaussian_spots(obj, n_spots, cfg)
            n_vacuoles = gen.rng.integers(*cfg['n_vacuoles_range'])
            obj = gen._add_vacuoles(obj, n_vacuoles, cfg)
            obj = gen._add_surface_noise(obj)
            obj = np.clip(obj, 0, 1).astype(np.float32)

            axes[row, col].imshow(obj, cmap='gray', vmin=0, vmax=1)
            axes[row, col].axis('off')
            if col == 0:
                state_name = {'normal': 'Normal Cell', 'dividing': 'Dividing Cell', 'curved': 'Curved Cell'}
                axes[row, col].set_ylabel(state_name[state], fontsize=12, fontweight='bold',
                                          rotation=90, labelpad=60)

    fig.suptitle('E. coli Cell Morphology Diversity', fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_04_cell_states.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_05_bio_structure_detail():
    """图5: 生物结构细节（内部结构、表面噪声）"""
    print("Generating fig_05_bio_structure_detail.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    rng = np.random.default_rng(42)
    gen = BioSampleGenerator(seed=42, sample_size_px=SAMPLE_SIZE_PX)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Create a capsule body only
    obj_base = gen._create_capsule_2d(150, 45, (292, 292), 0)
    obj_base = gen._add_membrane_gradient(obj_base)

    # With Gaussian spots
    obj_spots = obj_base.copy()
    cfg = gen._get_scaled_config()
    for _ in range(6):
        cy = rng.integers(260, 324)
        cx = rng.integers(220, 364)
        sigma = 6.0
        y_coords, x_coords = np.ogrid[:585, :585]
        spot = np.exp(-((y_coords - cy)**2 + (x_coords - cx)**2) / (2 * sigma**2)) * 0.25
        mask_bio = obj_spots > 0.1
        obj_spots[mask_bio] += spot[mask_bio]
    if obj_spots.max() > 0:
        obj_spots = obj_spots / obj_spots.max()

    # With vacuoles
    obj_vacuoles = obj_spots.copy()
    for cy, cx, r in [(280, 310, 8), (300, 260, 6)]:
        y_coords, x_coords = np.ogrid[:585, :585]
        vmask = ((y_coords - cy)**2 + (x_coords - cx)**2) <= r**2
        obj_vacuoles[vmask] = np.minimum(obj_vacuoles[vmask], 0.3)

    # Full sample with surface noise (builds on step 3, not a new sample)
    obj_full = gen._add_surface_noise(obj_vacuoles.copy())
    obj_full = np.clip(obj_full, 0, 1).astype(np.float32)

    # Display
    images = [obj_base, obj_spots, obj_vacuoles, obj_full]
    titles = ['(1) Capsule Body\n(membrane gradient)', '(2) + Gaussian Spots\n(ribosomes)',
              '(3) + Vacuoles\n(low density)', '(4) + Surface Noise\n(complete)']

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[0, i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].axis('off')

        # Histogram
        vals = img[img > 0.01]
        if len(vals) > 0:
            axes[1, i].hist(vals.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='none')
            axes[1, i].set_xlabel('Density')
            axes[1, i].set_ylabel('Count')
            axes[1, i].set_title('Range: [{:.3f}, {:.3f}]'.format(img.min(), img.max()), fontsize=9)
        else:
            axes[1, i].text(0.5, 0.5, 'Empty', ha='center', va='center')

    fig.suptitle('Bio Structure Construction Steps', fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_05_bio_structure_detail.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_06_bio_diversity():
    """图6: 生物样品多样性展示"""
    print("Generating fig_06_bio_diversity.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))

    for i in range(12):
        row, col = divmod(i, 6)
        seed = i * 137 + 42
        gen = BioSampleGenerator(seed=seed, sample_size_px=SAMPLE_SIZE_PX)
        obj = gen.generate()

        axes[row, col].imshow(obj, cmap='gray', vmin=0, vmax=1)
        axes[row, col].axis('off')
        n_nonzero = np.sum(obj > 0.1)
        axes[row, col].set_title('#{} ({}px)'.format(i+1, n_nonzero), fontsize=9)

    fig.suptitle('Bio Sample Diversity (12 random samples, size=70px)', fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_06_bio_diversity.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_07_augmentation():
    """图7: 数据增强效果展示"""
    print("Generating fig_07_augmentation.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    gen = BioSampleGenerator(seed=42, sample_size_px=SAMPLE_SIZE_PX)
    obj = gen.generate()

    augmentor = DataAugmentor(seed=42)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))

    # Original
    axes[0, 0].imshow(obj, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].axis('off')

    # Augmented samples
    for i in range(9):
        row, col = divmod(i + 1, 5)
        aug = augmentor.augment(obj, seed=i * 100 + 7)
        axes[row, col].imshow(aug, cmap='gray', vmin=0, vmax=1)
        axes[row, col].set_title('Augmented {}'.format(i+1))
        axes[row, col].axis('off')

    fig.suptitle('Data Augmentation Examples', fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_07_augmentation.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_08_diffraction_principle():
    """图8: 衍射模拟原理展示"""
    print("Generating fig_08_diffraction_principle.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    gen = BioSampleGenerator(seed=42, sample_size_px=SAMPLE_SIZE_PX)
    obj = gen.generate()

    sim = DiffractionSimulator()
    I_clean = sim.simulate(obj)
    q_values, radial_profile = sim.get_radial_profile(I_clean)
    params = sim.compute_physical_scale()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.subplots_adjust(wspace=0.4)  # Prevent (c) colorbar from overlapping (d) y-axis

    # (a) Object
    axes[0].imshow(obj, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('(a) Object\n2D density map')
    axes[0].axis('off')

    # (b) Diffraction (log)
    I_log = np.log10(1 + I_clean)
    im = axes[1].imshow(I_log, cmap='inferno')
    axes[1].set_title('(b) Diffraction Pattern\nI = |FFT|^2 (log scale)')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='log10(1+I)', shrink=0.8)

    # (c) Amplitude (sqrt of intensity)
    A = np.sqrt(I_clean)
    A_log = np.log10(1 + A)
    im = axes[2].imshow(A_log, cmap='viridis')
    axes[2].set_title('(c) Amplitude\n|FFT| (log scale)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], label='log10(1+|A|)', shrink=0.8)

    # (d) Radial profile - fixed spacing issue
    valid = radial_profile > 0
    axes[3].semilogy(q_values[valid] * 1e-9, radial_profile[valid], 'b-', linewidth=1.5)
    axes[3].set_xlabel('q (nm$^{-1}$)')
    axes[3].set_ylabel('Intensity (arb. units)')
    axes[3].set_title('(d) Radial Profile\n(azimuthal average)')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle('FFT-based Diffraction Simulation', fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_08_diffraction_principle.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_09_size_comparison():
    """图9: 固定70px样品的pipeline全流程展示（替代原尺寸对比）"""
    print("Generating fig_09_pipeline_70px.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    # Show multiple samples at the fixed 70px size through the pipeline
    n_samples = 6
    fig, axes = plt.subplots(4, n_samples, figsize=(3 * n_samples, 12))

    for i in range(n_samples):
        seed = 42 + i * 100
        gen = BioSampleGenerator(seed=seed, sample_size_px=SAMPLE_SIZE_PX)
        obj = gen.generate()

        augmentor = DataAugmentor(seed=seed)
        obj_aug = augmentor.augment(obj, seed=seed)

        sim = DiffractionSimulator()
        I_clean = sim.simulate(obj_aug)

        norm = IntensityNormalizer()
        I_norm = norm.normalize(I_clean)

        noise_applier = NoiseAndBeamstopApplier(seed=seed)
        I_noisy, beamstop_mask, I_sc, _ = noise_applier.apply(I_norm, seed=seed)

        # Row 1: Object
        axes[0, i].imshow(obj, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title('Sample {}'.format(i+1), fontweight='bold')
        axes[0, i].axis('off')

        # Row 2: Clean diffraction
        I_log = np.log10(1 + I_clean)
        axes[1, i].imshow(I_log, cmap='inferno')
        axes[1, i].axis('off')

        # Row 3: Beamstop overlay
        overlay = np.stack([
            (I_log / (I_log.max() + 1e-10)),
            (I_log / (I_log.max() + 1e-10)) * 0.5,
            (I_log / (I_log.max() + 1e-10)),
        ], axis=-1)
        overlay[beamstop_mask, 0] = 0.0
        overlay[beamstop_mask, 1] = 0.3
        overlay[beamstop_mask, 2] = 1.0
        axes[2, i].imshow(np.clip(overlay, 0, 1))
        axes[2, i].axis('off')

        # Row 4: Noisy diffraction
        axes[3, i].imshow(np.log10(1 + I_noisy), cmap='inferno')
        axes[3, i].axis('off')

    row_labels = ['Object (70px)', 'Clean Diffraction', 'Beamstop Overlay', 'Noisy (I_sc={:.1e})'.format(I_sc)]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=60)

    fig.suptitle('Pipeline Output: Fixed 70px Sample Size', fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_09_size_comparison.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_10_noise_comparison():
    """图10: 不同I_sc噪声水平对比"""
    print("Generating fig_10_noise_comparison.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    gen = BioSampleGenerator(seed=42, sample_size_px=SAMPLE_SIZE_PX)
    obj = gen.generate()

    sim = DiffractionSimulator()
    I_clean = sim.simulate(obj)

    norm = IntensityNormalizer()
    I_norm = norm.normalize(I_clean)

    I_sc_values = [1e7, 5e7, 1e8, 1e9]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, I_sc in enumerate(I_sc_values):
        rng = np.random.default_rng(42)

        I_expected = I_norm * I_sc
        I_poisson = rng.poisson(np.maximum(I_expected, 0).astype(np.float64)).astype(np.float64)
        I_noisy = I_poisson + rng.normal(0, 5.0, I_poisson.shape)
        I_noisy = np.maximum(I_noisy, 0)

        # Top row: noisy patterns
        im = axes[0, i].imshow(np.log10(1 + I_noisy), cmap='inferno')
        axes[0, i].set_title('I_sc = {:.0e}'.format(I_sc))
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], shrink=0.8)

        # Bottom row: difference (noise only)
        diff = I_noisy - I_expected
        vmax = np.percentile(np.abs(diff), 99)
        im = axes[1, i].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        # Standard per-pixel SNR: signal / sqrt(signal + read_noise^2)
        valid_mask = I_expected > 0
        if np.any(valid_mask):
            snr_map = I_expected[valid_mask] / np.sqrt(I_expected[valid_mask] + 5.0**2)
            mean_snr = np.mean(snr_map)
        else:
            mean_snr = 0
        axes[1, i].set_title('Noise, Mean SNR={:.1f}'.format(mean_snr))
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], shrink=0.8)

    fig.suptitle('Noise Comparison at Different Photon Counts (70px sample)',
                 fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_10_noise_comparison.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_11_beamstop():
    """图11: Beamstop mask详情"""
    print("Generating fig_11_beamstop.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    mask_file = SIMULATOR_DIR / 'beamstop_mask-585x585.mat'
    if not mask_file.exists():
        print("  -> SKIPPED (beamstop mask file not found)")
        return

    applier = NoiseAndBeamstopApplier(seed=42)
    base_mask = applier.load_beamstop_mask()

    # Create gradient
    from scipy.ndimage import distance_transform_edt
    gradient_wide = np.clip(distance_transform_edt(1 - base_mask) / 10, 0, 1)
    gradient_wide[base_mask > 0.5] = 0

    gradient_narrow = np.clip(distance_transform_edt(1 - base_mask) / 3, 0, 1)
    gradient_narrow[base_mask > 0.5] = 0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Binary mask
    axes[0].imshow(base_mask, cmap='gray')
    axes[0].set_title('(a) Binary Mask\n(blocked: {:,} px)'.format(int(np.sum(base_mask > 0.5))))
    axes[0].axis('off')

    # Gradient wide
    axes[1].imshow(gradient_wide, cmap='gray')
    axes[1].set_title('(b) Gradient (width=10px)\nSmooth transition')
    axes[1].axis('off')

    # Gradient narrow
    axes[2].imshow(gradient_narrow, cmap='gray')
    axes[2].set_title('(c) Gradient (width=3px)\nSharp transition')
    axes[2].axis('off')

    # Applied to diffraction
    gen = BioSampleGenerator(seed=42, sample_size_px=SAMPLE_SIZE_PX)
    obj = gen.generate()
    sim = DiffractionSimulator()
    I_clean = sim.simulate(obj)
    I_log = np.log10(1 + I_clean)

    I_masked = I_log * gradient_wide
    axes[3].imshow(I_masked, cmap='inferno')
    axes[3].set_title('(d) Applied to Diffraction\n(log scale)')
    axes[3].axis('off')

    fig.suptitle('Beamstop Mask with Gradient Transition', fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_11_beamstop.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_12_pipeline_complete():
    """图12: 完整pipeline单样本展示"""
    print("Generating fig_12_pipeline_complete.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    seed = 42

    # Step 1: Generate
    gen = BioSampleGenerator(seed=seed, sample_size_px=SAMPLE_SIZE_PX)
    obj = gen.generate()

    # Step 2: Augment
    augmentor = DataAugmentor(seed=seed)
    obj_aug = augmentor.augment(obj, seed=seed + 1000)

    # Step 3: Diffraction
    sim = DiffractionSimulator()
    I_clean = sim.simulate(obj_aug)

    # Step 4: Normalize
    norm = IntensityNormalizer()
    I_norm = norm.normalize(I_clean)

    # Step 5: Random mask (skip for clarity)
    I_masked = I_norm.copy()

    # Step 6: Noise with FIXED I_sc = 1e9 for display
    I_sc = 1e9
    rng_local = np.random.default_rng(seed)
    I_expected = I_masked * I_sc
    I_noisy = rng_local.poisson(np.maximum(I_expected, 0).astype(np.float64)).astype(np.float64)
    I_noisy = I_noisy + rng_local.normal(0, 5.0, I_noisy.shape)
    I_noisy = np.maximum(I_noisy, 0).astype(np.float32)

    # Step 7: Apply beamstop only
    applier = NoiseAndBeamstopApplier(seed=seed)
    beamstop_mask_base = applier.load_beamstop_mask()
    gradient_mask = applier._create_beamstop_gradient(beamstop_mask_base)
    I_noisy = (I_noisy * gradient_mask).astype(np.float32)
    beamstop_mask = beamstop_mask_base > 0.5
    metadata = {
        'I_sc': I_sc,
        'readout_noise_sigma': 5.0,
        'n_bad_pixels': 0,
        'n_bad_line_pixels': 0,
        'beamstop_masked_pixels': int(np.sum(beamstop_mask)),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # (a) Object
    axes[0, 0].imshow(obj, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('(a) Bio Sample\n[{:.2f}, {:.2f}]'.format(obj.min(), obj.max()), fontsize=10)
    axes[0, 0].axis('off')

    # (b) Augmented
    axes[0, 1].imshow(obj_aug, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('(b) After Augmentation', fontsize=10)
    axes[0, 1].axis('off')

    # (c) Clean diffraction
    I_log = np.log10(1 + I_clean)
    im = axes[0, 2].imshow(I_log, cmap='inferno')
    axes[0, 2].set_title('(c) Clean Diffraction\nMax={:.1e}'.format(I_clean.max()), fontsize=10)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)

    # (d) Normalized
    im = axes[0, 3].imshow(np.log10(1 + I_norm * 1e6), cmap='inferno')
    axes[0, 3].set_title('(d) Normalized\nSum={:.2e}'.format(np.sum(I_norm)), fontsize=10)
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], shrink=0.8)

    # (e) Beamstop mask
    axes[1, 0].imshow(beamstop_mask.astype(float), cmap='gray')
    axes[1, 0].set_title('(e) Beamstop Mask\nBlocked={} px'.format(beamstop_mask.sum()), fontsize=10)
    axes[1, 0].axis('off')

    # (f) Noisy diffraction
    im = axes[1, 1].imshow(np.log10(1 + I_noisy), cmap='inferno')
    axes[1, 1].set_title('(f) Noisy Diffraction\nI_sc={:.2e}'.format(I_sc), fontsize=10)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

    # (g) Preprocessing - log transform
    I_noisy_log = np.log10(1 + I_noisy)
    im = axes[1, 2].imshow(I_noisy_log, cmap='viridis')
    axes[1, 2].set_title('(g) Log Transform\nlog10(1+I)', fontsize=10)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

    # (h) Statistics
    axes[1, 3].axis('off')
    stats_text = (
        f"Pipeline Output Summary\n"
        f"{'='*30}\n\n"
        f"Sample size: {SAMPLE_SIZE_PX} px\n"
        f"I_sc = {I_sc:.2e} (fixed)\n"
        f"Read noise = 5.0 ADU\n"
        f"Beamstop = {metadata.get('beamstop_masked_pixels', 0)} px\n\n"
        f"Noisy range:\n"
        f"  [{I_noisy.min():.1f}, {I_noisy.max():.1f}]\n"
        f"Valid mean: {I_noisy[~beamstop_mask].mean():.1f}\n\n"
        f"Grid: 585 x 585\n"
        f"Dtype: float32"
    )
    axes[1, 3].text(0.05, 0.95, stats_text, transform=axes[1, 3].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 3].set_title('(h) Statistics')

    fig.suptitle('Complete Pipeline - Single Sample Walkthrough (70px)',
                 fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_12_pipeline_complete.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_13_preprocessing():
    """图13: 数据预处理流程"""
    print("Generating fig_13_preprocessing.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    gen = BioSampleGenerator(seed=42, sample_size_px=SAMPLE_SIZE_PX)
    obj = gen.generate()
    sim = DiffractionSimulator()
    I_clean = sim.simulate(obj)
    norm = IntensityNormalizer()
    I_norm = norm.normalize(I_clean)

    noise_applier = NoiseAndBeamstopApplier(seed=42)
    I_noisy, mask, I_sc, _ = noise_applier.apply(I_norm)

    # Preprocessing steps
    I_noisy_log = np.log10(1 + I_noisy)
    I_clean_log = np.log10(1 + I_clean)

    # Standardization using valid (non-beamstop) pixels
    valid = ~mask
    mean_train = np.mean(I_noisy_log[valid])
    std_train = np.std(I_noisy_log[valid])

    I_noisy_norm = (I_noisy_log - mean_train) / std_train
    I_clean_norm = (I_clean_log - mean_train) / std_train

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # (a) Raw noisy
    im = axes[0, 0].imshow(I_noisy, cmap='inferno', vmin=0, vmax=np.percentile(I_noisy[valid], 99))
    axes[0, 0].set_title('(a) Raw Noisy I\n[linear scale]')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # (b) Log transform
    im = axes[0, 1].imshow(I_noisy_log, cmap='inferno')
    axes[0, 1].set_title('(b) log10(1 + I)\n[dynamic range compressed]')
    axes[0, 1].axis('off')
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    # (c) Standardized
    vmax = 3
    im = axes[0, 2].imshow(I_noisy_norm, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 2].set_title('(c) Standardized\n(I_log - {:.2f}) / {:.2f}'.format(mean_train, std_train))
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], shrink=0.8)

    # (d) Clean target (standardized)
    im = axes[0, 3].imshow(I_clean_norm, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 3].set_title('(d) Clean Target\n(same standardization)')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], shrink=0.8)

    # (e) Histograms
    axes[1, 0].hist(I_noisy[valid].flatten(), bins=100, color='steelblue', alpha=0.7, log=True)
    axes[1, 0].set_title('(e) Raw Intensity Histogram')
    axes[1, 0].set_xlabel('Intensity')
    axes[1, 0].set_ylabel('Count (log)')

    axes[1, 1].hist(I_noisy_log[valid].flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[1, 1].axvline(mean_train, color='red', linestyle='--', label='mean={:.2f}'.format(mean_train))
    axes[1, 1].axvline(mean_train + std_train, color='orange', linestyle='--', label='std={:.2f}'.format(std_train))
    axes[1, 1].set_title('(f) Log-Transformed Histogram')
    axes[1, 1].set_xlabel('log10(1+I)')
    axes[1, 1].legend()

    axes[1, 2].hist(I_noisy_norm[valid].flatten(), bins=100, color='steelblue', alpha=0.7)
    axes[1, 2].axvline(0, color='red', linestyle='--', label='mean=0')
    axes[1, 2].axvline(1, color='orange', linestyle='--')
    axes[1, 2].axvline(-1, color='orange', linestyle='--', label='std=1')
    axes[1, 2].set_title('(g) Standardized Histogram')
    axes[1, 2].set_xlabel('(I_log - mean) / std')
    axes[1, 2].legend()

    # (h) HDF5 structure
    axes[1, 3].axis('off')
    hdf5_text = (
        f"HDF5 Dataset Structure\n"
        f"{'='*30}\n\n"
        f"bio_diffraction_v1.h5\n"
        f"  train/\n"
        f"    input  [N, 585, 585]\n"
        f"    target [N, 585, 585]\n"
        f"    mask   [N, 585, 585]\n"
        f"  val/  (same)\n"
        f"  test/ (same)\n"
        f"  config/\n"
        f"    mean_train = {mean_train:.4f}\n"
        f"    std_train  = {std_train:.4f}\n"
        f"    I_sc range = [1e7, 1e9]\n"
        f"    Physical params..."
    )
    axes[1, 3].text(0.05, 0.95, hdf5_text, transform=axes[1, 3].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 3].set_title('(h) HDF5 Structure')

    fig.suptitle('Data Preprocessing: Log Transform + Standardization',
                 fontsize=14, fontweight='bold')
    plt.savefig(OUT_DIR / 'fig_13_preprocessing.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


def fig_14_physical_parameters():
    """图14: 物理参数总览"""
    print("Generating fig_14_physical_parameters.png ...")

    if not HAS_MODULES:
        print("  -> SKIPPED (no project modules)")
        return

    params = compute_physical_parameters()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    text = (
        r"$\bf{Physical\ Parameters\ -\ Shanghai\ SXFEL}$" + "\n\n"
        r"$\bf{Detector}$" + "\n"
        "  Original: 4096 x 4096 pixels @ 15 um/pixel\n"
        "  Binned: 585 x 585 pixels @ 105 um/pixel\n"
        "  FOV: {:.2f} mm\n\n".format(params['fov_mm']) +
        r"$\bf{X-ray}$" + "\n"
        "  Wavelength: {:.1f} nm (soft X-ray)\n".format(params['wavelength_nm']) +
        "  Detector distance: {:.0f} cm\n\n".format(params['detector_distance_m']*100) +
        r"$\bf{Reciprocal\ Space}$" + "\n"
        "  dq (pixel size): {:.6f} nm$^{{-1}}$\n".format(params['dq_nm_inv']) +
        "  q_max: {:.4f} nm$^{{-1}}$\n".format(params['q_max_nm_inv']) +
        "  Max scattering angle: {:.4f} deg\n\n".format(params['theta_max_deg']) +
        r"$\bf{Noise\ Model}$" + "\n"
        "  I_sc range: [1e7, 1e9] photons\n"
        "  Gaussian read noise: $\\sigma$ = 5.0 ADU\n"
        "  Bad pixel prob: 0.1%\n"
        "  Bad line prob: 0.5%\n\n"
        r"$\bf{Validation}$" + "\n"
        "  FOV consistency: 4096 x 15um = 585 x 105um = {:.2f}mm".format(params['fov_mm'])
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(OUT_DIR / 'fig_14_physical_parameters.png', bbox_inches='tight')
    plt.close()
    print("  -> saved")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Generating all figures for group meeting report")
    print("Parameters: sample_size={}px, I_sc_range=[1e7, 1e9]".format(SAMPLE_SIZE_PX))
    print("Output directory: {}".format(OUT_DIR))
    print("=" * 60)
    print()

    figures = [
        fig_01_pipeline_overview,
        fig_02_experimental_data,
        fig_03_noise_calibration,
        fig_04_cell_states,
        fig_05_bio_structure_detail,
        fig_06_bio_diversity,
        fig_07_augmentation,
        fig_08_diffraction_principle,
        fig_09_size_comparison,
        fig_10_noise_comparison,
        fig_11_beamstop,
        fig_12_pipeline_complete,
        fig_13_preprocessing,
        fig_14_physical_parameters,
    ]

    success = 0
    failed = 0
    for func in figures:
        try:
            func()
            success += 1
        except Exception as e:
            print("  -> ERROR: {}".format(e))
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print("=" * 60)
    print("Done! {} figures generated, {} failed.".format(success, failed))
    print("Output: {}".format(OUT_DIR))
    print("=" * 60)


if __name__ == '__main__':
    main()
