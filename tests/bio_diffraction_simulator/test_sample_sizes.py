"""
Test Sample Sizes - Generate diffraction patterns for different bacteria sizes.

This script tests the full pipeline with different sample_size_px values
(30, 40, 50, ..., 100) to find the optimal size where the beamstop
only blocks the central (zero-order) diffraction peak, matching experimental data.

Usage:
    python test_sample_sizes.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from config import BASE_DIR
from bio_sample_generator import BioSampleGenerator
from data_augmentor import DataAugmentor
from diffraction_simulator import DiffractionSimulator
from intensity_normalizer import IntensityNormalizer
from random_mask_applier import RandomMaskApplier
from noise_beamstop_applier import NoiseAndBeamstopApplier
from utils import visualize_sample


def run_single_size(
    sample_size_px: int,
    seed: int = 42,
    output_dir: Path = None
):
    """
    Run the full pipeline for a single sample size.

    Args:
        sample_size_px: Target bacteria length in pixels.
        seed: Random seed.
        output_dir: Directory to save output.

    Returns:
        Dictionary with all pipeline data.
    """
    sample_seed = seed + sample_size_px * 100

    # Generate
    bio_generator = BioSampleGenerator(seed=sample_seed, sample_size_px=sample_size_px)
    obj = bio_generator.generate(seed=sample_seed)

    # Augment
    augmentor = DataAugmentor(seed=sample_seed)
    obj_aug = augmentor.augment(obj, seed=sample_seed)

    # Diffraction
    diffraction_sim = DiffractionSimulator()
    I_clean = diffraction_sim.simulate(obj_aug)

    # Normalize
    intensity_norm = IntensityNormalizer()
    I_norm = intensity_norm.normalize(I_clean)

    # Random mask
    random_mask_applier = RandomMaskApplier(seed=sample_seed)
    I_masked, _ = random_mask_applier.apply(I_norm, prob=0.5, seed=sample_seed)

    # Noise + beamstop
    noise_beamstop_applier = NoiseAndBeamstopApplier(seed=sample_seed)
    I_noisy, beamstop_mask, I_sc, metadata = noise_beamstop_applier.apply(
        I_masked, seed=sample_seed
    )

    # Save individual visualization
    if output_dir is not None:
        save_path = output_dir / f'sample_{sample_size_px:03d}px.png'
        fig = visualize_sample(
            obj, I_clean, I_noisy, beamstop_mask,
            save_path=save_path,
            title=f'Sample Size: {sample_size_px} px',
            I_sc=I_sc
        )
        plt.close(fig)

    return {
        'sample_size_px': sample_size_px,
        'obj': obj,
        'I_clean': I_clean,
        'I_noisy': I_noisy,
        'beamstop_mask': beamstop_mask,
        'I_sc': I_sc,
    }


def create_comparison_figure(results: list, output_dir: Path):
    """
    Create a side-by-side comparison figure for all sample sizes.

    Args:
        results: List of result dictionaries from run_single_size.
        output_dir: Directory to save the figure.
    """
    n_sizes = len(results)
    fig, axes = plt.subplots(4, n_sizes, figsize=(3 * n_sizes, 12))

    for i, r in enumerate(results):
        size = r['sample_size_px']
        obj = r['obj']
        I_clean = r['I_clean']
        I_noisy = r['I_noisy']
        beamstop_mask = r['beamstop_mask']

        # Row 1: Object (zoomed to center region)
        axes[0, i].imshow(obj, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{size} px', fontsize=11, fontweight='bold')
        axes[0, i].axis('off')

        # Row 2: Clean diffraction (log scale)
        I_log = np.log10(1 + I_clean)
        axes[1, i].imshow(I_log, cmap='inferno')
        axes[1, i].axis('off')

        # Row 3: Noisy diffraction (log scale)
        I_noisy_log = np.log10(1 + I_noisy)
        axes[3, i].imshow(I_noisy_log, cmap='inferno')
        axes[3, i].axis('off')

        # Row 4: Beamstop mask overlay on clean diffraction
        overlay = I_log.copy()
        # Highlight beamstop region
        overlay_display = np.stack([
            (I_log / (I_log.max() + 1e-10)),       # R channel (diffraction)
            (I_log / (I_log.max() + 1e-10)) * 0.5,  # G channel
            (I_log / (I_log.max() + 1e-10)),        # B channel
        ], axis=-1)
        # Mark beamstop in blue
        overlay_display[beamstop_mask, 0] = 0.0
        overlay_display[beamstop_mask, 1] = 0.3
        overlay_display[beamstop_mask, 2] = 1.0
        axes[2, i].imshow(np.clip(overlay_display, 0, 1))
        axes[2, i].axis('off')

    # Row labels
    row_labels = ['Object', 'Clean Diffraction', 'Beamstop Overlay', 'Noisy Diffraction']
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold', rotation=90, labelpad=60)

    fig.suptitle('Sample Size Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / 'size_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison figure saved: {save_path}")


def main():
    """Main entry point."""
    output_dir = BASE_DIR / 'test_output_sizes'
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = list(range(40, 91, 10))  # 40, 50, ..., 90
    seed = 42

    print("=" * 60)
    print("Sample Size Test - Bio Diffraction Simulator")
    print("=" * 60)
    print(f"Sizes to test: {sizes}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")
    print()

    results = []
    for size in tqdm(sizes, desc="Testing sizes"):
        result = run_single_size(size, seed=seed, output_dir=output_dir)
        results.append(result)

        # Print summary
        obj = result['obj']
        I_clean = result['I_clean']
        n_blocked = result['beamstop_mask'].sum()
        print(f"  Size {size:3d}px: "
              f"obj_pixels={np.sum(obj > 0.1):6d}, "
              f"I_clean_max={I_clean.max():.2e}, "
              f"beamstop_blocked={n_blocked}")

    # Create comparison figure
    print("\nGenerating comparison figure...")
    create_comparison_figure(results, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
