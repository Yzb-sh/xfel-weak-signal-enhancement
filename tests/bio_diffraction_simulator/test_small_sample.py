"""
Test Small Sample - Small sample test for bio diffraction simulator.

This script tests the complete pipeline with a small number of samples (N=10)
and generates comprehensive validation reports.

Usage:
    python test_small_sample.py --num_samples 10 --output_dir ./test_output
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from config import (
    EXP_CONFIG, BASE_DIR, run_all_validations,
    compute_physical_parameters, print_physical_report
)
from bio_sample_generator import BioSampleGenerator
from data_augmentor import DataAugmentor
from diffraction_simulator import DiffractionSimulator
from intensity_normalizer import IntensityNormalizer
from random_mask_applier import RandomMaskApplier
from noise_beamstop_applier import NoiseAndBeamstopApplier
from utils import (
    validate_output_range,
    validate_diffraction_nonnegative,
    validate_poisson_statistics,
    validate_normalization,
    validate_intensity_normalization,
    visualize_sample,
    visualize_pipeline_steps,
    generate_physical_report,
    generate_statistical_report,
    preprocess_for_training
)


def test_pipeline_components(seed: int = 42) -> Dict[str, bool]:
    """
    Test each pipeline component individually.

    Args:
        seed: Random seed.

    Returns:
        Dictionary with test results.
    """
    print("\n" + "=" * 60)
    print("Testing Pipeline Components")
    print("=" * 60)

    results = {}

    # Test 1: BioSampleGenerator
    print("\n1. Testing BioSampleGenerator...")
    try:
        generator = BioSampleGenerator(seed=seed)
        obj = generator.generate()
        range_result = validate_output_range(obj)
        results['bio_generator'] = range_result['passed']
        print(f"   Output shape: {obj.shape}, range: [{obj.min():.3f}, {obj.max():.3f}]")
        print(f"   Result: {'PASSED' if results['bio_generator'] else 'FAILED'}")
    except Exception as e:
        results['bio_generator'] = False
        print(f"   ERROR: {e}")

    # Test 2: DataAugmentor
    print("\n2. Testing DataAugmentor...")
    try:
        augmentor = DataAugmentor(seed=seed)
        obj_aug = augmentor.augment(obj)
        results['data_augmentor'] = obj_aug.shape == obj.shape
        print(f"   Output shape: {obj_aug.shape}")
        print(f"   Result: {'PASSED' if results['data_augmentor'] else 'FAILED'}")
    except Exception as e:
        results['data_augmentor'] = False
        print(f"   ERROR: {e}")

    # Test 3: DiffractionSimulator
    print("\n3. Testing DiffractionSimulator...")
    try:
        simulator = DiffractionSimulator()
        I_clean = simulator.simulate(obj_aug)
        nonneg_result = validate_diffraction_nonnegative(I_clean)
        results['diffraction_simulator'] = nonneg_result['passed']
        print(f"   Output shape: {I_clean.shape}, min: {I_clean.min():.2e}")
        print(f"   Result: {'PASSED' if results['diffraction_simulator'] else 'FAILED'}")
    except Exception as e:
        results['diffraction_simulator'] = False
        print(f"   ERROR: {e}")

    # Test 4: IntensityNormalizer
    print("\n4. Testing IntensityNormalizer...")
    try:
        normalizer = IntensityNormalizer()
        I_norm = normalizer.normalize(I_clean)
        norm_result = validate_intensity_normalization(I_norm)
        results['intensity_normalizer'] = norm_result['passed']
        print(f"   Sum after normalization: {np.sum(I_norm):.10f}")
        print(f"   Result: {'PASSED' if results['intensity_normalizer'] else 'FAILED'}")
    except Exception as e:
        results['intensity_normalizer'] = False
        print(f"   ERROR: {e}")

    # Test 5: RandomMaskApplier
    print("\n5. Testing RandomMaskApplier...")
    try:
        mask_applier = RandomMaskApplier(seed=seed)
        I_masked, random_mask = mask_applier.apply(I_norm, prob=0.5)
        results['random_mask_applier'] = I_masked.shape == I_norm.shape
        print(f"   Mask applied: {random_mask.any()}, masked pixels: {random_mask.sum()}")
        print(f"   Result: {'PASSED' if results['random_mask_applier'] else 'FAILED'}")
    except Exception as e:
        results['random_mask_applier'] = False
        print(f"   ERROR: {e}")

    # Test 6: NoiseAndBeamstopApplier
    print("\n6. Testing NoiseAndBeamstopApplier...")
    try:
        noise_applier = NoiseAndBeamstopApplier(seed=seed)
        I_noisy, beamstop_mask, I_sc, metadata = noise_applier.apply(I_norm)
        results['noise_beamstop_applier'] = I_noisy.shape == I_norm.shape
        print(f"   I_sc: {I_sc:.2e}, beamstop pixels: {beamstop_mask.sum()}")
        print(f"   Result: {'PASSED' if results['noise_beamstop_applier'] else 'FAILED'}")
    except Exception as e:
        results['noise_beamstop_applier'] = False
        print(f"   ERROR: {e}")

    return results


def run_small_sample_test(
    num_samples: int = 10,
    output_dir: Optional[Path] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run complete small sample test.

    Args:
        num_samples: Number of samples to generate.
        output_dir: Directory for output files.
        seed: Random seed.

    Returns:
        Dictionary with test results.
    """
    if output_dir is None:
        output_dir = BASE_DIR / 'test_output'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bio Diffraction Simulator - Small Sample Test")
    print("=" * 60)
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {seed}")

    # Step 1: Run physical validations
    print("\n" + "=" * 60)
    print("Step 1: Physical Validation")
    print("=" * 60)

    physical_results = run_all_validations()

    # Generate physical report
    physical_report_path = output_dir / 'physical_validation_report.txt'
    generate_physical_report(physical_report_path)

    # Step 2: Test pipeline components
    component_results = test_pipeline_components(seed)

    # Step 3: Generate samples with validation
    print("\n" + "=" * 60)
    print("Step 3: Generating Samples with Validation")
    print("=" * 60)

    # Initialize components
    bio_generator = BioSampleGenerator(seed=seed)
    augmentor = DataAugmentor(seed=seed)
    diffraction_sim = DiffractionSimulator()
    intensity_norm = IntensityNormalizer()
    random_mask_applier = RandomMaskApplier(seed=seed)
    noise_beamstop_applier = NoiseAndBeamstopApplier(seed=seed)

    validation_results = []
    samples_data = []

    for i in tqdm(range(num_samples), desc="Generating samples"):
        sample_seed = seed + i * 1000

        # Generate sample
        obj = bio_generator.generate(seed=sample_seed)
        obj_aug = augmentor.augment(obj, seed=sample_seed)
        I_clean = diffraction_sim.simulate(obj_aug)
        I_norm = intensity_norm.normalize(I_clean)
        I_masked, _ = random_mask_applier.apply(I_norm, prob=0.5, seed=sample_seed)
        I_noisy, beamstop_mask, I_sc, metadata = noise_beamstop_applier.apply(
            I_masked, seed=sample_seed
        )

        # Run validations
        sample_validation = {
            'sample_idx': i,
            'range_validation': validate_output_range(obj),
            'nonnegative_validation': validate_diffraction_nonnegative(I_clean),
            'intensity_norm_validation': validate_intensity_normalization(I_norm),
            'poisson_validation': validate_poisson_statistics(I_noisy, beamstop_mask),
        }

        validation_results.append(sample_validation)
        samples_data.append({
            'obj': obj,
            'obj_aug': obj_aug,
            'I_clean': I_clean,
            'I_norm': I_norm,
            'I_noisy': I_noisy,
            'beamstop_mask': beamstop_mask,
            'I_sc': I_sc,
            'metadata': metadata
        })

    # Step 4: Visualize samples
    print("\n" + "=" * 60)
    print("Step 4: Generating Visualizations")
    print("=" * 60)

    for i, sample in enumerate(samples_data[:5]):  # Visualize first 5
        save_path = output_dir / f'sample_{i:03d}.png'
        fig = visualize_sample(
            sample['obj'],
            sample['I_clean'],
            sample['I_noisy'],
            sample['beamstop_mask'],
            save_path=save_path,
            title=f"Sample {i}",
            I_sc=sample['I_sc']
        )
        plt.close(fig)
        print(f"  Saved: {save_path}")

    # Visualize pipeline for first sample
    pipeline_path = output_dir / 'pipeline_visualization.png'
    fig = visualize_pipeline_steps(
        samples_data[0]['obj'],
        samples_data[0]['obj_aug'],
        samples_data[0]['I_clean'],
        samples_data[0]['I_norm'],
        samples_data[0]['I_noisy'],
        samples_data[0]['beamstop_mask'],
        save_path=pipeline_path
    )
    plt.close(fig)
    print(f"  Saved: {pipeline_path}")

    # Step 5: Generate statistical report
    print("\n" + "=" * 60)
    print("Step 5: Generating Statistical Report")
    print("=" * 60)

    stats_report_path = output_dir / 'statistical_report.txt'
    generate_statistical_report(validation_results, stats_report_path)

    # Step 6: Test preprocessing
    print("\n" + "=" * 60)
    print("Step 6: Testing Preprocessing")
    print("=" * 60)

    # Compute mean/std from samples
    all_noisy = np.array([s['I_noisy'] for s in samples_data])
    all_masks = np.array([s['beamstop_mask'] for s in samples_data])

    valid_pixels = all_noisy[all_masks < 0.5]
    mean_train = np.mean(np.log10(1 + valid_pixels))
    std_train = np.std(np.log10(1 + valid_pixels))

    print(f"  Mean (log): {mean_train:.4f}")
    print(f"  Std (log): {std_train:.4f}")

    # Test standardization
    norm_validations = []
    for sample in samples_data:
        clean_norm, noisy_norm, _, _ = preprocess_for_training(
            sample['I_clean'], sample['I_noisy'], mean_train, std_train
        )
        norm_result = validate_normalization(noisy_norm[~sample['beamstop_mask']])
        norm_validations.append(norm_result)

    n_passed_norm = sum(1 for r in norm_validations if r['passed'])
    print(f"  Normalization validation: {n_passed_norm}/{num_samples} passed")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_component_passed = all(component_results.values())
    all_physical_passed = physical_results['all_passed']
    all_samples_passed = all(
        all(v['passed'] for v in r.values() if isinstance(v, dict) and 'passed' in v)
        for r in validation_results
    )

    summary = {
        'num_samples': num_samples,
        'physical_validation': all_physical_passed,
        'component_tests': component_results,
        'sample_validations': {
            'n_passed': sum(1 for r in validation_results if all(
                v['passed'] for v in r.values() if isinstance(v, dict) and 'passed' in v
            )),
            'n_total': num_samples
        },
        'normalization_validations': {
            'n_passed': n_passed_norm,
            'n_total': num_samples
        },
        'all_passed': all_component_passed and all_physical_passed
    }

    print(f"\nPhysical validation: {'PASSED' if all_physical_passed else 'FAILED'}")
    print(f"Component tests: {'PASSED' if all_component_passed else 'FAILED'}")
    print(f"Sample validations: {summary['sample_validations']['n_passed']}/{num_samples} passed")
    print(f"Normalization: {n_passed_norm}/{num_samples} passed")
    print(f"\nOverall: {'ALL TESTS PASSED' if summary['all_passed'] else 'SOME TESTS FAILED'}")

    # Save summary
    summary_path = output_dir / 'test_summary.json'
    with open(summary_path, 'w') as f:
        # Convert numpy types to Python types for JSON
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        json.dump(convert_types(summary), f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print(f"All outputs saved to: {output_dir}")

    return summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run small sample test for bio diffraction simulator'
    )
    parser.add_argument(
        '--num_samples', type=int, default=10,
        help='Number of samples to generate (default: 10)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./test_output',
        help='Output directory for test results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir

    result = run_small_sample_test(
        num_samples=args.num_samples,
        output_dir=output_dir,
        seed=args.seed
    )

    if not result['all_passed']:
        exit(1)


if __name__ == "__main__":
    main()
