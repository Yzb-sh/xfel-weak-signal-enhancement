"""
Generate Dataset - Main program for bio diffraction dataset generation.

This script generates HDF5 datasets containing:
- input: Noisy diffraction patterns (I_noisy_norm)
- target: Clean diffraction patterns (I_clean_norm)
- mask: Beamstop masks

The dataset is split into train/val/test sets.
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py

from src.simulation.bio_config import (
    EXP_CONFIG, BACTERIA_CONFIG, BASE_DIR, run_all_validations,
    compute_physical_parameters, get_config_summary
)
from src.simulation.bio_sample_generator import BioSampleGenerator
from src.simulation.data_augmentor import DataAugmentor
from src.simulation.bio_diffraction_simulator import DiffractionSimulator
from src.simulation.intensity_normalizer import IntensityNormalizer
from src.simulation.random_mask_applier import RandomMaskApplier
from src.simulation.noise_beamstop_applier import NoiseAndBeamstopApplier
from src.simulation.bio_utils import (
    preprocess_for_training,
    visualize_sample,
    generate_physical_report,
    generate_statistical_report,
    validate_output_range,
    validate_diffraction_nonnegative,
    validate_poisson_statistics,
    validate_normalization
)


class DatasetGenerator:
    """
    Complete pipeline for generating bio diffraction datasets.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the dataset generator.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Read sample_size_px from config: None (fixed default), int (fixed size), or (low, high) tuple (random range)
        self.sample_size_px_config = BACTERIA_CONFIG.get('sample_size_px', None)

        # Initialize all components (bio_generator created per-sample if size varies)
        self.augmentor = DataAugmentor(seed=seed)
        self.diffraction_sim = DiffractionSimulator()
        self.intensity_norm = IntensityNormalizer()
        self.random_mask = RandomMaskApplier(seed=seed)
        self.noise_beamstop = NoiseAndBeamstopApplier(seed=seed)

    def generate_single_sample(
        self,
        sample_idx: int,
        apply_random_mask: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Generate a single sample through the complete pipeline.

        Args:
            sample_idx: Sample index (used for seeding).
            apply_random_mask: Whether to apply random detector mask.

        Returns:
            Tuple of (I_clean_norm, I_noisy, beamstop_mask, metadata).
        """
        # Set seeds for reproducibility
        sample_seed = self.seed + sample_idx

        # Determine sample_size_px for this sample
        size_cfg = self.sample_size_px_config
        if isinstance(size_cfg, (list, tuple)) and len(size_cfg) == 2:
            sample_size_px = int(self.rng.integers(size_cfg[0], size_cfg[1] + 1))
        elif isinstance(size_cfg, int):
            sample_size_px = size_cfg
        else:
            sample_size_px = None

        # Step 1: Generate bio sample with appropriate size
        bio_generator = BioSampleGenerator(seed=sample_seed, sample_size_px=sample_size_px)
        obj = bio_generator.generate(seed=sample_seed)

        # Step 2: Data augmentation
        obj_aug = self.augmentor.augment(obj, seed=sample_seed + 1000)

        # Step 3: Diffraction simulation
        I_clean = self.diffraction_sim.simulate(obj_aug)

        # Step 4: Intensity normalization (CRITICAL: before noise)
        I_norm = self.intensity_norm.normalize(I_clean)

        # Step 5: Random mask (detector defects) - optional
        if apply_random_mask:
            I_masked, random_mask_record = self.random_mask.apply(
                I_norm, prob=0.5, seed=sample_seed + 2000
            )
        else:
            I_masked = I_norm.copy()
            random_mask_record = np.zeros_like(I_norm, dtype=bool)

        # Step 6: Noise and beamstop
        I_noisy, beamstop_mask, I_sc, noise_metadata = self.noise_beamstop.apply(
            I_masked, seed=sample_seed + 3000
        )

        # Combine metadata
        metadata = {
            'sample_idx': sample_idx,
            'sample_size_px': int(sample_size_px) if sample_size_px is not None else None,
            'I_sc': float(I_sc),
            'random_mask_applied': bool(random_mask_record.any()),
            'random_mask_pixels': int(np.sum(random_mask_record)),
            **noise_metadata
        }

        return I_clean, I_noisy, beamstop_mask, metadata

    def generate_split(
        self,
        n_samples: int,
        split_name: str,
        start_idx: int = 0,
        apply_random_mask: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate samples for a single split (train/val/test).

        Args:
            n_samples: Number of samples to generate.
            split_name: Name of the split.
            start_idx: Starting index for seeding.
            apply_random_mask: Whether to apply random masks.

        Returns:
            Tuple of (clean_patterns, noisy_patterns, masks, metadata_list).
        """
        clean_patterns = []
        noisy_patterns = []
        masks = []
        metadata_list = []

        print(f"\nGenerating {split_name} split ({n_samples} samples)...")

        for i in tqdm(range(n_samples), desc=split_name):
            sample_idx = start_idx + i

            I_clean, I_noisy, mask, metadata = self.generate_single_sample(
                sample_idx, apply_random_mask
            )

            clean_patterns.append(I_clean)
            noisy_patterns.append(I_noisy)
            masks.append(mask.astype(np.float32))
            metadata_list.append(metadata)

        return (
            np.array(clean_patterns, dtype=np.float32),
            np.array(noisy_patterns, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            metadata_list
        )


def compute_standardization_params(
    noisy_patterns: np.ndarray,
    masks: np.ndarray
) -> Tuple[float, float]:
    """
    Compute mean and std for standardization from training data.

    Uses only the training set's noisy patterns, excluding beamstop regions.

    Args:
        noisy_patterns: Noisy diffraction patterns (N, H, W).
        masks: Beamstop masks (N, H, W).

    Returns:
        Tuple of (mean, std) for log-transformed data.
    """
    print("Computing standardization parameters from training data...")

    # Log transform
    noisy_log = np.log10(1 + noisy_patterns)

    # Mask out beamstop regions
    valid_pixels = noisy_log[masks < 0.5]

    mean = np.mean(valid_pixels)
    std = np.std(valid_pixels)

    print(f"  Mean (log): {mean:.4f}")
    print(f"  Std (log): {std:.4f}")

    return float(mean), float(std)


def save_dataset(
    output_file: Path,
    train_data: Tuple,
    val_data: Tuple,
    test_data: Tuple,
    mean_train: float,
    std_train: float,
    config_dict: Dict[str, Any]
) -> None:
    """
    Save the dataset to HDF5 file.

    Args:
        output_file: Output file path.
        train_data: Tuple of (clean, noisy, masks, metadata) for training.
        val_data: Validation data tuple.
        test_data: Test data tuple.
        mean_train: Mean for standardization.
        std_train: Std for standardization.
        config_dict: Configuration dictionary.
    """
    print(f"\nSaving dataset to {output_file}...")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Create splits
        for split_name, (clean, noisy, masks, metadata) in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            # Preprocess: log transform and standardize
            clean_norm = []
            noisy_norm = []

            for i in range(len(clean)):
                c_norm, n_norm, _, _ = preprocess_for_training(
                    clean[i], noisy[i], mean_train, std_train
                )
                clean_norm.append(c_norm)
                noisy_norm.append(n_norm)

            clean_norm = np.array(clean_norm, dtype=np.float32)
            noisy_norm = np.array(noisy_norm, dtype=np.float32)

            # Create group
            group = f.create_group(split_name)

            # Save data
            group.create_dataset('input', data=noisy_norm, compression='gzip')
            group.create_dataset('target', data=clean_norm, compression='gzip')
            group.create_dataset('mask', data=masks, compression='gzip')

            # Save metadata
            meta_group = group.create_group('metadata')
            for i, meta in enumerate(metadata):
                # Convert numpy types to native Python types for JSON serialization
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (np.integer, np.int64, np.int32)):
                        clean_meta[k] = int(v)
                    elif isinstance(v, (np.floating, np.float64, np.float32)):
                        clean_meta[k] = float(v)
                    elif isinstance(v, np.ndarray):
                        clean_meta[k] = v.tolist()
                    else:
                        clean_meta[k] = v
                meta_group.create_dataset(f'sample_{i}', data=json.dumps(clean_meta))

        # Save config
        config_group = f.create_group('config')
        config_group.attrs['mean_train'] = mean_train
        config_group.attrs['std_train'] = std_train
        config_group.attrs['seed'] = config_dict.get('seed', 42)
        config_group.attrs['num_train'] = len(train_data[0])
        config_group.attrs['num_val'] = len(val_data[0])
        config_group.attrs['num_test'] = len(test_data[0])

        # Save physical parameters
        phys_params = compute_physical_parameters()
        for key, value in phys_params.items():
            if isinstance(value, (int, float, str)):
                config_group.attrs[key] = value

    print(f"Dataset saved successfully!")
    print(f"  Train: {len(train_data[0])} samples")
    print(f"  Val: {len(val_data[0])} samples")
    print(f"  Test: {len(test_data[0])} samples")


def generate_dataset(
    num_train: int,
    num_val: int,
    num_test: int,
    output_file: Path,
    seed: int = 42,
    run_validation: bool = True
) -> Dict[str, Any]:
    """
    Generate the complete bio diffraction dataset.

    Args:
        num_train: Number of training samples.
        num_val: Number of validation samples.
        num_test: Number of test samples.
        output_file: Output HDF5 file path.
        seed: Random seed.
        run_validation: Whether to run physical validation.

    Returns:
        Dictionary with generation statistics.
    """
    start_time = time.time()

    print("=" * 60)
    print("Bio Diffraction Dataset Generator")
    print("=" * 60)
    print(get_config_summary())

    # Run physical validations
    if run_validation:
        print("\nRunning physical validations...")
        run_all_validations()

    # Initialize generator
    generator = DatasetGenerator(seed=seed)

    # Generate training data
    train_data = generator.generate_split(
        num_train, 'train', start_idx=0, apply_random_mask=True
    )

    # Generate validation data
    val_data = generator.generate_split(
        num_val, 'val', start_idx=num_train, apply_random_mask=True
    )

    # Generate test data
    test_data = generator.generate_split(
        num_test, 'test', start_idx=num_train + num_val, apply_random_mask=True
    )

    # Compute standardization parameters from training data
    mean_train, std_train = compute_standardization_params(train_data[1], train_data[2])

    # Save dataset
    config_dict = {
        'seed': seed,
        'num_train': num_train,
        'num_val': num_val,
        'num_test': num_test,
    }

    save_dataset(
        output_file, train_data, val_data, test_data,
        mean_train, std_train, config_dict
    )

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Output file: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        'output_file': str(output_file),
        'num_train': num_train,
        'num_val': num_val,
        'num_test': num_test,
        'mean_train': mean_train,
        'std_train': std_train,
        'elapsed_time': elapsed_time,
    }


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description='Generate bio diffraction dataset'
    )
    parser.add_argument(
        '--num_train', type=int, default=10000,
        help='Number of training samples'
    )
    parser.add_argument(
        '--num_val', type=int, default=1000,
        help='Number of validation samples'
    )
    parser.add_argument(
        '--num_test', type=int, default=1000,
        help='Number of test samples'
    )
    parser.add_argument(
        '--output_file', type=str, default='./bio_diffraction_v1.h5',
        help='Output HDF5 file path'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--skip_validation', action='store_true',
        help='Skip physical validation'
    )

    args = parser.parse_args()

    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        # Resolve relative paths from project root
        project_root = Path(__file__).parent.parent
        output_file = project_root / output_file

    generate_dataset(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        output_file=output_file,
        seed=args.seed,
        run_validation=not args.skip_validation
    )


if __name__ == "__main__":
    main()
