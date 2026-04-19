"""
Generate Dataset - Main program for bio diffraction dataset generation.

This script generates HDF5 datasets containing:
- input: Noisy diffraction patterns (I_noisy_norm)
- target: Clean diffraction patterns (I_clean_norm)
- mask: Beamstop masks

Features:
- Streaming write: samples are written to HDF5 in batches, not all at once
- Low memory: peak usage ~2.3 GB regardless of dataset size
- Resume support: if interrupted, re-run picks up from where it left off
- Full config snapshot: config.json saved alongside the HDF5 file
"""

import argparse
import sys
import time
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import h5py

from src.simulation.bio_config import (
    EXP_CONFIG, BACTERIA_CONFIG, BASE_DIR, run_all_validations,
    compute_physical_parameters, get_config_summary, get_full_config
)
from src.simulation.bio_sample_generator import BioSampleGenerator
from src.simulation.data_augmentor import DataAugmentor
from src.simulation.bio_diffraction_simulator import DiffractionSimulator
from src.simulation.intensity_normalizer import IntensityNormalizer
from src.simulation.random_mask_applier import RandomMaskApplier
from src.simulation.noise_beamstop_applier import NoiseAndBeamstopApplier


# =============================================================================
# Welford Online Variance (Vectorized Batch Combination)
# =============================================================================

class BatchWelfordAccumulator:
    """Online mean/variance using parallel combination of batch statistics.

    Each batch is computed with vectorized numpy; batches are combined
    with the parallel Welford formula in O(1) scalar arithmetic.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_batch(self, batch_log: np.ndarray, batch_mask: np.ndarray):
        """Process one batch of (B, H, W) log-transformed data.

        Uses parallel combination:
          n_total = n_a + n_b
          delta = mean_b - mean_a
          mean_total = mean_a + delta * (n_b / n_total)
          M2_total = M2_a + M2_b + delta^2 * (n_a * n_b / n_total)
        """
        valid_pixels = batch_log[batch_mask < 0.5].ravel()
        n_b = len(valid_pixels)
        if n_b == 0:
            return

        mean_b = np.mean(valid_pixels)
        M2_b = np.sum((valid_pixels - mean_b) ** 2)

        n_a = self.n
        delta = mean_b - self.mean

        self.mean = self.mean + delta * (n_b / (n_a + n_b))
        self.M2 = self.M2 + M2_b + delta ** 2 * (n_a * n_b / (n_a + n_b))
        self.n = n_a + n_b

    def finalize(self) -> Tuple[float, float]:
        """Return (mean, std)."""
        if self.n < 2:
            return float(self.mean), 1.0
        return float(self.mean), float(np.sqrt(max(self.M2 / self.n, 1e-20)))


# =============================================================================
# Generation State (for crash recovery)
# =============================================================================

@dataclass
class GenerationState:
    """Persistent state for resume-after-crash."""
    seed: int = 42
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    batch_size: int = 500

    train_completed: int = 0
    val_completed: int = 0
    test_completed: int = 0

    # Welford accumulators
    welford_n: int = 0
    welford_mean: float = 0.0
    welford_M2: float = 0.0

    # Standardization (set after pass 1)
    mean_train: Optional[float] = None
    std_train: Optional[float] = None
    standardization_applied: bool = False

    @classmethod
    def load(cls, path: Path) -> Optional['GenerationState']:
        if path.exists():
            with open(path, 'r') as f:
                return cls(**json.load(f))
        return None

    def save(self, path: Path):
        d = asdict(self)
        # Ensure all values are JSON-serializable
        for k, v in d.items():
            if isinstance(v, (np.floating, np.float32, np.float64)):
                d[k] = float(v)
            elif isinstance(v, (np.integer,)):
                d[k] = int(v)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)


# =============================================================================
# Config Snapshot
# =============================================================================

def save_config(
    output_dir: Path,
    version: str,
    seed: int,
    num_train: int,
    num_val: int,
    num_test: int,
    batch_size: int,
    mean_train: Optional[float] = None,
    std_train: Optional[float] = None,
) -> Path:
    """Save a complete config snapshot to config.json."""
    full_config = get_full_config()
    full_config.update({
        'version': version,
        'timestamp': datetime.now().isoformat(),
        'generation': {
            'seed': seed,
            'num_train': num_train,
            'num_val': num_val,
            'num_test': num_test,
            'batch_size': batch_size,
        },
        'standardization': {
            'mean_train': mean_train,
            'std_train': std_train,
            'method': 'log10(1+I) then (I-mean)/std',
        },
        'data_shape': [EXP_CONFIG['train_size'], EXP_CONFIG['train_size']],
        'dtype': 'float32',
    })

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)

    return config_path


# =============================================================================
# Metadata Helper
# =============================================================================

def _clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy types to native Python types for JSON serialization."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        elif isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        else:
            clean[k] = v
    return clean


# =============================================================================
# Dataset Generator (Streaming)
# =============================================================================

class DatasetGenerator:
    """Complete pipeline for generating bio diffraction datasets."""

    def __init__(self, seed: int = 42, use_gpu: bool = False):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sample_size_px_config = BACTERIA_CONFIG.get('sample_size_px', None)
        self.use_gpu = use_gpu

        self.augmentor = DataAugmentor(seed=seed, use_gpu=use_gpu)
        self.diffraction_sim = DiffractionSimulator(use_gpu=use_gpu)
        self.intensity_norm = IntensityNormalizer()
        self.random_mask = RandomMaskApplier(seed=seed, use_gpu=use_gpu)
        self.noise_beamstop = NoiseAndBeamstopApplier(seed=seed, use_gpu=use_gpu)

        # Reusable bio generator
        self._bio_generator = None
        self._last_sample_size_px = None

    def _get_bio_generator(self, sample_size_px: Optional[int]) -> BioSampleGenerator:
        """Get or create a BioSampleGenerator, reusing when possible."""
        if self._bio_generator is None or sample_size_px != self._last_sample_size_px:
            self._bio_generator = BioSampleGenerator(sample_size_px=sample_size_px, use_gpu=self.use_gpu)
            self._last_sample_size_px = sample_size_px
        return self._bio_generator

    def generate_single_sample(
        self,
        sample_idx: int,
        apply_random_mask: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate a single sample through the complete pipeline."""
        sample_seed = self.seed + sample_idx

        # Determine sample_size_px for this sample
        size_cfg = self.sample_size_px_config
        if isinstance(size_cfg, range):
            sample_size_px = int(self.rng.integers(size_cfg.start, size_cfg.stop + 1))
        elif isinstance(size_cfg, (list, tuple)) and len(size_cfg) == 2:
            sample_size_px = int(self.rng.integers(size_cfg[0], size_cfg[1] + 1))
        elif isinstance(size_cfg, int):
            sample_size_px = size_cfg
        else:
            sample_size_px = None

        bio_gen = self._get_bio_generator(sample_size_px)

        # Step 1-3: Generate, augment, and simulate with retry
        for attempt in range(10):
            bio_gen.set_seed(sample_seed)
            obj = bio_gen.generate(seed=sample_seed)
            obj_aug = self.augmentor.augment(obj, seed=sample_seed + 1000 + attempt)
            I_clean = self.diffraction_sim.simulate(obj_aug)
            if np.sum(I_clean) > 1e-10:
                break

        # Step 4: Intensity normalization (CRITICAL: before noise)
        I_norm = self.intensity_norm.normalize(I_clean)

        # Step 5: Apply Poisson + Gaussian noise only
        I_noisy, I_sc, noise_metadata, defect_mask = self.noise_beamstop.apply_noise_only(
            I_norm, seed=sample_seed + 2000
        )

        # Step 6: Random mask (detector defects)
        if apply_random_mask:
            I_noisy, random_mask_record = self.random_mask.apply(
                I_noisy, prob=0.5, seed=sample_seed + 3000
            )
        else:
            random_mask_record = np.zeros_like(I_noisy, dtype=bool)

        # Step 7: Apply beam stop mask
        I_noisy, beamstop_mask, beamstop_metadata = self.noise_beamstop.apply_beamstop_only(I_noisy)

        final_mask = beamstop_mask | random_mask_record | defect_mask

        metadata = {
            'sample_idx': sample_idx,
            'sample_size_px': int(sample_size_px) if sample_size_px is not None else None,
            'I_sc': float(I_sc),
            'random_mask_applied': bool(random_mask_record.any()),
            'random_mask_pixels': int(np.sum(random_mask_record)),
            'total_mask_pixels': int(np.sum(final_mask)),
            **noise_metadata,
            **beamstop_metadata
        }

        return I_clean, I_noisy, final_mask, metadata

    def generate_split_streaming(
        self,
        n_samples: int,
        split_name: str,
        h5_file: h5py.File,
        start_idx: int,
        batch_size: int,
        apply_random_mask: bool,
        welford: Optional[BatchWelfordAccumulator],
        resume_from: int = 0
    ) -> None:
        """Generate samples and write incrementally to HDF5.

        Memory: O(batch_size) -- only one batch in memory at a time.
        """
        grid_size = EXP_CONFIG['train_size']
        group = h5_file.require_group(split_name)

        # Create resizable datasets if they don't exist
        if 'input' not in group:
            maxshape = (None, grid_size, grid_size)
            kw = dict(dtype=np.float32, chunks=(1, grid_size, grid_size),
                      compression='gzip', compression_opts=4)
            group.create_dataset('input', shape=(0, grid_size, grid_size),
                                 maxshape=maxshape, **kw)
            group.create_dataset('target', shape=(0, grid_size, grid_size),
                                 maxshape=maxshape, **kw)
            group.create_dataset('mask', shape=(0, grid_size, grid_size),
                                 maxshape=maxshape, **kw)
            group.create_group('metadata')

        input_ds = group['input']
        target_ds = group['target']
        mask_ds = group['mask']
        meta_group = group['metadata']
        current_count = input_ds.shape[0]

        print(f"\nGenerating {split_name} split ({n_samples} samples, "
              f"batch_size={batch_size}, resume_from={resume_from})...")

        pbar = tqdm(total=n_samples, desc=split_name, initial=resume_from,
                    unit='samples')

        for batch_start in range(resume_from, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            bs = batch_end - batch_start

            batch_input = np.empty((bs, grid_size, grid_size), dtype=np.float32)
            batch_target = np.empty((bs, grid_size, grid_size), dtype=np.float32)
            batch_mask = np.empty((bs, grid_size, grid_size), dtype=np.float32)

            for i in range(bs):
                sample_idx = start_idx + batch_start + i
                I_clean, I_noisy, mask, metadata = self.generate_single_sample(
                    sample_idx, apply_random_mask
                )
                batch_target[i] = I_clean
                batch_input[i] = I_noisy
                batch_mask[i] = mask.astype(np.float32)

                meta_group.create_dataset(
                    f'sample_{current_count + i}',
                    data=json.dumps(_clean_metadata(metadata))
                )

            # Update Welford statistics (train split only)
            if welford is not None:
                batch_input_log = np.log10(1 + batch_input)
                welford.update_batch(batch_input_log, batch_mask)

            # Resize and write
            new_size = current_count + bs
            input_ds.resize(new_size, axis=0)
            target_ds.resize(new_size, axis=0)
            mask_ds.resize(new_size, axis=0)

            input_ds[current_count:new_size] = batch_input
            target_ds[current_count:new_size] = batch_target
            mask_ds[current_count:new_size] = batch_mask

            current_count = new_size
            h5_file.flush()
            pbar.update(bs)

        pbar.close()


# =============================================================================
# In-place Standardization
# =============================================================================

def apply_standardization_inplace(
    h5_file: h5py.File,
    mean_train: float,
    std_train: float,
    chunk_size: int = 500
) -> None:
    """Apply log transform + standardization to all data in-place.

    Reads chunk_size samples at a time, transforms, writes back.
    Memory: O(chunk_size).
    """
    print("\nApplying standardization in-place...")

    for split_name in ['train', 'val', 'test']:
        if split_name not in h5_file:
            continue

        group = h5_file[split_name]
        n_samples = group['input'].shape[0]
        print(f"  {split_name}: {n_samples} samples")

        for start in tqdm(range(0, n_samples, chunk_size),
                          desc=f'{split_name} standardize'):
            end = min(start + chunk_size, n_samples)

            raw_input = group['input'][start:end]
            raw_target = group['target'][start:end]

            input_norm = ((np.log10(1 + raw_input) - mean_train) / std_train).astype(np.float32)
            target_norm = ((np.log10(1 + raw_target) - mean_train) / std_train).astype(np.float32)

            group['input'][start:end] = input_norm
            group['target'][start:end] = target_norm

    print("  Standardization complete.")


# =============================================================================
# Main Orchestration
# =============================================================================

def generate_dataset(
    num_train: int,
    num_val: int,
    num_test: int,
    output_dir: Path,
    seed: int = 42,
    batch_size: int = 500,
    run_validation: bool = True,
    use_gpu: bool = False
) -> Dict[str, Any]:
    """Generate the complete bio diffraction dataset with streaming write."""
    start_time = time.time()

    # Derive version name from directory name
    version = output_dir.name

    print("=" * 60)
    print("Bio Diffraction Dataset Generator")
    print("=" * 60)
    print(get_config_summary())

    # GPU check
    if use_gpu:
        from src.simulation.backend import check_gpu_available
        ok, msg = check_gpu_available()
        if not ok:
            print(f"\nERROR: GPU requested but not available: {msg}")
            sys.exit(1)
        print(f"\nGPU acceleration enabled: {msg}")
    else:
        print("\nRunning on CPU.")

    if run_validation:
        print("\nRunning physical validations...")
        run_all_validations()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_file_path = output_dir / f'{version}.h5'
    state_file = output_dir / 'generation_state.json'

    # Save config snapshot immediately (even if generation crashes, you know the params)
    save_config(output_dir, version, seed, num_train, num_val, num_test, batch_size)

    # Try to resume from previous state
    state = GenerationState.load(state_file)
    if state is None:
        state = GenerationState(
            seed=seed, num_train=num_train, num_val=num_val,
            num_test=num_test, batch_size=batch_size
        )
        print(f"\nStarting fresh generation.")
    else:
        print(f"\nResuming from state: train={state.train_completed}, "
              f"val={state.val_completed}, test={state.test_completed}")

    generator = DatasetGenerator(seed=seed, use_gpu=use_gpu)

    with h5py.File(h5_file_path, 'a') as h5f:
        # --- Pass 1: Generate raw data + compute Welford stats ---
        welford = BatchWelfordAccumulator()
        if state.welford_n > 0:
            welford.n = state.welford_n
            welford.mean = state.welford_mean
            welford.M2 = state.welford_M2

        # Train split
        if state.train_completed < num_train:
            generator.generate_split_streaming(
                num_train, 'train', h5f, start_idx=0,
                batch_size=batch_size, apply_random_mask=True,
                welford=welford, resume_from=state.train_completed
            )
            state.train_completed = num_train
            state.welford_n = welford.n
            state.welford_mean = welford.mean
            state.welford_M2 = welford.M2
            state.save(state_file)

        # Val split
        if state.val_completed < num_val:
            generator.generate_split_streaming(
                num_val, 'val', h5f, start_idx=num_train,
                batch_size=batch_size, apply_random_mask=True,
                welford=None, resume_from=state.val_completed
            )
            state.val_completed = num_val
            state.save(state_file)

        # Test split
        if state.test_completed < num_test:
            generator.generate_split_streaming(
                num_test, 'test', h5f, start_idx=num_train + num_val,
                batch_size=batch_size, apply_random_mask=True,
                welford=None, resume_from=state.test_completed
            )
            state.test_completed = num_test
            state.save(state_file)

        # Finalize Welford -> global mean/std
        mean_train, std_train = welford.finalize()
        print(f"\nStandardization parameters:")
        print(f"  Mean (log): {mean_train:.4f}")
        print(f"  Std  (log): {std_train:.4f}")

        state.mean_train = mean_train
        state.std_train = std_train
        state.save(state_file)

        # Update config.json with actual standardization values
        save_config(output_dir, version, seed, num_train, num_val, num_test,
                    batch_size, mean_train, std_train)

        # Store config as HDF5 attributes
        config_group = h5f.require_group('config')
        config_group.attrs['mean_train'] = mean_train
        config_group.attrs['std_train'] = std_train
        config_group.attrs['seed'] = seed
        config_group.attrs['num_train'] = num_train
        config_group.attrs['num_val'] = num_val
        config_group.attrs['num_test'] = num_test
        config_group.attrs['data_mode'] = 'standardized'

        phys_params = compute_physical_parameters()
        for key, value in phys_params.items():
            if isinstance(value, (int, float, str)):
                config_group.attrs[key] = value

        h5f.flush()

        # --- Pass 2: Apply standardization in-place ---
        if not state.standardization_applied:
            apply_standardization_inplace(h5f, mean_train, std_train,
                                          chunk_size=batch_size)
            state.standardization_applied = True
            state.save(state_file)

    # Clean up state file on success
    state_file.unlink(missing_ok=True)

    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed_time:.1f} seconds")
    print(f"Output directory: {output_dir}")
    print(f"HDF5 file: {h5_file_path}")
    print(f"File size: {h5_file_path.stat().st_size / 1024 / 1024:.1f} MB")

    return {
        'output_dir': str(output_dir),
        'h5_file': str(h5_file_path),
        'num_train': num_train,
        'num_val': num_val,
        'num_test': num_test,
        'mean_train': mean_train,
        'std_train': std_train,
        'elapsed_time': elapsed_time,
    }


# =============================================================================
# CLI
# =============================================================================

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
        '--output_dir', type=str, default=None,
        help='Output directory (e.g. data/bio_diffraction_v1). '
             'Defaults to data/bio_diffraction_v1 under project root.'
    )
    parser.add_argument(
        '--output_file', type=str, default=None,
        help='(Legacy) Output HDF5 file path. Prefer --output_dir.'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--batch_size', type=int, default=500,
        help='Number of samples per write batch (controls memory usage)'
    )
    parser.add_argument(
        '--skip_validation', action='store_true',
        help='Skip physical validation'
    )
    parser.add_argument(
        '--use_gpu', action='store_true',
        help='Enable GPU acceleration (requires CuPy + CUDA)'
    )

    args = parser.parse_args()
    project_root = Path(__file__).parent.parent

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output_file:
        # Legacy mode: derive directory from file path
        of = Path(args.output_file)
        if not of.is_absolute():
            of = project_root / of
        # Use the parent directory, creating a versioned folder
        version = of.stem  # e.g. "bio_diffraction_v1"
        output_dir = of.parent / version
    else:
        output_dir = project_root / 'data' / 'bio_diffraction_v1'

    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    generate_dataset(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        output_dir=output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        run_validation=not args.skip_validation,
        use_gpu=args.use_gpu
    )


if __name__ == "__main__":
    main()
