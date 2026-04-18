# Bio Diffraction Simulator

This directory contains a pipeline for generating synthetic 2D diffraction patterns of model E. coli cells with realistic X-ray noise for deep learning denoising tasks.

## Key Entry Points

- `generate_dataset.py`: Main script to generate large HDF5 datasets.
- `test_small_sample.py`: First script to run — validates the entire pipeline.
- `config.py`: Central configuration and physical parameters.

## Quick Commands

```bash
# Test the pipeline with 10 samples
python test_small_sample.py --num_samples 10

# Generate a full dataset
python generate_dataset.py --num_train 10000 --num_val 1000 --num_test 1000
```

## Pipeline Overview

1. **BioSampleGenerator** → 2D E. coli density map [0,1]
2. **DataAugmentor** → Random rotation/translation/scaling
3. **DiffractionSimulator** → FFT-based clean diffraction
4. **IntensityNormalizer** → Normalize sum to 1 (required for Poisson noise)
5. **RandomMaskApplier** → Simulate detector defects (50% probability)
6. **NoiseAndBeamstopApplier** → Poisson + Gaussian noise + beamstop mask
7. **Preprocessing** → Log transform + standardization → HDF5

## Important Notes

- The intensity normalization (Step 4) is **critical** for physically correct Poisson noise.
- Noise model reuses `src/physics/noise_model.py`.
- Beamstop mask loaded from `beamstop_mask-585x585.mat`.

For complete documentation, see README.md.
