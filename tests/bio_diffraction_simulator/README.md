# Bio Diffraction Simulator

A pipeline for generating synthetic 2D diffraction patterns of E. coli cells with realistic X-ray noise for deep learning denoising.

## Overview

**Core Functionality**: Generate paired clean/noisy diffraction patterns with beamstop masks.

**Key Features**:
- Physics-based Poisson-Gaussian noise model
- Parameters aligned with Shanghai Soft X-ray FEL
- Modular 7-step pipeline with built-in validation
- HDF5 output format ready for deep learning

---

## File Structure & Modules

| File | Core Purpose | Input | Output | Key Functions | Notes |
|------|--------------|-------|--------|---------------|-------|
| `config.py` | Store all physical parameters and validation | None | EXP_CONFIG, BACTERIA_CONFIG | `validate_fov_consistency()`, `validate_sampling_theorem()`, `run_all_validations()` | Runs physical self-consistency checks before any generation |
| `bio_sample_generator.py` | Generate E. coli 2D projection density | Random seed | obj (585x585, float32, [0,1]) | `BioSampleGenerator.generate()` | Independent 2D implementation; includes capsule body, Perlin noise surface, Gaussian spots, vacuoles |
| `data_augmentor.py` | Apply random spatial transforms | obj | obj_augmented (same size) | `DataAugmentor.augment()` | Sequential: rotation → translation → scaling |
| `diffraction_simulator.py` | FFT-based diffraction simulation | obj_aug | I_clean (non-negative) | `DiffractionSimulator.simulate()` | Includes physical scale calibration (dx_real, dq, q_max) |
| `intensity_normalizer.py` | Normalize intensity sum to 1 | I_clean | I_clean_normalized (sum=1) | `IntensityNormalizer.normalize()` | **CRITICAL**: Must run before noise; enables physical Poisson noise |
| `random_mask_applier.py` | Simulate detector defects | I_norm | I_masked, random_mask | `RandomMaskApplier.apply()` | 50% probability; circle/rectangle/irregular shapes |
| `noise_beamstop_applier.py` | Apply noise + beamstop | I_masked | I_noisy, beamstop_mask, I_sc | `NoiseAndBeamstopApplier.apply()` | Reuses `src/physics/noise_model.py`; loads beamstop from .mat file |
| `utils.py` | Visualization, validation, preprocessing | Various | Reports, figures | `visualize_sample()`, `validate_*()`, `preprocess_for_training()` | All validation functions |
| `generate_dataset.py` | Main pipeline for HDF5 generation | Config | HDF5 file | `generate_dataset()`, `DatasetGenerator` | Entry point for large-scale generation |
| `test_small_sample.py` | Small sample testing with validation | Config | Test outputs | `run_small_sample_test()` | **Run first** to validate pipeline |

---

## Data Generation Pipeline (7 Steps)

### Step 1: Generate Bio Sample Projection (BioSampleGenerator)

**Input**: Random parameters (dimensions, morphology variants)

**Process**:
- Draw a 2D capsule shape (rectangle + semicircular ends) on 585×585 grid
- Add Perlin-like noise to simulate irregular cell surface
- Add 3-10 internal Gaussian spots (simulating intracellular structures)
- Add 1-3 circular vacuoles (low density regions, value < 0.3)
- Randomly select cell state: normal / dividing (two connected capsules) / curved (arc shape)

**Output**: `obj` — 2D projection electron density map, value range [0, 1]

**Physical Meaning**: Represents the sample's integrated projection along the X-ray beam direction.

---

### Step 2: Data Augmentation (DataAugmentor)

**Input**: `obj`

**Process**:
- Random rotation: [0°, 360°), bilinear interpolation
- Random translation: [-10%, +10%] of grid size
- Random scaling: [0.9, 1.1]

**Output**: `obj_augmented`

**Purpose**: Simulate different orientations and positions of the sample in the beam, improving model generalization.

---

### Step 3: Diffraction Simulation (DiffractionSimulator)

**Input**: `obj_augmented`

**Process**:
```python
A = np.fft.fft2(obj_augmented)  # 2D FFT
A = np.fft.fftshift(A)          # Center zero frequency
I_clean = np.abs(A)**2          # Intensity = |amplitude|²
```

**Physical Scale Calibration** (computed in `config.compute_physical_parameters()`):
- Real space pixel size: `dx_real = detector_target_pixel_size ≈ 105 μm`
- Reciprocal space pixel size: `dq = 1.0 / (train_size × dx_real)` (spatial frequency convention)
- Nyquist maximum frequency: `q_max_nyquist = π / dx_real ≈ 2.99×10⁴ m⁻¹`
- Cross-validation with experimental geometry parameters ensures physical correctness

**Output**: `I_clean` — "clean" diffraction intensity (no noise, no beamstop)

**Physical Meaning**: Ideal diffraction pattern under perfect conditions.

---

### Step 4: Intensity Normalization (IntensityNormalizer)

**Input**: `I_clean`

**Process**:
```python
I_clean_normalized = I_clean / np.sum(I_clean)
```

**Output**: `I_clean_normalized` (sum = 1)

**Physical Necessity**:

> **This step is crucial for physically correct Poisson noise.**
>
> After normalization, `I_clean_normalized` represents the **probability distribution** of photons hitting each detector pixel.
>
> In Step 6, we generate a total photon count `I_sc ~ Uniform(1e4, 1e7)`. The expected photons per pixel is:
> ```
> expected_photons = I_clean_normalized × I_sc
> ```
> Only with this normalization does `I_sc` have a clear physical meaning: **total scattered photons**.

---

### Step 5: Random Mask Application (RandomMaskApplier)

**Input**: `I_clean_normalized`

**Process**:
- With 50% probability, generate random detector defect mask
- Shapes: circles, rectangles, irregular polygons (1-3 shapes combined)
- Set masked pixel intensities to 0

**Output**: `I_masked`, `random_mask`

**Note**: This simulates **detector defects**, completely different from the **beamstop mask** (experimental beam block).

---

### Step 6: Noise and Beamstop Application (NoiseAndBeamstopApplier)

#### Sub-step 6.1: Poisson Noise
- **Method**: Reuses `AnalyticNoiseModel.add_poisson_gaussian()` from `src/physics/noise_model.py`
- **Process**:
  1. Generate total photon count: `I_sc ~ Uniform(1e4, 1e7)`
  2. Calculate expected photons per pixel: `expected_photons = I_masked × I_sc`
  3. Poisson sampling: `I_poisson = Poisson(expected_photons)`
- **Physical Meaning**: Simulates photon counting quantum (shot) noise. **Characteristic: variance = mean**

#### Sub-step 6.2: Gaussian Readout Noise
- **Process**: Add Gaussian noise on top of Poisson result: `I_noisy = I_poisson + N(0, 5.0)`
- **Physical Meaning**: Simulates detector electronics noise

#### Sub-step 6.3: Bad Pixels/Lines
- **Process**:
  - Bad pixels: 0.1% probability per pixel; randomly set to 0 (dead) or 1e4-1e5 (hot)
  - Bad lines: 0.5% probability; randomly zero an entire row or column

#### Sub-step 6.4: Beamstop Mask
- **Process**:
  1. Load binary mask from `beamstop_mask-585x585.mat`
  2. Create 3-10 pixel gradient transition zone at edges
  3. Apply: blocked region → 0; gradient zone → weighted attenuation
- **Physical Meaning**: Simulates experimental beam block that prevents direct beam from damaging the detector

**Final Output**: `I_noisy`, `beamstop_mask`, `I_sc`

---

### Step 7: Preprocessing and Saving (in generate_dataset.py)

#### Sub-step 7.1: Log Transform
- **Process**: `I_log = np.log10(1 + I)`
- **Purpose**: Compress the diffraction pattern's dynamic range (several orders of magnitude) for neural network processing

#### Sub-step 7.2: Standardization

- **Process**:
  1. Compute `mean_train` and `std_train` **only from training set's I_noisy_log**
  2. Apply to **all** data (train/val/test, input and target):
     ```python
     I_norm = (I_log - mean_train) / std_train
     ```
- **Purpose**: Ensures consistent data distribution — standard practice for deep learning training
- **Critical**: Using the same (mean_train, std_train) for all splits prevents data leakage

#### Sub-step 7.3: HDF5 Save

Structure shown in [Output Files & Statistics](#output-files--statistics).

---

## Configuration Parameters

### EXP_CONFIG — Experiment Parameters (aligned with Shanghai Soft X-ray FEL)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `detector_original_pixel_size` | 15e-6 m | 15 μm - from detector specs |
| `detector_original_size` | 4096 | Original detector array size |
| `detector_target_size` | 585 | Binned size for simulation |
| `detector_target_pixel_size` | ~1.05e-4 m | ~105 μm - calculated: 4096×15μm / 585 |
| `wavelength` | 2.7e-9 m | 2.7 nm - X-ray wavelength |
| `detector_distance` | 0.32 m | 32 cm - sample-to-detector distance |
| `poisson_I_sc_range` | [1e4, 1e7] | Total photon count range |
| `gaussian_read_noise_sigma` | 5.0 | Readout noise std (ADU) |
| `bad_pixel_prob` | 0.001 | Single bad pixel probability |
| `bad_line_prob` | 0.005 | Bad line probability |

> **Parameter Origin Note**: Core parameters (wavelength 2.7nm, detector distance 0.32m, original pixel size 15μm) are aligned with Shanghai Soft X-ray Free Electron Laser facility. `detector_target_pixel_size` is calculated to preserve FOV after binning: 4096×15μm = 585×105μm.

### BACTERIA_CONFIG — Cell Morphology (pixel-based for 585×585 grid)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `length_range_px` | (80, 200) | Bacteria length in pixels |
| `diameter_range_px` | (30, 60) | Bacteria diameter in pixels |
| `cell_states` | ['normal', 'dividing', 'curved'] | Cell state types |
| `n_gaussian_spots_range` | (3, 10) | Number of internal structures |
| `n_vacuoles_range` | (1, 3) | Number of low-density regions |

### AUGMENT_CONFIG — Data Augmentation

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rotation_range` | (0, 360) | Rotation angle range (degrees) |
| `translation_range` | (-0.1, 0.1) | Translation range (fraction of size) |
| `scale_range` | (0.9, 1.1) | Scale range |

### RANDOM_MASK_CONFIG — Detector Defects

| Parameter | Value | Description |
|-----------|-------|-------------|
| `apply_probability` | 0.5 | Probability to apply random mask |
| `n_shapes_range` | (1, 3) | Number of shapes to combine |
| `shape_types` | ['circle', 'rectangle', 'irregular'] | Shape types |

---

## Output Files & Statistics

### Visualization Images (`sample_xxx.png`)

4-panel display showing: `obj`, `I_clean` (log), `I_noisy` (log), `beamstop_mask`, plus statistics panel.

### Statistical Report (`statistical_report.txt`) — Indicator Definitions

| Category | Indicator | Definition | Expected Value |
|----------|-----------|------------|----------------|
| **Object Stats** | `min`, `max` | Projection density min/max | Always in [0, 1] — validates BioSampleGenerator |
| | `mean` | Average density | Reflects sample "thickness" |
| | `std` | Density standard deviation | Reflects internal structure contrast |
| **Clean Diffraction** | `total_photons (pre-norm)` | Sum of I_clean before normalization | Arbitrary units; checks magnitude |
| **Noisy Diffraction** | `total_photons (post-noise)` | Total counts after noise | Should be close to I_sc used — validates noise generation |
| | `variance/mean ratio` | Computed in non-masked region | **~1 for pure Poisson noise**; significant deviation indicates noise model issues |
| **Normalization** | `mean`, `std` | After standardization | **~0 and ~1** respectively; validates preprocessing |

### HDF5 Dataset Structure

```
bio_diffraction_v1.h5
├── train/
│   ├── input      [N_train, 585, 585] float32  # I_noisy_norm (standardized log)
│   ├── target     [N_train, 585, 585] float32  # I_clean_norm (standardized log)
│   ├── mask       [N_train, 585, 585] float32  # beamstop_mask (binary)
│   └── metadata/  # JSON strings: I_sc, random_mask_applied, etc.
├── val/           # Same structure as train
├── test/          # Same structure as train
└── config/
    ├── mean_train     # Mean used for standardization
    ├── std_train      # Std used for standardization
    ├── seed           # Random seed
    ├── dx_real_m      # Real space pixel size (meters)
    ├── dq_inv_m       # Reciprocal space pixel size (1/m)
    └── ...            # Other physical parameters
```

---

## Usage Guide

### Environment Setup

Required packages:
```bash
pip install numpy scipy h5py tqdm matplotlib scikit-image
```

### Small Sample Test (MUST RUN FIRST)

```bash
cd tests/bio_diffraction_simulator
python test_small_sample.py --num_samples 10 --output_dir ./test_output --seed 42
```

**Expected Output**:
- Console shows: `Overall: ALL TESTS PASSED`
- Files generated in `./test_output/`:
  - `sample_000.png` to `sample_009.png`
  - `pipeline_visualization.png`
  - `physical_validation_report.txt`
  - `statistical_report.txt`
  - `test_summary.json`

**Checkpoints** (open and verify):
1. `physical_validation_report.txt`: Both `FOV Consistency` and `Sampling` show **PASSED**
2. `statistical_report.txt`: All samples pass Range, Non-negative, Poisson, Intensity norm validations
3. `sample_xxx.png`: Images clearly show progression from object → clean diffraction → noisy diffraction

### Full Dataset Generation

```bash
python generate_dataset.py --num_train 10000 --num_val 1000 --num_test 1000 --output_file ./bio_diffraction_v1.h5 --seed 42
```

**Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_train` | 10000 | Training samples |
| `--num_val` | 1000 | Validation samples |
| `--num_test` | 1000 | Test samples |
| `--output_file` | `./bio_diffraction_v1.h5` | Output HDF5 path |
| `--seed` | 42 | Random seed |
| `--skip_validation` | False | Skip physical validation |

---

## Physics Background & References

### Experimental Parameters

Aligned with Shanghai Soft X-ray Free Electron Laser (SXFEL) soft X-ray beamline.

### Noise Model References

- Implementation reuses `src/physics/noise_model.py` — `AnalyticNoiseModel` class
- Theoretical basis: Poisson-Gaussian noise model for X-ray detectors
- Literature: "Denoising low-intensity diffraction signals using k-space deep learning"

### Physical Validation

- **FOV Consistency**: `4096 × 15μm = 585 × 105μm ≈ 61.44 mm`
- **Sampling Theorem**: `q_max_nyquist = π / dx_real ≈ 2.99×10⁴ m⁻¹`

---

## Loading the Dataset

```python
import h5py
import numpy as np

# Load dataset
with h5py.File('bio_diffraction_v1.h5', 'r') as f:
    # Load training data
    train_input = f['train/input'][:]    # Noisy diffraction (standardized)
    train_target = f['train/target'][:]  # Clean diffraction (standardized)
    train_mask = f['train/mask'][:]      # Beamstop mask

    # Load standardization parameters
    mean_train = f['config'].attrs['mean_train']
    std_train = f['config'].attrs['std_train']

    # To denormalize:
    # I_log = I_norm * std_train + mean_train
    # I = 10**I_log - 1
```

---

## Troubleshooting

### Common Issues

1. **"Beam stop mask file not found"**: Ensure `beamstop_mask-585x585.mat` exists in the same directory.

2. **"FOV inconsistency detected"**: Check that config parameters match the validation requirements.

3. **Low variance/mean ratio in Poisson validation**: This is expected for high dynamic range diffraction images. The validation checks for presence of noise rather than exact Poisson statistics.

4. **Import errors for `src.physics.noise_model`**: Run from the project root directory, or ensure the Python path includes the project root.
