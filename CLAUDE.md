# DeepPhase-X — Project Guide

> One-file reference for AI agents working on this project. Read this first, then proceed.

## Project Overview

**Goal**: Physics-guided deep learning for XFEL coherent diffraction imaging denoising, improving downstream phase retrieval convergence and resolution.

**Target**: Publishable research (Nature Communications / Science Advances tier).

**Stack**: Python 3.10+, PyTorch >=2.0, CuPy (GPU), HDF5, NumPy/SciPy.

## Architecture

```
DeepPhase-X/
├── src/
│   ├── simulation/           # Physics simulation pipeline
│   ├── models/               # Neural network architectures
│   ├── losses/               # Loss functions
│   ├── evaluation/           # Metrics & visualization
│   └── reconstruction/       # Phase retrieval (HIO/ER)
├── scripts/                  # Entry points (generate, train, evaluate)
├── configs/                  # YAML configs (simulation, model, training)
├── data/
│   ├── raw/                  # Experimental .mat files
│   └── simulated/            # Generated HDF5 datasets
├── experiments/              # outputs: checkpoints/, logs/, results/
├── reference/                # Reference papers & code
│   ├── 参考文献/              # Original PDFs
│   └── 参考文献_summaries/    # Parsed summaries (.md)
├── docs_guide/               # Research plans, proposals
├── tests/                    # pytest suite
├── pyproject.toml
└── requirements.txt
```

## Module Index

| File | Key Symbols | Purpose |
|------|------------|---------|
| `src/simulation/simulator.py` | `XRaySimulator` | PDB-based X-ray diffraction simulation |
| `src/simulation/bio_sample_generator.py` | `BioSampleGenerator` | E. coli 2D density map generation |
| `src/simulation/bio_diffraction_simulator.py` | `BioDiffractionSimulator` | FFT diffraction (bio pipeline) |
| `src/simulation/bio_config.py` | `BioConfig` | Physical parameters (Shanghai SXFEL aligned) |
| `src/simulation/noise_model.py` | `AnalyticNoiseModel` | Poisson + Gaussian noise model |
| `src/simulation/noise_beamstop_applier.py` | `NoiseAndBeamstopApplier` | Noise injection + beamstop masking |
| `src/simulation/intensity_normalizer.py` | `IntensityNormalizer` | Normalize sum=1 for Poisson sampling |
| `src/simulation/random_mask_applier.py` | `RandomMaskApplier` | Simulate detector defects |
| `src/simulation/data_augmentor.py` | `DataAugmentor` | Rotation/translation/scaling |
| `src/simulation/bio_utils.py` | validation/viz/preprocess utils | Quality checks & visualization |
| `src/simulation/beam_stop.py` | `BeamStop` | Beamstop utilities |
| `src/simulation/backend.py` | backend utils | CuPy/NumPy backend switching |
| `src/models/layers.py` | `PartialConv2d`, `CentroSymmetricConv` | Custom convolution layers |
| `src/models/unet_physics.py` | `PhysicsUNet`, `LightweightPhysicsUNet` | Physics-guided U-Net denoiser |
| `src/models/noise_gan.py` | `NoiseGAN` | GAN-based noise modeling (NOT IN USE) |
| `src/losses/losses.py` | `PhysicsGuidedLoss` | Multi-component physics loss |
| `src/evaluation/metrics.py` | `DiffractionMetrics` | PSNR, SSIM, R-factor, PRTF, FSC |
| `src/evaluation/visualization.py` | visualization utils | Publication-quality plots |
| `src/reconstruction/hio_er.py` | `PhaseRetrieval` | HIO/ER phase retrieval algorithms |
| `src/reconstruction/support.py` | `SupportEstimator` | Support estimation for CDI |
| `scripts/generate_bio_dataset.py` | CLI | Bio dataset generation (streaming + resume) |
| `scripts/main_pipeline.py` | CLI | Main training/inference pipeline |
| `scripts/dataset.py` | utils | Dataset handling utilities |
| `scripts/checkpoint.py` | utils | Checkpoint management |

## Data Flow

```
Experimental .mat → feature extraction → simulation parameters
BioSampleGenerator → DataAugmentor → BioDiffractionSimulator (FFT)
→ IntensityNormalizer → NoiseAndBeamstopApplier → RandomMaskApplier
→ preprocess (log10 transform) → HDF5 (clean/noisy/mask pairs)

Training: HDF5 → DataLoader → PhysicsUNet(noisy, mask) → pred_linear
→ PhysicsGuidedLoss(pred, clean, mask) → backprop

Evaluation: denoised → metrics (PSNR/SSIM/R-factor)
→ PhaseRetrieval → PRTF/FSC/convergence analysis
```

## Physical Constraints

- Diffraction patterns exhibit **centrosymmetry** (Friedel's law)
- Beamstop causes **missing pixels** in low-frequency region
- Noise: **Poisson** (photon counting) + **Gaussian** (readout)
- Autocorrelation support = 2x object size
- Input to network: `log10(1 + I_noisy)`; output: linear scale via inverse transform

## Code Conventions

- **GPU first**: Use `torch.cuda` and CuPy where available; check `backend.py` for GPU/CPU switching
- **Reproducibility**: Every experiment records config.json, random seed, commit hash
- **Checkpointing**: All training supports interrupt/resume; delete checkpoints after successful completion
- **Naming**: `PascalCase` for classes, `snake_case` for functions/variables
- **Config**: All hyperparameters in `configs/*.yaml`, not hardcoded
- **Testing**: `pytest tests/` before any merge to master
- **Branches**: All changes on feature branches; merge only after test pass + review

## Research Status

| Step | Description | Status |
|------|-------------|--------|
| 0 | Project structure & infrastructure | ✅ Done |
| 1 | Data simulation pipeline (bio diffraction) | ✅ Done |
| 1b | Experimental feature calibration & quality check | Pending |
| 2 | Neural network architecture (PhysicsUNet) | Scaffolded, needs training |
| 3 | Loss functions & training loop | Scaffolded, needs training |
| 4 | Lightweight evaluation metrics | Partially done |
| 5 | Heavy evaluation (PRTF, 1000-run Monte Carlo) | Not started |
| 6 | Ablation experiments & hyperparameter tuning | Not started |
| 7 | Comparison with traditional methods | Not started |

## Key Configs

| Config | Path | Purpose |
|--------|------|---------|
| Simulation | `configs/simulation_config.yaml` | Physical parameters, noise levels |
| Model | `configs/model_config.yaml` | Network architecture params |
| Training | `configs/training_config.yaml` | Optimizer, LR schedule, epochs |
| GAN | `configs/gan_config.yaml` | NoiseGAN params (NOT IN USE) |
