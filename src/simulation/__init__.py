"""X-ray diffraction simulation engine."""

# Bio simulation pipeline components (always available)
from .bio_config import (
    EXP_CONFIG, BACTERIA_CONFIG, AUGMENT_CONFIG, RANDOM_MASK_CONFIG,
    BASE_DIR, run_all_validations, compute_physical_parameters,
    print_physical_report, get_config_summary,
)
from .bio_sample_generator import BioSampleGenerator, generate_bio_sample
from .data_augmentor import DataAugmentor
from .bio_diffraction_simulator import DiffractionSimulator as BioDiffractionSimulator
from .intensity_normalizer import IntensityNormalizer
from .random_mask_applier import RandomMaskApplier
from .noise_beamstop_applier import NoiseAndBeamstopApplier
from .bio_utils import (
    preprocess_for_training, visualize_sample,
    generate_physical_report, generate_statistical_report,
    validate_output_range, validate_diffraction_nonnegative,
    validate_poisson_statistics, validate_normalization,
)

# Existing PDB-based simulation (requires src.config)
try:
    from .simulator import XRaySimulator
    from .beam_stop import apply_beam_stop, create_beam_stop_mask
    _HAS_PDB_SIMULATOR = True
except ImportError:
    _HAS_PDB_SIMULATOR = False

__all__ = [
    # Existing
    "XRaySimulator",
    "AnalyticNoiseModel",
    "apply_beam_stop",
    "create_beam_stop_mask",
    # Bio simulation
    "EXP_CONFIG", "BACTERIA_CONFIG", "AUGMENT_CONFIG", "RANDOM_MASK_CONFIG",
    "BASE_DIR", "BioSampleGenerator", "generate_bio_sample", "DataAugmentor",
    "BioDiffractionSimulator", "IntensityNormalizer",
    "RandomMaskApplier", "NoiseAndBeamstopApplier",
    "preprocess_for_training", "visualize_sample",
    "generate_physical_report", "generate_statistical_report",
    "validate_output_range", "validate_diffraction_nonnegative",
    "validate_poisson_statistics", "validate_normalization",
]
