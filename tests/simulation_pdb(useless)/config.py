"""
衍射数据模拟配置文件

所有可调参数集中管理，便于实验和调优
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_BASE_DIR = PROJECT_ROOT / "tests" / "simulation" / "output"


@dataclass
class ExperimentConfig:
    """实验参数配置 - 与您的实验数据匹配"""
    wavelength_nm: float = 2.7
    detector_distance_cm: float = 32.0
    pixel_size_um: float = 15.0
    reference_image_size: int = 4096
    reference_pixel_size_um: float = 15.0
    

@dataclass
class SimulationConfig:
    """模拟参数配置"""
    image_size: int = 512
    oversampling_ratio: float = 1.0
    resolution_angstrom: float = 1.0
    random_seed: int = 42
    

@dataclass
class DataGenerationConfig:
    """数据生成配置"""
    total_samples: int = 100
    enable_random_projection: bool = True
    projection_axes: List[int] = field(default_factory=lambda: [0, 1, 2])
    enable_random_size: bool = False
    image_size_options: List[int] = field(default_factory=lambda: [512])
    

@dataclass
class StructureSourceConfig:
    """结构来源配置"""
    use_emdb: bool = True
    use_procedural: bool = True
    use_physics_model: bool = True
    
    emdb_ratio: float = 0.4
    procedural_ratio: float = 0.4
    physics_model_ratio: float = 0.2
    

@dataclass
class EMDBConfig:
    """EMDB数据源配置"""
    enabled: bool = True
    download_dir: Path = field(default_factory=lambda: OUTPUT_BASE_DIR / "emdb_data")
    cache_dir: Path = field(default_factory=lambda: OUTPUT_BASE_DIR / "emdb_cache")
    
    target_size_range_nm: Tuple[float, float] = (50.0, 2000.0)
    
    emdb_categories: dict = field(default_factory=lambda: {
        "virus": {"size_range_nm": (50, 300), "count_ratio": 0.3},
        "organelle": {"size_range_nm": (100, 500), "count_ratio": 0.3},
        "complex": {"size_range_nm": (20, 100), "count_ratio": 0.2},
        "bacteria": {"size_range_nm": (500, 2000), "count_ratio": 0.2},
    })
    
    sample_emdb_ids: List[str] = field(default_factory=lambda: [
        "EMD-1234",
        "EMD-3000",
        "EMD-5001",
    ])


@dataclass
class ProceduralGeneratorConfig:
    """程序化生成器配置"""
    enabled: bool = True
    
    particle_types: List[str] = field(default_factory=lambda: [
        'sphere', 'ellipsoid', 'irregular', 
        'cell', 'virus', 'organelle'
    ])
    
    size_range_nm: Tuple[float, float] = (200.0, 1500.0)
    aspect_ratio_range: Tuple[float, float] = (1.0, 3.0)
    irregularity_range: Tuple[float, float] = (0.05, 0.3)
    
    add_internal_structure: bool = True
    internal_feature_count_range: Tuple[int, int] = (3, 10)
    
    cell_types: List[str] = field(default_factory=lambda: [
        'simple', 'with_organelles', 'with_membrane'
    ])
    
    virus_types: List[str] = field(default_factory=lambda: [
        'spherical', 'icosahedral', 'enveloped'
    ])
    
    organelle_types: List[str] = field(default_factory=lambda: [
        'mitochondria', 'nucleus', 'ribosome', 'vesicle'
    ])


@dataclass
class PhysicsModelConfig:
    """物理参数化模型配置"""
    enabled: bool = True
    
    center_intensity_range: Tuple[float, float] = (30000.0, 50000.0)
    background_intensity_range: Tuple[float, float] = (1000.0, 2000.0)
    core_radius_range: Tuple[float, float] = (15.0, 35.0)
    halo_decay_range: Tuple[float, float] = (40.0, 80.0)
    
    structure_strength_range: Tuple[float, float] = (0.1, 0.4)
    speckle_count_range: Tuple[int, int] = (3, 10)
    ring_count_range: Tuple[int, int] = (0, 3)


@dataclass
class NoiseConfig:
    """噪声配置"""
    poisson_noise_enabled: bool = True
    gaussian_noise_enabled: bool = False
    bad_pixels_enabled: bool = False
    structured_noise_enabled: bool = False
    
    exposure_level_range: Tuple[float, float] = (10000.0, 300000.0)
    
    gaussian_std_range: Tuple[float, float] = (0.0, 10.0)
    
    bad_pixel_ratio: float = 0.001
    
    structured_noise_correlation_length: float = 20.0
    structured_noise_strength: float = 0.1


@dataclass
class DataAugmentationConfig:
    """数据增强配置 - 在FFT之前执行"""
    rotation_enabled: bool = True
    flip_enabled: bool = True
    scale_enabled: bool = False
    
    rotation_angles: List[int] = field(default_factory=lambda: [0, 90, 180, 270])
    flip_axes: List[int] = field(default_factory=lambda: [0, 1])
    scale_range: Tuple[float, float] = (0.9, 1.1)


@dataclass
class ValidationConfig:
    """验证配置"""
    enabled: bool = True
    
    min_contrast: float = 0.05
    intensity_max_range: Tuple[float, float] = (5000.0, 500000.0)
    intensity_mean_range: Tuple[float, float] = (500.0, 20000.0)
    decay_ratio_range: Tuple[float, float] = (1.5, 50.0)
    
    visualize_samples: int = 10
    visualization_dir: Path = field(default_factory=lambda: OUTPUT_BASE_DIR / "visualization")


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: Path = field(default_factory=lambda: OUTPUT_BASE_DIR / "dataset")
    output_filename: str = "diffraction_data.h5"
    
    save_metadata: bool = True
    save_clean_data: bool = True
    save_noisy_data: bool = True
    
    compression: str = "gzip"
    compression_level: int = 4


@dataclass
class Config:
    """主配置类"""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    structure_source: StructureSourceConfig = field(default_factory=StructureSourceConfig)
    emdb: EMDBConfig = field(default_factory=EMDBConfig)
    procedural: ProceduralGeneratorConfig = field(default_factory=ProceduralGeneratorConfig)
    physics_model: PhysicsModelConfig = field(default_factory=PhysicsModelConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    def __post_init__(self):
        for field_name in ['emdb', 'validation', 'output']:
            field_obj = getattr(self, field_name)
            for attr_name in dir(field_obj):
                if attr_name.endswith('_dir') and not attr_name.startswith('_'):
                    path = getattr(field_obj, attr_name)
                    if isinstance(path, Path):
                        path.mkdir(parents=True, exist_ok=True)


def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


def get_test_config() -> Config:
    """获取测试配置（小规模）"""
    config = Config()
    config.data_generation.total_samples = 100
    config.validation.visualize_samples = 5
    return config


if __name__ == "__main__":
    config = get_default_config()
    print("配置加载成功")
    print(f"图像尺寸: {config.simulation.image_size}")
    print(f"总样本数: {config.data_generation.total_samples}")
    print(f"曝光水平范围: {config.noise.exposure_level_range}")
    print(f"输出目录: {config.output.output_dir}")
