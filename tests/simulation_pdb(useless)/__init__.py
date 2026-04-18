"""
衍射数据模拟模块

用于生成X射线衍射模拟数据，支持：
1. EMDB数据库结构
2. 程序化生物结构生成
3. 物理参数化模型
4. 正确的泊松噪声模型
5. 数据增强和验证
"""

from config import (
    Config,
    get_default_config,
    get_test_config,
)

from noise import (
    PoissonNoiseModel,
    GaussianNoiseModel,
    CompositeNoiseModel,
    verify_poisson_noise,
)

from emdb_loader import (
    EMDBLoader,
    EMDBDownloader,
    create_synthetic_emdb_like_structure,
)

from procedural_gen import (
    VirusGenerator,
    CellGenerator,
    OrganelleGenerator,
    CompositeStructureGenerator,
)

from diffraction import (
    FFTDiffractionCalculator,
    DataAugmenter,
    PhysicsBasedDiffractionGenerator,
    DiffractionPipeline,
)

from validation import (
    DiffractionValidator,
    DatasetStatistics,
    visualize_validation,
)

from pipeline import (
    DiffractionDataPipeline,
    run_pipeline,
)

__all__ = [
    'Config',
    'get_default_config',
    'get_test_config',
    'PoissonNoiseModel',
    'GaussianNoiseModel',
    'CompositeNoiseModel',
    'verify_poisson_noise',
    'EMDBLoader',
    'EMDBDownloader',
    'create_synthetic_emdb_like_structure',
    'VirusGenerator',
    'CellGenerator',
    'OrganelleGenerator',
    'CompositeStructureGenerator',
    'FFTDiffractionCalculator',
    'DataAugmenter',
    'PhysicsBasedDiffractionGenerator',
    'DiffractionPipeline',
    'DiffractionValidator',
    'DatasetStatistics',
    'visualize_validation',
    'DiffractionDataPipeline',
    'run_pipeline',
]

__version__ = '1.0.0'
