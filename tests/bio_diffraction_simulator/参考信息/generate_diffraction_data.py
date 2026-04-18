"""
衍射图案生成脚本

功能：
1. 从PDB结构生成衍射图案
2. 添加水化层（可选）
3. 添加泊松噪声和高斯噪声
4. 保存为HDF5格式

输入：
- PDB文件
- 结构配置文件（JSON）

输出：
- HDF5数据集文件
"""

import os
import sys
import json
import time
import logging
import random
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter
import h5py

import matplotlib
import matplotlib.font_manager as fm
import shutil
shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)
fm.fontManager.addfont(r'c:\Windows\Fonts\MSYHL.TTC')
matplotlib.rc('font', family='Microsoft YaHei', size=12)
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from Bio.PDB import PDBParser
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("错误: BioPython未安装。请运行: pip install biopython")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 全局配置参数 - 可在此修改
# =============================================================================

STRUCTURE_CONFIG_FILE = project_root / "data" / "pdb_structures" / "structure_config.json"

OUTPUT_DIR = project_root / "data" / "diffraction_dataset"

OUTPUT_FILE = OUTPUT_DIR / "diffraction_data.h5"

TOTAL_SAMPLE_PAIRS = 100

RANDOM_SEED = 42

ENABLE_RANDOM_PROJECTION = True

DEFAULT_PROJECTION_AXIS = 2

PROJECTION_AXES = [0, 1, 2]

ENABLE_RANDOM_EXPOSURE = True

DEFAULT_EXPOSURE_LEVEL = 100000

EXPOSURE_LEVEL_RANGE = (10000, 500000)

POISSON_NOISE_ENABLED = True

ENABLE_RANDOM_GAUSSIAN = True

DEFAULT_GAUSSIAN_NOISE = 0.0

GAUSSIAN_NOISE_RANGE = (0.0, 10.0)

ENABLE_RANDOM_SIZE = False

DEFAULT_IMAGE_SIZE = 512

IMAGE_SIZE_OPTIONS = [512, 1024]

OVERSAMPLING_RATIO = 1.0

RESOLUTION = 1.0

VISUALIZATION_ENABLED = True

VISUALIZATION_SAMPLES = 100

VISUALIZATION_DIR = OUTPUT_DIR / "visualization"

WAVELENGTH_NM = 2.7

DETECTOR_DISTANCE_CM = 32.0

PIXEL_SIZE_UM = 15.0

REFERENCE_IMAGE_SIZE = 4096

REFERENCE_PIXEL_SIZE_UM = 15.0

USE_LARGE_PARTICLE_SIMULATION = True

PARTICLE_SIZE_RANGE_NM = (200.0, 1000.0)

PARTICLE_TYPES = ['sphere', 'ellipsoid', 'irregular']

PARTICLE_ASPECT_RATIO_RANGE = (1.0, 2.5)

PARTICLE_IRREGULARITY_RANGE = (0.05, 0.2)

ADD_INTERNAL_STRUCTURE = True

USE_REALISTIC_DIFFRACTION = True

CENTER_INTENSITY_RANGE = (30000.0, 50000.0)

BACKGROUND_INTENSITY_RANGE = (1000.0, 1500.0)

DECAY_RATE_RANGE = (50.0, 100.0)

STRUCTURE_STRENGTH_RANGE = (0.1, 0.3)


# =============================================================================
# 原子参数
# =============================================================================

ATOMIC_ELECTRONS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
    'FE': 26, 'ZN': 30, 'MG': 12, 'CA': 20, 'MN': 25,
    'CU': 29, 'SE': 34, 'CL': 17, 'NA': 11, 'K': 19,
}

ATOMIC_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
    'P': 1.80, 'FE': 2.00, 'ZN': 1.39, 'MG': 1.73, 'CA': 2.31,
    'MN': 1.79, 'CU': 1.40, 'SE': 1.90, 'CL': 1.75, 'NA': 2.27, 'K': 2.75,
}


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class SampleMetadata:
    sample_id: int
    pdb_id: str
    projection_axis: int
    exposure_level: float
    gaussian_noise_std: float
    has_hydration_layer: bool
    hydration_thickness: float
    hydration_density: float
    image_size: int


# =============================================================================
# 衍射模拟器
# =============================================================================

class DiffractionSimulator:
    """衍射图案模拟器"""
    
    def __init__(self, grid_size: int = 512, resolution: float = 1.0, oversampling: float = 2.0):
        self.grid_size = grid_size
        self.resolution = resolution
        self.oversampling = oversampling
        if HAS_BIOPYTHON:
            self.parser = PDBParser(QUIET=True)
        else:
            self.parser = None
    
    def parse_pdb(self, pdb_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """解析PDB文件，提取原子坐标和权重"""
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython is required")
        
        structure = self.parser.get_structure("protein", str(pdb_path))
        
        coords = []
        weights = []
        elements = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    hetflag = residue.id[0]
                    if hetflag != ' ' and hetflag != 'W':
                        continue
                    
                    for atom in residue:
                        coord = atom.get_coord()
                        element = atom.element.upper().strip()
                        
                        if element == '':
                            element = atom.get_name()[0].upper()
                        
                        weight = ATOMIC_ELECTRONS.get(element, 6)
                        
                        coords.append(coord)
                        weights.append(weight)
                        elements.append(element)
        
        return np.array(coords), np.array(weights), elements
    
    def create_density_map(
        self,
        coords: np.ndarray,
        weights: np.ndarray,
        projection_axis: int = 2,
        scale_factor: float = 1.0
    ) -> np.ndarray:
        """创建电子密度图
        
        参数:
            coords: 原子坐标 (Å)
            weights: 原子权重 (电子数)
            projection_axis: 投影轴 (0=X, 1=Y, 2=Z)
            scale_factor: 结构放大倍数
        """
        coords_centered = coords - coords.mean(axis=0)
        
        coords_centered = coords_centered * scale_factor
        
        axes = [0, 1, 2]
        axes.remove(projection_axis)
        coords_2d = coords_centered[:, axes]
        
        max_extent = np.abs(coords_2d).max()
        
        grid_occupancy = 0.8
        scale = (self.grid_size * grid_occupancy / 2) / max_extent if max_extent > 0 else 1.0
        coords_scaled = coords_2d * scale + self.grid_size / 2
        
        density = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        
        for (x, y), w in zip(coords_scaled, weights):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size:
                density[iy, ix] += w
        
        sigma = max(1.0, 1.5 / self.resolution * scale_factor)
        density = gaussian_filter(density, sigma=sigma)
        
        if density.max() > 0:
            density = density / density.max()
        
        return density
    
    def add_hydration_layer(
        self,
        density: np.ndarray,
        coords: np.ndarray,
        thickness: float = 3.0,
        density_value: float = 0.334
    ) -> np.ndarray:
        """添加水化层"""
        coords_centered = coords - coords.mean(axis=0)
        
        axes_3d = [0, 1, 2]
        max_radius = np.sqrt((coords_centered ** 2).sum(axis=1)).max()
        
        x = np.linspace(-self.grid_size/2, self.grid_size/2, self.grid_size)
        y = np.linspace(-self.grid_size/2, self.grid_size/2, self.grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        inner_radius = max_radius * 1.05
        outer_radius = inner_radius + thickness / self.resolution
        
        hydration_mask = (R >= inner_radius) & (R <= outer_radius)
        
        hydration_density = np.zeros_like(density)
        hydration_density[hydration_mask] = density_value
        
        hydration_density = gaussian_filter(hydration_density, sigma=1.0)
        
        combined = density + hydration_density * 0.1
        
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def generate_diffraction(self, density: np.ndarray) -> np.ndarray:
        """生成衍射图案"""
        padded_size = int(self.grid_size * self.oversampling)
        padded = np.zeros((padded_size, padded_size), dtype=density.dtype)
        offset = (padded_size - self.grid_size) // 2
        padded[offset:offset + self.grid_size, offset:offset + self.grid_size] = density
        
        f_map = fft2(padded)
        f_map = fftshift(f_map)
        
        intensity = np.abs(f_map) ** 2
        
        if self.oversampling > 1.0:
            center = padded_size // 2
            half_size = self.grid_size // 2
            intensity = intensity[center - half_size:center + half_size,
                                  center - half_size:center + half_size]
        
        return intensity
    
    def generate_realistic_diffraction(
        self,
        center_intensity: float = 40000.0,
        background_intensity: float = 1200.0,
        decay_rate: float = 300.0,
        structure_strength: float = 0.1,
        seed: int = None
    ) -> np.ndarray:
        """直接生成符合实验数据特征的衍射图案
        
        参数:
            center_intensity: 中心强度
            background_intensity: 背景强度
            decay_rate: 强度衰减率
            structure_strength: 结构强度 (0-1)
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        
        center = self.grid_size / 2
        x = np.linspace(-center, center, self.grid_size)
        y = np.linspace(-center, center, self.grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        core_radius = np.random.uniform(20, 30)
        core_intensity = center_intensity * np.exp(-R**2 / (2 * core_radius**2))
        
        halo_decay = np.random.uniform(40, 60)
        halo_intensity = (center_intensity * 0.15) * np.exp(-R / halo_decay)
        
        radial_profile = core_intensity + halo_intensity + background_intensity
        
        n_speckles = np.random.randint(3, 8)
        for _ in range(n_speckles):
            speckle_r = np.random.uniform(0, core_radius * 2)
            speckle_theta = np.random.uniform(0, 2 * np.pi)
            speckle_x = speckle_r * np.cos(speckle_theta)
            speckle_y = speckle_r * np.sin(speckle_theta)
            speckle_size = np.random.uniform(5, 15)
            speckle_intensity = np.random.uniform(0.5, 2.0)
            
            dist_to_speckle = np.sqrt((X - speckle_x)**2 + (Y - speckle_y)**2)
            speckle = speckle_intensity * np.exp(-dist_to_speckle**2 / (2 * speckle_size**2))
            radial_profile = radial_profile * (1 + speckle * structure_strength * 0.5)
        
        n_rings = np.random.randint(0, 2)
        for _ in range(n_rings):
            ring_r = np.random.uniform(20, 60)
            ring_width = np.random.uniform(3, 10)
            ring_intensity = np.random.uniform(0.8, 1.2)
            
            ring = np.exp(-(R - ring_r)**2 / (2 * ring_width**2))
            radial_profile = radial_profile * (1 + ring * (ring_intensity - 1) * structure_strength * 0.3)
        
        noise = np.random.randn(self.grid_size, self.grid_size)
        noise_strength = structure_strength * 0.5 * np.exp(-R / core_radius)
        radial_profile = radial_profile * (1 + noise * noise_strength)
        
        return radial_profile
    
    def create_large_particle_density(
        self,
        particle_type: str = 'ellipsoid',
        size_nm: float = 500.0,
        aspect_ratio: float = 1.0,
        irregularity: float = 0.1,
        internal_structure: bool = True
    ) -> np.ndarray:
        """创建大尺度颗粒的电子密度图
        
        参数:
            particle_type: 颗粒类型 ('sphere', 'ellipsoid', 'irregular')
            size_nm: 颗粒尺寸 (nm)
            aspect_ratio: 长短轴比 (用于椭球)
            irregularity: 不规则度 (0-1)
            internal_structure: 是否添加内部结构
        """
        density = np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
        
        center = self.grid_size / 2
        
        grid_occupancy = 0.7
        radius_pixels = (self.grid_size * grid_occupancy / 2)
        
        x = np.linspace(-center, center, self.grid_size)
        y = np.linspace(-center, center, self.grid_size)
        X, Y = np.meshgrid(x, y)
        
        if particle_type == 'sphere':
            R = np.sqrt(X**2 + Y**2)
            mask = R <= radius_pixels
            
        elif particle_type == 'ellipsoid':
            R = np.sqrt((X / aspect_ratio)**2 + Y**2)
            mask = R <= radius_pixels
            
        elif particle_type == 'irregular':
            R = np.sqrt(X**2 + Y**2)
            np.random.seed(int(size_nm * 1000) % 2**31)
            noise = np.random.randn(self.grid_size, self.grid_size) * irregularity * radius_pixels
            mask = (R + noise) <= radius_pixels
        else:
            R = np.sqrt(X**2 + Y**2)
            mask = R <= radius_pixels
        
        density[mask] = 1.0
        
        if internal_structure:
            np.random.seed(int(size_nm * 100 + aspect_ratio * 10) % 2**31)
            
            n_features = np.random.randint(3, 8)
            for _ in range(n_features):
                cx = np.random.uniform(-radius_pixels * 0.5, radius_pixels * 0.5)
                cy = np.random.uniform(-radius_pixels * 0.5, radius_pixels * 0.5)
                cr = np.random.uniform(radius_pixels * 0.1, radius_pixels * 0.3)
                feature_mask = ((X - cx)**2 + (Y - cy)**2) <= cr**2
                density[feature_mask & mask] = np.random.uniform(0.5, 1.5)
            
            n_rings = np.random.randint(0, 3)
            for _ in range(n_rings):
                ring_r = np.random.uniform(radius_pixels * 0.3, radius_pixels * 0.8)
                ring_w = np.random.uniform(radius_pixels * 0.05, radius_pixels * 0.15)
                ring_mask = np.abs(R - ring_r) <= ring_w
                density[ring_mask & mask] *= np.random.uniform(0.7, 1.3)
        
        sigma = max(2.0, radius_pixels * 0.02)
        density = gaussian_filter(density, sigma=sigma)
        
        if density.max() > 0:
            density = density / density.max()
        
        return density


# =============================================================================
# 噪声模型
# =============================================================================

class NoiseModel:
    """噪声模型"""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def add_poisson_noise(self, intensity: np.ndarray, exposure_level: float) -> Tuple[np.ndarray, np.ndarray]:
        """添加泊松噪声，返回缩放后的干净图像和含噪图像"""
        intensity_norm = intensity / (intensity.max() + 1e-10)
        photon_counts = intensity_norm * exposure_level
        photon_counts = np.maximum(photon_counts, 0)
        noisy = self.rng.poisson(photon_counts).astype(np.float64)
        clean_scaled = photon_counts
        return clean_scaled, noisy
    
    def add_gaussian_noise(self, intensity: np.ndarray, std: float) -> np.ndarray:
        """添加高斯噪声"""
        if std <= 0:
            return intensity
        noise = self.rng.normal(0, std, intensity.shape)
        noisy = intensity + noise
        noisy = np.maximum(noisy, 0)
        return noisy


# =============================================================================
# 数据生成流水线
# =============================================================================

class DiffractionDataPipeline:
    """衍射数据生成流水线"""
    
    def __init__(
        self,
        config_file: Path,
        output_dir: Path,
        random_seed: int = 42
    ):
        self.config_file = config_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        
        self.noise_model = NoiseModel(seed=random_seed)
        
        self.structure_configs = self._load_structure_configs()
    
    def _load_structure_configs(self) -> List[Dict]:
        """加载结构配置"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get("structures", [])
    
    def _get_random_projection_axis(self, structure_config: Dict) -> int:
        """获取随机投影轴"""
        if ENABLE_RANDOM_PROJECTION:
            return random.choice(PROJECTION_AXES)
        return DEFAULT_PROJECTION_AXIS
    
    def _get_random_exposure_level(self, structure_config: Dict) -> float:
        """获取随机曝光水平"""
        if ENABLE_RANDOM_EXPOSURE:
            min_exp, max_exp = EXPOSURE_LEVEL_RANGE
            return random.uniform(min_exp, max_exp)
        return DEFAULT_EXPOSURE_LEVEL
    
    def _get_random_gaussian_noise(self, structure_config: Dict) -> float:
        """获取随机高斯噪声"""
        if ENABLE_RANDOM_GAUSSIAN:
            min_std, max_std = GAUSSIAN_NOISE_RANGE
            return random.uniform(min_std, max_std)
        return DEFAULT_GAUSSIAN_NOISE
    
    def _get_random_image_size(self) -> int:
        """获取随机图像尺寸"""
        if ENABLE_RANDOM_SIZE:
            return random.choice(IMAGE_SIZE_OPTIONS)
        return DEFAULT_IMAGE_SIZE
    
    def _get_hydration_params(self, structure_config: Dict) -> Tuple[float, float]:
        """获取水化层参数"""
        if not structure_config.get("has_hydration_layer", False):
            return 0.0, 0.0
        
        thickness_range = structure_config.get("hydration_thickness_range", [3.0, 4.0])
        density_range = structure_config.get("hydration_density_range", [0.33, 0.38])
        
        thickness = random.uniform(thickness_range[0], thickness_range[1])
        density = random.uniform(density_range[0], density_range[1])
        
        return thickness, density
    
    def generate_sample(
        self,
        sample_id: int,
        structure_config: Dict,
        pdb_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray, SampleMetadata]:
        """生成单个样本"""
        pdb_id = structure_config["pdb_id"]
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        
        projection_axis = self._get_random_projection_axis(structure_config)
        exposure_level = self._get_random_exposure_level(structure_config)
        gaussian_std = self._get_random_gaussian_noise(structure_config)
        image_size = self._get_random_image_size()
        
        has_hydration = structure_config.get("has_hydration_layer", False)
        hydration_thickness, hydration_density = self._get_hydration_params(structure_config)
        
        simulator = DiffractionSimulator(
            grid_size=image_size,
            resolution=RESOLUTION,
            oversampling=OVERSAMPLING_RATIO
        )
        
        if USE_REALISTIC_DIFFRACTION:
            center_intensity = self.rng.uniform(*CENTER_INTENSITY_RANGE)
            background_intensity = self.rng.uniform(*BACKGROUND_INTENSITY_RANGE)
            decay_rate = self.rng.uniform(*DECAY_RATE_RANGE)
            structure_strength = self.rng.uniform(*STRUCTURE_STRENGTH_RANGE)
            
            clean_pattern = simulator.generate_realistic_diffraction(
                center_intensity=center_intensity,
                background_intensity=background_intensity,
                decay_rate=decay_rate,
                structure_strength=structure_strength,
                seed=sample_id
            )
            particle_type = "realistic"
        elif USE_LARGE_PARTICLE_SIMULATION:
            particle_type = self.rng.choice(PARTICLE_TYPES)
            particle_size = self.rng.uniform(*PARTICLE_SIZE_RANGE_NM)
            aspect_ratio = self.rng.uniform(*PARTICLE_ASPECT_RATIO_RANGE)
            irregularity = self.rng.uniform(*PARTICLE_IRREGULARITY_RANGE)
            
            density = simulator.create_large_particle_density(
                particle_type=particle_type,
                size_nm=particle_size,
                aspect_ratio=aspect_ratio,
                irregularity=irregularity,
                internal_structure=ADD_INTERNAL_STRUCTURE
            )
            clean_pattern = simulator.generate_diffraction(density)
        else:
            if not pdb_path.exists():
                raise FileNotFoundError(f"PDB file not found: {pdb_path}")
            
            coords, weights, elements = simulator.parse_pdb(pdb_path)
            density = simulator.create_density_map(coords, weights, projection_axis)
            
            if has_hydration and hydration_thickness > 0:
                density = simulator.add_hydration_layer(
                    density, coords,
                    thickness=hydration_thickness,
                    density_value=hydration_density
                )
            clean_pattern = simulator.generate_diffraction(density)
        
        if POISSON_NOISE_ENABLED:
            clean_scaled, noisy_pattern = self.noise_model.add_poisson_noise(clean_pattern, exposure_level)
        else:
            intensity_norm = clean_pattern / (clean_pattern.max() + 1e-10)
            clean_scaled = intensity_norm * exposure_level
            noisy_pattern = clean_scaled.copy()
        
        if gaussian_std > 0:
            noisy_pattern = self.noise_model.add_gaussian_noise(noisy_pattern, gaussian_std)
        
        metadata = SampleMetadata(
            sample_id=sample_id,
            pdb_id=pdb_id if not USE_LARGE_PARTICLE_SIMULATION else f"particle_{particle_type}",
            projection_axis=projection_axis,
            exposure_level=exposure_level,
            gaussian_noise_std=gaussian_std,
            has_hydration_layer=has_hydration,
            hydration_thickness=hydration_thickness,
            hydration_density=hydration_density,
            image_size=image_size
        )
        
        return clean_scaled, noisy_pattern, metadata
    
    def generate_dataset(
        self,
        total_pairs: int,
        pdb_dir: Path,
        output_file: Path
    ) -> Dict[str, Any]:
        """生成完整数据集"""
        logger.info(f"开始生成数据集: {total_pairs} 对样本")
        
        clean_patterns = []
        noisy_patterns = []
        metadata_list = []
        
        n_structures = len(self.structure_configs)
        samples_per_structure = max(1, total_pairs // n_structures)
        
        sample_id = 0
        for struct_idx, struct_config in enumerate(self.structure_configs):
            if sample_id >= total_pairs:
                break
            
            n_samples = min(samples_per_structure, total_pairs - sample_id)
            
            for i in range(n_samples):
                if sample_id >= total_pairs:
                    break
                
                try:
                    clean, noisy, meta = self.generate_sample(
                        sample_id, struct_config, pdb_dir
                    )
                    
                    clean_patterns.append(clean)
                    noisy_patterns.append(noisy)
                    metadata_list.append(meta)
                    
                    if (sample_id + 1) % 10 == 0:
                        logger.info(f"进度: {sample_id + 1}/{total_pairs}")
                    
                    sample_id += 1
                    
                except Exception as e:
                    logger.error(f"生成样本失败 {struct_config['pdb_id']}: {e}")
        
        self._save_dataset(clean_patterns, noisy_patterns, metadata_list, output_file)
        
        return {
            "total_samples": len(clean_patterns),
            "output_file": str(output_file)
        }
    
    def _save_dataset(
        self,
        clean_patterns: List[np.ndarray],
        noisy_patterns: List[np.ndarray],
        metadata_list: List[SampleMetadata],
        output_file: Path
    ):
        """保存数据集"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('clean', data=np.array(clean_patterns))
            f.create_dataset('noisy', data=np.array(noisy_patterns))
            
            meta_group = f.create_group('metadata')
            for i, meta in enumerate(metadata_list):
                meta_group.create_dataset(f'sample_{i}', data=json.dumps(asdict(meta)))
        
        logger.info(f"数据集已保存: {output_file}")


# =============================================================================
# 可视化
# =============================================================================

def visualize_samples(
    clean_patterns: List[np.ndarray],
    noisy_patterns: List[np.ndarray],
    metadata_list: List[SampleMetadata],
    output_dir: Path,
    n_samples: int = 5
):
    """可视化样本"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_samples = min(n_samples, len(clean_patterns))
    
    for i in range(n_samples):
        clean = clean_patterns[i]
        noisy = noisy_patterns[i]
        meta = metadata_list[i]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        ax = axes[0, 0]
        clean_log = np.log1p(clean)
        im = ax.imshow(clean_log, cmap='inferno')
        ax.set_title(f"干净衍射图案 (log)\n{meta.pdb_id} | 投影轴: {meta.projection_axis}")
        plt.colorbar(im, ax=ax, label='log(1+光子数)')
        
        ax = axes[0, 1]
        noisy_log = np.log1p(noisy)
        im = ax.imshow(noisy_log, cmap='inferno')
        ax.set_title(f"含噪衍射图案 (log)\n曝光: {meta.exposure_level:.0f} | 高斯σ: {meta.gaussian_noise_std:.2f}")
        plt.colorbar(im, ax=ax, label='log(1+光子数)')
        
        ax = axes[0, 2]
        diff = noisy - clean
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        ax.set_title(f"噪声分布\n范围: [{diff.min():.0f}, {diff.max():.0f}]")
        plt.colorbar(im, ax=ax, label='光子数差异')
        
        ax = axes[1, 0]
        radial_clean = compute_radial_profile(clean)
        radial_noisy = compute_radial_profile(noisy)
        
        ax.semilogy(radial_clean, label='干净', alpha=0.8, linewidth=1.5)
        ax.semilogy(radial_noisy, label='含噪', alpha=0.8, linewidth=1.5)
        ax.set_xlabel('径向距离 (像素)')
        ax.set_ylabel('强度 (光子数, 对数尺度)')
        ax.set_title('径向强度分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.1)
        
        ax = axes[1, 1]
        center_size = 50
        center_clean = clean[clean.shape[0]//2-center_size//2:clean.shape[0]//2+center_size//2,
                            clean.shape[1]//2-center_size//2:clean.shape[1]//2+center_size//2]
        im = ax.imshow(center_clean, cmap='inferno')
        ax.set_title(f"中心区域放大 (50x50)\n最大强度: {clean.max():.0f}")
        plt.colorbar(im, ax=ax, label='光子数')
        
        ax = axes[1, 2]
        stats_text = (
            f"样本统计信息\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"PDB ID: {meta.pdb_id}\n"
            f"投影轴: {meta.projection_axis}\n"
            f"曝光水平: {meta.exposure_level:.0f}\n"
            f"高斯噪声σ: {meta.gaussian_noise_std:.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"干净图像:\n"
            f"  最大值: {clean.max():.0f}\n"
            f"  最小值: {clean.min():.0f}\n"
            f"  平均值: {clean.mean():.1f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"含噪图像:\n"
            f"  最大值: {noisy.max():.0f}\n"
            f"  最小值: {noisy.min():.0f}\n"
            f"  平均值: {noisy.mean():.1f}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"水化层: {'是' if meta.has_hydration_layer else '否'}"
        )
        if meta.has_hydration_layer:
            stats_text += f"\n  厚度: {meta.hydration_thickness:.1f}Å\n  密度: {meta.hydration_density:.2f}"
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title('统计信息')
        
        fig.suptitle(f"衍射图案模拟 - 样本 {i:04d}", fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = output_dir / f"sample_{i:04d}_{meta.pdb_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化保存: {save_path}")


def compute_radial_profile(image: np.ndarray) -> np.ndarray:
    """计算径向强度分布"""
    h, w = image.shape
    cy, cx = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    
    radial_sum = np.bincount(r.ravel(), image.ravel())
    radial_count = np.bincount(r.ravel())
    
    radial_profile = radial_sum / (radial_count + 1e-10)
    
    return radial_profile


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("=" * 60)
    print("衍射图案生成")
    print("=" * 60)
    
    print(f"\n配置参数:")
    print(f"  结构配置文件: {STRUCTURE_CONFIG_FILE}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  样本对数量: {TOTAL_SAMPLE_PAIRS}")
    print(f"  随机种子: {RANDOM_SEED}")
    print(f"  随机投影: {ENABLE_RANDOM_PROJECTION}")
    print(f"  随机曝光: {ENABLE_RANDOM_EXPOSURE}")
    print(f"  随机高斯噪声: {ENABLE_RANDOM_GAUSSIAN}")
    print(f"  默认图像尺寸: {DEFAULT_IMAGE_SIZE}")
    
    if not STRUCTURE_CONFIG_FILE.exists():
        logger.error(f"结构配置文件不存在: {STRUCTURE_CONFIG_FILE}")
        logger.error("请先运行 select_pdb_structures.py 生成配置文件")
        return
    
    pdb_dir = Path(STRUCTURE_CONFIG_FILE).parent / "pdb_raw"
    
    pipeline = DiffractionDataPipeline(
        config_file=STRUCTURE_CONFIG_FILE,
        output_dir=OUTPUT_DIR,
        random_seed=RANDOM_SEED
    )
    
    result = pipeline.generate_dataset(
        total_pairs=TOTAL_SAMPLE_PAIRS,
        pdb_dir=pdb_dir,
        output_file=OUTPUT_FILE
    )
    
    if VISUALIZATION_ENABLED:
        logger.info("\n生成可视化...")
        
        with h5py.File(OUTPUT_FILE, 'r') as f:
            clean_patterns = f['clean'][:VISUALIZATION_SAMPLES]
            noisy_patterns = f['noisy'][:VISUALIZATION_SAMPLES]
            
            metadata_list = []
            meta_group = f['metadata']
            for i in range(VISUALIZATION_SAMPLES):
                meta_str = meta_group[f'sample_{i}'][()]
                if isinstance(meta_str, bytes):
                    meta_str = meta_str.decode('utf-8')
                meta_dict = json.loads(meta_str)
                metadata_list.append(SampleMetadata(**meta_dict))
        
        visualize_samples(
            clean_patterns, noisy_patterns, metadata_list,
            VISUALIZATION_DIR, VISUALIZATION_SAMPLES
        )
    
    print("\n" + "=" * 60)
    print("完成!")
    print(f"总样本数: {result['total_samples']}")
    print(f"输出文件: {result['output_file']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
