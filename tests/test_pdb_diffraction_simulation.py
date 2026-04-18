"""
PDB生物分子衍射数据仿真测试模块

功能：
1. 从PDB结构生成原子级衍射图案
2. 添加水化层建模
3. 多尺度噪声注入（泊松+高斯+结构化噪声）
4. 验证仿真数据的物理合理性

参考文献：
- Cherukara et al., "Denoising low-intensity diffraction signals using k-space deep learning", PRR 2021
- Skopi/SingFEL: https://github.com/duaneloh/singfel
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from scipy.fft import fft2, fftshift, ifft2
from scipy.ndimage import gaussian_filter

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
    print("警告: BioPython未安装，部分功能不可用。请运行: pip install biopython")

from src.config.config_loader import SimulationConfig


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


@dataclass
class AtomInfo:
    atom_coords: np.ndarray
    atom_weights: np.ndarray
    atom_elements: List[str]
    atom_radii: np.ndarray


@dataclass
class HydrationShellConfig:
    enabled: bool = True
    thickness: float = 3.0
    density: float = 0.334
    smooth_sigma: float = 1.0


@dataclass
class NoiseConfig:
    exposure_level: float = 100.0
    readout_noise: float = 2.0
    dark_current: float = 0.1
    quantum_efficiency: float = 0.9
    background_level: float = 0.0
    background_scatter_radius: float = 0.0


class EnhancedXRaySimulator:
    """
    增强版X射线衍射模拟器
    
    特性：
    1. 原子级衍射模拟（使用原子散射因子）
    2. 水化层建模
    3. 多尺度噪声注入
    4. 物理约束验证
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        if HAS_BIOPYTHON:
            self.parser = PDBParser(QUIET=True)
        else:
            self.parser = None
    
    def parse_pdb_atoms(self, pdb_path: Path) -> AtomInfo:
        if not HAS_BIOPYTHON:
            raise ImportError("BioPython is required for PDB parsing")
        
        structure = self.parser.get_structure("protein", str(pdb_path))
        
        coords = []
        weights = []
        elements = []
        radii = []
        
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
                        radius = ATOMIC_RADII.get(element, 1.7)
                        
                        coords.append(coord)
                        weights.append(weight)
                        elements.append(element)
                        radii.append(radius)
        
        if len(coords) == 0:
            raise ValueError(f"No atoms found in PDB file: {pdb_path}")
        
        return AtomInfo(
            atom_coords=np.array(coords),
            atom_weights=np.array(weights),
            atom_elements=elements,
            atom_radii=np.array(radii)
        )
    
    def calculate_atomic_form_factor(self, q_magnitude: np.ndarray, element: str) -> np.ndarray:
        """
        计算原子散射因子 f(q)
        
        使用Cromer-Mann参数化：
        f(q) = sum_i(a_i * exp(-b_i * (q/4π)^2)) + c
        
        Args:
            q_magnitude: 散射矢量大小 (Å^-1)
            element: 元素符号
        
        Returns:
            原子散射因子
        """
        cromer_mann_params = {
            'H':  {'a': [0.493, 0.323, 0.140, 0.041], 'b': [10.511, 26.707, 3.337, 0.0], 'c': 0.003},
            'C':  {'a': [2.310, 1.020, 1.589, 0.865], 'b': [20.844, 10.208, 0.569, 51.651], 'c': 0.216},
            'N':  {'a': [12.213, 3.132, 2.013, 1.166], 'b': [0.006, 9.893, 28.997, 0.583], 'c': -11.529},
            'O':  {'a': [3.049, 2.287, 1.546, 0.867], 'b': [13.277, 5.701, 0.324, 32.909], 'c': 0.251},
            'S':  {'a': [6.905, 4.203, 1.958, 0.576], 'b': [1.468, 22.215, 0.254, 56.172], 'c': 0.259},
            'P':  {'a': [6.435, 4.179, 1.780, 1.491], 'b': [1.907, 27.157, 0.526, 68.165], 'c': 1.115},
            'FE': {'a': [11.768, 7.240, 3.483, 2.112], 'b': [4.761, 0.274, 18.635, 54.979], 'c': 1.397},
            'ZN': {'a': [11.973, 7.331, 4.353, 2.144], 'b': [4.568, 0.243, 18.123, 50.839], 'c': 1.198},
            'MG': {'a': [5.420, 2.174, 1.518, 1.444], 'b': [2.828, 79.271, 0.448, 39.799], 'c': 0.444},
            'CA': {'a': [8.627, 4.916, 2.593, 1.490], 'b': [10.442, 0.359, 28.654, 58.920], 'c': 0.374},
        }
        
        if element not in cromer_mann_params:
            params = cromer_mann_params['C']
        else:
            params = cromer_mann_params[element]
        
        s_squared = (q_magnitude / (4 * np.pi)) ** 2
        
        f = params['c']
        for a, b in zip(params['a'], params['b']):
            f = f + a * np.exp(-b * s_squared)
        
        return f
    
    def create_density_from_atoms(
        self,
        atom_info: AtomInfo,
        grid_size: Optional[int] = None,
        resolution: Optional[float] = None,
        projection_axis: int = 2,
        use_form_factor: bool = False
    ) -> np.ndarray:
        """
        从原子坐标创建电子密度图
        
        Args:
            atom_info: 原子信息
            grid_size: 网格大小
            resolution: 分辨率 (Å/pixel)
            projection_axis: 投影轴 (0=x, 1=y, 2=z)
            use_form_factor: 是否使用原子散射因子
        
        Returns:
            2D电子密度图
        """
        grid_size = grid_size or self.config.grid_size
        resolution = resolution or self.config.resolution
        
        coords = atom_info.atom_coords.copy()
        weights = atom_info.atom_weights.copy()
        
        coords = coords - coords.mean(axis=0)
        
        axes = [0, 1, 2]
        axes.remove(projection_axis)
        coords_2d = coords[:, axes]
        
        max_extent = np.abs(coords_2d).max()
        scale = (grid_size / 2 - 2) / max_extent if max_extent > 0 else 1.0
        coords_scaled = coords_2d * scale + grid_size / 2
        
        density = np.zeros((grid_size, grid_size), dtype=np.float64)
        
        for i, ((x, y), w) in enumerate(zip(coords_scaled, weights)):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < grid_size and 0 <= iy < grid_size:
                element = atom_info.atom_elements[i]
                radius = atom_info.atom_radii[i]
                sigma = max(0.5, radius / resolution)
                
                local_size = int(4 * sigma) + 1
                local_size = min(local_size, 15)
                
                y_local = np.arange(-local_size//2, local_size//2 + 1)
                x_local = np.arange(-local_size//2, local_size//2 + 1)
                X_local, Y_local = np.meshgrid(x_local, y_local)
                
                dist_sq = X_local**2 + Y_local**2
                gaussian = np.exp(-dist_sq / (2 * sigma**2))
                gaussian = gaussian / gaussian.sum()
                
                y_start = max(0, iy - local_size//2)
                y_end = min(grid_size, iy + local_size//2 + 1)
                x_start = max(0, ix - local_size//2)
                x_end = min(grid_size, ix + local_size//2 + 1)
                
                g_y_start = local_size//2 - (iy - y_start)
                g_y_end = local_size//2 + (y_end - iy)
                g_x_start = local_size//2 - (ix - x_start)
                g_x_end = local_size//2 + (x_end - ix)
                
                density[y_start:y_end, x_start:x_end] += w * gaussian[g_y_start:g_y_end, g_x_start:g_x_end]
        
        if density.max() > 0:
            density = density / density.max()
        
        return density
    
    def add_hydration_shell(
        self,
        density: np.ndarray,
        atom_info: AtomInfo,
        config: Optional[HydrationShellConfig] = None,
        resolution: Optional[float] = None
    ) -> np.ndarray:
        """
        添加水化层
        
        模拟生物分子表面的水分子层，这是真实XFEL实验中的重要特征。
        
        Args:
            density: 原始电子密度图
            atom_info: 原子信息
            config: 水化层配置
            resolution: 分辨率
        
        Returns:
            包含水化层的电子密度图
        """
        config = config or HydrationShellConfig()
        resolution = resolution or self.config.resolution
        
        if not config.enabled:
            return density
        
        coords = atom_info.atom_coords
        coords_centered = coords - coords.mean(axis=0)
        
        max_radius = np.sqrt((coords_centered ** 2).sum(axis=1)).max()
        
        grid_size = density.shape[0]
        x = np.linspace(-grid_size/2, grid_size/2, grid_size)
        y = np.linspace(-grid_size/2, grid_size/2, grid_size)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        
        inner_radius = max_radius * 1.05
        outer_radius = inner_radius + config.thickness / resolution
        
        hydration_mask = (R >= inner_radius) & (R <= outer_radius)
        
        hydration_density = np.zeros_like(density)
        hydration_density[hydration_mask] = config.density
        
        hydration_density = gaussian_filter(hydration_density, sigma=config.smooth_sigma)
        
        combined = density + hydration_density * 0.1
        
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def generate_diffraction_pattern(
        self,
        density: np.ndarray,
        oversampling_ratio: Optional[float] = None
    ) -> np.ndarray:
        """
        生成衍射图案
        
        使用FFT计算衍射强度 |F(q)|^2
        
        Args:
            density: 电子密度图
            oversampling_ratio: 过采样率
        
        Returns:
            衍射强度图案
        """
        oversampling_ratio = oversampling_ratio or self.config.oversampling_ratio
        
        original_size = density.shape[0]
        
        if oversampling_ratio > 1.0:
            padded_size = int(original_size * oversampling_ratio)
            padded = np.zeros((padded_size, padded_size), dtype=density.dtype)
            offset = (padded_size - original_size) // 2
            padded[offset:offset + original_size, offset:offset + original_size] = density
        else:
            padded = density.copy()
        
        f_map = fft2(padded)
        f_map = fftshift(f_map)
        
        intensity = np.abs(f_map) ** 2
        
        return intensity
    
    def apply_beam_stop(
        self,
        pattern: np.ndarray,
        radius: Optional[int] = None
    ) -> np.ndarray:
        """应用光束阻挡器"""
        radius = radius or self.config.beam_stop_radius
        
        result = pattern.copy()
        h, w = pattern.shape
        cy, cx = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        result[mask] = 0
        
        return result


class EnhancedNoiseModel:
    """
    增强版噪声模型
    
    特性：
    1. 泊松噪声（光子散粒噪声）
    2. 高斯噪声（探测器读出噪声）
    3. 结构化背景散射
    4. 支持多曝光水平
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def add_poisson_noise(
        self,
        intensity: np.ndarray,
        exposure_level: float = 100.0
    ) -> np.ndarray:
        """添加泊松噪声"""
        photon_counts = intensity * exposure_level
        photon_counts = np.maximum(photon_counts, 0)
        noisy = self.rng.poisson(photon_counts).astype(np.float64)
        return noisy
    
    def add_gaussian_noise(
        self,
        intensity: np.ndarray,
        std: float = 2.0
    ) -> np.ndarray:
        """添加高斯噪声"""
        noise = self.rng.normal(0, std, intensity.shape)
        return intensity + noise
    
    def add_background_scatter(
        self,
        intensity: np.ndarray,
        level: float = 0.1,
        radius: float = 0.3
    ) -> np.ndarray:
        """
        添加背景散射
        
        模拟载气/溶剂产生的结构化背景
        
        Args:
            intensity: 衍射图案
            level: 背景强度水平
            radius: 背景散射半径（相对于图像尺寸）
        
        Returns:
            添加背景后的图案
        """
        h, w = intensity.shape
        cy, cx = h // 2, w // 2
        
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        
        max_r = min(h, w) // 2
        r_normalized = r / max_r
        
        background = level * np.exp(-r_normalized ** 2 / (2 * radius ** 2))
        
        return intensity + background * intensity.max()
    
    def add_full_noise(
        self,
        intensity: np.ndarray,
        config: Optional[NoiseConfig] = None,
        beamstop_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        添加完整噪声模型
        
        正确的物理顺序：
        1. 背景散射
        2. 泊松噪声（光子计数）
        3. 高斯噪声（探测器读出）
        
        注意：beamstop区域不添加任何噪声，因为那里没有探测器
        
        Args:
            intensity: 衍射强度（已归一化）
            config: 噪声配置
            beamstop_mask: beamstop区域的布尔mask（True=beamstop区域）
        
        Returns:
            含噪衍射图案
        """
        config = config or NoiseConfig()
        
        noisy = intensity.copy()
        
        if config.background_level > 0:
            noisy = self.add_background_scatter(
                noisy,
                level=config.background_level,
                radius=config.background_scatter_radius
            )
        
        noisy = self.add_poisson_noise(
            noisy / (noisy.max() + 1e-10),
            exposure_level=config.exposure_level
        )
        
        noisy = self.add_gaussian_noise(noisy, std=config.readout_noise)
        
        noisy = np.maximum(noisy, 0)
        
        if beamstop_mask is not None:
            noisy[beamstop_mask] = 0
        
        return noisy


class DiffractionSimulationPipeline:
    """
    完整的衍射数据仿真流水线
    """
    
    def __init__(
        self,
        sim_config: Optional[SimulationConfig] = None,
        hydration_config: Optional[HydrationShellConfig] = None,
        noise_config: Optional[NoiseConfig] = None
    ):
        self.sim_config = sim_config or SimulationConfig()
        self.hydration_config = hydration_config or HydrationShellConfig()
        self.noise_config = noise_config or NoiseConfig()
        
        self.simulator = EnhancedXRaySimulator(self.sim_config)
        self.noise_model = EnhancedNoiseModel()
    
    def process_pdb(
        self,
        pdb_path: Path,
        add_hydration: bool = True,
        add_noise: bool = True,
        apply_beam_stop: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        处理单个PDB文件，生成完整的仿真数据
        
        正确的物理顺序：
        电子密度 → FFT → 衍射强度 → beamstop → 泊松噪声 → 高斯噪声
        
        注意：beamstop必须在噪声之前应用，因为：
        1. 泊松噪声与信号强度成正比，beamstop区域信号为0
        2. beamstop后面没有探测器，不会有任何计数
        
        Args:
            pdb_path: PDB文件路径
            add_hydration: 是否添加水化层
            add_noise: 是否添加噪声
            apply_beam_stop: 是否应用光束阻挡器
        
        Returns:
            包含所有中间结果的字典
        """
        atom_info = self.simulator.parse_pdb_atoms(pdb_path)
        
        density = self.simulator.create_density_from_atoms(
            atom_info,
            grid_size=self.sim_config.grid_size,
            resolution=self.sim_config.resolution
        )
        
        if add_hydration:
            density_with_hydration = self.simulator.add_hydration_shell(
                density, atom_info, self.hydration_config
            )
        else:
            density_with_hydration = density
        
        clean_pattern = self.simulator.generate_diffraction_pattern(
            density_with_hydration,
            oversampling_ratio=self.sim_config.oversampling_ratio
        )
        
        beamstop_mask = None
        if apply_beam_stop:
            h, w = clean_pattern.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            beamstop_mask = (x - cx) ** 2 + (y - cy) ** 2 <= self.sim_config.beam_stop_radius ** 2
            clean_pattern_masked = clean_pattern.copy()
            clean_pattern_masked[beamstop_mask] = 0
        else:
            clean_pattern_masked = clean_pattern
        
        clean_normalized = clean_pattern_masked / (clean_pattern_masked.max() + 1e-10)
        
        if add_noise:
            noisy_pattern = self.noise_model.add_full_noise(
                clean_normalized,
                self.noise_config,
                beamstop_mask=beamstop_mask
            )
        else:
            noisy_pattern = clean_normalized * self.noise_config.exposure_level
        
        return {
            'density': density,
            'density_with_hydration': density_with_hydration,
            'clean_pattern': clean_pattern,
            'clean_pattern_masked': clean_pattern_masked,
            'clean_normalized': clean_normalized,
            'noisy_pattern': noisy_pattern,
            'beamstop_mask': beamstop_mask,
            'atom_count': len(atom_info.atom_coords),
            'pdb_id': pdb_path.stem
        }
    
    def generate_dataset(
        self,
        pdb_files: List[Path],
        exposure_levels: Optional[List[float]] = None,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        生成完整数据集
        
        Args:
            pdb_files: PDB文件列表
            exposure_levels: 曝光水平列表（用于数据增强）
            save_path: 保存路径
        
        Returns:
            数据集字典
        """
        if exposure_levels is None:
            exposure_levels = [50, 100, 200, 500]
        
        clean_patterns = []
        noisy_patterns = []
        densities = []
        beamstop_masks = []
        metadata = []
        
        for i, pdb_file in enumerate(pdb_files):
            print(f"处理 [{i+1}/{len(pdb_files)}]: {pdb_file.stem}")
            
            for exposure in exposure_levels:
                self.noise_config.exposure_level = exposure
                
                result = self.process_pdb(pdb_file)
                
                clean_patterns.append(result['clean_normalized'])
                noisy_patterns.append(result['noisy_pattern'])
                densities.append(result['density'])
                beamstop_masks.append(result['beamstop_mask'] if result['beamstop_mask'] is not None else np.zeros_like(result['clean_normalized'], dtype=bool))
                metadata.append({
                    'pdb_id': result['pdb_id'],
                    'atom_count': result['atom_count'],
                    'exposure_level': exposure
                })
        
        dataset = {
            'clean': np.array(clean_patterns),
            'noisy': np.array(noisy_patterns),
            'densities': np.array(densities),
            'beamstop_masks': np.array(beamstop_masks),
            'metadata': metadata
        }
        
        if save_path:
            import h5py
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('clean', data=dataset['clean'])
                f.create_dataset('noisy', data=dataset['noisy'])
                f.create_dataset('densities', data=dataset['densities'])
                f.create_dataset('beamstop_masks', data=dataset['beamstop_masks'].astype(np.uint8))
                
                meta_group = f.create_group('metadata')
                for i, m in enumerate(metadata):
                    meta_group.create_dataset(f'sample_{i}', data=str(m))
            
            print(f"数据集已保存到: {save_path}")
        
        return dataset


def visualize_simulation_result(result: Dict[str, np.ndarray], save_path: Optional[Path] = None):
    """可视化仿真结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    im = ax.imshow(result['density'], cmap='viridis')
    ax.set_title(f"电子密度图\n{result['pdb_id']} ({result['atom_count']} atoms)")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.imshow(result['density_with_hydration'], cmap='viridis')
    ax.set_title("电子密度图（含水化层）")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    clean_log = np.log1p(result['clean_pattern'])
    im = ax.imshow(clean_log, cmap='inferno')
    ax.set_title("干净衍射图案 (log scale)")
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 0]
    clean_masked_log = np.log1p(result['clean_pattern_masked'])
    im = ax.imshow(clean_masked_log, cmap='inferno')
    ax.set_title("干净衍射图案（含beam stop）")
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 1]
    noisy_log = np.log1p(result['noisy_pattern'])
    im = ax.imshow(noisy_log, cmap='inferno')
    ax.set_title("含噪衍射图案 (log scale)")
    if result.get('beamstop_mask') is not None:
        from matplotlib.patches import Circle
        h, w = result['noisy_pattern'].shape
        circle = Circle((w//2, h//2), np.sqrt(result['beamstop_mask'].sum()/np.pi), 
                        fill=False, color='cyan', linewidth=2, linestyle='--')
        ax.add_patch(circle)
    plt.colorbar(im, ax=ax)
    
    ax = axes[1, 2]
    clean_norm = result['clean_normalized']
    noisy = result['noisy_pattern']
    
    radial_clean = compute_radial_profile(clean_norm)
    radial_noisy = compute_radial_profile(noisy / (noisy.max() + 1e-10))
    
    ax.semilogy(radial_clean, label='Clean', alpha=0.8)
    ax.semilogy(radial_noisy, label='Noisy', alpha=0.8)
    ax.set_xlabel('Radial distance (pixels)')
    ax.set_ylabel('Intensity (log scale)')
    ax.set_title('径向强度分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    
    plt.show()


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


def test_single_pdb():
    """测试单个PDB文件的仿真"""
    print("=" * 60)
    print("测试: 单个PDB文件衍射仿真")
    print("=" * 60)
    
    pdb_dir = project_root / "data" / "test_pdb" / "pdb_raw"
    
    pdb_files = list(pdb_dir.glob("*.pdb"))
    
    if not pdb_files:
        print(f"错误: 在 {pdb_dir} 中未找到PDB文件")
        print("请先运行数据下载脚本获取PDB文件")
        return None
    
    pdb_file = pdb_files[0]
    print(f"\n使用PDB文件: {pdb_file}")
    
    sim_config = SimulationConfig(
        grid_size=128,
        resolution=1.0,
        oversampling_ratio=2.0,
        beam_stop_radius=5
    )
    
    hydration_config = HydrationShellConfig(
        enabled=True,
        thickness=3.0,
        density=0.334,
        smooth_sigma=1.0
    )
    
    noise_config = NoiseConfig(
        exposure_level=100.0,
        readout_noise=2.0,
        background_level=0.05,
        background_scatter_radius=0.3
    )
    
    pipeline = DiffractionSimulationPipeline(
        sim_config=sim_config,
        hydration_config=hydration_config,
        noise_config=noise_config
    )
    
    result = pipeline.process_pdb(pdb_file)
    
    print(f"\n仿真结果:")
    print(f"  PDB ID: {result['pdb_id']}")
    print(f"  原子数: {result['atom_count']}")
    print(f"  电子密度图尺寸: {result['density'].shape}")
    print(f"  衍射图案尺寸: {result['clean_pattern'].shape}")
    print(f"  干净图案强度范围: [{result['clean_normalized'].min():.6f}, {result['clean_normalized'].max():.6f}]")
    print(f"  含噪图案强度范围: [{result['noisy_pattern'].min():.2f}, {result['noisy_pattern'].max():.2f}]")
    
    output_dir = project_root / "data" / "simulation_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize_simulation_result(
        result,
        save_path=output_dir / f"{result['pdb_id']}_simulation.png"
    )
    
    return result


def test_dataset_generation():
    """测试数据集生成"""
    print("\n" + "=" * 60)
    print("测试: 数据集生成")
    print("=" * 60)
    
    pdb_dir = project_root / "data" / "test_pdb" / "pdb_raw"
    pdb_files = list(pdb_dir.glob("*.pdb"))
    
    if not pdb_files:
        print(f"错误: 在 {pdb_dir} 中未找到PDB文件")
        return None
    
    pdb_files = pdb_files[:3]
    print(f"\n使用 {len(pdb_files)} 个PDB文件")
    
    sim_config = SimulationConfig(
        grid_size=128,
        resolution=1.0,
        oversampling_ratio=2.0
    )
    
    pipeline = DiffractionSimulationPipeline(sim_config=sim_config)
    
    output_path = project_root / "data" / "simulation_test" / "test_dataset.h5"
    
    dataset = pipeline.generate_dataset(
        pdb_files,
        exposure_levels=[50, 100, 200],
        save_path=output_path
    )
    
    print(f"\n数据集统计:")
    print(f"  样本数: {len(dataset['clean'])}")
    print(f"  数据形状: {dataset['clean'].shape}")
    
    return dataset


def validate_physical_properties(result: Dict[str, np.ndarray]):
    """验证仿真数据的物理合理性"""
    print("\n" + "=" * 60)
    print("物理合理性验证")
    print("=" * 60)
    
    clean = result['clean_normalized']
    noisy = result['noisy_pattern']
    
    if result.get('beamstop_mask') is not None:
        beamstop_mask = result['beamstop_mask']
        beamstop_signal_clean = clean[beamstop_mask].sum()
        beamstop_signal_noisy = noisy[beamstop_mask].sum()
        beamstop_pixels = beamstop_mask.sum()
        
        print(f"\nBeamstop区域验证:")
        print(f"  Beamstop像素数: {beamstop_pixels}")
        print(f"  Beamstop区域干净信号: {beamstop_signal_clean:.6f} (应为0)")
        print(f"  Beamstop区域含噪信号: {beamstop_signal_noisy:.6f} (应为0)")
        
        if beamstop_signal_noisy == 0:
            print(f"  ✓ Beamstop区域噪声处理正确!")
        else:
            print(f"  ✗ 警告: Beamstop区域存在非零值!")
    
    autocorr_clean = np.abs(fftshift(ifft2(np.sqrt(clean + 1e-10))))
    autocorr_noisy = np.abs(fftshift(ifft2(np.sqrt(noisy + 1e-10))))
    
    h, w = clean.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    support_radius = h // 4
    support_mask = r <= support_radius
    
    energy_inside = autocorr_clean[support_mask].sum()
    energy_total = autocorr_clean.sum()
    support_ratio = energy_inside / energy_total
    
    print(f"\n自相关函数支撑域验证:")
    print(f"  支撑域内能量占比: {support_ratio:.2%}")
    print(f"  预期: > 90% (紧支集约束)")
    
    radial_clean = compute_radial_profile(clean)
    radial_noisy = compute_radial_profile(noisy / (noisy.max() + 1e-10))
    
    correlation = np.corrcoef(radial_clean[:len(radial_noisy)], radial_noisy)[0, 1]
    print(f"\n径向分布相关性: {correlation:.4f}")
    
    snr = noisy.mean() / (noisy.std() + 1e-10)
    print(f"\n信噪比估计: {snr:.2f}")
    
    return {
        'support_ratio': support_ratio,
        'radial_correlation': correlation,
        'snr_estimate': snr
    }


if __name__ == "__main__":
    result = test_single_pdb()
    
    if result:
        validate_physical_properties(result)
    
    test_dataset_generation()
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
