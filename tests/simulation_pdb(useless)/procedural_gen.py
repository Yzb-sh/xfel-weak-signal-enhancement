"""
程序化生物结构生成器

功能：
1. 生成多样化的生物结构电子密度图
2. 包括：细胞、病毒、细胞器等
3. 支持随机参数化，确保结构多样性
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter, zoom


@dataclass
class StructureMetadata:
    """结构元数据"""
    structure_type: str
    size_nm: float
    grid_size: int
    parameters: Dict[str, Any]


class BaseStructureGenerator:
    """结构生成器基类"""
    
    def __init__(self, grid_size: int = 512, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        
        self.center = grid_size // 2
        x = np.linspace(-self.center, self.center, grid_size)
        y = np.linspace(-self.center, self.center, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
    
    def _create_base_grid(self) -> np.ndarray:
        """创建空白网格"""
        return np.zeros((self.grid_size, self.grid_size), dtype=np.float64)
    
    def _apply_gaussian_blur(self, density: np.ndarray, sigma_factor: float = 0.02) -> np.ndarray:
        """应用高斯模糊"""
        sigma = max(1.0, self.grid_size * sigma_factor * 0.5)
        return gaussian_filter(density, sigma=sigma)
    
    def _normalize(self, density: np.ndarray) -> np.ndarray:
        """归一化到[0, 1]"""
        if density.max() > 0:
            density = density / density.max()
        return density


class VirusGenerator(BaseStructureGenerator):
    """
    病毒结构生成器
    
    病毒特征：
    - 衣壳：球形、二十面体、螺旋形
    - 包膜：脂质双层膜（部分病毒）
    - 表面蛋白：刺突（如冠状病毒）
    - 内部基因组：核酸密度
    """
    
    VIRUS_TYPES = ['spherical', 'icosahedral', 'enveloped', 'helical']
    
    def generate(
        self,
        size_nm: float = 100.0,
        virus_type: Optional[str] = None,
        grid_occupancy: float = 0.7
    ) -> Tuple[np.ndarray, StructureMetadata]:
        """
        生成病毒结构
        
        参数:
            size_nm: 病毒尺寸（nm）
            virus_type: 病毒类型
            grid_occupancy: 网格占用比例
        """
        if virus_type is None:
            virus_type = self.rng.choice(self.VIRUS_TYPES)
        
        density = self._create_base_grid()
        radius = self.grid_size * grid_occupancy / 2
        
        if virus_type == 'spherical':
            density = self._generate_spherical_virus(radius)
        elif virus_type == 'icosahedral':
            density = self._generate_icosahedral_virus(radius)
        elif virus_type == 'enveloped':
            density = self._generate_enveloped_virus(radius)
        elif virus_type == 'helical':
            density = self._generate_helical_virus(radius)
        else:
            density = self._generate_spherical_virus(radius)
        
        density = self._apply_gaussian_blur(density)
        density = self._normalize(density)
        
        metadata = StructureMetadata(
            structure_type=f'virus_{virus_type}',
            size_nm=size_nm,
            grid_size=self.grid_size,
            parameters={'virus_type': virus_type, 'radius_pixels': radius}
        )
        
        return density, metadata
    
    def _generate_spherical_virus(self, radius: float) -> np.ndarray:
        """生成球形病毒"""
        density = self._create_base_grid()
        
        capsid_mask = self.R <= radius
        density[capsid_mask] = 1.0
        
        n_spikes = self.rng.integers(0, 20)
        for _ in range(n_spikes):
            angle = self.rng.uniform(0, 2 * np.pi)
            spike_r = radius * self.rng.uniform(0.95, 1.15)
            spike_x = spike_r * np.cos(angle)
            spike_y = spike_r * np.sin(angle)
            spike_size = self.rng.uniform(2, 6)
            
            dist = np.sqrt((self.X - spike_x)**2 + (self.Y - spike_y)**2)
            spike = np.exp(-dist**2 / (2 * spike_size**2))
            density += spike * 0.4
        
        inner_radius = radius * self.rng.uniform(0.3, 0.6)
        inner_mask = self.R <= inner_radius
        density[inner_mask] *= self.rng.uniform(0.6, 1.2)
        
        return density
    
    def _generate_icosahedral_virus(self, radius: float) -> np.ndarray:
        """生成二十面体病毒"""
        density = self._create_base_grid()
        
        n_vertices = 12
        for i in range(n_vertices):
            angle = 2 * np.pi * i / n_vertices + self.rng.uniform(-0.1, 0.1)
            r = radius * self.rng.uniform(0.85, 1.0)
            vx = r * np.cos(angle)
            vy = r * np.sin(angle)
            
            dist = np.sqrt((self.X - vx)**2 + (self.Y - vy)**2)
            vertex = np.exp(-dist**2 / (2 * radius * 0.15)**2)
            density += vertex * 0.5
        
        capsid_mask = self.R <= radius * 0.9
        density[capsid_mask] += 0.5
        
        return density
    
    def _generate_enveloped_virus(self, radius: float) -> np.ndarray:
        """生成包膜病毒"""
        density = self._create_base_grid()
        
        envelope_mask = self.R <= radius
        density[envelope_mask] = 0.3
        
        membrane_width = radius * 0.05
        membrane = np.exp(-(self.R - radius * 0.95)**2 / (2 * membrane_width**2))
        density += membrane * 0.5
        
        capsid_radius = radius * self.rng.uniform(0.5, 0.7)
        capsid_mask = self.R <= capsid_radius
        density[capsid_mask] += 0.5
        
        n_spikes = self.rng.integers(10, 40)
        for _ in range(n_spikes):
            angle = self.rng.uniform(0, 2 * np.pi)
            spike_base_r = radius * 0.95
            spike_length = self.rng.uniform(radius * 0.1, radius * 0.25)
            spike_x = spike_base_r * np.cos(angle)
            spike_y = spike_base_r * np.sin(angle)
            spike_size = self.rng.uniform(2, 5)
            
            dist = np.sqrt((self.X - spike_x)**2 + (self.Y - spike_y)**2)
            spike = np.exp(-dist**2 / (2 * spike_size**2))
            density += spike * 0.6
        
        return density
    
    def _generate_helical_virus(self, radius: float) -> np.ndarray:
        """生成螺旋病毒"""
        density = self._create_base_grid()
        
        aspect = self.rng.uniform(2.0, 4.0)
        angle = self.rng.uniform(0, np.pi)
        
        X_rot = self.X * np.cos(angle) + self.Y * np.sin(angle)
        Y_rot = -self.X * np.sin(angle) + self.Y * np.cos(angle)
        
        R_ellipse = np.sqrt((X_rot / aspect)**2 + Y_rot**2)
        mask = R_ellipse <= radius
        density[mask] = 1.0
        
        n_helices = self.rng.integers(2, 6)
        for i in range(n_helices):
            helix_r = radius * (0.3 + 0.15 * i)
            helix_width = radius * 0.08
            ring = np.exp(-(R_ellipse - helix_r)**2 / (2 * helix_width**2))
            density += ring * 0.3
        
        return density


class CellGenerator(BaseStructureGenerator):
    """
    细胞结构生成器
    
    细胞特征：
    - 细胞膜
    - 细胞核
    - 细胞器：线粒体、内质网、高尔基体等
    - 细胞质
    """
    
    CELL_TYPES = ['simple', 'with_organelles', 'with_membrane', 'irregular']
    
    def generate(
        self,
        size_nm: float = 1000.0,
        cell_type: Optional[str] = None,
        grid_occupancy: float = 0.7
    ) -> Tuple[np.ndarray, StructureMetadata]:
        """生成细胞结构"""
        if cell_type is None:
            cell_type = self.rng.choice(self.CELL_TYPES)
        
        density = self._create_base_grid()
        radius = self.grid_size * grid_occupancy / 2
        
        if cell_type == 'simple':
            density = self._generate_simple_cell(radius)
        elif cell_type == 'with_organelles':
            density = self._generate_cell_with_organelles(radius)
        elif cell_type == 'with_membrane':
            density = self._generate_cell_with_membrane(radius)
        elif cell_type == 'irregular':
            density = self._generate_irregular_cell(radius)
        else:
            density = self._generate_simple_cell(radius)
        
        density = self._apply_gaussian_blur(density)
        density = self._normalize(density)
        
        metadata = StructureMetadata(
            structure_type=f'cell_{cell_type}',
            size_nm=size_nm,
            grid_size=self.grid_size,
            parameters={'cell_type': cell_type, 'radius_pixels': radius}
        )
        
        return density, metadata
    
    def _generate_simple_cell(self, radius: float) -> np.ndarray:
        """生成简单细胞"""
        density = self._create_base_grid()
        
        cell_mask = self.R <= radius
        density[cell_mask] = 0.5
        
        nucleus_radius = radius * self.rng.uniform(0.2, 0.35)
        nucleus_mask = self.R <= nucleus_radius
        density[nucleus_mask] = 1.0
        
        return density
    
    def _generate_cell_with_organelles(self, radius: float) -> np.ndarray:
        """生成带细胞器的细胞"""
        density = self._generate_simple_cell(radius)
        
        n_organelles = self.rng.integers(3, 10)
        for _ in range(n_organelles):
            org_angle = self.rng.uniform(0, 2 * np.pi)
            org_dist = self.rng.uniform(radius * 0.3, radius * 0.8)
            org_x = org_dist * np.cos(org_angle)
            org_y = org_dist * np.sin(org_angle)
            org_r = self.rng.uniform(radius * 0.05, radius * 0.15)
            
            dist = np.sqrt((self.X - org_x)**2 + (self.Y - org_y)**2)
            organelle = np.exp(-dist**2 / (2 * org_r**2))
            density += organelle * self.rng.uniform(0.3, 0.7)
        
        return density
    
    def _generate_cell_with_membrane(self, radius: float) -> np.ndarray:
        """生成带明显细胞膜的细胞"""
        density = self._create_base_grid()
        
        membrane_width = radius * 0.03
        membrane = np.exp(-(self.R - radius)**2 / (2 * membrane_width**2))
        density += membrane * 0.8
        
        cytoplasm_mask = self.R <= radius * 0.95
        density[cytoplasm_mask] += 0.3
        
        nucleus_radius = radius * self.rng.uniform(0.2, 0.3)
        nucleus_mask = self.R <= nucleus_radius
        density[nucleus_mask] += 0.5
        
        return density
    
    def _generate_irregular_cell(self, radius: float) -> np.ndarray:
        """生成不规则形状细胞"""
        density = self._create_base_grid()
        
        n_blobs = self.rng.integers(3, 7)
        for i in range(n_blobs):
            blob_angle = 2 * np.pi * i / n_blobs + self.rng.uniform(-0.3, 0.3)
            blob_dist = radius * self.rng.uniform(0.3, 0.7)
            blob_x = blob_dist * np.cos(blob_angle)
            blob_y = blob_dist * np.sin(blob_angle)
            blob_r = radius * self.rng.uniform(0.3, 0.6)
            
            dist = np.sqrt((self.X - blob_x)**2 + (self.Y - blob_y)**2)
            blob = np.exp(-dist**2 / (2 * blob_r**2))
            density += blob
        
        return density


class OrganelleGenerator(BaseStructureGenerator):
    """
    细胞器结构生成器
    
    包括：
    - 线粒体：双层膜 + 嵴
    - 细胞核：核膜 + 染色质
    - 核糖体：大小亚基
    - 囊泡：单层膜
    """
    
    ORGANELLE_TYPES = ['mitochondria', 'nucleus', 'ribosome', 'vesicle', 'er']
    
    def generate(
        self,
        size_nm: float = 200.0,
        organelle_type: Optional[str] = None,
        grid_occupancy: float = 0.7
    ) -> Tuple[np.ndarray, StructureMetadata]:
        """生成细胞器结构"""
        if organelle_type is None:
            organelle_type = self.rng.choice(self.ORGANELLE_TYPES)
        
        density = self._create_base_grid()
        radius = self.grid_size * grid_occupancy / 2
        
        if organelle_type == 'mitochondria':
            density = self._generate_mitochondria(radius)
        elif organelle_type == 'nucleus':
            density = self._generate_nucleus(radius)
        elif organelle_type == 'ribosome':
            density = self._generate_ribosome(radius)
        elif organelle_type == 'vesicle':
            density = self._generate_vesicle(radius)
        elif organelle_type == 'er':
            density = self._generate_endoplasmic_reticulum(radius)
        else:
            density = self._generate_mitochondria(radius)
        
        density = self._apply_gaussian_blur(density)
        density = self._normalize(density)
        
        metadata = StructureMetadata(
            structure_type=f'organelle_{organelle_type}',
            size_nm=size_nm,
            grid_size=self.grid_size,
            parameters={'organelle_type': organelle_type, 'radius_pixels': radius}
        )
        
        return density, metadata
    
    def _generate_mitochondria(self, radius: float) -> np.ndarray:
        """生成线粒体"""
        density = self._create_base_grid()
        
        aspect = self.rng.uniform(1.5, 3.0)
        angle = self.rng.uniform(0, np.pi)
        
        X_rot = self.X * np.cos(angle) + self.Y * np.sin(angle)
        Y_rot = -self.X * np.sin(angle) + self.Y * np.cos(angle)
        
        R_ellipse = np.sqrt((X_rot / aspect)**2 + Y_rot**2)
        
        outer_membrane = np.exp(-(R_ellipse - radius)**2 / (2 * radius * 0.05)**2)
        density += outer_membrane * 0.6
        
        inner_radius = radius * 0.85
        inner_mask = R_ellipse <= inner_radius
        density[inner_mask] += 0.3
        
        n_cristae = self.rng.integers(3, 8)
        for _ in range(n_cristae):
            cristae_angle = self.rng.uniform(0, 2 * np.pi)
            cristae_r = self.rng.uniform(inner_radius * 0.3, inner_radius * 0.8)
            cristae_x = cristae_r * np.cos(cristae_angle)
            cristae_y = cristae_r * np.sin(cristae_angle)
            cristae_width = radius * 0.1
            
            dist = np.sqrt((self.X - cristae_x)**2 + (self.Y - cristae_y)**2)
            cristae = np.exp(-dist**2 / (2 * cristae_width**2))
            density += cristae * 0.4
        
        return density
    
    def _generate_nucleus(self, radius: float) -> np.ndarray:
        """生成细胞核"""
        density = self._create_base_grid()
        
        nuclear_mask = self.R <= radius
        density[nuclear_mask] = 0.4
        
        membrane_width = radius * 0.03
        membrane = np.exp(-(self.R - radius)**2 / (2 * membrane_width**2))
        density += membrane * 0.5
        
        nucleolus_radius = radius * self.rng.uniform(0.2, 0.35)
        nucleolus_mask = self.R <= nucleolus_radius
        density[nucleolus_mask] += 0.5
        
        n_chromatin = self.rng.integers(5, 15)
        for _ in range(n_chromatin):
            chr_angle = self.rng.uniform(0, 2 * np.pi)
            chr_dist = self.rng.uniform(nucleolus_radius * 1.2, radius * 0.9)
            chr_x = chr_dist * np.cos(chr_angle)
            chr_y = chr_dist * np.sin(chr_angle)
            chr_r = self.rng.uniform(radius * 0.05, radius * 0.15)
            
            dist = np.sqrt((self.X - chr_x)**2 + (self.Y - chr_y)**2)
            chromatin = np.exp(-dist**2 / (2 * chr_r**2))
            density += chromatin * 0.3
        
        return density
    
    def _generate_ribosome(self, radius: float) -> np.ndarray:
        """生成核糖体"""
        density = self._create_base_grid()
        
        large_subunit_radius = radius * 0.6
        large_mask = self.R <= large_subunit_radius
        density[large_mask] = 0.8
        
        small_offset_x = radius * 0.3
        small_offset_y = radius * 0.2
        small_radius = radius * 0.4
        
        dist_small = np.sqrt((self.X - small_offset_x)**2 + (self.Y - small_offset_y)**2)
        small_mask = dist_small <= small_radius
        density[small_mask] = 0.6
        
        return density
    
    def _generate_vesicle(self, radius: float) -> np.ndarray:
        """生成囊泡"""
        density = self._create_base_grid()
        
        membrane_width = radius * 0.05
        membrane = np.exp(-(self.R - radius)**2 / (2 * membrane_width**2))
        density += membrane * 0.7
        
        inner_mask = self.R <= radius * 0.9
        density[inner_mask] += 0.3
        
        return density
    
    def _generate_endoplasmic_reticulum(self, radius: float) -> np.ndarray:
        """生成内质网"""
        density = self._create_base_grid()
        
        n_sheets = self.rng.integers(3, 7)
        for i in range(n_sheets):
            sheet_angle = self.rng.uniform(0, 2 * np.pi)
            sheet_dist = radius * (0.3 + 0.15 * i)
            sheet_width = radius * 0.05
            
            sheet_x = sheet_dist * np.cos(sheet_angle)
            sheet_y = sheet_dist * np.sin(sheet_angle)
            
            dist = np.abs(self.X * np.cos(sheet_angle) + self.Y * np.sin(sheet_angle) - sheet_dist)
            sheet = np.exp(-dist**2 / (2 * sheet_width**2))
            density += sheet * 0.5
        
        return density


class CompositeStructureGenerator:
    """组合结构生成器 - 统一接口"""
    
    def __init__(self, grid_size: int = 512, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        
        self.virus_gen = VirusGenerator(grid_size, seed=seed)
        self.cell_gen = CellGenerator(grid_size, seed=seed + 1 if seed else None)
        self.organelle_gen = OrganelleGenerator(grid_size, seed=seed + 2 if seed else None)
    
    def generate_random(
        self,
        size_nm: float = 500.0,
        structure_category: Optional[str] = None
    ) -> Tuple[np.ndarray, StructureMetadata]:
        """
        随机生成结构
        
        参数:
            size_nm: 结构尺寸
            structure_category: 结构类别 ('virus', 'cell', 'organelle')
        """
        if structure_category is None:
            structure_category = self.rng.choice(['virus', 'cell', 'organelle'])
        
        if structure_category == 'virus':
            return self.virus_gen.generate(size_nm)
        elif structure_category == 'cell':
            return self.cell_gen.generate(size_nm)
        elif structure_category == 'organelle':
            return self.organelle_gen.generate(size_nm)
        else:
            return self.virus_gen.generate(size_nm)


if __name__ == "__main__":
    print("测试程序化结构生成器...")
    
    gen = CompositeStructureGenerator(grid_size=256, seed=42)
    
    for category in ['virus', 'cell', 'organelle']:
        density, meta = gen.generate_random(size_nm=500, structure_category=category)
        print(f"{category}: shape={density.shape}, max={density.max():.3f}, type={meta.structure_type}")
    
    print("\n程序化结构生成器测试完成")
