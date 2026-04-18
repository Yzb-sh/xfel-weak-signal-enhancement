"""
EMDB数据加载模块

功能：
1. 从EMDB数据库下载电子密度图
2. 将电子密度图转换为适合衍射模拟的格式
3. 生成随机2D投影

EMDB (Electron Microscopy Data Bank) 存储 cryo-EM 3D结构数据
尺寸范围：nm - μm级别，适合您的实验数据
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import urllib.request
import urllib.error
import json
import gzip
import struct

try:
    import mrcfile
    HAS_MRCFILE = True
except ImportError:
    HAS_MRCFILE = False
    print("警告: mrcfile未安装。请运行: pip install mrcfile")


@dataclass
class EMDBEntry:
    """EMDB条目信息"""
    emdb_id: str
    title: str
    resolution: float
    size_nm: Tuple[float, float, float]
    voxel_size_nm: Tuple[float, float, float]
    contour_level: float
    filename: str


class EMDBDownloader:
    """EMDB数据下载器"""
    
    BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures"
    
    def __init__(self, download_dir: Path):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_map(self, emdb_id: str, force: bool = False) -> Optional[Path]:
        """
        下载EMDB map文件
        
        参数:
            emdb_id: EMDB ID，如 "EMD-1234"
            force: 是否强制重新下载
        
        返回:
            下载文件的路径，失败返回None
        """
        emdb_num = emdb_id.replace("EMD-", "").replace("emd-", "")
        filename = f"emd_{emdb_num}.map.gz"
        output_path = self.download_dir / filename.replace(".gz", "")
        
        if output_path.exists() and not force:
            return output_path
        
        url = f"{self.BASE_URL}/EMD-{emdb_num}/map/{filename}"
        
        try:
            print(f"正在下载: {url}")
            compressed_path = self.download_dir / filename
            urllib.request.urlretrieve(url, compressed_path)
            
            print(f"正在解压...")
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            compressed_path.unlink()
            print(f"下载完成: {output_path}")
            return output_path
            
        except urllib.error.URLError as e:
            print(f"下载失败: {e}")
            return None
        except Exception as e:
            print(f"处理失败: {e}")
            return None
    
    def get_entry_info(self, emdb_id: str) -> Optional[Dict[str, Any]]:
        """获取EMDB条目信息"""
        emdb_num = emdb_id.replace("EMD-", "").replace("emd-", "")
        url = f"{self.BASE_URL}/EMD-{emdb_num}/header/emd-{emdb_num}.xml"
        
        try:
            import xml.etree.ElementTree as ET
            
            with urllib.request.urlopen(url, timeout=30) as response:
                xml_content = response.read()
            
            root = ET.fromstring(xml_content)
            
            ns = {'emd': 'http://www.emdatabank.org/nsd/emd/emd-1.0'}
            
            title = root.find('.//emd:title', ns)
            resolution = root.find('.//emd:resolution', ns)
            
            return {
                'emdb_id': emdb_id,
                'title': title.text if title is not None else '',
                'resolution': float(resolution.text) if resolution is not None else 0.0,
            }
            
        except Exception as e:
            print(f"获取条目信息失败: {e}")
            return None


class EMDBLoader:
    """EMDB数据加载器"""
    
    def __init__(self, download_dir: Path, cache_dir: Optional[Path] = None):
        self.download_dir = Path(download_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.download_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = EMDBDownloader(self.download_dir)
    
    def load_map(
        self, 
        emdb_id: str, 
        target_size_nm: Optional[float] = None,
        force_download: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        加载EMDB map文件
        
        参数:
            emdb_id: EMDB ID
            target_size_nm: 目标尺寸（nm），如果指定则进行缩放
            force_download: 是否强制重新下载
        
        返回:
            density: 3D电子密度图
            metadata: 元数据字典
        """
        if not HAS_MRCFILE:
            raise ImportError("需要安装mrcfile库: pip install mrcfile")
        
        map_path = self.downloader.download_map(emdb_id, force=force_download)
        if map_path is None:
            raise FileNotFoundError(f"无法下载EMDB map: {emdb_id}")
        
        with mrcfile.open(map_path) as mrc:
            density = mrc.data.copy()
            voxel_size = (mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z)
        
        metadata = {
            'emdb_id': emdb_id,
            'original_shape': density.shape,
            'voxel_size_nm': voxel_size,
            'original_size_nm': (
                density.shape[0] * voxel_size[0],
                density.shape[1] * voxel_size[1],
                density.shape[2] * voxel_size[2]
            )
        }
        
        if target_size_nm is not None:
            density = self._resize_density(density, voxel_size, target_size_nm)
            metadata['resized'] = True
            metadata['target_size_nm'] = target_size_nm
        
        density = self._normalize_density(density)
        
        return density, metadata
    
    def _resize_density(
        self, 
        density: np.ndarray, 
        voxel_size: Tuple[float, float, float],
        target_size_nm: float
    ) -> np.ndarray:
        """调整密度图尺寸"""
        from scipy.ndimage import zoom
        
        current_size = max(density.shape[i] * voxel_size[i] for i in range(3))
        scale_factor = target_size_nm / current_size
        
        if abs(scale_factor - 1.0) < 0.01:
            return density
        
        zoomed = zoom(density, scale_factor, order=1)
        return zoomed
    
    def _normalize_density(self, density: np.ndarray) -> np.ndarray:
        """归一化密度图"""
        density = density - density.min()
        if density.max() > 0:
            density = density / density.max()
        return density
    
    def generate_2d_projection(
        self, 
        density_3d: np.ndarray, 
        rotation_angles: Optional[Tuple[float, float, float]] = None,
        projection_axis: int = 2
    ) -> np.ndarray:
        """
        从3D密度图生成2D投影
        
        参数:
            density_3d: 3D电子密度图
            rotation_angles: 欧拉角 (theta, phi, psi)，单位：度
            projection_axis: 投影轴 (0=X, 1=Y, 2=Z)
        
        返回:
            2D投影
        """
        from scipy.ndimage import rotate
        
        if rotation_angles is not None:
            theta, phi, psi = rotation_angles
            
            density_3d = rotate(density_3d, theta, axes=(1, 2), reshape=False, order=1)
            density_3d = rotate(density_3d, phi, axes=(0, 2), reshape=False, order=1)
            density_3d = rotate(density_3d, psi, axes=(0, 1), reshape=False, order=1)
        
        projection = np.sum(density_3d, axis=projection_axis)
        
        return projection
    
    def generate_random_projection(
        self, 
        density_3d: np.ndarray,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """
        生成随机2D投影
        
        返回:
            projection: 2D投影
            angles: 使用的欧拉角
        """
        if rng is None:
            rng = np.random.default_rng()
        
        theta = rng.uniform(0, 360)
        phi = rng.uniform(0, 360)
        psi = rng.uniform(0, 360)
        
        projection = self.generate_2d_projection(density_3d, (theta, phi, psi))
        
        return projection, (theta, phi, psi)


class EMDBStructureGenerator:
    """EMDB结构生成器 - 用于衍射模拟"""
    
    SAMPLE_EMDB_IDS = [
        "EMD-1234",
        "EMD-3000",
        "EMD-5001",
    ]
    
    def __init__(
        self, 
        download_dir: Path,
        cache_dir: Optional[Path] = None,
        target_size_range_nm: Tuple[float, float] = (50.0, 2000.0),
        seed: Optional[int] = None
    ):
        self.loader = EMDBLoader(download_dir, cache_dir)
        self.target_size_range = target_size_range_nm
        self.rng = np.random.default_rng(seed)
        
        self._cache: Dict[str, Tuple[np.ndarray, Dict]] = {}
    
    def load_and_cache(self, emdb_id: str) -> Tuple[np.ndarray, Dict]:
        """加载并缓存EMDB数据"""
        if emdb_id not in self._cache:
            target_size = self.rng.uniform(*self.target_size_range)
            self._cache[emdb_id] = self.loader.load_map(emdb_id, target_size_nm=target_size)
        return self._cache[emdb_id]
    
    def generate_structure(
        self, 
        emdb_id: Optional[str] = None,
        target_size_nm: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        生成用于衍射模拟的结构
        
        返回:
            density_2d: 2D电子密度投影
            metadata: 元数据
        """
        if emdb_id is None:
            emdb_id = self.rng.choice(self.SAMPLE_EMDB_IDS)
        
        if target_size_nm is None:
            target_size_nm = self.rng.uniform(*self.target_size_range)
        
        density_3d, metadata_3d = self.loader.load_map(emdb_id, target_size_nm=target_size_nm)
        
        projection, angles = self.loader.generate_random_projection(density_3d, self.rng)
        
        metadata = {
            **metadata_3d,
            'projection_angles': angles,
            'projection_shape': projection.shape
        }
        
        return projection, metadata


def create_synthetic_emdb_like_structure(
    size_nm: float,
    structure_type: str = 'virus',
    grid_size: int = 256,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    创建合成的EMDB风格结构（用于测试，无需下载）
    
    参数:
        size_nm: 结构尺寸（nm）
        structure_type: 结构类型
        grid_size: 网格大小
        seed: 随机种子
    
    返回:
        density_2d: 2D电子密度投影
        metadata: 元数据
    """
    rng = np.random.default_rng(seed)
    
    density = np.zeros((grid_size, grid_size), dtype=np.float64)
    center = grid_size // 2
    
    x = np.linspace(-center, center, grid_size)
    y = np.linspace(-center, center, grid_size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    grid_occupancy = 0.7
    radius = grid_size * grid_occupancy / 2
    
    if structure_type == 'virus':
        mask = R <= radius
        density[mask] = 1.0
        
        n_spikes = rng.integers(10, 30)
        for _ in range(n_spikes):
            angle = rng.uniform(0, 2 * np.pi)
            spike_r = rng.uniform(radius * 0.8, radius * 1.1)
            spike_x = spike_r * np.cos(angle)
            spike_y = spike_r * np.sin(angle)
            spike_size = rng.uniform(3, 8)
            
            dist = np.sqrt((X - spike_x)**2 + (Y - spike_y)**2)
            spike = np.exp(-dist**2 / (2 * spike_size**2))
            density += spike * 0.5
        
        n_internal = rng.integers(1, 3)
        for _ in range(n_internal):
            inner_r = rng.uniform(radius * 0.3, radius * 0.6)
            inner_mask = R <= inner_r
            density[inner_mask] *= rng.uniform(0.5, 1.5)
    
    elif structure_type == 'organelle':
        aspect = rng.uniform(1.0, 2.5)
        angle = rng.uniform(0, np.pi)
        
        X_rot = X * np.cos(angle) + Y * np.sin(angle)
        Y_rot = -X * np.sin(angle) + Y * np.cos(angle)
        
        R_ellipse = np.sqrt((X_rot / aspect)**2 + Y_rot**2)
        mask = R_ellipse <= radius
        density[mask] = 1.0
        
        n_membranes = rng.integers(1, 4)
        for i in range(n_membranes):
            mem_r = radius * (0.3 + 0.2 * i)
            mem_width = rng.uniform(2, 5)
            ring = np.exp(-(R_ellipse - mem_r)**2 / (2 * mem_width**2))
            density += ring * 0.3
    
    elif structure_type == 'cell':
        mask = R <= radius
        density[mask] = 1.0
        
        n_organelles = rng.integers(3, 8)
        for _ in range(n_organelles):
            org_x = rng.uniform(-radius * 0.6, radius * 0.6)
            org_y = rng.uniform(-radius * 0.6, radius * 0.6)
            org_r = rng.uniform(radius * 0.1, radius * 0.25)
            
            dist = np.sqrt((X - org_x)**2 + (Y - org_y)**2)
            org_mask = dist <= org_r
            density[org_mask] = rng.uniform(0.5, 1.5)
    
    else:
        mask = R <= radius
        density[mask] = 1.0
    
    from scipy.ndimage import gaussian_filter
    sigma = max(1.0, radius * 0.02)
    density = gaussian_filter(density, sigma=sigma)
    
    if density.max() > 0:
        density = density / density.max()
    
    metadata = {
        'structure_type': structure_type,
        'size_nm': size_nm,
        'grid_size': grid_size,
        'synthetic': True
    }
    
    return density, metadata


if __name__ == "__main__":
    print("测试EMDB模块...")
    
    print("\n创建合成结构测试:")
    for stype in ['virus', 'organelle', 'cell']:
        density, meta = create_synthetic_emdb_like_structure(
            size_nm=500,
            structure_type=stype,
            seed=42
        )
        print(f"  {stype}: shape={density.shape}, max={density.max():.3f}")
    
    print("\nEMDB模块测试完成")
