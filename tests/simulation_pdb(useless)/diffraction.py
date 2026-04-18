"""
FFT衍射计算模块

功能：
1. 从电子密度图计算衍射图案
2. 物理参数化模型直接生成衍射图案
3. 数据增强（旋转、翻转等）- 在FFT之前执行
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.fft import fft2, fftshift
from scipy.ndimage import rotate, zoom


@dataclass
class DiffractionResult:
    """衍射计算结果"""
    intensity: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    metadata: Dict[str, Any]


class FFTDiffractionCalculator:
    """
    FFT衍射计算器
    
    物理原理：
    - 衍射图案 = |F(q)|²
    - F(q) = ∫ρ(r)exp(-iq·r)dr = FFT[ρ(r)]
    
    其中：
    - ρ(r) 是电子密度
    - F(q) 是结构因子
    - I(q) = |F(q)|² 是衍射强度
    """
    
    def __init__(
        self, 
        grid_size: int = 512,
        oversampling: float = 1.0
    ):
        self.grid_size = grid_size
        self.oversampling = oversampling
    
    def calculate(
        self, 
        density: np.ndarray,
        apply_shift: bool = True
    ) -> DiffractionResult:
        """
        计算衍射图案
        
        参数:
            density: 电子密度图（2D）
            apply_shift: 是否应用fftshift
        
        返回:
            DiffractionResult对象
        """
        if density.shape != (self.grid_size, self.grid_size):
            density = self._resize_density(density)
        
        padded = self._pad_density(density)
        
        F = fft2(padded)
        
        if apply_shift:
            F = fftshift(F)
        
        amplitude = np.abs(F)
        intensity = amplitude ** 2
        phase = np.angle(F)
        
        if self.oversampling > 1.0:
            intensity = self._crop_to_original(intensity)
            amplitude = self._crop_to_original(amplitude)
            phase = self._crop_to_original(phase)
        
        metadata = {
            'original_shape': density.shape,
            'padded_shape': padded.shape,
            'oversampling': self.oversampling
        }
        
        return DiffractionResult(
            intensity=intensity,
            amplitude=amplitude,
            phase=phase,
            metadata=metadata
        )
    
    def _resize_density(self, density: np.ndarray) -> np.ndarray:
        """调整密度图尺寸"""
        if density.shape == (self.grid_size, self.grid_size):
            return density
        
        scale_x = self.grid_size / density.shape[0]
        scale_y = self.grid_size / density.shape[1]
        
        resized = zoom(density, (scale_x, scale_y), order=1)
        return resized
    
    def _pad_density(self, density: np.ndarray) -> np.ndarray:
        """零填充"""
        if self.oversampling <= 1.0:
            return density
        
        padded_size = int(self.grid_size * self.oversampling)
        padded = np.zeros((padded_size, padded_size), dtype=density.dtype)
        
        offset = (padded_size - self.grid_size) // 2
        padded[offset:offset + self.grid_size, offset:offset + self.grid_size] = density
        
        return padded
    
    def _crop_to_original(self, data: np.ndarray) -> np.ndarray:
        """裁剪回原始尺寸"""
        center = data.shape[0] // 2
        half_size = self.grid_size // 2
        
        return data[center - half_size:center + half_size,
                    center - half_size:center + half_size]


class DataAugmenter:
    """
    数据增强器
    
    重要：所有增强操作都在FFT之前执行（对电子密度图操作）
    这样可以保证物理正确性
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def augment(
        self,
        density: np.ndarray,
        rotation_enabled: bool = True,
        flip_enabled: bool = True,
        scale_enabled: bool = False,
        rotation_angles: list = [0, 90, 180, 270],
        flip_axes: list = [0, 1],
        scale_range: tuple = (0.9, 1.1)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        执行数据增强
        
        参数:
            density: 输入电子密度图
            rotation_enabled: 是否启用旋转
            flip_enabled: 是否启用翻转
            scale_enabled: 是否启用尺度变换
            rotation_angles: 可选的旋转角度
            flip_axes: 可选的翻转轴
            scale_range: 尺度变换范围
        
        返回:
            augmented: 增强后的密度图
            params: 使用的增强参数
        """
        augmented = density.copy()
        params = {}
        
        if rotation_enabled:
            angle = self.rng.choice(rotation_angles)
            if angle != 0:
                augmented = rotate(augmented, angle, reshape=False, order=1)
            params['rotation_angle'] = angle
        
        if flip_enabled:
            if self.rng.random() > 0.5:
                flip_axis = self.rng.choice(flip_axes)
                augmented = np.flip(augmented, axis=flip_axis)
                params['flip_axis'] = flip_axis
            else:
                params['flip_axis'] = None
        
        if scale_enabled:
            scale = self.rng.uniform(*scale_range)
            augmented = zoom(augmented, scale, order=1)
            
            target_size = density.shape[0]
            if augmented.shape[0] != target_size:
                crop_size = min(augmented.shape[0], target_size)
                start = (augmented.shape[0] - crop_size) // 2
                augmented = augmented[start:start + crop_size, start:start + crop_size]
                
                if augmented.shape[0] < target_size:
                    pad_size = target_size - augmented.shape[0]
                    augmented = np.pad(augmented, pad_size // 2, mode='constant')
            
            params['scale'] = scale
        
        return augmented, params


class PhysicsBasedDiffractionGenerator:
    """
    物理参数化模型 - 直接生成衍射图案
    
    基于实验数据特征建模，不依赖具体结构
    """
    
    def __init__(self, grid_size: int = 512, seed: Optional[int] = None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed)
        
        self.center = grid_size // 2
        x = np.linspace(-self.center, self.center, grid_size)
        y = np.linspace(-self.center, self.center, grid_size)
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
    
    def generate(
        self,
        center_intensity: float = 40000.0,
        background_intensity: float = 1500.0,
        core_radius: float = 25.0,
        halo_decay: float = 50.0,
        structure_strength: float = 0.2,
        speckle_count: int = 5,
        ring_count: int = 1
    ) -> np.ndarray:
        """
        生成物理参数化衍射图案
        
        参数:
            center_intensity: 中心强度
            background_intensity: 背景强度
            core_radius: 核心半径（像素）
            halo_decay: 晕衰减率
            structure_strength: 结构强度 (0-1)
            speckle_count: 散斑数量
            ring_count: 衍射环数量
        """
        core_intensity = center_intensity * np.exp(-self.R**2 / (2 * core_radius**2))
        
        halo_intensity = (center_intensity * 0.15) * np.exp(-self.R / halo_decay)
        
        radial_profile = core_intensity + halo_intensity + background_intensity
        
        pattern = radial_profile.copy()
        
        for _ in range(speckle_count):
            speckle_r = self.rng.uniform(0, core_radius * 2.5)
            speckle_theta = self.rng.uniform(0, 2 * np.pi)
            speckle_x = speckle_r * np.cos(speckle_theta)
            speckle_y = speckle_r * np.sin(speckle_theta)
            speckle_size = self.rng.uniform(5, 20)
            speckle_intensity = self.rng.uniform(0.3, 1.5)
            
            dist = np.sqrt((self.X - speckle_x)**2 + (self.Y - speckle_y)**2)
            speckle = speckle_intensity * np.exp(-dist**2 / (2 * speckle_size**2))
            pattern = pattern * (1 + speckle * structure_strength * 0.5)
        
        for _ in range(ring_count):
            ring_r = self.rng.uniform(core_radius * 0.8, core_radius * 3)
            ring_width = self.rng.uniform(3, 15)
            ring_intensity = self.rng.uniform(0.8, 1.3)
            
            ring = np.exp(-(self.R - ring_r)**2 / (2 * ring_width**2))
            pattern = pattern * (1 + ring * (ring_intensity - 1) * structure_strength * 0.3)
        
        noise = self.rng.standard_normal((self.grid_size, self.grid_size))
        noise_strength = structure_strength * 0.3 * np.exp(-self.R / (core_radius * 2))
        pattern = pattern * (1 + noise * noise_strength)
        
        pattern = np.maximum(pattern, 0)
        
        return pattern
    
    def generate_random(
        self,
        center_intensity_range: tuple = (30000.0, 50000.0),
        background_intensity_range: tuple = (1000.0, 2000.0),
        core_radius_range: tuple = (15.0, 35.0),
        halo_decay_range: tuple = (40.0, 80.0),
        structure_strength_range: tuple = (0.1, 0.4),
        speckle_count_range: tuple = (3, 10),
        ring_count_range: tuple = (0, 3)
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        随机参数生成衍射图案
        """
        params = {
            'center_intensity': self.rng.uniform(*center_intensity_range),
            'background_intensity': self.rng.uniform(*background_intensity_range),
            'core_radius': self.rng.uniform(*core_radius_range),
            'halo_decay': self.rng.uniform(*halo_decay_range),
            'structure_strength': self.rng.uniform(*structure_strength_range),
            'speckle_count': self.rng.integers(*speckle_count_range),
            'ring_count': self.rng.integers(*ring_count_range)
        }
        
        pattern = self.generate(**params)
        
        return pattern, params


class DiffractionPipeline:
    """
    衍射计算流水线
    
    整合：
    1. 数据增强（FFT之前）
    2. FFT衍射计算
    3. 物理参数化模型
    """
    
    def __init__(
        self,
        grid_size: int = 512,
        oversampling: float = 1.0,
        seed: Optional[int] = None
    ):
        self.fft_calc = FFTDiffractionCalculator(grid_size, oversampling)
        self.augmenter = DataAugmenter(seed)
        self.physics_gen = PhysicsBasedDiffractionGenerator(grid_size, seed)
    
    def from_density(
        self,
        density: np.ndarray,
        augment: bool = True,
        augment_params: Optional[Dict] = None
    ) -> DiffractionResult:
        """
        从电子密度图计算衍射
        
        参数:
            density: 电子密度图
            augment: 是否执行数据增强
            augment_params: 增强参数（None则随机）
        """
        if augment:
            if augment_params is None:
                density, aug_info = self.augmenter.augment(density)
            else:
                density, aug_info = self.augmenter.augment(
                    density, **augment_params
                )
        else:
            aug_info = {}
        
        result = self.fft_calc.calculate(density)
        result.metadata['augmentation'] = aug_info
        
        return result
    
    def from_physics_model(
        self,
        randomize: bool = True,
        **kwargs
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        从物理参数化模型生成衍射
        """
        if randomize:
            return self.physics_gen.generate_random(**kwargs)
        else:
            pattern = self.physics_gen.generate(**kwargs)
            return pattern, kwargs


if __name__ == "__main__":
    print("测试衍射计算模块...")
    
    pipeline = DiffractionPipeline(grid_size=256, seed=42)
    
    density = np.zeros((256, 256))
    center = 128
    y, x = np.ogrid[:256, :256]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    density[r < 80] = 1.0
    
    result = pipeline.from_density(density, augment=True)
    print(f"FFT衍射: shape={result.intensity.shape}, max={result.intensity.max():.1f}")
    
    pattern, params = pipeline.from_physics_model(randomize=True)
    print(f"物理模型: shape={pattern.shape}, max={pattern.max():.1f}")
    print(f"参数: {params}")
    
    print("\n衍射计算模块测试完成")
