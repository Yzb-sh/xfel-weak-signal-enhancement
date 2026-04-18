"""
噪声模型模块

核心要点：
1. 泊松噪声：噪音强度分布与衍射数据强度分布相关
   - 高强度区域：期望光子数多，噪声绝对值大，相对噪声小
   - 低强度区域：期望光子数少，噪声绝对值小，相对噪声大

2. 物理正确性：
   - 泊松分布：P(k|λ) = λ^k * e^(-λ) / k!
   - 期望 = 方差 = λ
   - 信噪比 SNR = sqrt(λ)，与光子数的平方根成正比
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NoiseResult:
    """噪声添加结果"""
    clean_scaled: np.ndarray
    noisy: np.ndarray
    exposure_level: float
    gaussian_std: float
    snr_map: Optional[np.ndarray] = None


class PoissonNoiseModel:
    """
    泊松噪声模型
    
    物理原理：
    - X射线光子的到达服从泊松过程
    - 每个像素探测到的光子数服从泊松分布
    - 泊松分布的期望值 = 该像素的"真实"光子数
    
    关键特性：
    - 噪声强度与信号强度相关（信号依赖噪声）
    - 高强度区域：噪声绝对值大，但相对噪声小（SNR高）
    - 低强度区域：噪声绝对值小，但相对噪声大（SNR低）
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def add_noise(
        self, 
        clean_pattern: np.ndarray, 
        exposure_level: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加泊松噪声
        
        参数:
            clean_pattern: 干净的衍射图案（任意强度单位）
            exposure_level: 曝光水平（最大强度位置的平均光子数）
        
        返回:
            clean_scaled: 缩放后的干净图案（光子数单位）
            noisy: 含噪图案（光子数单位）
        
        物理过程:
            1. 将干净图案归一化到[0, 1]
            2. 乘以exposure_level，得到每个像素的期望光子数λ
            3. 泊松采样：每个像素的光子数k ~ Poisson(λ)
        """
        max_val = clean_pattern.max()
        if max_val <= 0:
            raise ValueError("衍射图案最大值必须大于0")
        
        normalized = clean_pattern / max_val
        
        expected_photons = normalized * exposure_level
        
        expected_photons = np.maximum(expected_photons, 0)
        
        noisy = self.rng.poisson(expected_photons.astype(np.float64))
        
        clean_scaled = expected_photons.copy()
        
        return clean_scaled.astype(np.float64), noisy.astype(np.float64)
    
    def compute_snr(self, expected_photons: np.ndarray) -> np.ndarray:
        """
        计算信噪比
        
        对于泊松分布：
        - 信号 = λ（期望光子数）
        - 噪声 = sqrt(λ)（标准差）
        - SNR = λ / sqrt(λ) = sqrt(λ)
        
        这意味着：
        - 高强度区域（λ大）：SNR高
        - 低强度区域（λ小）：SNR低
        """
        return np.sqrt(np.maximum(expected_photons, 0))
    
    def compute_noise_statistics(
        self, 
        clean_scaled: np.ndarray, 
        noisy: np.ndarray
    ) -> dict:
        """
        计算噪声统计信息
        
        用于验证噪声模型的正确性
        """
        noise = noisy - clean_scaled
        
        expected_variance = clean_scaled
        
        actual_variance = noise ** 2
        
        snr = self.compute_snr(clean_scaled)
        
        return {
            'noise_mean': noise.mean(),
            'noise_std': noise.std(),
            'expected_noise_std': np.sqrt(clean_scaled.mean()),
            'signal_mean': clean_scaled.mean(),
            'signal_max': clean_scaled.max(),
            'snr_mean': snr.mean(),
            'snr_at_max': np.sqrt(clean_scaled.max()),
            'variance_ratio': actual_variance.mean() / expected_variance.mean() if expected_variance.mean() > 0 else 0,
        }


class GaussianNoiseModel:
    """
    高斯噪声模型
    
    物理来源：探测器读出噪声
    特点：与信号强度无关（信号独立噪声）
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def add_noise(
        self, 
        pattern: np.ndarray, 
        std: float
    ) -> np.ndarray:
        """
        添加高斯噪声
        
        参数:
            pattern: 输入图案
            std: 噪声标准差
        
        返回:
            含噪图案
        """
        if std <= 0:
            return pattern.copy()
        
        noise = self.rng.normal(0, std, pattern.shape)
        noisy = pattern + noise
        noisy = np.maximum(noisy, 0)
        
        return noisy


class BadPixelModel:
    """
    坏像素模型
    
    物理来源：探测器缺陷
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def add_bad_pixels(
        self, 
        pattern: np.ndarray, 
        bad_pixel_ratio: float,
        bad_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        添加坏像素
        
        参数:
            pattern: 输入图案
            bad_pixel_ratio: 坏像素比例
            bad_value: 坏像素的值（默认为0，即死像素）
        
        返回:
            含坏像素的图案, 坏像素掩码
        """
        if bad_pixel_ratio <= 0:
            return pattern.copy(), np.zeros(pattern.shape, dtype=bool)
        
        bad_mask = self.rng.random(pattern.shape) < bad_pixel_ratio
        
        noisy = pattern.copy()
        noisy[bad_mask] = bad_value
        
        return noisy, bad_mask


class StructuredNoiseModel:
    """
    结构化噪声模型
    
    物理来源：背景散射、杂散光
    特点：具有空间相关性
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
    
    def add_noise(
        self, 
        pattern: np.ndarray, 
        correlation_length: float,
        strength: float
    ) -> np.ndarray:
        """
        添加结构化噪声
        
        参数:
            pattern: 输入图案
            correlation_length: 空间相关长度（像素）
            strength: 噪声强度（相对于信号的比例）
        
        返回:
            含结构化噪声的图案
        """
        if strength <= 0:
            return pattern.copy()
        
        from scipy.ndimage import gaussian_filter
        
        random_noise = self.rng.random(pattern.shape)
        
        structured_noise = gaussian_filter(random_noise, correlation_length)
        
        structured_noise = structured_noise / structured_noise.max() * pattern.max() * strength
        
        noisy = pattern + structured_noise
        noisy = np.maximum(noisy, 0)
        
        return noisy


class CompositeNoiseModel:
    """
    组合噪声模型
    
    按顺序添加多种噪声：
    1. 泊松噪声（光子散粒噪声）- 必须首先添加
    2. 高斯噪声（探测器读出噪声）
    3. 结构化噪声（背景散射）
    4. 坏像素（探测器缺陷）
    """
    
    def __init__(
        self,
        poisson_enabled: bool = True,
        gaussian_enabled: bool = False,
        bad_pixels_enabled: bool = False,
        structured_enabled: bool = False,
        seed: Optional[int] = None
    ):
        self.poisson_enabled = poisson_enabled
        self.gaussian_enabled = gaussian_enabled
        self.bad_pixels_enabled = bad_pixels_enabled
        self.structured_enabled = structured_enabled
        
        self.poisson = PoissonNoiseModel(seed=seed)
        self.gaussian = GaussianNoiseModel(seed=seed + 1 if seed is not None else None)
        self.bad_pixels = BadPixelModel(seed=seed + 2 if seed is not None else None)
        self.structured = StructuredNoiseModel(seed=seed + 3 if seed is not None else None)
    
    def add_noise(
        self,
        clean_pattern: np.ndarray,
        exposure_level: float,
        gaussian_std: float = 0.0,
        bad_pixel_ratio: float = 0.0,
        structured_correlation: float = 20.0,
        structured_strength: float = 0.1
    ) -> NoiseResult:
        """
        添加组合噪声
        
        参数:
            clean_pattern: 干净的衍射图案
            exposure_level: 曝光水平
            gaussian_std: 高斯噪声标准差
            bad_pixel_ratio: 坏像素比例
            structured_correlation: 结构化噪声相关长度
            structured_strength: 结构化噪声强度
        
        返回:
            NoiseResult对象
        """
        current = clean_pattern.copy()
        clean_scaled = None
        snr_map = None
        
        if self.poisson_enabled and exposure_level > 0:
            clean_scaled, current = self.poisson.add_noise(current, exposure_level)
            snr_map = self.poisson.compute_snr(clean_scaled)
        else:
            clean_scaled = current.copy()
        
        if self.structured_enabled and structured_strength > 0:
            current = self.structured.add_noise(
                current, structured_correlation, structured_strength
            )
        
        if self.gaussian_enabled and gaussian_std > 0:
            current = self.gaussian.add_noise(current, gaussian_std)
        
        bad_mask = None
        if self.bad_pixels_enabled and bad_pixel_ratio > 0:
            current, bad_mask = self.bad_pixels.add_bad_pixels(current, bad_pixel_ratio)
        
        return NoiseResult(
            clean_scaled=clean_scaled,
            noisy=current,
            exposure_level=exposure_level,
            gaussian_std=gaussian_std,
            snr_map=snr_map
        )


def verify_poisson_noise():
    """
    验证泊松噪声模型的正确性
    
    测试方法：
    1. 生成模拟衍射图案
    2. 多次添加泊松噪声
    3. 验证噪声统计特性是否符合泊松分布
    """
    print("=" * 60)
    print("泊松噪声模型验证")
    print("=" * 60)
    
    np.random.seed(42)
    
    size = 256
    center = size // 2
    x = np.linspace(-center, center, size)
    y = np.linspace(-center, center, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    clean_pattern = 40000 * np.exp(-R**2 / (2 * 30**2)) + 1500
    
    model = PoissonNoiseModel(seed=42)
    
    exposure_level = 100000
    clean_scaled, noisy = model.add_noise(clean_pattern, exposure_level)
    
    stats = model.compute_noise_statistics(clean_scaled, noisy)
    
    print(f"\n曝光水平: {exposure_level}")
    print(f"信号最大值: {stats['signal_max']:.1f}")
    print(f"信号平均值: {stats['signal_mean']:.1f}")
    print(f"\n噪声统计:")
    print(f"  噪声均值: {stats['noise_mean']:.4f} (理论值: 0)")
    print(f"  噪声标准差: {stats['noise_std']:.2f}")
    print(f"  期望噪声标准差: {stats['expected_noise_std']:.2f}")
    print(f"\n信噪比:")
    print(f"  平均SNR: {stats['snr_mean']:.2f}")
    print(f"  最大信号处SNR: {stats['snr_at_max']:.2f}")
    print(f"\n方差比 (实际/理论): {stats['variance_ratio']:.4f} (理论值: 1.0)")
    
    n_trials = 1000
    test_pixel = (center, center)
    samples = []
    
    for i in range(n_trials):
        _, noisy_sample = model.add_noise(clean_pattern, exposure_level)
        samples.append(noisy_sample[test_pixel])
    
    samples = np.array(samples)
    expected_value = clean_scaled[test_pixel]
    
    print(f"\n单像素统计验证 (位置: {test_pixel}):")
    print(f"  期望值: {expected_value:.1f}")
    print(f"  样本均值: {samples.mean():.1f}")
    print(f"  期望标准差: {np.sqrt(expected_value):.1f}")
    print(f"  样本标准差: {samples.std():.1f}")
    
    print("\n" + "=" * 60)
    print("验证完成！")
    print("=" * 60)


if __name__ == "__main__":
    verify_poisson_noise()
