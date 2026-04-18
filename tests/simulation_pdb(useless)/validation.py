"""
验证模块

功能：
1. 验证模拟衍射数据的质量
2. 简化的评估指标：
   - 有清晰的衍射结构（不是平滑背景）
   - 强度范围与实验数据近似
   - 径向衰减特征合理
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    contrast: float
    intensity_max: float
    intensity_mean: float
    decay_ratio: float
    has_structure: bool
    intensity_ok: bool
    decay_ok: bool
    details: Dict[str, Any]


class DiffractionValidator:
    """
    衍射图案验证器
    
    简化的验证标准：
    1. 有衍射结构：对比度 > 阈值
    2. 强度范围：最大值和平均值在合理范围内
    3. 径向衰减：中心到边缘的衰减比例合理
    """
    
    def __init__(
        self,
        min_contrast: float = 0.3,
        intensity_max_range: Tuple[float, float] = (5000.0, 500000.0),
        intensity_mean_range: Tuple[float, float] = (500.0, 10000.0),
        decay_ratio_range: Tuple[float, float] = (3.0, 30.0)
    ):
        self.min_contrast = min_contrast
        self.intensity_max_range = intensity_max_range
        self.intensity_mean_range = intensity_mean_range
        self.decay_ratio_range = decay_ratio_range
    
    def validate(self, pattern: np.ndarray) -> ValidationResult:
        """
        验证衍射图案
        
        参数:
            pattern: 衍射图案
        
        返回:
            ValidationResult对象
        """
        contrast = self._calculate_contrast(pattern)
        
        intensity_max = pattern.max()
        intensity_mean = pattern.mean()
        
        decay_ratio = self._calculate_decay_ratio(pattern)
        
        has_structure = contrast > self.min_contrast
        
        intensity_ok = (self.intensity_max_range[0] <= intensity_max <= self.intensity_max_range[1] and
                       self.intensity_mean_range[0] <= intensity_mean <= self.intensity_mean_range[1])
        
        decay_ok = self.decay_ratio_range[0] <= decay_ratio <= self.decay_ratio_range[1]
        
        is_valid = has_structure and intensity_ok and decay_ok
        
        details = {
            'contrast_threshold': self.min_contrast,
            'intensity_max_range': self.intensity_max_range,
            'intensity_mean_range': self.intensity_mean_range,
            'decay_ratio_range': self.decay_ratio_range
        }
        
        return ValidationResult(
            is_valid=is_valid,
            contrast=contrast,
            intensity_max=intensity_max,
            intensity_mean=intensity_mean,
            decay_ratio=decay_ratio,
            has_structure=has_structure,
            intensity_ok=intensity_ok,
            decay_ok=decay_ok,
            details=details
        )
    
    def _calculate_contrast(self, pattern: np.ndarray) -> float:
        """
        计算对比度
        
        对比度 = 标准差 / 平均值
        高对比度表示有清晰的衍射结构
        """
        if pattern.mean() == 0:
            return 0.0
        return pattern.std() / pattern.mean()
    
    def _calculate_decay_ratio(self, pattern: np.ndarray, r_center: int = 5, r_edge: int = 50) -> float:
        """
        计算径向衰减比例
        
        中心强度 / 边缘强度
        
        参数:
            pattern: 衍射图案
            r_center: 中心区域半径
            r_edge: 边缘测量半径
        """
        center = pattern.shape[0] // 2
        
        y, x = np.ogrid[:pattern.shape[0], :pattern.shape[1]]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        center_mask = r <= r_center
        center_intensity = pattern[center_mask].mean() if center_mask.sum() > 0 else 0
        
        edge_mask = (r >= r_edge - 5) & (r <= r_edge + 5)
        edge_intensity = pattern[edge_mask].mean() if edge_mask.sum() > 0 else 1
        
        if edge_intensity == 0:
            return float('inf')
        
        return center_intensity / edge_intensity
    
    def calculate_radial_profile(self, pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算径向强度分布
        
        返回:
            r: 半径数组
            profile: 径向强度分布
        """
        center = pattern.shape[0] // 2
        y, x = np.ogrid[:pattern.shape[0], :pattern.shape[1]]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        r_int = r.astype(int)
        max_r = r_int.max()
        
        radial_sum = np.bincount(r_int.ravel(), pattern.ravel())
        radial_count = np.bincount(r_int.ravel())
        
        profile = radial_sum / np.maximum(radial_count, 1)
        r_values = np.arange(len(profile))
        
        return r_values, profile


class DatasetStatistics:
    """数据集统计信息"""
    
    def __init__(self):
        self.samples_validated = 0
        self.samples_passed = 0
        self.contrast_values = []
        self.intensity_max_values = []
        self.intensity_mean_values = []
        self.decay_ratio_values = []
    
    def add_result(self, result: ValidationResult):
        """添加验证结果"""
        self.samples_validated += 1
        if result.is_valid:
            self.samples_passed += 1
        
        self.contrast_values.append(result.contrast)
        self.intensity_max_values.append(result.intensity_max)
        self.intensity_mean_values.append(result.intensity_mean)
        self.decay_ratio_values.append(result.decay_ratio)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'total_samples': self.samples_validated,
            'passed_samples': self.samples_passed,
            'pass_rate': self.samples_passed / max(self.samples_validated, 1),
            'contrast': {
                'mean': np.mean(self.contrast_values) if self.contrast_values else 0,
                'std': np.std(self.contrast_values) if self.contrast_values else 0,
                'min': np.min(self.contrast_values) if self.contrast_values else 0,
                'max': np.max(self.contrast_values) if self.contrast_values else 0,
            },
            'intensity_max': {
                'mean': np.mean(self.intensity_max_values) if self.intensity_max_values else 0,
                'std': np.std(self.intensity_max_values) if self.intensity_max_values else 0,
                'min': np.min(self.intensity_max_values) if self.intensity_max_values else 0,
                'max': np.max(self.intensity_max_values) if self.intensity_max_values else 0,
            },
            'intensity_mean': {
                'mean': np.mean(self.intensity_mean_values) if self.intensity_mean_values else 0,
                'std': np.std(self.intensity_mean_values) if self.intensity_mean_values else 0,
                'min': np.min(self.intensity_mean_values) if self.intensity_mean_values else 0,
                'max': np.max(self.intensity_mean_values) if self.intensity_mean_values else 0,
            },
            'decay_ratio': {
                'mean': np.mean(self.decay_ratio_values) if self.decay_ratio_values else 0,
                'std': np.std(self.decay_ratio_values) if self.decay_ratio_values else 0,
                'min': np.min(self.decay_ratio_values) if self.decay_ratio_values else 0,
                'max': np.max(self.decay_ratio_values) if self.decay_ratio_values else 0,
            }
        }


def visualize_validation(
    pattern: np.ndarray,
    result: ValidationResult,
    output_path: Optional[Path] = None
):
    """
    可视化验证结果
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1 = axes[0]
    im1 = ax1.imshow(pattern, cmap='jet', norm=plt.matplotlib.colors.LogNorm())
    ax1.set_title(f'衍射图案\n最大值: {result.intensity_max:.0f}')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = axes[1]
    center = pattern.shape[0] // 2
    y, x = np.ogrid[:pattern.shape[0], :pattern.shape[1]]
    r = np.sqrt((x - center)**2 + (y - center)**2)
    
    r_int = r.astype(int)
    radial_sum = np.bincount(r_int.ravel(), pattern.ravel())
    radial_count = np.bincount(r_int.ravel())
    profile = radial_sum / np.maximum(radial_count, 1)
    
    ax2.semilogy(profile[:len(profile)//2])
    ax2.set_xlabel('半径 (像素)')
    ax2.set_ylabel('平均强度')
    ax2.set_title(f'径向分布\n衰减比: {result.decay_ratio:.1f}')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    status_color = 'green' if result.is_valid else 'red'
    ax3.axis('off')
    
    status_text = "✓ 通过" if result.is_valid else "✗ 未通过"
    info_text = f"""
    验证结果: {status_text}
    
    对比度: {result.contrast:.3f} (阈值: {result.details['contrast_threshold']})
    衍射结构: {'有' if result.has_structure else '无'}
    
    最大强度: {result.intensity_max:.0f}
    平均强度: {result.intensity_mean:.0f}
    强度范围: {'✓' if result.intensity_ok else '✗'}
    
    衰减比例: {result.decay_ratio:.1f}
    衰减范围: {'✓' if result.decay_ok else '✗'}
    """
    
    ax3.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("测试验证模块...")
    
    validator = DiffractionValidator()
    
    rng = np.random.default_rng(42)
    size = 256
    center = size // 2
    x = np.linspace(-center, center, size)
    y = np.linspace(-center, center, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    good_pattern = 40000 * np.exp(-R**2 / (2 * 30**2)) + 1500
    good_pattern *= (1 + rng.random((size, size)) * 0.2)
    
    result = validator.validate(good_pattern)
    print(f"\n好的衍射图案:")
    print(f"  通过: {result.is_valid}")
    print(f"  对比度: {result.contrast:.3f}")
    print(f"  衰减比: {result.decay_ratio:.1f}")
    
    bad_pattern = np.ones((size, size)) * 1000
    
    result = validator.validate(bad_pattern)
    print(f"\n差的衍射图案:")
    print(f"  通过: {result.is_valid}")
    print(f"  对比度: {result.contrast:.3f}")
    print(f"  衰减比: {result.decay_ratio:.1f}")
    
    print("\n验证模块测试完成")
