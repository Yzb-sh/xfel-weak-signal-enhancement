"""
数据生成流水线

整合所有模块，生成完整的衍射数据集

流程：
1. 结构来源选择（EMDB / 程序化生成 / 物理模型）
2. 数据增强（FFT之前）
3. FFT衍射计算
4. 噪声添加
5. 验证
6. 保存
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import h5py
import time
import logging

from config import Config, get_test_config
from noise import CompositeNoiseModel
from emdb_loader import create_synthetic_emdb_like_structure
from procedural_gen import CompositeStructureGenerator
from diffraction import DiffractionPipeline
from validation import DiffractionValidator, DatasetStatistics, visualize_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SampleMetadata:
    """样本元数据"""
    sample_id: int
    source_type: str
    structure_type: str
    size_nm: float
    
    exposure_level: float
    gaussian_std: float
    
    rotation_angle: int
    flip_axis: Optional[int]
    
    contrast: float
    intensity_max: float
    intensity_mean: float
    decay_ratio: float
    is_valid: bool


class DiffractionDataPipeline:
    """
    衍射数据生成流水线
    
    整合所有模块，生成完整的训练数据集
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        self.rng = np.random.default_rng(config.simulation.random_seed)
        
        self.noise_model = CompositeNoiseModel(
            poisson_enabled=config.noise.poisson_noise_enabled,
            gaussian_enabled=config.noise.gaussian_noise_enabled,
            bad_pixels_enabled=config.noise.bad_pixels_enabled,
            structured_enabled=config.noise.structured_noise_enabled,
            seed=config.simulation.random_seed
        )
        
        self.structure_gen = CompositeStructureGenerator(
            grid_size=config.simulation.image_size,
            seed=config.simulation.random_seed
        )
        
        self.diffraction_pipeline = DiffractionPipeline(
            grid_size=config.simulation.image_size,
            oversampling=config.simulation.oversampling_ratio,
            seed=config.simulation.random_seed
        )
        
        self.validator = DiffractionValidator(
            min_contrast=config.validation.min_contrast,
            intensity_max_range=config.validation.intensity_max_range,
            intensity_mean_range=config.validation.intensity_mean_range,
            decay_ratio_range=config.validation.decay_ratio_range
        )
        
        self.stats = DatasetStatistics()
        
        self._setup_directories()
    
    def _setup_directories(self):
        """创建必要的目录"""
        self.config.output.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.validation.visualization_dir.mkdir(parents=True, exist_ok=True)
    
    def _select_source_type(self) -> str:
        """随机选择结构来源类型"""
        r = self.rng.random()
        cumsum = 0.0
        
        ratios = {
            'emdb': self.config.structure_source.emdb_ratio,
            'procedural': self.config.structure_source.procedural_ratio,
            'physics_model': self.config.structure_source.physics_model_ratio
        }
        
        for source_type, ratio in ratios.items():
            cumsum += ratio
            if r < cumsum:
                return source_type
        
        return 'procedural'
    
    def _generate_structure(self, source_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """生成电子密度结构"""
        size_nm = self.rng.uniform(
            self.config.procedural.size_range_nm[0],
            self.config.procedural.size_range_nm[1]
        )
        
        if source_type == 'emdb':
            if self.config.emdb.enabled:
                structure_type = self.rng.choice(['virus', 'organelle', 'cell'])
                density, meta = create_synthetic_emdb_like_structure(
                    size_nm=size_nm,
                    structure_type=structure_type,
                    grid_size=self.config.simulation.image_size,
                    seed=self.rng.integers(0, 2**31)
                )
                meta['source_type'] = 'emdb_synthetic'
                return density, meta
            else:
                source_type = 'procedural'
        
        if source_type == 'procedural':
            category = self.rng.choice(['virus', 'cell', 'organelle'])
            density, struct_meta = self.structure_gen.generate_random(
                size_nm=size_nm,
                structure_category=category
            )
            meta = {
                'source_type': 'procedural',
                'structure_type': struct_meta.structure_type,
                'size_nm': struct_meta.size_nm,
                'grid_size': struct_meta.grid_size,
                'parameters': struct_meta.parameters
            }
            return density, meta
        
        if source_type == 'physics_model':
            pattern, params = self.diffraction_pipeline.from_physics_model(
                randomize=True,
                center_intensity_range=self.config.physics_model.center_intensity_range,
                background_intensity_range=self.config.physics_model.background_intensity_range,
                core_radius_range=self.config.physics_model.core_radius_range,
                halo_decay_range=self.config.physics_model.halo_decay_range,
                structure_strength_range=self.config.physics_model.structure_strength_range,
                speckle_count_range=self.config.physics_model.speckle_count_range,
                ring_count_range=self.config.physics_model.ring_count_range
            )
            meta = {
                'source_type': 'physics_model',
                'structure_type': 'physics_model',
                'size_nm': size_nm,
                'physics_params': params
            }
            return pattern, meta
        
        density = np.zeros((self.config.simulation.image_size, self.config.simulation.image_size))
        return density, {'source_type': 'unknown', 'structure_type': 'unknown'}
    
    def _apply_augmentation(self, density: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """应用数据增强（FFT之前）"""
        from diffraction import DataAugmenter
        
        augmenter = DataAugmenter(seed=self.rng.integers(0, 2**31))
        
        return augmenter.augment(
            density,
            rotation_enabled=self.config.augmentation.rotation_enabled,
            flip_enabled=self.config.augmentation.flip_enabled,
            scale_enabled=self.config.augmentation.scale_enabled,
            rotation_angles=self.config.augmentation.rotation_angles,
            flip_axes=self.config.augmentation.flip_axes,
            scale_range=self.config.augmentation.scale_range
        )
    
    def _calculate_diffraction(self, density: np.ndarray, source_type: str) -> np.ndarray:
        """计算衍射图案"""
        if source_type == 'physics_model':
            return density
        
        result = self.diffraction_pipeline.from_density(
            density,
            augment=False
        )
        
        intensity = result.intensity
        
        target_max = self.rng.uniform(30000, 50000)
        background = self.rng.uniform(1000, 2000)
        
        if intensity.max() > 0:
            scale = target_max / intensity.max()
            intensity = intensity * scale
        
        intensity = intensity + background
        
        return intensity
    
    def _add_noise(
        self, 
        pattern: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """添加噪声"""
        exposure_level = self.rng.uniform(
            *self.config.noise.exposure_level_range
        )
        
        gaussian_std = 0.0
        if self.config.noise.gaussian_noise_enabled:
            gaussian_std = self.rng.uniform(
                *self.config.noise.gaussian_std_range
            )
        
        result = self.noise_model.add_noise(
            pattern,
            exposure_level=exposure_level,
            gaussian_std=gaussian_std,
            bad_pixel_ratio=self.config.noise.bad_pixel_ratio if self.config.noise.bad_pixels_enabled else 0.0,
            structured_correlation=self.config.noise.structured_noise_correlation_length,
            structured_strength=self.config.noise.structured_noise_strength
        )
        
        return result.clean_scaled, result.noisy, exposure_level, gaussian_std
    
    def generate_sample(self, sample_id: int) -> Tuple[np.ndarray, np.ndarray, SampleMetadata]:
        """
        生成单个样本
        
        返回:
            clean: 干净的衍射图案
            noisy: 含噪的衍射图案
            metadata: 元数据
        """
        source_type = self._select_source_type()
        
        density, struct_meta = self._generate_structure(source_type)
        
        if source_type != 'physics_model' and self.config.augmentation.rotation_enabled:
            density, aug_meta = self._apply_augmentation(density)
        else:
            aug_meta = {'rotation_angle': 0, 'flip_axis': None}
        
        clean_pattern = self._calculate_diffraction(density, source_type)
        
        clean, noisy, exposure_level, gaussian_std = self._add_noise(clean_pattern)
        
        validation_result = self.validator.validate(noisy)
        self.stats.add_result(validation_result)
        
        metadata = SampleMetadata(
            sample_id=sample_id,
            source_type=struct_meta.get('source_type', 'unknown'),
            structure_type=struct_meta.get('structure_type', 'unknown'),
            size_nm=struct_meta.get('size_nm', 0.0),
            exposure_level=exposure_level,
            gaussian_std=gaussian_std,
            rotation_angle=aug_meta.get('rotation_angle', 0),
            flip_axis=aug_meta.get('flip_axis'),
            contrast=validation_result.contrast,
            intensity_max=validation_result.intensity_max,
            intensity_mean=validation_result.intensity_mean,
            decay_ratio=validation_result.decay_ratio,
            is_valid=validation_result.is_valid
        )
        
        return clean, noisy, metadata
    
    def generate_dataset(
        self,
        total_samples: Optional[int] = None,
        visualize: bool = True,
        visualize_every: int = 10
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[SampleMetadata]]:
        """
        生成完整数据集
        
        参数:
            total_samples: 总样本数（None则使用配置）
            visualize: 是否可视化部分样本
            visualize_every: 每隔多少样本可视化一次
        """
        if total_samples is None:
            total_samples = self.config.data_generation.total_samples
        
        clean_list = []
        noisy_list = []
        metadata_list = []
        
        logger.info(f"开始生成数据集，共 {total_samples} 个样本")
        start_time = time.time()
        
        for i in range(total_samples):
            clean, noisy, metadata = self.generate_sample(i)
            
            clean_list.append(clean)
            noisy_list.append(noisy)
            metadata_list.append(metadata)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (total_samples - i - 1)
                logger.info(f"进度: {i+1}/{total_samples} ({(i+1)/total_samples*100:.1f}%), "
                           f"已用时间: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
            
            if visualize and (i + 1) % visualize_every == 0:
                output_path = self.config.validation.visualization_dir / f"sample_{i:04d}.png"
                result = self.validator.validate(noisy)
                visualize_validation(noisy, result, output_path)
        
        elapsed = time.time() - start_time
        logger.info(f"数据集生成完成，总用时: {elapsed:.1f}s")
        
        summary = self.stats.get_summary()
        logger.info(f"验证通过率: {summary['pass_rate']*100:.1f}%")
        
        return clean_list, noisy_list, metadata_list
    
    def save_dataset(
        self,
        clean_list: List[np.ndarray],
        noisy_list: List[np.ndarray],
        metadata_list: List[SampleMetadata],
        output_path: Optional[Path] = None
    ):
        """保存数据集到HDF5文件"""
        if output_path is None:
            output_path = self.config.output.output_dir / self.config.output.output_filename
        
        logger.info(f"保存数据集到: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            f.attrs['total_samples'] = len(clean_list)
            f.attrs['image_size'] = self.config.simulation.image_size
            f.attrs['creation_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            if self.config.output.save_clean_data:
                clean_group = f.create_group('clean')
                for i, clean in enumerate(clean_list):
                    clean_group.create_dataset(
                        f'sample_{i:05d}',
                        data=clean.astype(np.float32),
                        compression=self.config.output.compression,
                        compression_opts=self.config.output.compression_level
                    )
            
            if self.config.output.save_noisy_data:
                noisy_group = f.create_group('noisy')
                for i, noisy in enumerate(noisy_list):
                    noisy_group.create_dataset(
                        f'sample_{i:05d}',
                        data=noisy.astype(np.float32),
                        compression=self.config.output.compression,
                        compression_opts=self.config.output.compression_level
                    )
            
            if self.config.output.save_metadata:
                import json
                
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.integer):
                            return int(obj)
                        elif isinstance(obj, np.floating):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, np.bool_):
                            return bool(obj)
                        return super().default(obj)
                
                meta_group = f.create_group('metadata')
                meta_json = [asdict(meta) for meta in metadata_list]
                for i, meta_dict in enumerate(meta_json):
                    if meta_dict['flip_axis'] is None:
                        meta_dict['flip_axis'] = -1
                meta_group.create_dataset(
                    'all_metadata',
                    data=json.dumps(meta_json, cls=NumpyEncoder)
                )
            
            summary = self.stats.get_summary()
            stats_group = f.create_group('statistics')
            for key, value in summary.items():
                if isinstance(value, dict):
                    sub_group = stats_group.create_group(key)
                    for k, v in value.items():
                        sub_group.attrs[k] = v
                else:
                    stats_group.attrs[key] = value
        
        logger.info(f"数据集保存完成，文件大小: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def run_pipeline(config: Optional[Config] = None) -> Tuple[str, Dict[str, Any]]:
    """
    运行完整的数据生成流水线
    
    返回:
        output_path: 输出文件路径
        statistics: 统计信息
    """
    if config is None:
        config = get_test_config()
    
    pipeline = DiffractionDataPipeline(config)
    
    clean_list, noisy_list, metadata_list = pipeline.generate_dataset(
        visualize=config.validation.enabled,
        visualize_every=max(1, config.data_generation.total_samples // config.validation.visualize_samples)
    )
    
    output_path = config.output.output_dir / config.output.output_filename
    pipeline.save_dataset(clean_list, noisy_list, metadata_list, output_path)
    
    statistics = pipeline.stats.get_summary()
    
    return str(output_path), statistics


if __name__ == "__main__":
    print("=" * 60)
    print("衍射数据生成流水线测试")
    print("=" * 60)
    
    config = get_test_config()
    print(f"\n配置信息:")
    print(f"  总样本数: {config.data_generation.total_samples}")
    print(f"  图像尺寸: {config.simulation.image_size}")
    print(f"  曝光水平范围: {config.noise.exposure_level_range}")
    print(f"  输出目录: {config.output.output_dir}")
    
    output_path, stats = run_pipeline(config)
    
    print("\n" + "=" * 60)
    print("生成完成!")
    print("=" * 60)
    print(f"输出文件: {output_path}")
    print(f"通过率: {stats['pass_rate']*100:.1f}%")
    print(f"平均对比度: {stats['contrast']['mean']:.3f}")
    print(f"平均衰减比: {stats['decay_ratio']['mean']:.1f}")
