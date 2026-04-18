"""
主运行脚本

运行衍射数据模拟，生成训练数据集
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config, get_default_config, get_test_config
from pipeline import run_pipeline, DiffractionDataPipeline


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='生成X射线衍射模拟数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_simulation.py                    # 使用默认配置
  python run_simulation.py --samples 1000     # 生成1000个样本
  python run_simulation.py --size 1024        # 使用1024x1024图像尺寸
  python run_simulation.py --test             # 使用测试配置（小规模）
        """
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=None,
        help='总样本数（默认：配置文件中的值）'
    )
    
    parser.add_argument(
        '--size', '-s',
        type=int,
        default=None,
        help='图像尺寸（默认：512）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（默认：42）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件路径'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='使用测试配置（小规模，100个样本）'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='禁用可视化'
    )
    
    parser.add_argument(
        '--exposure-min',
        type=float,
        default=None,
        help='最小曝光水平'
    )
    
    parser.add_argument(
        '--exposure-max',
        type=float,
        default=None,
        help='最大曝光水平'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 70)
    print("X射线衍射数据模拟器")
    print("=" * 70)
    
    if args.test:
        config = get_test_config()
        print("\n使用测试配置")
    else:
        config = get_default_config()
    
    if args.samples is not None:
        config.data_generation.total_samples = args.samples
    if args.size is not None:
        config.simulation.image_size = args.size
    if args.seed is not None:
        config.simulation.random_seed = args.seed
    if args.output is not None:
        config.output.output_filename = args.output
    if args.no_visualize:
        config.validation.enabled = False
    if args.exposure_min is not None:
        config.noise.exposure_level_range = (
            args.exposure_min,
            config.noise.exposure_level_range[1]
        )
    if args.exposure_max is not None:
        config.noise.exposure_level_range = (
            config.noise.exposure_level_range[0],
            args.exposure_max
        )
    
    print("\n配置参数:")
    print(f"  总样本数: {config.data_generation.total_samples}")
    print(f"  图像尺寸: {config.simulation.image_size} x {config.simulation.image_size}")
    print(f"  随机种子: {config.simulation.random_seed}")
    print(f"  曝光水平范围: {config.noise.exposure_level_range}")
    print(f"  泊松噪声: {'启用' if config.noise.poisson_noise_enabled else '禁用'}")
    print(f"  高斯噪声: {'启用' if config.noise.gaussian_noise_enabled else '禁用'}")
    print(f"  坏像素: {'启用' if config.noise.bad_pixels_enabled else '禁用'}")
    print(f"  结构化噪声: {'启用' if config.noise.structured_noise_enabled else '禁用'}")
    print(f"  数据增强: 旋转={config.augmentation.rotation_enabled}, "
          f"翻转={config.augmentation.flip_enabled}")
    print(f"  输出目录: {config.output.output_dir}")
    print(f"  输出文件: {config.output.output_filename}")
    
    print("\n结构来源配置:")
    print(f"  EMDB: {config.structure_source.emdb_ratio*100:.0f}%")
    print(f"  程序化生成: {config.structure_source.procedural_ratio*100:.0f}%")
    print(f"  物理模型: {config.structure_source.physics_model_ratio*100:.0f}%")
    
    print("\n" + "-" * 70)
    print("开始生成数据...")
    print("-" * 70)
    
    output_path, statistics = run_pipeline(config)
    
    print("\n" + "=" * 70)
    print("生成完成!")
    print("=" * 70)
    print(f"\n输出文件: {output_path}")
    print(f"\n统计信息:")
    print(f"  总样本数: {statistics['total_samples']}")
    print(f"  通过验证: {statistics['passed_samples']}")
    print(f"  通过率: {statistics['pass_rate']*100:.1f}%")
    print(f"\n对比度统计:")
    print(f"  平均: {statistics['contrast']['mean']:.3f}")
    print(f"  标准差: {statistics['contrast']['std']:.3f}")
    print(f"  范围: [{statistics['contrast']['min']:.3f}, {statistics['contrast']['max']:.3f}]")
    print(f"\n衰减比统计:")
    print(f"  平均: {statistics['decay_ratio']['mean']:.1f}")
    print(f"  标准差: {statistics['decay_ratio']['std']:.1f}")
    print(f"  范围: [{statistics['decay_ratio']['min']:.1f}, {statistics['decay_ratio']['max']:.1f}]")
    print(f"\n强度统计:")
    print(f"  最大值平均: {statistics['intensity_max']['mean']:.0f}")
    print(f"  平均值平均: {statistics['intensity_mean']['mean']:.0f}")
    
    return output_path, statistics


if __name__ == "__main__":
    main()
