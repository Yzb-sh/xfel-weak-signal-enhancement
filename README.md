# DeepPhase-X

**XFEL 弱信号增强与相位恢复系统**

一个基于物理驱动的深度学习系统，用于 X 射线相干衍射成像（CDI）的弱信号去噪和相位恢复。

## 核心特性

- **生物衍射仿真引擎**: E. coli 2D投影密度图 → FFT衍射 → 泊松+高斯噪声 → Beamstop掩码
- **物理引导 U-Net**: 使用 PartialConv2d 和 CentroSymmetricConv 的去噪网络
- **相位恢复**: HIO/ER 算法用于结构重建
- **全面的评估指标**: R-factor、PRTF、PSNR、SSIM 等

## 安装

```bash
cd "DeepPhase-X (XFEL Weak Signal Enhancement & Phasing)"
pip install -r requirements.txt
```

## 快速开始

### 1. 生成生物衍射仿真数据集

```bash
# 小规模测试（10+5+5 样本）
python scripts/generate_bio_dataset.py --num_train 10 --num_val 5 --num_test 5 --output_file data/simulated/bio_test.h5

# 完整数据集（训练集10000 + 验证集1000 + 测试集1000）
python scripts/generate_bio_dataset.py --num_train 10000 --num_val 1000 --num_test 1000 --output_file data/simulated/bio_diffraction_v1.h5
```

### 2. 运行测试

```bash
python -m pytest tests/ -v
```

### 3. 训练模型

```bash
python scripts/main_pipeline.py train --data data/simulated/bio_diffraction_v1.h5 --checkpoint-dir experiments/exp_01
```

## 代码示例

```python
from src.simulation import (
    BioSampleGenerator, BioDiffractionSimulator,
    IntensityNormalizer, NoiseAndBeamstopApplier,
    preprocess_for_training
)
import numpy as np

# 1. 生成生物样品（E. coli 2D密度图）
bio_gen = BioSampleGenerator(seed=42)
obj = bio_gen.generate(seed=42)  # (585, 585) float32, values in [0, 1]

# 2. FFT衍射模拟
sim = BioDiffractionSimulator()
I_clean = sim.simulate(obj)  # (585, 585) 衍射强度

# 3. 强度归一化（泊松噪声前必须执行）
norm = IntensityNormalizer()
I_norm = norm.normalize(I_clean)  # sum = 1.0

# 4. 噪声 + Beamstop
noise_bs = NoiseAndBeamstopApplier(seed=42)
I_noisy, mask, I_sc, metadata = noise_bs.apply(I_norm)

# 5. 训练预处理
clean_norm, noisy_norm, mean, std = preprocess_for_training(I_clean, I_noisy)
```

## 项目结构

```
DeepPhase-X/
├── configs/                    # YAML 配置文件
│   ├── simulation_config.yaml  # 物理仿真参数
│   ├── model_config.yaml       # 模型架构参数
│   ├── gan_config.yaml         # NoiseGAN 参数
│   └── training_config.yaml    # 训练超参数
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始实验数据 & Beamstop掩码
│   │   └── beamstop_mask-585x585.mat
│   └── simulated/              # 生成的H5数据集
│
├── src/                        # 源代码
│   ├── simulation/             # 物理仿真 & 生物衍射仿真流水线
│   │   ├── simulator.py        # X射线衍射模拟 (PDB-based)
│   │   ├── noise_model.py      # 噪声模型 (AnalyticNoiseModel)
│   │   ├── beam_stop.py        # Beam Stop 工具
│   │   ├── bio_config.py       # 生物仿真配置 & 物理校验
│   │   ├── bio_sample_generator.py  # E. coli 2D密度图生成
│   │   ├── data_augmentor.py   # 数据增强（旋转/平移/缩放）
│   │   ├── bio_diffraction_simulator.py  # FFT衍射模拟 (bio)
│   │   ├── intensity_normalizer.py  # 强度归一化 (sum=1)
│   │   ├── random_mask_applier.py   # 随机探测器缺陷模拟
│   │   ├── noise_beamstop_applier.py  # 泊松+高斯噪声+Beamstop
│   │   └── bio_utils.py        # 验证/可视化/预处理工具
│   ├── models/                 # 深度学习模型
│   │   ├── layers.py           # PartialConv2d 等自定义层
│   │   ├── unet_physics.py     # 物理引导 U-Net
│   │   └── noise_gan.py        # NoiseGAN
│   ├── losses/                 # 损失函数
│   │   └── losses.py           # 物理引导损失函数
│   ├── evaluation/             # 评估指标
│   │   ├── metrics.py          # PSNR, SSIM, R-factor 等
│   │   └── visualization.py    # 可视化
│   └── reconstruction/         # 相位恢复算法
│       ├── hio_er.py           # HIO/ER 算法
│       └── support.py          # 支撑估计
│
├── scripts/                    # 执行脚本
│   ├── generate_bio_dataset.py # 生物衍射数据集生成（主入口）
│   ├── main_pipeline.py        # 主训练/推理流水线
│   ├── generate_diffraction_data.py  # PDB衍射数据生成
│   └── pdb_fetcher.py          # PDB文件下载
│
├── tests/                      # 测试文件夹
│   └── bio_diffraction_simulator/  # 仿真脚本（含参考代码和测试）
├── experiments/                # 实验结果
├── docs_guide/                 # 文档与实验方案
├── pyproject.toml              # 项目配置
└── requirements.txt            # Python 依赖
```

## 核心模块说明

| 模块 | 功能 | 主要类/函数 |
|------|------|------------|
| `src/simulation/` | 生物衍射仿真 | `BioSampleGenerator`, `BioDiffractionSimulator`, `NoiseAndBeamstopApplier` |
| `src/simulation/` | 噪声模型 | `AnalyticNoiseModel` |
| `src/models/` | 深度学习 | `PhysicsUNet`, `PartialConv2d` |
| `src/losses/` | 损失函数 | `PhysicsGuidedLoss` |
| `src/evaluation/` | 评估指标 | `DiffractionMetrics` |
| `src/reconstruction/` | 相位恢复 | `PhaseRetrieval`, `SupportEstimator` |

## 仿真流水线

数据生成的完整流水线：

1. **BioSampleGenerator** → E. coli 2D密度图 [0,1]（正常/分裂/弯曲三种形态）
2. **DataAugmentor** → 随机旋转(0-360°)、平移(±10%)、缩放(0.9-1.1x)
3. **BioDiffractionSimulator** → FFT衍射强度图
4. **IntensityNormalizer** → 归一化 sum=1（泊松噪声前必须执行）
5. **RandomMaskApplier** → 50%概率添加随机探测器缺陷
6. **NoiseAndBeamstopApplier** → 泊松噪声 + 高斯噪声 + Beamstop掩码
7. **Preprocessing** → log10(1+x) + 标准化 → HDF5

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 生物仿真流水线测试（在tests/bio_diffraction_simulator/目录下运行）
cd tests/bio_diffraction_simulator
python test_small_sample.py --num_samples 10
```

## 详细文档

- **实验方案**: [docs_guide/完整实验方案.md](docs_guide/完整实验方案.md)
- **使用指南**: [USAGE_GUIDE.md](USAGE_GUIDE.md)

## License

MIT License
