# -*- coding: utf-8 -*-
"""
实验数据噪声分析脚本
分析背景扣除后的衍射数据噪声水平，用于确定泊松噪声模拟参数 I_sc

数据说明:
- 实验数据: Ecoli_AB_20min_7897_background_subtraction_result.mat (背景扣除后的衍射数据)
- Mask数据: Ecoli_AB_20min_7897_20260322_140309_missing-latest.mat (beamstop区域mask)

分析日期: 2026-03-22
"""

import scipy.io as sio
import numpy as np
import os

def analyze_noise_level(data_file, mask_file):
    """
    分析实验数据的噪声水平
    
    Parameters:
    -----------
    data_file : str
        背景扣除后的衍射数据文件路径
    mask_file : str
        beamstop区域mask文件路径
    
    Returns:
    --------
    dict : 包含噪声分析结果的字典
    """
    print('='*70)
    print('背景扣除后数据噪声分析')
    print('='*70)
    
    data = sio.loadmat(data_file)['data_sb']
    mask = sio.loadmat(mask_file)['mask']
    
    I = data.astype(np.float64)
    valid_mask = (mask == 0)
    I_valid = I[valid_mask]
    
    h, w = I.shape
    print(f'\n图像尺寸: {h} x {w}')
    print(f'有效区域像素数: {np.sum(valid_mask):,}')
    print(f'beamstop区域像素数: {np.sum(mask):,}')
    
    print('\n' + '='*70)
    print('一、整体统计')
    print('='*70)
    
    print(f'\n有效区域统计:')
    print(f'  最小值: {np.min(I_valid):.4f}')
    print(f'  最大值: {np.max(I_valid):.4f}')
    print(f'  均值: {np.mean(I_valid):.4f}')
    print(f'  标准差: {np.std(I_valid):.4f}')
    print(f'  中位数: {np.median(I_valid):.4f}')
    
    neg_count = np.sum(I_valid < 0)
    zero_count = np.sum(I_valid == 0)
    print(f'\n负值像素数: {neg_count:,} ({neg_count/len(I_valid)*100:.2f}%)')
    print(f'零值像素数: {zero_count:,} ({zero_count/len(I_valid)*100:.2f}%)')
    
    print('\n' + '='*70)
    print('二、背景区域噪声分析')
    print('='*70)
    
    center_y, center_x = h // 2, w // 2
    
    background_regions = {
        '左上角': (slice(100, 400), slice(100, 400)),
        '右上角': (slice(100, 400), slice(w-400, w-100)),
        '左下角': (slice(h-400, h-100), slice(100, 400)),
        '右下角': (slice(h-400, h-100), slice(w-400, w-100)),
    }
    
    bg_stats = {}
    for name, (y_slice, x_slice) in background_regions.items():
        region = I[y_slice, x_slice]
        region_mask = mask[y_slice, x_slice]
        valid_region = region[region_mask == 0]
        
        mean_val = np.mean(valid_region)
        var_val = np.var(valid_region)
        std_val = np.std(valid_region)
        
        poisson_ratio = var_val / mean_val if mean_val != 0 else 0
        cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
        
        bg_stats[name] = {
            'mean': mean_val,
            'var': var_val,
            'std': std_val,
            'ratio': poisson_ratio,
            'cv': cv,
            'n_pixels': len(valid_region)
        }
        
        print(f'\n【{name}】')
        print(f'  像素数: {len(valid_region):,}')
        print(f'  均值: {mean_val:.4f}')
        print(f'  方差: {var_val:.4f}')
        print(f'  标准差: {std_val:.4f}')
        print(f'  变异系数 (CV): {cv:.4f}')
        print(f'  方差/均值: {poisson_ratio:.4f}')
    
    all_means = [s['mean'] for s in bg_stats.values()]
    all_vars = [s['var'] for s in bg_stats.values()]
    all_cvs = [s['cv'] for s in bg_stats.values()]
    
    avg_bg_mean = np.mean(all_means)
    avg_bg_var = np.mean(all_vars)
    avg_bg_cv = np.mean(all_cvs)
    
    print(f'\n背景区域汇总:')
    print(f'  平均均值: {avg_bg_mean:.4f}')
    print(f'  平均方差: {avg_bg_var:.4f}')
    print(f'  平均CV: {avg_bg_cv:.4f}')
    
    print('\n' + '='*70)
    print('三、泊松噪声参数估算')
    print('='*70)
    
    bg_corner = I[100:400, 100:400]
    bg_corner_valid = bg_corner[mask[100:400, 100:400] == 0]
    
    bg_mean = np.mean(bg_corner_valid)
    bg_std = np.std(bg_corner_valid)
    bg_var = np.var(bg_corner_valid)
    
    print(f'\n背景区域详细统计:')
    print(f'  均值: {bg_mean:.4f}')
    print(f'  标准差: {bg_std:.4f}')
    print(f'  方差: {bg_var:.4f}')
    print(f'  CV (std/mean): {bg_std/bg_mean:.4f}')
    
    print('\n【背景扣除的噪声传播】')
    print('  原始信号 S ~ Poisson(lambda)')
    print('  背景估计 B ~ Poisson(mu)')
    print('  背景扣除后: D = S - B')
    print('  对于纯背景区域: Var(D) = lambda + mu = 2*mu')
    
    original_bg_photons = bg_var / 2
    print(f'\n  观测方差 = {bg_var:.2f}')
    print(f'  原始背景光子数 mu = {original_bg_photons:.1f}')
    
    print('\n' + '='*70)
    print('四、I_sc 参数选择指南')
    print('='*70)
    
    total_signal = np.sum(I_valid)
    print(f'\n背景扣除后总强度: {total_signal:.4e}')
    
    bg_norm = bg_mean / total_signal
    print(f'背景像素归一化强度比例: {bg_norm:.6e}')
    
    print('\n假设背景像素归一化强度 = 1e-7 (典型值):')
    bg_frac = 1e-7
    
    I_sc_list = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
    print('\n  I_sc     | 背景光子数 | 背景sigma | 噪声水平')
    print('  ---------|------------|-----------|----------')
    
    for I_sc in I_sc_list:
        bg_photons = I_sc * bg_frac
        bg_sigma = np.sqrt(bg_photons)
        
        if bg_sigma < 1:
            level = '极高噪声'
        elif bg_sigma < 3:
            level = '高噪声'
        elif bg_sigma < 10:
            level = '中等噪声'
        elif bg_sigma < 30:
            level = '低噪声'
        else:
            level = '极低噪声'
        
        print(f'  {I_sc:>8.0e} | {bg_photons:>10.4f} | {bg_sigma:>9.2f} | {level}')
    
    print(f'\n实验数据背景 sigma = {bg_std:.1f}')
    print(f'匹配的 I_sc = {(bg_std**2)/bg_frac:.2e}')
    
    print('\n' + '='*70)
    print('五、最终建议')
    print('='*70)
    
    print(f'''
【推荐 I_sc 范围】

  1. 匹配实验数据噪声:
     I_sc = {(bg_std**2)/bg_frac:.2e}
     
  2. 论文目标场景 (低强度、高噪声):
     I_sc = 1e4 ~ 1e7
     
  3. 建议实验范围:
     I_sc = [1e4, 1e5, 1e6, 1e7, 1e8]
     覆盖从极高噪声到中等噪声
''')
    
    return {
        'bg_mean': bg_mean,
        'bg_std': bg_std,
        'bg_var': bg_var,
        'original_bg_photons': original_bg_photons,
        'total_signal': total_signal,
        'recommended_I_sc': (bg_std**2)/bg_frac
    }


if __name__ == '__main__':
    data_file = r'Ecoli_AB_20min_7897_background_subtraction_result.mat'
    mask_file = r'Ecoli_AB_20min_7897_20260322_140309_missing-latest.mat'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, data_file)
    mask_file = os.path.join(script_dir, mask_file)
    
    results = analyze_noise_level(data_file, mask_file)
