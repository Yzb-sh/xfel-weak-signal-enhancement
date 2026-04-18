# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import scipy
#import cv2
from scipy.signal import convolve2d
from skimage.feature import peak_local_max
#from scipy.stats import chisquare
import scipy.io as io
import os
import glob
import re
import matplotlib
# 强制使用非交互式后端（Headless模式）
# 必须在导入pyplot之前设置
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.path import Path
from tqdm import tqdm
import matplotlib.patches as patches
from itertools import combinations
#import spe_loader as sl
#import heapq

# --------- Headless模式说明 ---------
# 本脚本已改造为Headless模式，不再弹出任何GUI窗口
# 所有交互式功能已移除或改为非交互式实现
# 如需交互式功能，请使用前端Web界面

def _is_interactive_backend():
    """判断当前后端是否为交互式后端（Headless模式下始终返回False）"""
    return False

def _assert_interactive_backend():
    """Headless模式下不支持交互式操作"""
    raise RuntimeError(
        "当前运行在Headless模式下，不支持交互式GUI操作。\n"
        "请使用前端Web界面进行交互式操作。"
    )

# 条件导入重构类，避免MPI错误
try:
    from Class_CDI_OSS_MPI import CDIReconstructor
    RECONSTRUCTION_AVAILABLE = True
except ImportError as e:
    print(f"重构模块导入失败: {e}")
    print("重构功能将不可用，但数据预处理功能仍然可用")
    RECONSTRUCTION_AVAILABLE = False
    CDIReconstructor = None
import multiprocessing as mp
import time

matplotlib.rc("font", family='Microsoft YaHei')

# ---------------------- 通用路径解析函数 ----------------------
def resolve_file_path(file_path, default_dir='uploads'):
    """
    解析文件路径，处理各种路径格式
    
    参数:
        file_path: 输入的文件路径
        default_dir: 默认目录（如果路径不是绝对路径且不包含目录前缀）
    
    返回:
        完整的绝对路径
    """
    if not file_path:
        return None
    
    # 如果是绝对路径，直接返回
    if os.path.isabs(file_path):
        return file_path
    
    # 如果已经包含 uploads/ 或 outputs/ 前缀，相对于 cwd 解析
    if (file_path.startswith('uploads/') or file_path.startswith('uploads\\') or
        file_path.startswith('outputs/') or file_path.startswith('outputs\\')):
        return os.path.join(os.getcwd(), file_path)
    
    # 否则添加默认目录前缀
    return os.path.join(os.getcwd(), default_dir, file_path)

# ---------------------- 工具函数 ----------------------
class MultiROISelector:
    def __init__(self, image):
        self.image = image.copy()
        self.processed_image = self._prepare_log_display(image)
        self.coord_pairs = []
        self.rects = []
        self.current_rect = None
        self.current_coords = []
        self.done = False
        
        # 初始化图形界面
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.manager.set_window_title('Multi-ROI Selector')
        
        # 显示图像并设置等比例坐标轴
        self.img_display = self.ax.imshow(
            self.processed_image, 
            cmap='jet', 
            aspect='equal',  # 关键修改点1：设置等比例显示
            origin='upper'
        )
        self.ax.set_title("Left: 选择区域 | Right: 撤销 | Wheel: 缩放")
        
        # 事件绑定
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)  # 新增滚轮事件
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)

        # 初始化缩放参数
        self.zoom_factor = 1.0
        self.zoom_center = (self.image.shape[1]/2, self.image.shape[0]/2)  # (x, y)
        
        # 创建控制面板
        self._create_control_panel()
        self._create_reset_button()
        
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        
        # 新增重置按钮
        self._create_reset_button()
        
        # 绑定键盘事件
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        # 阻塞显示，等待用户操作完成
        plt.show()
        
    def _prepare_log_display(self, image):
        """预处理图像用于对数显示"""
        # 拷贝并处理数据
        display_img = image.astype(float).copy()

        display_img[display_img < 1] = 1
        
        # Step 3: 应用对数变换
        log_img = np.log(display_img)
        return log_img
    
    def _create_reset_button(self):
        self.ax_reset = plt.axes([0.55, 0.01, 0.12, 0.06])  # 调整按钮位置
        self.btn_reset = Button(self.ax_reset, '重置缩放')
        self.btn_reset.on_clicked(self.reset_view)
        
    def _create_control_panel(self):
        """创建底部控制按钮"""
        self.ax_undo = plt.axes([0.70, 0.01, 0.12, 0.06])
        self.btn_undo = Button(self.ax_undo, '撤销')
        self.btn_undo.on_clicked(self.undo_last)

        self.ax_done = plt.axes([0.83, 0.01, 0.12, 0.06])
        self.btn_done = Button(self.ax_done, '完成')
        self.btn_done.on_clicked(self.finish_selection)

    def _run_event_loop(self):
        """自定义事件循环"""
        while not self.done and plt.fignum_exists(self.fig.number):
            plt.pause(0.1)  # 降低CPU占用
        
    def on_scroll(self, event):
        """滚轮缩放处理"""
        if event.inaxes != self.ax:
            return
        # 计算缩放中心（数据坐标）
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        
        # 更新缩放参数
        zoom_step = 0.2
        if event.button == 'up':
            self.zoom_factor *= (1 - zoom_step)
        elif event.button == 'down':
            self.zoom_factor *= (1 + zoom_step)
        
        # 限制缩放范围
        self.zoom_factor = np.clip(self.zoom_factor, 0.1, 10)
        
        # 计算新显示范围
        width = self.image.shape[1] * self.zoom_factor
        height = self.image.shape[0] * self.zoom_factor
        
        # 保持中心点不变
        x_center = xdata
        y_center = ydata
        
        # 设置新的显示范围
        self.ax.set_xlim(x_center - width/2, x_center + width/2)
        self.ax.set_ylim(y_center + height/2, y_center - height/2)  # 注意Y轴方向
        # 强制重绘
        self.fig.canvas.draw_idle()
        
    def reset_view(self, event=None):
        """重置到初始视图"""
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.zoom_factor = 1.0
        self.fig.canvas.draw_idle()

    def onkey(self, event):
        """键盘事件处理（新增R键重置）"""
        if event.key in ['r', 'R']:  # 按R键重置
            self.reset_view()
        elif event.key == 'escape':
            self._cancel_current_selection()
            
    def _update_aspect(self):
        """维持等比例显示"""
        self.ax.set_aspect('equal', adjustable='datalim')
        self.fig.canvas.draw_idle()
        
    def _start_new_rect(self, x, y):
        """初始化新矩形"""
        self.current_rect = Rectangle(
            (x, y), 0, 0, 
            edgecolor='#FF69B4', lw=2,
            linestyle=':', 
            facecolor='none'
        )
        self.ax.add_patch(self.current_rect)
        self._update_aspect()  # 添加矩形后更新比例
        
    def on_close(self, event):
        """窗口关闭事件处理"""
        self.done = True

    def finish_selection(self, event):
        """完成选择"""    
        self.done = True
        plt.close(self.fig)

    def onclick(self, event):
        """鼠标点击处理"""
        if self.done or event.inaxes != self.ax:
            return

        # 左键选择坐标
        if event.button == 1:  
            x, y = int(event.xdata), int(event.ydata)
            self.current_coords.append((x, y))

            if len(self.current_coords) == 1:
                self._start_new_rect(x, y)
            elif len(self.current_coords) == 2:
                self._finalize_rect()

        # 右键撤销最后一步
        elif event.button == 3:  
            self.undo_last(None)

    def _finalize_rect(self):
        """完成当前矩形"""
        x1, y1 = self.current_coords[0]
        x2, y2 = self.current_coords[1]
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        # 更新矩形样式
        self.current_rect.set_bounds(xmin, ymin, xmax-xmin, ymax-ymin)
        self.current_rect.set_linestyle('-')
        self.current_rect.set_edgecolor('#00FF00')  # 实线绿色

        # 存储记录
        self.coord_pairs.append(((xmin, ymin), (xmax, ymax)))
        self.rects.append(self.current_rect)

        # 重置状态
        self.current_coords = []
        self.current_rect = None
        self.fig.canvas.draw()


    def _cancel_current_selection(self):
        """取消当前未完成的选择"""
        if self.current_rect:
            self.current_rect.remove()
            self.current_rect = None
            self.current_coords = []
            self.fig.canvas.draw()

    def undo_last(self, event):
        """撤销最后一步"""
        if self.rects:
            last_rect = self.rects.pop()
            last_rect.remove()
            self.coord_pairs.pop()
            self.fig.canvas.draw()

def sub_multi_select_roi(img):
    """
    非阻塞多ROI选择函数
    返回:
        mask: 选中区域标记为-2的掩码矩阵
    """
    selector = MultiROISelector(img)
    
    # 生成掩码矩阵
    mask = np.zeros_like(img, dtype=np.int8)
    if hasattr(selector, 'coord_pairs'):
        for (p1, p2) in selector.coord_pairs:
            (xmin, ymin), (xmax, ymax) = p1, p2
            mask[ymin:ymax+1, xmin:xmax+1] = -2
    
    return mask  
   
class CenterFinder_auto:
    def __init__(self, image, search_range=10, region_ratio=0.1):
        self.image = image
        self.processed_image = self._prepare_log_display(image)
        self.search_range = search_range
        self.region_ratio = region_ratio  # 区域边长比例
        self.initial_center = None
        self.roi_mask = None
        self.best_center = None
        
    def _prepare_log_display(self, image):
        """预处理图像用于对数显示"""
        # 拷贝并处理数据
        display_img = image.astype(float).copy()

        display_img[display_img < 1] = 1
        
        # Step 3: 应用对数变换
        log_img = np.log(display_img)
        return log_img

   
    def _auto_detect_missing_region(self):
        """自动检测缺失数据区域并定位最大内切圆中心"""
        # 创建缺失区域掩码
        missing_mask = (self.image == -1).astype(np.uint8)
        
        # 寻找最大连通区域
        from skimage.measure import label, regionprops
        labeled = label(missing_mask)
        regions = regionprops(labeled)
        
        if not regions:
            raise ValueError("未检测到缺失数据区域")
        
        largest_region = max(regions, key=lambda x: x.area)
        y0, x0, y1, x1 = largest_region.bbox
        sub_mask = missing_mask[y0:y1, x0:x1]

        # 计算距离变换（找到最大内切圆）
        from scipy.ndimage import distance_transform_edt
        dist_transform = distance_transform_edt(sub_mask)
        
        # 找到最大距离点（局部最大值）
        max_coords = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
        radius = dist_transform[max_coords]
        cy_sub, cx_sub = max_coords
        
        # 转换到原始坐标系
        cy = y0 + cy_sub
        cx = x0 + cx_sub

        return int(cy), int(cx)

    def _auto_generate_symmetry_regions(self, center):
        """在局部区域内选取所有局部最大值，并生成匹配区域，仅可视化中心±500视场"""
        
        h, w = self.image.shape
        region_size = 150
        min_distance = 20
        n_peaks = 5
        max_radius = 500
    
        center_y, center_x = center
    
        # 图像预处理
        image = self.image.copy()
        image[image <= 0] = np.nan
    
        # 创建距离掩码，限制peak搜索范围
        y_coords, x_coords = np.ogrid[:h, :w]
        distance_mask = ((y_coords - center_y)**2 + (x_coords - center_x)**2) <= max_radius**2
        image_masked = image.copy()
        image_masked[~distance_mask] = 0
    
        # 寻找局部最大值
        coordinates = peak_local_max(
            image, 
            min_distance=min_distance,
            threshold_abs=10000,#threshold_abs,
            num_peaks=n_peaks
        )
        
        # 如果没有找到足够的高值点，降低阈值
        if len(coordinates) < n_peaks:
            coordinates = peak_local_max(
                image, 
                min_distance=min_distance,
                threshold_abs=5000,
                num_peaks=n_peaks
            )
            
        if len(coordinates) < 2:
            print("局部最大值太少，终止")
            exit()
    
        regions = []
        for y, x in coordinates:
            y0 = max(0, y - region_size // 2)
            x0 = max(0, x - region_size // 2)
            y1 = min(h, y0 + region_size)
            x1 = min(w, x0 + region_size)
            regions.append(((y0, x0), (y1, x1)))
    
        # 可视化视场区域
        v_half = 500
        v_y0 = max(0, center_y - v_half)
        v_x0 = max(0, center_x - v_half)
        v_y1 = min(h, center_y + v_half)
        v_x1 = min(w, center_x + v_half)
    
        view_img = self.image[v_y0:v_y1, v_x0:v_x1]
    
        plt.figure()
        plt.imshow(np.log(view_img + 3), cmap='jet')
    
        # 显示坐标点：只显示在视场内的
        for y, x in coordinates:
            if v_y0 <= y < v_y1 and v_x0 <= x < v_x1:
                plt.scatter(x - v_x0, y - v_y0, c='red', s=40, marker='x')
    
        # 显示regions（匹配视场部分）
        for (y0, x0), (y1, x1) in regions:
            if y1 < v_y0 or y0 > v_y1 or x1 < v_x0 or x0 > v_x1:
                continue  # 区域完全在视场外，不画
            # 计算相对位置
            ry0 = y0 - v_y0
            rx0 = x0 - v_x0
            height = y1 - y0
            width = x1 - x0
            rect = Rectangle((rx0, ry0), width, height,
                             edgecolor='cyan', facecolor='none', linewidth=1.5)
            plt.gca().add_patch(rect)
    
        plt.title("Peaks and Regions (Zoomed View)")
        plt.xlim([0, v_x1 - v_x0])
        plt.ylim([v_y1 - v_y0, 0])
        # 在非交互式模式下不显示图像
        if os.environ.get('NON_INTERACTIVE', 'false').lower() != 'true':
            plt.show()
            plt.pause(5)
        plt.close()
        return regions
    
    def _calculate_error(self, candidate_center):
        """计算对称误差（动态匹配有效区域）"""
        error_sum = 0.0
        valid_pixels = 0
        
        for (p1, p2) in self.roi_mask:
            # 原始区域坐标
            (xmin, ymin), (xmax, ymax) = p1, p2
            
            # 生成对称区域坐标（带边界保护）
            cy, cx = candidate_center
            sym_xmin = max(0, 2*cx - xmax)
            sym_xmax = min(self.image.shape[1]-1, 2*cx - xmin)
            sym_ymin = max(0, 2*cy - ymax)
            sym_ymax = min(self.image.shape[0]-1, 2*cy - ymin)
            
            # 提取有效区域（带边界检查）
            orig_region = self.image[ymin:ymax+1, xmin:xmax+1]
            sym_region = self.image[sym_ymin:sym_ymax+1, sym_xmin:sym_xmax+1]
            
            # 生成有效掩码（排除-1且>0）
            orig_mask = (orig_region != -1) & (orig_region > 0)
            sym_mask = (sym_region != -1) & (sym_region > 0)
            sym_mask = sym_mask[::-1, ::-1]
            # 创建对齐网格（动态调整形状）
            min_rows = min(orig_region.shape[0], sym_region.shape[0])
            min_cols = min(orig_region.shape[1], sym_region.shape[1])
            
            # 裁剪到共同区域
            orig_cropped = orig_region[:min_rows, :min_cols]
            sym_cropped = sym_region[:min_rows, :min_cols]
            sym_cropped = sym_cropped[::-1, ::-1]
            mask_intersection = orig_mask[:min_rows, :min_cols] & sym_mask[:min_rows, :min_cols]
    
            # 计算有效误差
            if np.any(mask_intersection):
                sqrt_orig = np.sqrt(orig_cropped[mask_intersection])
                sqrt_sym = np.sqrt(sym_cropped[mask_intersection])
                error_sum += np.sum(np.abs(sqrt_orig - sqrt_sym))
                valid_pixels += np.sum(mask_intersection)
        return error_sum / valid_pixels if valid_pixels > 0 else np.inf

    def optimize_center(self):
        """自动优化流程"""
        # Step 1: 自动检测初始中心
        try:
            self.initial_center = self._auto_detect_missing_region()
        except Exception as e:
            raise ValueError("自动检测初始中心失败，请检查缺失数据区域") from e
        
        # Step 2: 自动生成检测区域
        self.roi_mask = self._auto_generate_symmetry_regions(self.initial_center)
        if not self.roi_mask:
            raise ValueError("自动生成检测区域失败")
        
        # Step 3: 搜索最佳中心
        min_error = np.inf
        best_center = self.initial_center
        error_map = np.full((2*self.search_range+1, 2*self.search_range+1), np.nan)
        
        # 创建进度条
        total_points = (2*self.search_range+1)**2
        with tqdm(total=total_points, desc="优化中心") as pbar:
            for dy in range(-self.search_range, self.search_range+1):
                for dx in range(-self.search_range, self.search_range+1):
                    candidate_y = self.initial_center[0] + dy
                    candidate_x = self.initial_center[1] + dx
                    
                    # 边界检查
                    if not (0 <= candidate_y < self.image.shape[0] and 
                            0 <= candidate_x < self.image.shape[1]):
                        continue
                    
                    # 计算误差
                    current_error = self._calculate_error((candidate_y, candidate_x))
                    error_map[dy+self.search_range, dx+self.search_range] = current_error
                    
                    if current_error < min_error:
                        min_error = current_error
                        best_center = (candidate_y, candidate_x)
                    pbar.update(1)
        
        
        print(f"初始中心: {self.initial_center}")
        print(f"优化中心: {best_center}")
        print(f"最小误差: {min_error:.4f}")
        
        return best_center, error_map

class CenterFinder:
    def __init__(self, image, search_range=10):
        self.image = image
        self.processed_image = self._prepare_log_display(image)
        self.search_range = search_range
        self.initial_center = None
        self.roi_mask = None
        self.best_center = None
        
        # 新增缩放相关参数
        self.zoom_factor = 1.0
        self.original_xlim = None
        self.original_ylim = None
        self.ax = None  # 新增绘图轴引用
        
    def _prepare_log_display(self, image):
        """预处理图像用于对数显示"""
        # 拷贝并处理数据
        display_img = image.astype(float).copy()

        display_img[display_img < 1] = 1
        
        # Step 3: 应用对数变换
        log_img = np.log(display_img)
        return log_img
    
    def _get_initial_center(self):
        """交互式获取初始中心点（含滚轮缩放功能）"""
        fig, self.ax = plt.subplots()
        self.ax.imshow(self.processed_image, cmap='jet', aspect='equal')
        self.ax.set_title("Left Click: Select Center | Wheel: Zoom | R: Reset")
        
        # 保存初始视图范围
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        
        # 事件绑定
        fig.canvas.mpl_connect('button_press_event', self._on_click)
        fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        fig.canvas.mpl_connect('key_press_event', self._on_key)
        fig.canvas.mpl_connect('close_event', self._on_close)
        
        # 添加重置按钮
        self._add_reset_button(fig)
        
        # 非阻塞事件循环
        self.initial_center = None
        self._run_event_loop(fig)
        return self.initial_center

    def _add_reset_button(self, fig):
        """添加重置视图按钮"""
        ax_reset = plt.axes([0.82, 0.01, 0.15, 0.06])
        btn_reset = Button(ax_reset, 'Reset View (R)')
        btn_reset.on_clicked(lambda e: self._reset_view())

    def _run_event_loop(self, fig):
        """自定义非阻塞事件循环"""
        while plt.fignum_exists(fig.number) and self.initial_center is None:
            plt.pause(0.1)

    def _on_click(self, event):
        """鼠标点击事件处理"""
        if event.inaxes == self.ax and event.button == 1:
            y = int(np.clip(event.ydata, 0, self.image.shape[0]-1))
            x = int(np.clip(event.xdata, 0, self.image.shape[1]-1))
            self.initial_center = (y, x)
            plt.close()

    def _on_scroll(self, event):
        """滚轮缩放处理"""
        if event.inaxes != self.ax:
            return

        # 计算缩放因子
        zoom_step = 0.15
        base_scale = 1 + zoom_step if event.button == 'up' else 1/(1 + zoom_step)
        new_zoom = self.zoom_factor * base_scale
        new_zoom = np.clip(new_zoom, 0.5, 5.0)

        # 计算缩放中心
        x_center = event.xdata
        y_center = event.ydata
        if x_center is None or y_center is None:
            return

        # 计算新显示范围
        x_range = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / base_scale
        y_range = (self.ax.get_ylim()[0] - self.ax.get_ylim()[1]) / base_scale

        # 应用缩放
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center + y_range/2, y_center - y_range/2)
        self.zoom_factor = new_zoom
        self.ax.figure.canvas.draw_idle()

    def _on_key(self, event):
        """键盘事件处理"""
        if event.key in ['r', 'R']:
            self._reset_view()

    def _on_close(self, event):
        """窗口关闭事件处理（修复版）"""
        # 仅在未选择中心点时显示提示
        if self.initial_center is None:
            print("Window closed without selection!")
            
    def _reset_view(self):
        """重置视图到初始状态"""
        if self.original_xlim and self.original_ylim:
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.zoom_factor = 1.0
            self.ax.figure.canvas.draw_idle()

    def _select_roi_regions(self):
        """选择对称比较区域"""
        selector = MultiROISelector(self.image)
        self.roi_mask = selector.coord_pairs
        return self.roi_mask

    def _calculate_error(self, candidate_center):
        """计算对称误差（动态匹配有效区域）"""
        error_sum = 0.0
        valid_pixels = 0
        
        for (p1, p2) in self.roi_mask:
            # 原始区域坐标
            (xmin, ymin), (xmax, ymax) = p1, p2
            
            # 生成对称区域坐标（带边界保护）
            cy, cx = candidate_center
            sym_xmin = max(0, 2*cx - xmax)
            sym_xmax = min(self.image.shape[1]-1, 2*cx - xmin)
            sym_ymin = max(0, 2*cy - ymax)
            sym_ymax = min(self.image.shape[0]-1, 2*cy - ymin)
            
            # 提取有效区域（带边界检查）
            orig_region = self.image[ymin:ymax+1, xmin:xmax+1]
            sym_region = self.image[sym_ymin:sym_ymax+1, sym_xmin:sym_xmax+1]
            
            # 生成有效掩码（排除-1且>0）
            orig_mask = (orig_region != -1) & (orig_region > 0)
            sym_mask = (sym_region != -1) & (sym_region > 0)
            sym_mask = sym_mask[::-1, ::-1]
            # 创建对齐网格（动态调整形状）
            min_rows = min(orig_region.shape[0], sym_region.shape[0])
            min_cols = min(orig_region.shape[1], sym_region.shape[1])
            
            # 裁剪到共同区域
            orig_cropped = orig_region[:min_rows, :min_cols]
            sym_cropped = sym_region[:min_rows, :min_cols]
            sym_cropped = sym_cropped[::-1, ::-1]
            mask_intersection = orig_mask[:min_rows, :min_cols] & sym_mask[:min_rows, :min_cols]

            # 计算有效误差
            if np.any(mask_intersection):
                sqrt_orig = np.sqrt(orig_cropped[mask_intersection])
                sqrt_sym = np.sqrt(sym_cropped[mask_intersection])
                error_sum += np.sum(np.abs(sqrt_orig - sqrt_sym))
                valid_pixels += np.sum(mask_intersection)
        return error_sum / valid_pixels if valid_pixels > 0 else np.inf

    def optimize_center(self):
        """主优化流程"""
        print("手动寻找中心")
        # Step 1: 获取初始中心
        self._get_initial_center()
        if not self.initial_center:
            raise ValueError("No initial center selected!")
        
        # Step 2: 选择ROI区域
        self._select_roi_regions()
        if not self.roi_mask:
            raise ValueError("No ROI regions selected!")
        
        # Step 3: 在搜索范围内遍历候选中心
        min_error = np.inf
        best_center = self.initial_center
        error_map = np.zeros((2*self.search_range+1, 2*self.search_range+1))
        
        for dy in range(-self.search_range, self.search_range+1):
            for dx in range(-self.search_range, self.search_range+1):
                candidate_y = self.initial_center[0] + dy
                candidate_x = self.initial_center[1] + dx
                
                # 边界检查
                if (candidate_y < 0 or candidate_y >= self.image.shape[0] or
                    candidate_x < 0 or candidate_x >= self.image.shape[1]):
                    continue
                
                # 计算误差
                current_error = self._calculate_error((candidate_y, candidate_x))
                error_map[dy+self.search_range, dx+self.search_range] = current_error
                
                if current_error < min_error:
                    min_error = current_error
                    best_center = (candidate_y, candidate_x)
        
        
        print(f"Optimized Center (y,x): {best_center}")
        print(f"Minimum Error: {min_error:.4f}")
        
        return best_center, error_map

class MultiPolygonROISelector:
    def __init__(self, image):
        self.image = image
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # 使用与网页显示一致的彩色映射
        self.ax.imshow(image, cmap='jet', extent=[0, image.shape[1], image.shape[0], 0])
        self.ax.set_title("Left: 添加点 | Right: 撤销 | Middle: 完成当前 | 滚轮缩放")
        
        # 存储所有多边形数据
        self.all_polygons = []          # 所有完整多边形
        self.current_poly = []          # 当前正在绘制的多边形
        self.current_line = None        # 当前绘制线段
        self.poly_patches = []          # 所有多边形覆盖层
        self.point_plots = []           # 当前多边形的点标记

        # 视图状态参数
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        self.zoom_factor = 1.0

        # 创建控制面板
        self._create_control_panel()

        # 事件绑定
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)

        # 启动非阻塞事件循环
        plt.show(block=True)

    def _create_control_panel(self):
        """创建底部控制按钮面板"""
        btn_params = {
            'width': 0.10,
            'height': 0.05,
            'verticalalignment': 'bottom'
        }
        
        # 调整按钮位置，为新按钮腾出空间
        self.ax_re = plt.axes([0.18, 0.01, btn_params['width'], btn_params['height']])
        self.btn_re = Button(self.ax_re, '重置缩放(R)')
        self.btn_re.on_clicked(self.reset_view)
        
        # 新建多边形按钮
        self.ax_new = plt.axes([0.31, 0.01, btn_params['width'], btn_params['height']])
        self.btn_new = Button(self.ax_new, '完成并新建(N)')
        self.btn_new.on_clicked(self.start_new_polygon)
        
        # 撤销点按钮
        self.ax_undop = plt.axes([0.44, 0.01, btn_params['width'], btn_params['height']])
        self.btn_undop = Button(self.ax_undop, '撤销点(鼠标右键)')
        self.btn_undop.on_clicked(self.undo_last_point)
        
        # 撤销ROI按钮
        self.ax_undo = plt.axes([0.57, 0.01, btn_params['width'], btn_params['height']])
        self.btn_undo = Button(self.ax_undo, '撤销ROI(U)')
        self.btn_undo.on_clicked(self.undo_last_action)        
        
        # 新增：缩小一半并完成按钮
        self.ax_half = plt.axes([0.70, 0.01, btn_params['width'] * 1.2, btn_params['height']])
        self.btn_half = Button(self.ax_half, '缩小一半并完成(H)')
        self.btn_half.on_clicked(self.half_size_and_finish)
        
        # 完成全部按钮
        self.ax_done = plt.axes([0.84, 0.01, btn_params['width'], btn_params['height']])
        self.btn_done = Button(self.ax_done, '完成(D)')
        self.btn_done.on_clicked(self.finish_all)

    def _run_event_loop(self):
        """自定义非阻塞事件循环"""
        # 强制显示窗口
        plt.show(block=True)

    def on_scroll(self, event):
        """处理鼠标滚轮缩放事件"""
        if event.inaxes != self.ax:
            return

        # 计算缩放因子
        zoom_step = 0.15
        base_scale = 1 + zoom_step if event.button == 'up' else 1/(1 + zoom_step)
        new_zoom = self.zoom_factor * base_scale
        new_zoom = np.clip(new_zoom, 0.5, 5.0)

        # 计算新显示范围
        x_center = event.xdata
        y_center = event.ydata
        if x_center is None or y_center is None:
            return

        x_range = (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) / base_scale
        y_range = (self.ax.get_ylim()[0] - self.ax.get_ylim()[1]) / base_scale

        # 应用缩放
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center + y_range/2, y_center - y_range/2)
        self.zoom_factor = new_zoom
        self.fig.canvas.draw_idle()

    def reset_view(self, event=None):
        """重置视图到初始状态"""
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        self.zoom_factor = 1.0
        self.fig.canvas.draw_idle()

    def onclick(self, event):
        """处理鼠标点击事件"""
        if event.inaxes != self.ax:
            return

        # 左键添加点
        if event.button == 1:
            self._handle_left_click(event)
            
        # 中键完成当前多边形
        elif event.button == 2:
            self._finish_current_polygon()
            
        # 右键撤销最后点
        elif event.button == 3:
            self.undo_last_point()

    def _handle_left_click(self, event):
        """处理左键点击添加点"""
        x = int(np.clip(event.xdata, 0, self.image.shape[1]-1))
        y = int(np.clip(event.ydata, 0, self.image.shape[0]-1))

        # 检查是否闭合多边形
        if len(self.current_poly) >= 3 and self._check_closure((x, y)):
            self._finish_current_polygon()
            return

        # 添加新点
        self.current_poly.append((x, y))
        
        # 绘制点标记
        point = self.ax.plot(x, y, 'ro', markersize=6, alpha=0.7)[0]
        self.point_plots.append(point)
        
        # 更新连线
        self._update_current_line()
        self.fig.canvas.draw_idle()

    def _check_closure(self, new_point):
        """检查是否闭合多边形"""
        first_point = self.current_poly[0]
        distance = np.hypot(new_point[0]-first_point[0], new_point[1]-first_point[1])
        return distance < 10  # 10像素内视为闭合

    def _update_current_line(self):
        """更新当前多边形连线"""
        # 移除旧线段
        if self.current_line:
            self.current_line.remove()
            self.current_line = None
            
        # 绘制新线段
        if len(self.current_poly) >= 2:
            x = [p[0] for p in self.current_poly]
            y = [p[1] for p in self.current_poly]
            self.current_line, = self.ax.plot(x, y, 'r-', lw=1.5, alpha=0.7)
            
        # 闭合多边形
        if len(self.current_poly) >= 3:
            x = x + [self.current_poly[0][0]]
            y = y + [self.current_poly[0][1]]
            self.current_line.set_data(x, y)

    def _finish_current_polygon(self):
        """完成当前多边形"""
        if len(self.current_poly) >= 3:
            # 存储多边形数据
            self.all_polygons.append(self.current_poly.copy())
            
            # 创建覆盖层
            poly_patch = patches.Polygon(self.current_poly,
                                        closed=True,
                                        edgecolor='gold',
                                        facecolor='yellow',
                                        alpha=0.3)
            self.ax.add_patch(poly_patch)
            self.poly_patches.append(poly_patch)
            
            # 重置当前状态
            self.current_poly.clear()
            self._clear_current_elements()
            
            # 更新标题
            self.ax.set_title(f"已选择 {len(self.all_polygons)} 个多边形")
            self.fig.canvas.draw_idle()

    def _clear_current_elements(self):
        """清除当前绘制元素"""
        if self.current_line:
            self.current_line.remove()
            self.current_line = None
        for point in self.point_plots:
            point.remove()
        self.point_plots.clear()

    def start_new_polygon(self, event=None):
        """开始新多边形绘制"""
        if self.current_poly:
            self._finish_current_polygon()
        self.ax.set_title("正在绘制新多边形...")
        self.fig.canvas.draw_idle()

    def undo_last_point(self, event=None):
        """撤销最后添加的点"""
        if self.current_poly:
            self.current_poly.pop()
            # 移除最后一个点标记
            if self.point_plots:
                self.point_plots.pop().remove()
            self._update_current_line()
            self.fig.canvas.draw_idle()

    def undo_last_action(self, event=None):
        """撤销上一步完整操作"""
        if self.all_polygons:
            # 移除最后一个多边形
            self.all_polygons.pop()
            # 移除对应的覆盖层
            if self.poly_patches:
                self.poly_patches.pop().remove()
            self.ax.set_title(f"已撤销，剩余 {len(self.all_polygons)} 个多边形")
            self.fig.canvas.draw_idle()

    def finish_all(self, event=None):
        """完成所有选择并保存support文件"""
        if self.current_poly:
            self._finish_current_polygon()
        
        # 生成support掩码
        support_mask = self.get_combined_mask()
        support = support_mask.astype(np.float64)
        
        # 保存support文件
        output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_support_path = os.path.join(output_dir, 'manual_support.mat')
        
        try:
            scipy.io.savemat(output_support_path, {'support': support})
            print(f"[SUCCESS] Support文件已保存: {output_support_path}")
            print(f"[INFO] Support形状: {support.shape}")
            print(f"[INFO] Support中1的数量: {np.sum(support == 1)}")
        except Exception as e:
            print(f"[ERROR] 保存support文件失败: {e}")
        
        plt.close(self.fig)

    def half_size_and_finish(self, event=None):
        """缩小一半并完成选择"""
        if self.current_poly:
            self._finish_current_polygon()
        
        if not self.all_polygons:
            print("没有多边形可以缩小")
            return
            
        # 计算缩放前所有点的中心坐标
        all_points = []
        for poly in self.all_polygons:
            all_points.extend(poly)
        
        # 计算所有点的中心
        center_x = sum(point[0] for point in all_points) / len(all_points)
        center_y = sum(point[1] for point in all_points) / len(all_points)
        
        print(f"缩放前中心坐标: ({center_x:.2f}, {center_y:.2f})")
        
        # 缩小所有多边形并调整位置
        scaled_polygons = []
        for poly in self.all_polygons:
            scaled_poly = []
            for x, y in poly:
                # 缩小一半
                scaled_x = x * 0.5
                scaled_y = y * 0.5
                
                # 调整位置，使缩放后的点放回图像中部
                adjusted_x = scaled_x + center_x * 0.5
                adjusted_y = scaled_y + center_y * 0.5
                
                scaled_poly.append((adjusted_x, adjusted_y))
            scaled_polygons.append(scaled_poly)
        
        # 更新多边形列表为缩放后的坐标
        self.all_polygons = scaled_polygons
        
        # 更新视图中的多边形覆盖层
        self._update_polygon_patches()
        
        # 生成并保存support文件
        support_mask = self.get_combined_mask()
        support = support_mask.astype(np.float64)
        
        # 保存support文件
        output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_support_path = os.path.join(output_dir, 'manual_support.mat')
        
        try:
            scipy.io.savemat(output_support_path, {'support': support})
            print(f"[SUCCESS] 缩放后的Support文件已保存: {output_support_path}")
            print(f"[INFO] Support形状: {support.shape}")
            print(f"[INFO] Support中1的数量: {np.sum(support == 1)}")
        except Exception as e:
            print(f"[ERROR] 保存support文件失败: {e}")
        
        # 关闭窗口
        plt.close(self.fig)

    def onkey(self, event):
        """处理键盘事件"""
        if event.key in ['n', 'N']:
            self.start_new_polygon()
        elif event.key in ['d', 'D']:
            self.finish_all()
        elif event.key in ['h', 'H']:  # 新增键盘快捷键
            self.half_size_and_finish()
        elif event.key in ['u', 'U']:
            self.undo_last_action()
        elif event.key in ['r', 'R']:
            self.reset_view()
        elif event.key == 'escape':
            self.undo_last_point()

    def on_close(self, event):
        """处理窗口关闭事件"""
        if self.current_poly:
            self._finish_current_polygon()

    def get_combined_mask(self):
        """生成合并所有多边形的二值掩码"""
        h, w = self.image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # 生成网格坐标
        y_coords, x_coords = np.mgrid[:h, :w]
        points = np.vstack((x_coords.flatten(), y_coords.flatten())).T
        
        # 合并所有多边形区域
        for poly in self.all_polygons:
            path = Path(poly)
            mask |= path.contains_points(points).reshape(h, w).astype(np.uint8)
            
        return mask

    def get_polygons(self):
        """获取所有多边形坐标列表"""
        return [poly.copy() for poly in self.all_polygons]
    
    def _update_polygon_patches(self):
        """更新多边形覆盖层以反映缩放后的坐标"""
        # 清除所有现有覆盖层
        for patch in self.poly_patches:
            patch.remove()
        self.poly_patches = []
        
        # 使用缩放后的坐标重新绘制所有多边形
        for poly in self.all_polygons:
            poly_patch = patches.Polygon(poly,
                                        closed=True,
                                        edgecolor='gold',
                                        facecolor='yellow',
                                        alpha=0.3)
            self.ax.add_patch(poly_patch)
            self.poly_patches.append(poly_patch)
        
        # 重绘图像
        self.fig.canvas.draw_idle()
    
    def get_half_size_polygons(self):
        """获取缩小一半的多边形坐标列表"""
        if not self.all_polygons:
            return []
        
        # 计算缩放前所有点的中心坐标
        all_points = []
        for poly in self.all_polygons:
            all_points.extend(poly)
        
        center_x = sum(point[0] for point in all_points) / len(all_points)
        center_y = sum(point[1] for point in all_points) / len(all_points)
        
        # 创建缩小一半的多边形列表
        half_size_polygons = []
        for poly in self.all_polygons:
            scaled_poly = []
            for x, y in poly:
                scaled_x = x * 0.5 + center_x * 0.5
                scaled_y = y * 0.5 + center_y * 0.5
                scaled_poly.append((scaled_x, scaled_y))
            half_size_polygons.append(scaled_poly)
        
        return half_size_polygons

def auto_select_bg_region(pattern, quadrant, edge_ratio=0.2, num_samples=5):
    """
    自动选择背景区域
    quadrant: 当前分区的方位标识 ('11':左上, '12':右上, '21':左下, '22':右下)
    edge_ratio: 边缘区域占比（默认1/5=0.2）
    num_samples: 每个边缘方向采样数
    """
    h, w = pattern.shape
    edge_w = int(w * edge_ratio)
    edge_h = int(h * edge_ratio)
    
    candidate_regions = []
    
    # 根据分区位置确定候选区域
    if quadrant == '11':  # 左上分区
        # 左边缘（左侧edge_w宽度，全高）
        x_range = (0, edge_w)
        y_range = (0, h)
        for i in range(num_samples):
            y0 = np.random.randint(0, h-edge_h)
            x0 = x_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
        
        # 上边缘（上侧edge_h高度，全宽）
        x_range = (0, w)
        y_range = (h-edge_h, h)
        for i in range(num_samples):
            x0 = np.random.randint(0, w-edge_w)
            y0 = y_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
            
    elif quadrant == '12':  # 右上分区
        # 右边缘（右侧edge_w宽度，全高）
        x_range = (w-edge_w, w)
        y_range = (0, h)
        for i in range(num_samples):
            y0 = np.random.randint(0, h-edge_h)
            x0 = x_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
        
        # 上边缘（上侧edge_h高度，全宽）
        x_range = (0, w)
        y_range = (h-edge_h, h)
        for i in range(num_samples):
            x0 = np.random.randint(0, w-edge_w)
            y0 = y_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
            
    elif quadrant == '21':  # 左下分区
        # 左边缘（左侧edge_w宽度，全高）
        x_range = (0, edge_w)
        y_range = (0, h)
        for i in range(num_samples):
            y0 = np.random.randint(0, h-edge_h)
            x0 = x_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
        
        # 下边缘（下侧edge_h高度，全宽）
        x_range = (0, w)
        y_range = (0, edge_h)
        for i in range(num_samples):
            x0 = np.random.randint(0, w-edge_w)
            y0 = y_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
            
    elif quadrant == '22':  # 右下分区
        # 右边缘（右侧edge_w宽度，全高）
        x_range = (w-edge_w, w)
        y_range = (0, h)
        for i in range(num_samples):
            y0 = np.random.randint(0, h-edge_h)
            x0 = x_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
        
        # 下边缘（下侧edge_h高度，全宽）
        x_range = (0, w)
        y_range = (0, edge_h)
        for i in range(num_samples):
            x0 = np.random.randint(0, w-edge_w)
            y0 = y_range[0]
            candidate_regions.append( (x0, y0, edge_w, edge_h) )
    
    # 评估所有候选区域
    min_sum = np.inf
    best_region = None
    for reg in candidate_regions:
        x0, y0, rw, rh = reg
        sub = pattern[y0:y0+rh, x0:x0+rw]
        valid_pixels = sub[sub != -1]
        if len(valid_pixels) > 0:
            current_sum = np.sum(valid_pixels)
            if current_sum < min_sum:
                min_sum = current_sum
                best_region = reg
                
    return best_region

def Sub_BG(pattern, BG, quadrant, edge_ratio=0.2):
    """修改后的背景减除函数，增加调试信息"""
    # 自动选择背景区域
    bg_region = auto_select_bg_region(pattern, quadrant, edge_ratio)
    if bg_region is None:
        raise ValueError("无法自动找到合适的背景区域")
    
    # 生成mask
    mask = np.zeros_like(pattern, dtype=np.int8)
    x0, y0, rw, rh = bg_region
    mask[y0:y0+rh, x0:x0+rw] = -2
    
    # 计算背景系数
    index = (mask == -2)
    if np.sum(index) == 0:
        raise ValueError("自动选择的背景区域无效")
    
    aa1 = np.sum(pattern[index] * BG[index])
    aa2 = np.sum(BG[index] * BG[index])
    cc = aa1 / aa2 if aa2 != 0 else 0
    
    # 打印调试信息
    mean_pattern_roi = np.mean(pattern[index])
    mean_bg_roi = np.mean(BG[index])
    print(f"[DEBUG] Sub_BG Quadrant {quadrant}:")
    print(f"  - ROI位置: x0={x0}, y0={y0}, w={rw}, h={rh}")
    print(f"  - 信号均值: {mean_pattern_roi:.2f}")
    print(f"  - 背景均值: {mean_bg_roi:.2f}")
    print(f"  - 背景系数 C = {cc:.6f}")
    
    # 应用背景减除
    tmppattern = pattern - cc * BG
    dd = tmppattern
    
    # 统计减除效果
    original_mean = np.mean(pattern)
    subtracted_mean = np.mean(dd[dd >= 0])
    print(f"  - 原始数据均值: {original_mean:.2f}")
    print(f"  - 减除后均值: {subtracted_mean:.2f}")
    print(f"  - 减少量: {original_mean - subtracted_mean:.2f} ({(original_mean - subtracted_mean) / original_mean * 100:.1f}%)")
    
    dd[dd < 0] = 0
    return dd

def I2F(pattern, scale=1):
    pattern_tmp = pattern.copy()
    pattern_tmp[(pattern_tmp < 0) & (pattern_tmp != -1)] = 0
    pattern_tmp[pattern_tmp != -1] = np.sqrt(pattern_tmp[pattern_tmp != -1] / scale)
    return pattern_tmp

def centro_binning_odd(pattern, center, xyb):
    nbx, nby = xyb
    if nbx % 2 == 0 or nby % 2 == 0:
        raise ValueError("xybin must be odd number")
    xcenter, ycenter = center
    xsize, ysize = pattern.shape
    xhalf = min(xcenter - 1, xsize - xcenter)
    yhalf = min(ycenter - 1, ysize - ycenter)
    
    # 计算裁剪范围
    rxhalf = (( (2*xhalf + 1) // nbx - 1 ) * nbx - 1 ) // 2 if ( (2*xhalf +1) // nbx ) % 2 ==0 else ( (2*xhalf +1) // nbx * nbx -1 ) //2
    ryhalf = (( (2*yhalf + 1) // nby - 1 ) * nby - 1 ) // 2 if ( (2*yhalf +1) // nby ) % 2 ==0 else ( (2*yhalf +1) // nby * nby -1 ) //2
    
    pattern_crop = pattern[xcenter-rxhalf:xcenter+rxhalf+1, ycenter-ryhalf:ycenter+ryhalf+1]
    xsize_new, ysize_new = pattern_crop.shape
    
    # X方向分箱
    binx_pattern = np.zeros((xsize_new // nbx, ysize_new))
    for k in range(ysize_new):
        for h in range(0, xsize_new, nbx):
            bin_idx = h // nbx
            binx_pattern[bin_idx, k] = np.sum(pattern_crop[h:h+nbx, k])
            if -1 in pattern_crop[h:h+nbx, k]:
                binx_pattern[bin_idx, k] = -1
    
    # Y方向分箱
    binxy_pattern = np.zeros((xsize_new // nbx, ysize_new // nby))
    for h in range(binx_pattern.shape[0]):
        for k in range(0, ysize_new, nby):
            bin_idx = k // nby
            binxy_pattern[h, bin_idx] = np.sum(binx_pattern[h, k:k+nby])
            if -1 in binx_pattern[h, k:k+nby]:
                binxy_pattern[h, bin_idx] = -1
    
    return binxy_pattern

def split_matrix_into_quadrants(matrix):
    """
    将一个偶数尺寸的矩阵分为四个象限。
    """
    rows, cols = matrix.shape
    
    # 检查行和列是否为偶数
    if rows % 2 != 0 or cols % 2 != 0:
        raise ValueError("矩阵的行和列必须是偶数。")
    
    # 计算中间索引
    mid_row = rows // 2
    mid_col = cols // 2
    
    # 分割矩阵
    top_left = matrix[:mid_row, :mid_col]
    top_right = matrix[:mid_row, mid_col:]
    bottom_left = matrix[mid_row:, :mid_col]
    bottom_right = matrix[mid_row:, mid_col:]
    
    return top_left, top_right, bottom_left, bottom_right

def merge_quadrants(top_left, top_right, bottom_left, bottom_right):
    """
    将四个相同大小的矩阵合并为一个大矩阵，分别位于四个象限
    """
    # 验证输入矩阵形状是否一致
    if (top_left.shape != top_right.shape or 
        top_left.shape != bottom_left.shape or 
        top_left.shape != bottom_right.shape):
        raise ValueError("所有输入矩阵必须具有相同的形状")
    
    # 计算合并后的矩阵尺寸
    m, n = top_left.shape
    merged = np.zeros((2*m, 2*n), dtype=top_left.dtype)
    
    # 填充四个象限
    merged[:m, :n] = top_left         # 左上
    merged[:m, n:] = top_right        # 右上
    merged[m:, :n] = bottom_left      # 左下
    merged[m:, n:] = bottom_right     # 右下
    
    return merged

def crop_with_padding(data_sb: np.ndarray,
                      center: tuple[int, int],
                      size: int = 3605,
                      pad_value: float = -1) -> np.ndarray:
    # Validate size is odd
    if size % 2 == 0:
        size=size+1

    half = size // 2
    H, W = data_sb.shape
    
    # 处理center的不同格式
    try:
        if isinstance(center, np.ndarray):
            if center.ndim == 2 and center.shape[0] == 1:
                # center是(1, 2)格式，如[[1873 2137]]
                cy, cx = center[0, 0], center[0, 1]
            elif center.ndim == 1 and center.size == 2:
                # center是(2,)格式
                cy, cx = center[0], center[1]
            else:
                # 其他格式，尝试flatten
                center_flat = center.flatten()
                cy, cx = center_flat[0], center_flat[1]
        elif isinstance(center, (list, tuple)):
            cy, cx = center
        else:
            raise ValueError(f"Unsupported center format: {type(center)}")
            
        print(f"[CROP] 解析后的中心位置: cy={cy}, cx={cx}")
        
    except Exception as e:
        print(f"[ERROR] 解析center失败: {e}")
        print(f"[ERROR] center类型: {type(center)}")
        print(f"[ERROR] center值: {center}")
        raise ValueError(f"无法解析center参数: {center}")

    # Pad original array so we can always slice without boundary checks
    pad_top = half
    pad_bottom = half
    pad_left = half
    pad_right = half
    padded = np.pad(
        data_sb,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant', constant_values=pad_value
    )

    # Adjust center coordinates to padded array
    cy_p = cy + pad_top
    cx_p = cx + pad_left

    # Extract crop
    r0 = cy_p - half
    r1 = cy_p + half + 1  # +1 because slice end is exclusive
    c0 = cx_p - half
    c1 = cx_p + half + 1
    crop = padded[r0:r1, c0:c1]

    # Sanity check
    if crop.shape != (size, size):
        raise RuntimeError(f"Crop shape mismatch: expected ({size},{size}), got {crop.shape}")

    return crop

def create_folders_based_on_filenames(target_dir,target_name):
    # 指定目标目录
    new_dir = None  # 初始化变量
    try:
        # 查找所有符合条件的.mat文件
        mat_files = glob.glob(os.path.join(target_dir,target_name))
        
        if not mat_files:
            print(f"在目录 {target_dir} 中未找到匹配的.mat文件")
            # 创建一个默认目录
            default_dir = os.path.join(target_dir, 'default_output')
            if not os.path.exists(default_dir):
                os.makedirs(default_dir)
                print(f"已创建默认目录: {default_dir}")
            return default_dir

        print(f"找到 {len(mat_files)} 个匹配的文件")

        # 遍历所有找到的文件
        for file_path in mat_files:
            # 提取纯文件名
            file_name = os.path.basename(file_path)
            
            # 使用正则表达式提取数字部分
            match = re.search(r'_(\d+)\.mat$', file_name)
            
            if match:
                number_str = match.group(1)
                # 构建新目录路径
                new_dir = os.path.join(target_dir, number_str)
                
                # 创建目录（如果不存在）
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    print(f"已创建目录: {new_dir}")
                else:
                    print(f"目录已存在，跳过创建: {new_dir}")
                break  # 只处理第一个匹配的文件
            else:
                print(f"文件名格式不符合要求: {file_name}")

    except PermissionError:
        print("错误：没有足够的权限访问目录")
        new_dir = os.path.join(target_dir, 'fallback_output')
        os.makedirs(new_dir, exist_ok=True)
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        new_dir = os.path.join(target_dir, 'error_output')
        os.makedirs(new_dir, exist_ok=True)
    
    # 确保返回有效路径
    if new_dir is None:
        new_dir = os.path.join(target_dir, 'default_output')
        os.makedirs(new_dir, exist_ok=True)
    
    return new_dir

def run_single_reconstruction(args):
    """
    单次完整重构任务函数，用于多进程执行
    
    每个进程独立完成一个完整的重构任务，包括：
    - iter_times次外层迭代（每次外层迭代包含one_times次内层迭代）
    - 这是单核任务，不能在内部并行化
    """
    config, process_id = args
    
    # 为每个进程设置不同的随机种子，确保不同的随机初始化
    np.random.seed(process_id + int(time.time()))
    
    # 更新配置中的进程信息
    config_copy = config.copy()
    config_copy['process_id'] = process_id
    # 注意：iter_times和one_times保持不变，这是单个重构任务的完整参数
    # 每个进程都会独立运行完整的iter_times*one_times次迭代
    
    try:
        # 首先检查重构模块是否可用
        if not RECONSTRUCTION_AVAILABLE:
            return f"Process {process_id} failed: Reconstruction module not available"
        
        # 检查重构器类是否存在
        if CDIReconstructor is None:
            return f"Process {process_id} failed: CDIReconstructor class not available"
        
        # 尝试使用MPI版本的重构器（通常就是CDIReconstructor）
        reconstructor = CDIReconstructor(config_copy)
        reconstructor.run()
        return f"Process {process_id} completed successfully"
    except ImportError as e:
        return f"Process {process_id} failed: Import error - {str(e)}"
    except AttributeError as e:
        return f"Process {process_id} failed: Attribute error - {str(e)}"
    except Exception as e:
        return f"Process {process_id} failed: {type(e).__name__} - {str(e)}"

def run_multiprocess_reconstruction(base_config, total_reconstructions=1000, num_processes=None):
    """
    使用多进程运行大量独立的重构任务
    
    重构任务的多进程策略：
    - 每个核心独立运行一个完整的重构任务
    - 每个重构任务包含iter_times次外层迭代，每次外层迭代包含one_times次内层迭代
    - 多个核心并行运行，但每个重构任务本身是单核的
    - 通过运行多个独立的重构任务来获得多个重构结果
    
    参数:
        base_config: 基础配置字典，包含iter_times和one_times等单核重构参数
        total_reconstructions: 总重构次数（每次都是完整的重构）
        num_processes: 使用的进程数，默认为CPU核心数
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    print("多进程重构配置:")
    print(f"  - 总重构次数: {total_reconstructions}")
    print(f"  - 并行进程数: {num_processes}")
    print(f"  - 每个重构的外层迭代次数: {base_config.get('iter_times', 'N/A')}")
    print(f"  - 每次外层迭代的内层迭代次数: {base_config.get('one_times', 'N/A')}")
    print(f"  - 每个重构的总计算量: {base_config.get('iter_times', 1)+1} × {base_config.get('one_times', 1)} = {(base_config.get('iter_times', 1)+1) * base_config.get('one_times', 1)} 次迭代")
    print("开始并行执行...")
    
    # 准备任务参数
    tasks = [(base_config, i+1) for i in range(total_reconstructions)]
    
    # 创建进程池并执行任务
    with mp.Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示进度条
        results = []
        with tqdm(total=total_reconstructions, desc="重构进度") as pbar:
            # 提交所有任务
            async_results = [pool.apply_async(run_single_reconstruction, (task,)) for task in tasks]
            
            # 收集结果并更新进度条
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=3600)  # 1小时超时
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    results.append(f"Task failed: {str(e)}")
                    pbar.update(1)
    
    # 统计结果
    successful = sum(1 for r in results if "completed successfully" in r)
    failed = total_reconstructions - successful
    
    print(f"\n重构完成！")
    print(f"成功: {successful}/{total_reconstructions}")
    print(f"失败: {failed}/{total_reconstructions}")
    
    # 如果有失败的任务，显示错误信息
    if failed > 0:
        print("\n失败任务的错误信息:")
        failed_results = [r for r in results if "completed successfully" not in r]
        for i, error in enumerate(failed_results[:3]):  # 只显示前3个错误
            print(f"  {i+1}. {error}")
        if len(failed_results) > 3:
            print(f"  ... 还有 {len(failed_results)-3} 个类似错误")
    
    return results

def select_mat_file(directory):
    """
    查找并选择指定目录中以'F_'开头的.mat文件
    """
    # 构建匹配模式（不区分大小写）
    pattern = os.path.join(directory, 'F_*.mat')
    files = glob.glob(pattern)
    
    # 精确过滤出真正以'F_'开头的文件（避免匹配到'f_'）
    filtered_files = [f for f in files if os.path.basename(f).startswith('F_')]
    
    if not filtered_files:
        print(f"在目录 {directory} 中未找到以'F_'开头的.mat文件")
        return None
    
    if len(filtered_files) == 1:
        return filtered_files[0]
    
    # 显示选择菜单
    print(f"找到 {len(filtered_files)} 个匹配文件:")
    for i, file in enumerate(filtered_files, 1):
        print(f"[{i}] {os.path.basename(file)}")
    
    # 获取用户输入
    while True:
        try:
            choice = int(input("请选择文件编号 (输入0退出): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(filtered_files):
                return filtered_files[choice-1]
            print("输入无效，请重新输入")
        except ValueError:
            print("请输入有效数字")

# %%---------------------- 主程序 ----------------------
def binning_processing_only(diffraction_file=None, background_file=None, missing_file=None, **kwargs):
    """仅执行Binning处理功能
    
    支持两种中心坐标来源：
    1. 从kwargs中获取前端传入的center_x和center_y（优先）
    2. 从center_result.mat文件中读取（后备）
    """
    try:
        print("[BINNING] 开始Binning处理...")
        
        # 设置非交互式模式
        non_interactive = os.environ.get('NON_INTERACTIVE', 'false').lower() == 'true'
        print(f"[BINNING] 非交互式模式: {non_interactive}")
        
        # 使用传入的输出目录或默认目录
        output_dir = kwargs.get('output_dir') or os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        print(f"[BINNING] 输出目录: {output_dir}")
        
        # 检查是否有前端传入的中心坐标
        center_x = kwargs.get('center_x')
        center_y = kwargs.get('center_y')
        use_frontend_center = center_x is not None and center_y is not None
        
        if use_frontend_center:
            print(f"[BINNING] 使用前端传入的中心坐标: ({center_y}, {center_x})")
        
        # 首先检查是否有中心查找的结果
        center_result_path = os.path.join(output_dir, 'center_result.mat')
        print(f"[BINNING] 检查中心查找结果文件: {center_result_path}")
        
        if os.path.exists(center_result_path):
            # 使用中心查找的结果
            try:
                print("[BINNING] 加载中心查找结果...")
                center_data = scipy.io.loadmat(center_result_path)
                data_sb = center_data['data_sb']
                
                # 如果前端传入了中心坐标，使用前端的值；否则使用文件中的值
                if use_frontend_center:
                    center = np.array([[int(center_y), int(center_x)]])
                    print(f"[BINNING] 使用前端传入的中心坐标: {center}")
                else:
                    center = center_data['center']
                    print(f"[BINNING] 使用文件中的中心坐标: {center}")
                
                print(f"[BINNING] 中心查找结果加载成功，数据尺寸: {data_sb.shape}")
                print(f"[BINNING] 中心位置: {center}")
            except Exception as e:
                print(f"[ERROR] 加载中心查找结果失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            # 如果没有中心查找结果，需要先进行中心查找
            print("[INFO] 未找到中心查找结果，先执行中心查找...")
            try:
                if not center_finding_only(diffraction_file, background_file, missing_file, **kwargs):
                    print("[ERROR] 中心查找失败")
                    return False
                print("[BINNING] 中心查找完成，重新加载结果...")
                center_data = scipy.io.loadmat(center_result_path)
                data_sb = center_data['data_sb']
                
                # 如果前端传入了中心坐标，使用前端的值；否则使用文件中的值
                if use_frontend_center:
                    center = np.array([[int(center_y), int(center_x)]])
                    print(f"[BINNING] 使用前端传入的中心坐标: {center}")
                else:
                    center = center_data['center']
                    print(f"[BINNING] 使用文件中的中心坐标: {center}")
                
                print(f"[BINNING] 中心查找结果加载成功，数据尺寸: {data_sb.shape}")
                print(f"[BINNING] 中心位置: {center}")
            except Exception as e:
                print(f"[ERROR] 中心查找过程失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # 执行Binning处理
        cut_size = kwargs.get('cut_size', 3605)
        bin_pixel = kwargs.get('binning_size', 7)
        
        print(f"[BINNING] 开始Binning处理，裁剪大小: {cut_size}, Binning大小: {bin_pixel}")
        print(f"[BINNING] 输入数据尺寸: {data_sb.shape}")
        
        # 检查是否需要裁剪
        if cut_size <= 0 or cut_size >= data_sb.shape[0]:
            # 不进行裁剪，直接使用原始数据
            print("[INFO] 跳过裁剪，直接使用原始数据进行binning")
            I = data_sb.copy()
        else:
            # 中心裁剪
            print(f"[INFO] 执行中心裁剪，裁剪大小: {cut_size}")
            try:
                # 确保center格式正确
                print(f"[BINNING] 原始center格式: {type(center)}, 值: {center}")
                
                # 将center转换为正确的格式
                if isinstance(center, np.ndarray):
                    if center.ndim == 2 and center.shape[0] == 1:
                        # center是(1, 2)格式，如[[1873 2137]]
                        center_tuple = (int(center[0, 0]), int(center[0, 1]))
                    elif center.ndim == 1 and center.size == 2:
                        # center是(2,)格式
                        center_tuple = (int(center[0]), int(center[1]))
                    else:
                        # 其他格式，尝试flatten
                        center_flat = center.flatten()
                        center_tuple = (int(center_flat[0]), int(center_flat[1]))
                elif isinstance(center, (list, tuple)):
                    center_tuple = (int(center[0]), int(center[1]))
                else:
                    raise ValueError(f"Unsupported center format: {type(center)}")
                
                print(f"[BINNING] 转换后的center: {center_tuple}")
                
                I = crop_with_padding(data_sb, center_tuple, size=cut_size, pad_value=-1)
                print(f"[BINNING] 裁剪成功，裁剪后尺寸: {I.shape}")
            except Exception as e:
                print(f"[ERROR] 裁剪失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"[INFO] 裁剪后数据尺寸: {I.shape}")
        
        # Binning处理
        try:
            xc = I.shape[0] // 2
            yc = I.shape[1] // 2
            print(f"[INFO] 开始binning处理，中心位置: ({xc}, {yc}), binning大小: {bin_pixel}")
            
            # 检查binning大小是否为奇数
            if bin_pixel % 2 == 0:
                print(f"[WARNING] Binning大小 {bin_pixel} 不是奇数，自动调整为 {bin_pixel + 1}")
                bin_pixel = bin_pixel + 1
            
            print(f"[BINNING] 执行centro_binning_odd...")
            II = centro_binning_odd(I, (xc, yc), (bin_pixel, bin_pixel))
            print(f"[BINNING] centro_binning_odd完成，输出尺寸: {II.shape}")
            
            print(f"[BINNING] 执行I2F转换...")
            F = I2F(II, 1)
            F[(F <= 0) & (F != -1)] = 0
            print(f"[BINNING] I2F转换完成，最终尺寸: {F.shape}")
            
        except Exception as e:
            print(f"[ERROR] Binning处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"[SUCCESS] Binning处理完成，输出尺寸: {F.shape}")
        
        # 生成对比图像
        try:
            print("[BINNING] 开始生成对比图像...")
            plt.figure(figsize=(15, 5))
            
            # 左图：原始数据
            plt.subplot(131)
            plt.imshow(np.log(np.abs(data_sb) + 1), cmap='jet')
            plt.colorbar(label='Log Intensity')
            plt.title('原始数据')
            plt.xlabel('X')
            plt.ylabel('Y')
            
            # 标记中心位置
            # 处理center数组的不同格式
            try:
                if center.ndim == 2 and center.shape[0] == 1:
                    # center是(1, 2)格式
                    center_x = center[0, 1]
                    center_y = center[0, 0]
                elif center.ndim == 1 and center.size == 2:
                    # center是(2,)格式
                    center_x = center[1]
                    center_y = center[0]
                else:
                    # 其他格式，尝试flatten
                    center_flat = center.flatten()
                    center_x = center_flat[1]
                    center_y = center_flat[0]
                    
                plt.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=3, label=f'Center: ({center_x}, {center_y})')
                plt.legend()
            except Exception as e:
                print(f"[WARNING] 中心位置标记失败: {e}")
            
            # 中图：裁剪后的数据（或原始数据）
            plt.subplot(132)
            if cut_size <= 0 or cut_size >= data_sb.shape[0]:
                plt.imshow(np.log(np.abs(I) + 1), cmap='jet')
                plt.colorbar(label='Log Intensity')
                plt.title(f'原始数据 (无裁剪)')
            else:
                plt.imshow(np.log(np.abs(I) + 1), cmap='jet')
                plt.colorbar(label='Log Intensity')
                plt.title(f'裁剪后数据 ({I.shape[0]}x{I.shape[1]})')
            plt.xlabel('X')
            plt.ylabel('Y')
            
            # 右图：Binning后的数据
            plt.subplot(133)
            plt.imshow(np.log(np.abs(F) + 1), cmap='jet')
            plt.colorbar(label='Log Intensity')
            plt.title(f'Binning后数据 ({F.shape[0]}x{F.shape[1]}, bin={bin_pixel})')
            plt.xlabel('X')
            plt.ylabel('Y')
            
            plt.tight_layout()
            
            # 保存图像
            output_image_path = os.path.join(output_dir, 'binning_result.png')
            print(f"[BINNING] 保存图像到: {output_image_path}")
            plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            print("[BINNING] 图像保存成功")
            
        except Exception as e:
            print(f"[ERROR] 图像生成失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使图像生成失败，也继续保存数据
        
        # 保存处理后的数据
        try:
            output_data_path = os.path.join(output_dir, 'binning_result.mat')
            print(f"[BINNING] 保存数据到: {output_data_path}")
            # 注意：II 是 binning 后的强度（Sum），F 是振幅（Sqrt）
            # 前端显示应该使用 II（强度），重构算法使用 F（振幅）
            scipy.io.savemat(output_data_path, {
                'F': F,              # 振幅 = sqrt(强度)，供重构算法使用
                'II': II,            # Binning后的强度（Sum），供前端显示
                'I': I,              # 裁剪后的原始数据
                'center': center,
                'data_sb': data_sb,
                'binning_size': bin_pixel,
                'cut_size': cut_size
            })
            print(f"[BINNING] 数据保存成功")
            print(f"[BINNING] II (强度) 范围: min={II.min():.2f}, max={II.max():.2f}")
            print(f"[BINNING] F (振幅) 范围: min={F.min():.2f}, max={F.max():.2f}")
        except Exception as e:
            print(f"[ERROR] 数据保存失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"[SUCCESS] Binning处理完成，结果保存到: {output_image_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Binning处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def center_finding_only(diffraction_file=None, background_file=None, missing_file=None, **kwargs):
    """仅执行中心查找功能"""
    # 设置非交互式模式
    non_interactive = os.environ.get('NON_INTERACTIVE', 'false').lower() == 'true'
    
    # 使用传入的输出目录或默认目录
    output_dir = kwargs.get('output_dir') or os.path.join(os.getcwd(), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先检查是否有缺失数据处理的结果
    missing_result_path = os.path.join(output_dir, 'missing_result.mat')
    if os.path.exists(missing_result_path):
        # 使用缺失数据处理的结果
        try:
            data_sb = scipy.io.loadmat(missing_result_path)['data_sb']
            print('[SUCCESS] 加载缺失数据处理结果')
        except Exception as e:
            print(f"[ERROR] 加载缺失数据处理结果失败: {e}")
            return False
    else:
        # 如果没有缺失数据处理结果，需要先进行缺失数据处理
        print("[INFO] 未找到缺失数据处理结果，先执行缺失数据处理...")
        if not missing_data_processing_only(diffraction_file, background_file, missing_file, **kwargs):
            return False
        data_sb = scipy.io.loadmat(missing_result_path)['data_sb']
    
    # 执行中心查找
    search_range = kwargs.get('search_range', 45)
    region_ratio = kwargs.get('region_ratio', 0.1)
    
    print(f"开始中心查找，搜索范围: {search_range}, 区域比例: {region_ratio}")
    finder = CenterFinder_auto(data_sb, search_range=search_range, region_ratio=region_ratio)
    center, error_map = finder.optimize_center()
    
    print(f"[SUCCESS] 中心查找完成，找到中心: {center}")
    
    # 生成误差图
    plt.figure(figsize=(12, 5))
    
    # 左图：原始数据
    plt.subplot(121)
    plt.imshow(np.log(np.abs(data_sb) + 1), cmap='jet')
    plt.colorbar(label='Log Intensity')
    plt.title('原始数据')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 标记找到的中心
    plt.plot(center[1], center[0], 'r+', markersize=15, markeredgewidth=3, label=f'Center: ({center[1]}, {center[0]})')
    plt.legend()
    
    # 右图：误差图
    plt.subplot(122)
    plt.imshow(np.abs(error_map), cmap='jet')
    plt.colorbar(label='Error')
    plt.title('中心查找误差图')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 标记最佳中心位置
    min_error_pos = np.unravel_index(np.nanargmin(error_map), error_map.shape)
    plt.plot(min_error_pos[1], min_error_pos[0], 'r+', markersize=15, markeredgewidth=3, label=f'Min Error: ({min_error_pos[1]}, {min_error_pos[0]})')
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    output_image_path = os.path.join(output_dir, 'center_result.png')
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存中心坐标和误差图数据
    output_data_path = os.path.join(output_dir, 'center_result.mat')
    scipy.io.savemat(output_data_path, {
        'center': center,
        'error_map': error_map,
        'data_sb': data_sb
    })
    
    print(f"[SUCCESS] 中心查找完成，结果保存到: {output_image_path}")
    # 输出可解析的中心坐标（供前端解析）
    print(f"[CENTER_RESULT] center_y={center[0]}, center_x={center[1]}")
    return True

def missing_data_processing_only(diffraction_file=None, background_file=None, missing_file=None, manual_selection=False, **kwargs):
    """仅执行缺失数据处理功能"""
    # 设置非交互式模式
    non_interactive = os.environ.get('NON_INTERACTIVE', 'false').lower() == 'true'
    
    # 调试信息
    print(f"[DEBUG] missing_data_processing_only 参数:")
    print(f"[DEBUG] manual_selection = {manual_selection}")
    print(f"[DEBUG] non_interactive = {non_interactive}")
    print(f"[DEBUG] 环境变量 NON_INTERACTIVE = {os.environ.get('NON_INTERACTIVE', 'not set')}")
    
    # 使用传入的输出目录或默认目录
    output_dir = kwargs.get('output_dir') or os.path.join(os.getcwd(), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试加载已有的背景减除结果
    background_result_path = os.path.join(output_dir, 'background_subtraction_result.mat')
    
    if os.path.exists(background_result_path):
        # 使用已有的背景减除结果
        print(f"[INFO] 使用已有的背景减除结果: {background_result_path}")
        try:
            data_sb = scipy.io.loadmat(background_result_path)['data_sb']
            print('[SUCCESS] 加载背景减除结果')
        except Exception as e:
            print(f"[ERROR] 加载背景减除结果失败: {e}")
            return False
    else:
        # 如果没有背景减除结果，需要先执行背景减除
        print("[INFO] 未找到背景减除结果，需要先执行背景减除...")
        if not background_file:
            print("[ERROR] 未提供背景文件，无法执行背景减除")
            return False
        if not background_subtraction_only(diffraction_file, background_file, missing_file, **kwargs):
            return False
        try:
            data_sb = scipy.io.loadmat(background_result_path)['data_sb']
            print('[SUCCESS] 加载背景减除结果')
        except Exception as e:
            print(f"[ERROR] 加载背景减除结果失败: {e}")
            return False
    
    # 加载缺失数据
    if missing_file:
        miss_path = resolve_file_path(missing_file, 'uploads')
        print(f"[MISSING] 缺失数据路径: {miss_path}")
    else:
        print("[ERROR] 未提供缺失数据文件")
        return False
    
    try:
        miss_data = scipy.io.loadmat(miss_path)
        # 自动检测变量名
        if 'miss' in miss_data:
            miss = miss_data['miss']
        else:
            data_keys = [k for k in miss_data.keys() if not k.startswith('__')]
            if data_keys:
                miss = miss_data[data_keys[0]]
                print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为缺失数据")
            else:
                raise KeyError("MAT文件中没有找到有效的数据变量")
        print('[SUCCESS] 成功读取缺失数据')
    except (FileNotFoundError, KeyError) as e:
        print(f"[ERROR] 读取缺失数据失败: {e}")
        return False
    
    # 如果启用手动选择，先进行手动选择
    manual_missing_mask = None
    print(f"[DEBUG] 检查手动选择条件: manual_selection={manual_selection}, non_interactive={non_interactive}")
    if manual_selection and not non_interactive:
        print("[INFO] 启动手动缺失数据选择工具...")
        try:
            # 创建用于手动选择的图像（背景减除后的干净图像）
            display_image = np.log(np.abs(data_sb) + 1)
            
            # 创建选择器但不立即启动事件循环
            selector = MultiPolygonROISelector(display_image)
            
            # 等待用户完成选择
            print("[INFO] 请在Python窗口中绘制多边形选择缺失区域...")
            print("[INFO] 完成后关闭Python窗口即可")
            
            # 等待窗口关闭
            while plt.fignum_exists(selector.fig.number):
                plt.pause(0.1)
            
            # 获取用户选择的多边形
            selected_polygons = selector.get_polygons()
            
            if selected_polygons:
                print(f"[SUCCESS] 用户选择了 {len(selected_polygons)} 个缺失区域")
                
                # 创建手动选择的缺失数据mask
                manual_missing_mask = np.zeros(data_sb.shape, dtype=bool)
                
                # 将用户选择的区域添加到手动缺失数据mask中
                for i, polygon in enumerate(selected_polygons):
                    print(f"[DEBUG] 处理第 {i+1} 个多边形，包含 {len(polygon)} 个点")
                    print(f"[DEBUG] 多边形坐标: {polygon[:3]}...")  # 只打印前3个点
                    
                    # 创建多边形mask
                    from matplotlib.path import Path
                    poly_path = Path(polygon)
                    
                    # 创建网格坐标 (注意：numpy的mgrid是(y, x)顺序)
                    y_coords, x_coords = np.mgrid[0:data_sb.shape[0], 0:data_sb.shape[1]]
                    points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
                    
                    # 检查哪些点在多边形内
                    inside_polygon = poly_path.contains_points(points)
                    inside_polygon = inside_polygon.reshape(data_sb.shape)
                    
                    # 统计被选中的像素数量
                    selected_pixels = np.sum(inside_polygon)
                    print(f"[DEBUG] 多边形 {i+1} 选中了 {selected_pixels} 个像素")
                    
                    # 将多边形内的点添加到手动缺失数据mask中
                    manual_missing_mask |= inside_polygon
                    print(f"[INFO] 添加了多边形区域到手动缺失数据mask中")
                
                # 统计手动选择的像素数量
                manual_selected_pixels = np.sum(manual_missing_mask)
                print(f"[DEBUG] 手动选择总共选中了 {manual_selected_pixels} 个像素")
            else:
                print("[INFO] 用户未选择任何额外区域，将使用原始缺失数据")
        except Exception as e:
            print(f"[WARNING] 手动选择过程中出现错误: {e}")
            print("[INFO] 继续使用原始缺失数据进行处理")
    
    # 应用缺失数据mask
    print(f"[DEBUG] 应用缺失数据mask: manual_missing_mask is not None = {manual_missing_mask is not None}")
    if manual_missing_mask is not None:
        # 如果用户进行了手动选择，优先使用手动选择的区域
        print("[INFO] 使用手动选择的缺失区域，忽略原始miss.data文件")
        print(f"[DEBUG] manual_missing_mask 形状: {manual_missing_mask.shape}")
        print(f"[DEBUG] manual_missing_mask 中True的数量: {np.sum(manual_missing_mask)}")
        data_sb[manual_missing_mask] = -1
        print(f"[INFO] 应用了手动选择的缺失区域")
    else:
        # 如果没有手动选择，使用原始缺失数据文件
        print("[INFO] 使用原始miss.data文件进行缺失数据处理")
        miss_mask = (miss == 1)
        data_sb[miss_mask] = -1
    
    # 统计手动选择前后的缺失数据像素数量
    if manual_selection and not non_interactive:
        manual_missing_pixels = np.sum(data_sb == -1)
        print(f"[DEBUG] 手动选择后，缺失数据像素数量: {manual_missing_pixels}")
    
    # 扩展缺失区域（可选）
    original_mask = (data_sb == -1)
    kernel = np.ones((3, 3))
    expanded_mask = convolve2d(original_mask.astype(float), kernel, mode='same') > 0
    data_sb[expanded_mask] = -1
    
    # 统计最终缺失数据像素数量
    final_missing_pixels = np.sum(data_sb == -1)
    print(f"[DEBUG] 最终缺失数据像素数量: {final_missing_pixels}")
    
    # 生成预览图像
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log(np.abs(data_sb) + 1), cmap='jet')
    plt.colorbar(label='Log Intensity')
    plt.title('缺失数据处理结果')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 保存图像
    output_image_path = os.path.join(output_dir, 'missing_result.png')
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存处理后的数据
    output_data_path = os.path.join(output_dir, 'missing_result.mat')
    scipy.io.savemat(output_data_path, {'data_sb': data_sb})
    
    print(f"[SUCCESS] 缺失数据处理完成，结果保存到: {output_image_path}")
    return True

def background_subtraction_only(diffraction_file=None, background_file=None, missing_file=None, **kwargs):
    """仅执行背景减除功能
    
    支持两种模式：
    1. 使用背景文件模式：需要提供 background_file
    2. 自动算法模式：kwargs['auto_background']=True，基于边缘估计背景
    """
    try:
        print("[BACKGROUND] 开始背景减除处理...")
        print(f"[BACKGROUND] kwargs: {kwargs}")
        
        # 检查是否使用自动背景模式
        auto_background = kwargs.get('auto_background', False)
        print(f"[BACKGROUND] 自动背景模式: {auto_background}")
        
        # 设置非交互式模式
        non_interactive = os.environ.get('NON_INTERACTIVE', 'false').lower() == 'true'
        print(f"[BACKGROUND] 非交互式模式: {non_interactive}")
        
        # 使用传入的输出目录或默认目录
        output_dir = kwargs.get('output_dir')
        print(f"[BACKGROUND] kwargs.get('output_dir'): {output_dir}")
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), 'outputs')
            print(f"[BACKGROUND] 使用默认输出目录")
        os.makedirs(output_dir, exist_ok=True)
        print(f"[BACKGROUND] 最终输出目录: {output_dir}")
        
        # 加载衍射数据
        if diffraction_file:
            diff_path = resolve_file_path(diffraction_file, 'uploads')
            print(f"[BACKGROUND] 衍射数据路径: {diff_path}")
        else:
            print("[ERROR] 未提供衍射数据文件")
            return False
        
        try:
            diff_data = scipy.io.loadmat(diff_path)
            # 自动检测变量名：优先使用 'I'，否则使用第一个非系统变量
            if 'I' in diff_data:
                I = diff_data['I']
            else:
                # 过滤掉系统变量（以__开头和结尾的）
                data_keys = [k for k in diff_data.keys() if not k.startswith('__')]
                if data_keys:
                    I = diff_data[data_keys[0]]
                    print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为衍射数据")
                else:
                    raise KeyError("MAT文件中没有找到有效的数据变量")
            print('[SUCCESS] 成功读取衍射数据')
        except (FileNotFoundError, KeyError) as e:
            print(f"[ERROR] 读取衍射数据失败: {e}")
            return False
        
        edge_ratio = kwargs.get('edge_ratio', 0.2)
        data_s = I.astype(np.float64)
        
        # 根据模式选择背景数据来源
        if auto_background:
            # 自动算法模式：基于边缘估计背景
            print("[BACKGROUND] 使用自动算法估计背景...")
            # 使用边缘区域的均值作为背景估计
            h, w = data_s.shape
            edge_h = int(h * edge_ratio)
            edge_w = int(w * edge_ratio)
            
            # 提取四个边缘区域
            top_edge = data_s[:edge_h, :]
            bottom_edge = data_s[-edge_h:, :]
            left_edge = data_s[:, :edge_w]
            right_edge = data_s[:, -edge_w:]
            
            # 计算边缘均值作为背景估计
            bg_estimate = np.mean([
                np.mean(top_edge),
                np.mean(bottom_edge),
                np.mean(left_edge),
                np.mean(right_edge)
            ])
            
            # 创建均匀背景
            bg = np.full_like(data_s, bg_estimate)
            print(f"[BACKGROUND] 自动估计背景值: {bg_estimate:.2f}")
        else:
            # 使用背景文件模式
            if background_file:
                bg_path = resolve_file_path(background_file, 'uploads')
                print(f"[BACKGROUND] 背景数据路径: {bg_path}")
            else:
                print("[ERROR] 未提供背景数据文件")
                return False
            
            try:
                bg_data = scipy.io.loadmat(bg_path)
                # 自动检测变量名：优先使用 'bg'，否则使用第一个非系统变量
                if 'bg' in bg_data:
                    bg = bg_data['bg']
                else:
                    # 过滤掉系统变量（以__开头和结尾的）
                    data_keys = [k for k in bg_data.keys() if not k.startswith('__')]
                    if data_keys:
                        bg = bg_data[data_keys[0]]
                        print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为背景数据")
                    else:
                        raise KeyError("MAT文件中没有找到有效的数据变量")
                print('[SUCCESS] 成功读取背景数据')
            except (FileNotFoundError, KeyError) as e:
                print(f"[ERROR] 读取背景数据失败: {e}")
                return False
        
        # 背景减除处理
        data_bg = bg.astype(np.float64)

        # 分割图像为4个区域
        data_s_11, data_s_12, data_s_21, data_s_22 = split_matrix_into_quadrants(data_s)
        data_bg_11, data_bg_12, data_bg_21, data_bg_22 = split_matrix_into_quadrants(data_bg)

        # 背景减除
        data_sub_11 = Sub_BG(data_s_11, data_bg_11, quadrant='11', edge_ratio=edge_ratio)
        data_sub_12 = Sub_BG(data_s_12, data_bg_12, quadrant='12', edge_ratio=edge_ratio)
        data_sub_21 = Sub_BG(data_s_21, data_bg_21, quadrant='21', edge_ratio=edge_ratio)
        data_sub_22 = Sub_BG(data_s_22, data_bg_22, quadrant='22', edge_ratio=edge_ratio)

        # 合并图像
        data_sb = merge_quadrants(data_sub_11, data_sub_12, data_sub_21, data_sub_22)
        
        # 生成预览图像
        plt.figure(figsize=(10, 8))
        plt.imshow(np.log(np.abs(data_sb) + 1), cmap='jet')
        plt.colorbar(label='Log Intensity')
        plt.title('背景减除结果')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # 保存图像
        output_image_path = os.path.join(output_dir, 'background_subtraction_result.png')
        plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存处理后的数据
        output_data_path = os.path.join(output_dir, 'background_subtraction_result.mat')
        scipy.io.savemat(output_data_path, {'data_sb': data_sb})
        
        print(f"[SUCCESS] 背景减除完成，结果保存到: {output_image_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 背景减除处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_data_only(diffraction_file=None, background_file=None, missing_file=None, **kwargs):
    """仅加载数据，不进行处理"""
    # 设置默认路径
    upload_dir = os.path.join(os.getcwd(), 'uploads')
    # 使用传入的输出目录或默认目录
    output_dir = kwargs.get('output_dir') or os.path.join(os.getcwd(), 'outputs')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 清理之前的结果文件，避免缓存问题
    cleanup_files = [
        'background_subtraction_result.mat',
        'background_subtraction_result.png',
        'missing_result.mat',
        'missing_result.png',
        'manual_support.mat',
        'binning_result.mat',
        'binning_result.png',
        'center_result.mat',
        'center_result.png',
        'preview_diffraction.mat',
        'preview_diffraction.png',
        'preview_background.mat',
        'preview_background.png',
        'preview_missing.mat',
        'preview_missing.png'
    ]
    
    cleaned_count = 0
    for filename in cleanup_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"[INFO] 清理旧文件: {filename}")
                cleaned_count += 1
            except Exception as e:
                print(f"[WARNING] 清理文件失败 {filename}: {e}")
    
    print(f"[INFO] 开始新的处理流程，已清理 {cleaned_count} 个旧文件")
    print("开始加载数据...")
    
    # 加载衍射数据
    if diffraction_file:
        if os.path.isabs(diffraction_file):
            load_folder = os.path.dirname(diffraction_file) + '/'
            target_name = os.path.basename(diffraction_file)
        else:
            load_folder = upload_dir + '/'
            target_name = diffraction_file
        
        print(f"加载衍射文件: {load_folder}{target_name}")
        
        try:
            diff_data = scipy.io.loadmat(load_folder + target_name)
            # 自动检测变量名
            if 'I' in diff_data:
                I = diff_data['I']
            else:
                data_keys = [k for k in diff_data.keys() if not k.startswith('__')]
                if data_keys:
                    I = diff_data[data_keys[0]]
                    print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为衍射数据")
                else:
                    raise KeyError("MAT文件中没有找到有效的数据变量")
            print("[SUCCESS] 成功加载衍射数据")
            
            # 保存加载的数据用于预览
            preview_path = os.path.join(output_dir, 'preview_diffraction.mat')
            io.savemat(preview_path, {'I': I})
            
            # 生成预览图像（非交互式）
            plt.figure(figsize=(8, 6))
            plt.imshow(np.log(np.abs(I) + 1), cmap='jet')
            plt.title('衍射数据预览')
            plt.colorbar()
            plt.tight_layout()
            
            # 保存图像到outputs目录
            image_path = os.path.join(output_dir, 'preview_diffraction.png')
            plt.savefig(image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[SUCCESS] 预览图像已保存: {image_path}")
            
        except Exception as e:
            print(f"[ERROR] 加载衍射数据失败: {e}")
            return False
    
    # 加载背景数据
    if background_file:
        bg_path = resolve_file_path(background_file, 'uploads')
        print(f"加载背景文件: {bg_path}")
        
        try:
            bg_data = scipy.io.loadmat(bg_path)
            # 自动检测变量名
            if 'bg' in bg_data:
                bg = bg_data['bg']
            else:
                data_keys = [k for k in bg_data.keys() if not k.startswith('__')]
                if data_keys:
                    bg = bg_data[data_keys[0]]
                    print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为背景数据")
                else:
                    raise KeyError("MAT文件中没有找到有效的数据变量")
            print("[SUCCESS] 成功加载背景数据")
            
            # 保存背景数据
            bg_path_save = os.path.join(output_dir, 'preview_background.mat')
            io.savemat(bg_path_save, {'bg': bg})
            
            # 生成背景数据预览图像
            plt.figure(figsize=(8, 6))
            plt.imshow(np.log(np.abs(bg) + 1), cmap='jet')
            plt.title('背景数据预览')
            plt.colorbar()
            plt.tight_layout()
            
            # 保存图像到outputs目录
            bg_image_path = os.path.join(output_dir, 'preview_background.png')
            plt.savefig(bg_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[SUCCESS] 背景数据预览图像已保存: {bg_image_path}")
            
        except Exception as e:
            print(f"[ERROR] 加载背景数据失败: {e}")
            # 背景数据不是必需的，所以不返回False
    
    # 加载缺失数据
    if missing_file:
        miss_path = resolve_file_path(missing_file, 'uploads')
        print(f"加载缺失文件: {miss_path}")
        
        try:
            miss_data = scipy.io.loadmat(miss_path)
            # 自动检测变量名
            if 'miss' in miss_data:
                miss = miss_data['miss']
            else:
                data_keys = [k for k in miss_data.keys() if not k.startswith('__')]
                if data_keys:
                    miss = miss_data[data_keys[0]]
                    print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为缺失数据")
                else:
                    raise KeyError("MAT文件中没有找到有效的数据变量")
            print("[SUCCESS] 成功加载缺失数据")
            
            # 保存缺失数据
            miss_path_save = os.path.join(output_dir, 'preview_missing.mat')
            io.savemat(miss_path_save, {'miss': miss})
            
            # 生成缺失数据预览图像
            plt.figure(figsize=(8, 6))
            plt.imshow(miss, cmap='gray')
            plt.title('缺失数据预览')
            plt.colorbar()
            plt.tight_layout()
            
            # 保存图像到outputs目录
            miss_image_path = os.path.join(output_dir, 'preview_missing.png')
            plt.savefig(miss_image_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[SUCCESS] 缺失数据预览图像已保存: {miss_image_path}")
            
        except Exception as e:
            print(f"[ERROR] 加载缺失数据失败: {e}")
            # 缺失数据不是必需的，所以不返回False
    
    print("[SUCCESS] 数据加载完成")
    return True

def manual_support_selection_only(diffraction_file=None, background_file=None, missing_file=None, **kwargs):
    """仅执行手动选择support区域功能"""
    try:
        # 确保是交互式后端，否则直接报错提示用户安装依赖
        _assert_interactive_backend()

        # 强制设置为交互式模式（因为这是手动选择功能）
        non_interactive = False
        
        # 使用传入的输出目录或默认目录
        output_dir = kwargs.get('output_dir') or os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查是否有binning处理结果文件
        binning_result_path = os.path.join(output_dir, 'binning_result.mat')
        if os.path.exists(binning_result_path):
            # 使用binning处理结果中的F数据
            try:
                binning_data = scipy.io.loadmat(binning_result_path)
                F = binning_data['F']
                print('[SUCCESS] 加载binning处理结果中的F数据')
            except Exception as e:
                print(f"[ERROR] 加载binning处理结果失败: {e}")
                return False
        else:
            # 如果没有binning处理结果，需要先进行binning处理
            print("[INFO] 未找到binning处理结果，先执行binning处理...")
            if not binning_processing_only(diffraction_file, background_file, missing_file, **kwargs):
                return False
            binning_data = scipy.io.loadmat(binning_result_path)
            F = binning_data['F']
        
        print("开始手动选择Support区域...")
        print(f"基于binning后数据 (形状: {F.shape})")
        
        # 使用MultiPolygonROISelector进行手动选择（基于binning后的数据）
        print("启动交互式多边形选择工具...")
        print("操作说明：")
        print("- 左键点击：添加多边形顶点")
        print("- 右键点击：撤销最后一个点")
        print("- 中键点击：完成当前多边形")
        print("- 滚轮：缩放图像")
        print("- 按D键：完成所有选择")
        print("- 按H键：缩小一半并完成")
        print("- 按R键：重置视图")
        print("- 按U键：撤销最后一个多边形")
        
        selector = MultiPolygonROISelector(np.log(np.abs(F) + 2))
        
        # 等待用户完成选择（事件循环会阻塞直到用户完成）
        # 用户可以通过以下方式完成选择：
        # 1. 点击"完成"按钮
        # 2. 按D键完成
        # 3. 按H键缩小一半并完成
        # 4. 关闭窗口
        
        print("等待用户完成多边形选择...")
        
        # 获取用户选择的support mask
        support_mask = selector.get_combined_mask()
        
        # 将mask转换为support格式
        support = support_mask.astype(np.float64)
        
        # 保存手动选择的support
        output_support_path = os.path.join(output_dir, 'manual_support.mat')
        scipy.io.savemat(output_support_path, {'support': support})
        
        print(f"[SUCCESS] 手动选择Support完成，保存到: {output_support_path}")
        print(f"[INFO] Support形状: {support.shape}")
        print(f"[INFO] Support中1的数量: {np.sum(support == 1)}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 手动选择Support失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(diffraction_file=None, background_file=None, missing_file=None, **kwargs):
    """主处理函数，支持命令行参数"""
    # 设置默认路径
    upload_dir = os.path.join(os.getcwd(), 'uploads')
    output_dir = os.path.join(os.getcwd(), 'outputs')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否为非交互式模式（通过环境变量或参数）
    non_interactive = os.environ.get('NON_INTERACTIVE', 'false').lower() == 'true'
    
    # 加载数据
    if diffraction_file:
        # 如果提供了完整路径，直接使用
        if os.path.isabs(diffraction_file):
            load_folder = os.path.dirname(diffraction_file) + '/'
            target_name = os.path.basename(diffraction_file)
        else:
            # 否则从uploads目录加载
            load_folder = upload_dir + '/'
            target_name = diffraction_file
    else:
        # 默认值（保持向后兼容）
        load_folder = 'E:/GZC/2025上半年实验/HKUST/-10ps/'
        target_name = 'HKUST1_25W_-10ps_55uJ_2585.mat'
    
    print(f"加载文件: {load_folder}{target_name}")
    
    try:
        diff_data = scipy.io.loadmat(load_folder + target_name)
        # 自动检测变量名
        if 'I' in diff_data:
            I = diff_data['I']
        else:
            data_keys = [k for k in diff_data.keys() if not k.startswith('__')]
            if data_keys:
                I = diff_data[data_keys[0]]
                print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为衍射数据")
            else:
                raise KeyError("MAT文件中没有找到有效的数据变量")
        print("[SUCCESS] 成功加载衍射数据")
    except Exception as e:
        print(f"[ERROR] 加载衍射数据失败: {e}")
        return False
    
    save_folder = create_folders_based_on_filenames(load_folder, target_name)
    
    # 加载背景数据
    if background_file:
        bg_path = background_file if os.path.isabs(background_file) else os.path.join(upload_dir, background_file)
    else:
        bg_path = 'E:/GZC/2025上半年实验/bg.mat'
    
    try:
        bg_data = scipy.io.loadmat(bg_path)
        # 自动检测变量名
        if 'bg' in bg_data:
            bg = bg_data['bg']
        else:
            data_keys = [k for k in bg_data.keys() if not k.startswith('__')]
            if data_keys:
                bg = bg_data[data_keys[0]]
                print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为背景数据")
            else:
                raise KeyError("MAT文件中没有找到有效的数据变量")
        print('[SUCCESS] 成功读取bg')
    except (FileNotFoundError, KeyError):
        print("[INFO] bg 不存在，创建全 1 矩阵")
        bg = np.ones_like(I)
    
    # 加载缺失数据
    if missing_file:
        miss_path = missing_file if os.path.isabs(missing_file) else os.path.join(upload_dir, missing_file)
    else:
        miss_path = 'E:/GZC/2025上半年实验/HKUST/miss_-10ps.mat'
    
    try:
        miss_data = scipy.io.loadmat(miss_path)
        # 自动检测变量名
        if 'miss' in miss_data:
            miss = miss_data['miss']
        else:
            data_keys = [k for k in miss_data.keys() if not k.startswith('__')]
            if data_keys:
                miss = miss_data[data_keys[0]]
                print(f"[INFO] 使用变量名 '{data_keys[0]}' 作为缺失数据")
            else:
                raise KeyError("MAT文件中没有找到有效的数据变量")
        print('[SUCCESS] 成功读取miss')
    except (FileNotFoundError, KeyError):
        print("[INFO] miss 不存在，创建全 0 矩阵")
        miss = np.zeros_like(I)
    
    # 继续原有的处理逻辑...
    data_bg = bg.astype(np.float64)
    data_s = I.astype(np.float64)

    # 分割图像为4个区域
    data_s_11, data_s_12, data_s_21, data_s_22 = split_matrix_into_quadrants(data_s)
    data_bg_11, data_bg_12, data_bg_21, data_bg_22 = split_matrix_into_quadrants(data_bg)

    # 背景减除（添加quadrant参数和edge_ratio参数）
    edge_ratio = kwargs.get('edge_ratio', 0.2)  # 可调整参数，0.2表示边缘宽度为分区大小的1/5
    data_sub_11 = Sub_BG(data_s_11, data_bg_11, quadrant='11', edge_ratio=edge_ratio)
    data_sub_12 = Sub_BG(data_s_12, data_bg_12, quadrant='12', edge_ratio=edge_ratio)
    data_sub_21 = Sub_BG(data_s_21, data_bg_21, quadrant='21', edge_ratio=edge_ratio)
    data_sub_22 = Sub_BG(data_s_22, data_bg_22, quadrant='22', edge_ratio=edge_ratio)

    # 合并图像
    data_sb = merge_quadrants(data_sub_11,data_sub_12,data_sub_21,data_sub_22)
    
    # 处理缺失数据
    miss_mask = miss== 1  
    data_sb[miss_mask] = -1
    data_sb[data_sb >= 255000] = -1

    # 在非交互式模式下，跳过手动选择缺失数据区域
    if not non_interactive:
        # 手动选择额外的缺失数据区域（可选）
        # 如果自动识别的缺失区域不够准确，可以使用以下代码手动补充
        print("请手动选择额外的缺失数据区域...")
        selector_miss = MultiPolygonROISelector(np.abs(np.log(data_sb+2)))
        miss2 = selector_miss.get_combined_mask()
        data_sb[miss2==1] = -1
    else:
        print("非交互式模式：跳过手动选择缺失数据区域")

    # 扩展缺失区域
    original_mask = (data_sb == -1)
    kernel = np.ones((3, 3))
    expanded_mask = convolve2d(original_mask.astype(float), kernel, mode='same') > 0
    data_sb[expanded_mask] = -1
    data_sb[922, 2048:4095] = -1  # 坏线    

    # 中心裁剪和调整
    if non_interactive:
        # 非交互式模式：自动找中心
        print("非交互式模式：自动找中心")
        search_range = kwargs.get('search_range', 45)
        region_ratio = kwargs.get('region_ratio', 0.1)
        finder = CenterFinder_auto(data_sb, search_range=search_range, region_ratio=region_ratio)
    else:
        choice = input("是否自动找中心？(y/n): ").strip().lower()
        if choice == 'y':
            print("自动找中心")
            search_range = kwargs.get('search_range', 45)
            region_ratio = kwargs.get('region_ratio', 0.1)
            finder = CenterFinder_auto(data_sb, search_range=search_range, region_ratio=region_ratio)
        else:
            finder = CenterFinder(data_sb, search_range=30)
    
    center, error_map = finder.optimize_center()
    
    if not non_interactive:
        plt.figure()
        plt.imshow(np.abs(error_map),cmap='jet')
        plt.show()
        plt.pause(3)
        plt.close()
        
        if choice =='y':
            choice2 = input("是否再次手动找中心？(y/n): ").strip().lower()
            if choice2 == 'y':
                finder = CenterFinder(data_sb, search_range=30)
                center, error_map = finder.optimize_center()
                plt.figure()
                plt.imshow(np.abs(error_map),cmap='jet')
                plt.show()
                plt.pause(3)
                plt.close()
            else:
                print("中心寻找完毕")
    else:
        print("非交互式模式：中心寻找完成")
    
    cut_size = kwargs.get('cut_size', 3605)
    bin_pixel = kwargs.get('binning_size', 7)
    I = crop_with_padding(data_sb, center, size=cut_size, pad_value=-1)
    # binning处理
    xc = I.shape[0]// 2
    yc = I.shape[1]// 2
    II = centro_binning_odd(I, (xc, yc), (bin_pixel, bin_pixel))
    F = I2F(II, 1)
    F[(F <= 0) & (F != -1)] = 0
    
    if not non_interactive:
        plt.figure()
        plt.imshow(np.log(np.abs(F)+2), cmap='jet')
        plt.show()
        plt.pause(3)
        plt.close()

    # 保存中心和裁剪结果
    save_path=os.path.join(save_folder,'I_4096.mat')
    io.savemat(save_path,{'I':data_sb})
    save_path=os.path.join(save_folder,f'I_{cut_size}.mat')
    io.savemat(save_path,{'I':I})
    save_path=os.path.join(save_folder,'center.mat')
    io.savemat(save_path,{'center':center})
    save_path=os.path.join(save_folder,f'F_bin{bin_pixel}by{bin_pixel}.mat')
    io.savemat(save_path,{'F':F})
    
    print("数据处理完成")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='衍射数据处理脚本')
    parser.add_argument('--load', action='store_true', help='加载数据')
    parser.add_argument('--preview', action='store_true', help='生成预览')
    parser.add_argument('--background_subtract', action='store_true', help='背景减除')
    parser.add_argument('--subtract', action='store_true', help='执行减除')
    parser.add_argument('--missing_process', action='store_true', help='缺失数据处理')
    parser.add_argument('--process', action='store_true', help='执行处理')
    parser.add_argument('--center', action='store_true', help='中心查找')
    parser.add_argument('--find', action='store_true', help='执行查找')
    parser.add_argument('--binning', action='store_true', help='Binning处理')
    parser.add_argument('--manual_support', action='store_true', help='手动选择Support区域')
    parser.add_argument('--manual_missing', action='store_true', help='启用手动选择缺失区域')
    parser.add_argument('--auto_background', action='store_true', help='使用自动算法进行背景减除（不需要背景文件）')
    
    # 设置参数解析的严格模式
    parser.allow_abbrev = False
    
    # 文件参数
    parser.add_argument('--diffraction', type=str, help='衍射数据文件路径')
    parser.add_argument('--background', type=str, help='背景数据文件路径')
    parser.add_argument('--missing', type=str, help='缺失数据文件路径')
    
    # 处理参数
    parser.add_argument('--edge_ratio', type=float, default=0.2, help='边缘比例')
    parser.add_argument('--search_range', type=int, default=30, help='搜索范围')
    parser.add_argument('--region_ratio', type=float, default=0.1, help='区域比例')
    parser.add_argument('--binning_size', type=int, default=7, help='Binning大小')
    parser.add_argument('--cut_size', type=int, default=3605, help='裁剪大小')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录（会话文件夹）')
    parser.add_argument('--center_x', type=int, default=None, help='中心X坐标（前端传入）')
    parser.add_argument('--center_y', type=int, default=None, help='中心Y坐标（前端传入）')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Parsed args:")
    print(f"[DEBUG] - diffraction: {args.diffraction}")
    print(f"[DEBUG] - background: {args.background}")
    print(f"[DEBUG] - missing: {args.missing}")
    print(f"[DEBUG] - output_dir: {args.output_dir}")
    
    # 准备参数
    kwargs = {
        'edge_ratio': args.edge_ratio,
        'search_range': args.search_range,
        'region_ratio': args.region_ratio,
        'binning_size': args.binning_size,
        'cut_size': args.cut_size,
        'output_dir': args.output_dir,  # 传递输出目录
        'center_x': args.center_x,  # 前端传入的中心X坐标
        'center_y': args.center_y,  # 前端传入的中心Y坐标
        'auto_background': args.auto_background,  # 自动背景减除模式
    }
    
    print(f"[DEBUG] kwargs: {kwargs}")
    
    # 根据参数执行相应的处理步骤
    import sys
    success = False
    
    if args.load or args.preview:
        print("执行数据加载和预览...")
        success = load_data_only(diffraction_file=args.diffraction, background_file=args.background, missing_file=args.missing, **kwargs)
    elif args.background_subtract or args.subtract:
        print("执行背景减除...")
        success = background_subtraction_only(diffraction_file=args.diffraction, background_file=args.background, missing_file=args.missing, **kwargs)
    elif args.missing_process or args.process:
        print("执行缺失数据处理...")
        print(f"[DEBUG] args.manual_missing = {args.manual_missing}")
        success = missing_data_processing_only(diffraction_file=args.diffraction, background_file=args.background, missing_file=args.missing, manual_selection=args.manual_missing, **kwargs)
    elif args.center or args.find:
        print("执行中心查找...")
        success = center_finding_only(diffraction_file=args.diffraction, background_file=args.background, missing_file=args.missing, **kwargs)
    elif args.binning:
        print("执行Binning处理...")
        success = binning_processing_only(diffraction_file=args.diffraction, background_file=args.background, missing_file=args.missing, **kwargs)
    elif args.manual_support:
        print("执行手动选择Support区域...")
        success = manual_support_selection_only(diffraction_file=args.diffraction, background_file=args.background, missing_file=args.missing, **kwargs)
    else:
        # 默认执行完整流程
        success = main(diffraction_file=args.diffraction, background_file=args.background, missing_file=args.missing, **kwargs)
    
    # 根据处理结果设置退出码
    if not success:
        print("[ERROR] 处理失败")
        sys.exit(1)
    else:
        print("[SUCCESS] 处理完成")
        sys.exit(0)
