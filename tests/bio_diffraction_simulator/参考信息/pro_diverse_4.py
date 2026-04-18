import numpy as np
import condor
import h5py
import multiprocessing
import time
import os
import warnings
from tqdm import tqdm
from functools import partial
from scipy.spatial.transform import Rotation as R
import random
import psutil
import signal

warnings.filterwarnings("ignore")


def monitor_resources():
    """监控系统资源使用情况"""
    resource_info = {"process_memory": 0, "memory_percent_used": 0, "cpu_percent": 0}
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    resource_info["process_memory"] = memory_info.rss / 1024 / 1024
    try:
        virtual_memory = psutil.virtual_memory()
        resource_info["memory_percent_used"] = virtual_memory.percent
        resource_info["available_memory_mb"] = virtual_memory.available / 1024 / 1024
        resource_info["total_memory_mb"] = virtual_memory.total / 1024 / 1024
    except Exception:
        pass
    try:
        resource_info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    except Exception:
        pass
    return resource_info


# ==================== 基本几何形状 ====================

def create_sphere(diameter, dx=1e-9, num_voxels=None):
    """创建球体 - Condor内置支持"""
    if num_voxels is None:
        num_voxels = int(np.ceil(diameter / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = diameter / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    sphere = (distance <= radius).astype(np.float32)
    return sphere


def create_cube(edge_length, dx=1e-9, num_voxels=None):
    """创建立方体 - Condor内置支持"""
    if num_voxels is None:
        num_voxels = int(np.ceil(edge_length / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = edge_length / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    half_edge = edge_length / 2
    cube = ((np.abs(X) <= half_edge) & (np.abs(Y) <= half_edge) & (np.abs(Z) <= half_edge)).astype(np.float32)
    return cube


def create_spheroid(diameter_a, diameter_b, diameter_c, dx=1e-9, num_voxels=None):
    """创建椭球体 - Condor内置支持"""
    max_diameter = max(diameter_a, diameter_b, diameter_c)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_diameter / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_diameter / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius_a, radius_b, radius_c = diameter_a/2, diameter_b/2, diameter_c/2
    ellipsoid = ((X/radius_a)**2 + (Y/radius_b)**2 + (Z/radius_c)**2 <= 1).astype(np.float32)
    return ellipsoid


def create_cylinder(diameter, height, dx=1e-9, num_voxels=None):
    """创建圆柱体"""
    max_size = max(diameter, height)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    half_height = height / 2
    cylinder = ((X**2 + Y**2 <= radius**2) & (np.abs(Z) <= half_height)).astype(np.float32)
    return cylinder


# ==================== 新增几何形状 ====================

def create_cone(diameter, height, dx=1e-9, num_voxels=None):
    """创建圆锥体"""
    max_size = max(diameter, height)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    half_height = height / 2
    z_normalized = (Z + half_height) / height
    current_radius = radius * (1 - z_normalized)
    cone = ((X**2 + Y**2 <= current_radius**2) & (Z >= -half_height) & (Z <= half_height)).astype(np.float32)
    return cone


def create_cuboid(length, width, height, dx=1e-9, num_voxels=None):
    """创建长方体"""
    max_size = max(length, width, height)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cuboid = ((np.abs(X) <= length/2) & (np.abs(Y) <= width/2) & (np.abs(Z) <= height/2)).astype(np.float32)
    return cuboid


def create_hollow_sphere(outer_diameter, shell_thickness, dx=1e-9, num_voxels=None):
    """创建空心球体"""
    if num_voxels is None:
        num_voxels = int(np.ceil(outer_diameter / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = outer_diameter / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    outer_radius = outer_diameter / 2
    inner_radius = outer_radius - shell_thickness
    hollow_sphere = ((distance <= outer_radius) & (distance > inner_radius)).astype(np.float32)
    return hollow_sphere


def create_bipyramid(diameter, height, dx=1e-9, num_voxels=None):
    """创建双锥体(菱形)"""
    max_size = max(diameter, height)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    half_height = height / 2
    current_radius = radius * (1 - np.abs(Z) / half_height)
    bipyramid = ((X**2 + Y**2 <= current_radius**2) & (np.abs(Z) <= half_height)).astype(np.float32)
    return bipyramid


def create_truncated_cone(bottom_diameter, top_diameter, height, dx=1e-9, num_voxels=None):
    """创建截锥体(台锥)"""
    max_size = max(bottom_diameter, top_diameter, height)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    bottom_radius = bottom_diameter / 2
    top_radius = top_diameter / 2
    half_height = height / 2
    z_normalized = (Z + half_height) / height
    current_radius = bottom_radius + (top_radius - bottom_radius) * z_normalized
    truncated_cone = ((X**2 + Y**2 <= current_radius**2) & (Z >= -half_height) & (Z <= half_height)).astype(np.float32)
    return truncated_cone


def create_octahedron(diameter, dx=1e-9, num_voxels=None):
    """创建正八面体 - Condor内置支持(icosahedron)"""
    if num_voxels is None:
        num_voxels = int(np.ceil(diameter / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = diameter / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    octahedron_size = diameter / 2
    octahedron = (np.abs(X) + np.abs(Y) + np.abs(Z) <= octahedron_size).astype(np.float32)
    return octahedron


def create_tetrahedron(edge_length, dx=1e-9, num_voxels=None):
    """创建正四面体"""
    max_size = edge_length * 0.82
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    v1 = np.array([1, 0, -1/np.sqrt(2)]) * edge_length / (2*np.sqrt(2))
    v2 = np.array([-1, 0, -1/np.sqrt(2)]) * edge_length / (2*np.sqrt(2))
    v3 = np.array([0, 1, 1/np.sqrt(2)]) * edge_length / (2*np.sqrt(2))
    v4 = np.array([0, -1, 1/np.sqrt(2)]) * edge_length / (2*np.sqrt(2))
    n1 = np.cross(v2-v1, v3-v1)
    n1 = n1 / np.linalg.norm(n1)
    d1 = -np.dot(n1, v1)
    n2 = np.cross(v3-v1, v4-v1)
    n2 = n2 / np.linalg.norm(n2)
    d2 = -np.dot(n2, v1)
    n3 = np.cross(v4-v2, v3-v2)
    n3 = n3 / np.linalg.norm(n3)
    d3 = -np.dot(n3, v2)
    n4 = np.cross(v2-v4, v1-v4)
    n4 = n4 / np.linalg.norm(n4)
    d4 = -np.dot(n4, v4)
    point = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    side1 = np.dot(point, n1) + d1
    side2 = np.dot(point, n2) + d2
    side3 = np.dot(point, n3) + d3
    side4 = np.dot(point, n4) + d4
    inside = (side1 <= 0) & (side2 <= 0) & (side3 <= 0) & (side4 <= 0)
    tetrahedron = inside.reshape(X.shape).astype(np.float32)
    return tetrahedron


def create_torus(major_radius, minor_radius, dx=1e-9, num_voxels=None):
    """创建圆环体"""
    diameter = 2 * (major_radius + minor_radius)
    if num_voxels is None:
        num_voxels = int(np.ceil(diameter / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = diameter / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dist_from_origin_xz = np.sqrt(X**2 + Z**2)
    dist_from_circle = np.abs(dist_from_origin_xz - major_radius)
    torus = (dist_from_circle**2 + Y**2 <= minor_radius**2).astype(np.float32)
    return torus


def create_prism(diameter, height, sides=6, dx=1e-9, num_voxels=None):
    """创建正棱柱"""
    max_size = max(diameter, height)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    half_height = height / 2
    height_mask = (np.abs(Z) <= half_height)
    radius = diameter / 2
    angles = np.linspace(0, 2*np.pi, sides, endpoint=False)
    prism = np.ones_like(X, dtype=bool)
    for i in range(sides):
        theta1 = angles[i]
        theta2 = angles[(i + 1) % sides]
        mid_angle = (theta1 + theta2) / 2
        nx = -np.cos(mid_angle)
        ny = -np.sin(mid_angle)
        px = radius * np.cos(theta1)
        py = radius * np.sin(theta1)
        prism &= (nx * (X - px) + ny * (Y - py) <= 0)
    prism = (prism & height_mask).astype(np.float32)
    return prism


def create_rounded_cube(edge_length, corner_radius, dx=1e-9, num_voxels=None):
    """创建圆角立方体"""
    if num_voxels is None:
        num_voxels = int(np.ceil(edge_length / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = edge_length / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    cube_half_size = edge_length / 2
    dx_array = np.maximum(np.abs(X) - (cube_half_size - corner_radius), 0)
    dy_array = np.maximum(np.abs(Y) - (cube_half_size - corner_radius), 0)
    dz_array = np.maximum(np.abs(Z) - (cube_half_size - corner_radius), 0)
    distance = np.sqrt(dx_array**2 + dy_array**2 + dz_array**2)
    rounded_cube = (distance <= corner_radius).astype(np.float32)
    return rounded_cube


def create_nanobox(edge_length, wall_thickness, dx=1e-9, num_voxels=None):
    """创建纳米盒(空心立方体)"""
    if num_voxels is None:
        num_voxels = int(np.ceil(edge_length / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = edge_length / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    half_outer = edge_length / 2
    outer_cube = ((np.abs(X) <= half_outer) & (np.abs(Y) <= half_outer) & (np.abs(Z) <= half_outer))
    half_inner = half_outer - wall_thickness
    inner_cube = ((np.abs(X) <= half_inner) & (np.abs(Y) <= half_inner) & (np.abs(Z) <= half_inner))
    nanobox = (outer_cube & ~inner_cube).astype(np.float32)
    return nanobox


def create_rod(diameter, length, dx=1e-9, num_voxels=None):
    """创建纳米棒(圆柱体+两端半球)"""
    max_size = max(diameter, length)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    half_length = length / 2
    cylinder_part = (X**2 + Y**2 <= radius**2) & (np.abs(Z) <= (half_length - radius))
    cap1 = (X**2 + Y**2 + (Z - (half_length - radius))**2 <= radius**2) & (Z > (half_length - radius))
    cap2 = (X**2 + Y**2 + (Z + (half_length - radius))**2 <= radius**2) & (Z < -(half_length - radius))
    rod = (cylinder_part | cap1 | cap2).astype(np.float32)
    return rod


# ==================== 生物样品形状 ====================

def create_bacteria_rod(length, diameter, dx=1e-9, num_voxels=None):
    """创建杆状细菌形状（类似大肠杆菌）"""
    max_size = max(length, diameter)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    half_length = length / 2
    # 胶囊体模型：圆柱体 + 两端半球
    cylinder_part = (Y**2 + Z**2 <= radius**2) & (np.abs(X) <= (half_length - radius))
    cap1 = ((X - (half_length - radius))**2 + Y**2 + Z**2 <= radius**2) & (X > (half_length - radius))
    cap2 = ((X + (half_length - radius))**2 + Y**2 + Z**2 <= radius**2) & (X < -(half_length - radius))
    bacteria = (cylinder_part | cap1 | cap2).astype(np.float32)
    return bacteria


def create_bacteria_sphere(diameter, dx=1e-9, num_voxels=None):
    """创建球形细菌"""
    return create_sphere(diameter, dx, num_voxels)


def create_virus_capsid(diameter, spike_length=0, dx=1e-9, num_voxels=None):
    """创建病毒衣壳（如冠状病毒）"""
    max_size = diameter + 2 * spike_length
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # 基础球体（病毒核心）
    core_radius = diameter / 2
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    virus = (distance <= core_radius).astype(np.float32)
    # 添加刺突
    if spike_length > 0:
        n_spikes = 12
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_spikes):
            theta = 2 * np.pi * i / golden_ratio
            phi = np.arccos(1 - 2 * (i + 0.5) / n_spikes)
            spike_dir = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            spike_start = spike_dir * core_radius
            spike_end = spike_dir * (core_radius + spike_length)
            spike_radius_max = core_radius * 0.1
            spike_radius_min = core_radius * 0.02
            P = np.stack([X, Y, Z], axis=-1)
            v = spike_end - spike_start
            v_norm = np.linalg.norm(v)
            if v_norm > 0:
                v_unit = v / v_norm
                t = np.dot(P - spike_start, v_unit)
                t = np.clip(t, 0, v_norm)
                projection = spike_start + t[..., np.newaxis] * v_unit
                dist_to_axis = np.sqrt(np.sum((P - projection)**2, axis=-1))
                current_radius = spike_radius_max * (1 - t/v_norm) + spike_radius_min * (t/v_norm)
                spike = (dist_to_axis <= current_radius) & (t >= 0) & (t <= v_norm)
                virus = np.maximum(virus, spike.astype(np.float32))
    return virus


def create_cell_nucleus(diameter, membrane_thickness=0, dx=1e-9, num_voxels=None):
    """创建细胞核形状"""
    max_size = diameter
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    if membrane_thickness > 0:
        outer_radius = radius
        inner_radius = radius - membrane_thickness
        nucleus = ((distance <= outer_radius) & (distance > inner_radius)).astype(np.float32)
    else:
        nucleus = (distance <= radius).astype(np.float32)
    return nucleus


def create_mitochondria(length, diameter, cristae_density=0.3, dx=1e-9, num_voxels=None):
    """创建线粒体形状"""
    max_size = max(length, diameter)
    if num_voxels is None:
        num_voxels = int(np.ceil(max_size / dx))
        if num_voxels % 2 == 1:
            num_voxels += 1
    half_size = max_size / 2
    x = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    y = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    z = np.linspace(-half_size, half_size, num_voxels, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    radius = diameter / 2
    half_length = length / 2
    # 胶囊体基础形状
    cylinder_part = (Y**2 + Z**2 <= radius**2) & (np.abs(X) <= (half_length - radius))
    cap1 = ((X - (half_length - radius))**2 + Y**2 + Z**2 <= radius**2) & (X > (half_length - radius))
    cap2 = ((X + (half_length - radius))**2 + Y**2 + Z**2 <= radius**2) & (X < -(half_length - radius))
    mitochondria = (cylinder_part | cap1 | cap2).astype(np.float32)
    # 添加内部嵴结构
    if cristae_density > 0:
        n_cristae = max(3, int(cristae_density * 8))
        for i in range(n_cristae):
            cristae_pos = -half_length + (i + 0.5) * (length / (n_cristae + 1))
            cristae_radius = radius * 0.7
            cristae = ((X - cristae_pos)**2 / (radius*0.3)**2 + 
                      Y**2 / cristae_radius**2 + 
                      Z**2 / cristae_radius**2 <= 1)
            cristae = cristae & (mitochondria > 0)
            mitochondria[cristae] = 1.0
    return mitochondria


def create_particle(particle_type='sphere', diameter=500e-9, aspect_ratio=None, 
                   corner_radius=None, wall_thickness=None, sides=6, 
                   spike_length=None, cristae_density=0.3,
                   dx=1e-9, rotation_values=None, material='Au'):
    """
    创建多样化的粒子形状 - 使用Condor ParticleMap
    
    支持的形状类型:
    
    几何形状 (16种):
    - sphere: 球体 (Condor内置)
    - cube: 立方体
    - spheroid: 椭球体 (Condor内置)
    - cylinder: 圆柱体
    - cone: 圆锥体
    - cuboid: 长方体
    - hollow_sphere: 空心球体
    - bipyramid: 双锥体
    - truncated_cone: 截锥体
    - octahedron: 八面体
    - tetrahedron: 四面体
    - torus: 圆环体
    - prism: 正棱柱 (六边形等)
    - rounded_cube: 圆角立方体
    - nanobox: 纳米盒(空心立方体)
    - rod: 纳米棒
    
    生物样品 (5种):
    - bacteria_rod: 杆状细菌
    - bacteria_sphere: 球形细菌
    - virus_capsid: 病毒衣壳(带刺突)
    - cell_nucleus: 细胞核
    - mitochondria: 线粒体
    """
    
    # 确定实际尺寸
    if aspect_ratio is None:
        aspect_ratio = 1.0
    
    # 根据粒子类型创建几何形状
    if particle_type == 'sphere':
        # 使用Condor内置的ParticleSphere
        particle = condor.ParticleSphere(
            diameter=diameter,
            material_type=material.lower() if material in ['protein', 'virus', 'cell', 'water'] else None,
            number=1,
            arrival='synchronised',
            position=[0, 0, 0],
            rotation_formalism='quaternion',
            rotation_mode='extrinsic',
            rotation_values=rotation_values if rotation_values is not None else [[1, 0, 0, 0]]
        )
        print(f"Created {particle_type} using Condor ParticleSphere")
        return particle
        
    elif particle_type == 'spheroid':
        # 使用Condor内置的ParticleSpheroid
        flattening = aspect_ratio if aspect_ratio > 0 else 1.0
        particle = condor.ParticleSpheroid(
            diameter=diameter,
            flattening=flattening,
            material_type=material.lower() if material in ['protein', 'virus', 'cell', 'water'] else None,
            number=1,
            arrival='synchronised',
            position=[0, 0, 0],
            rotation_formalism='quaternion',
            rotation_mode='extrinsic',
            rotation_values=rotation_values if rotation_values is not None else [[1, 0, 0, 0]]
        )
        print(f"Created {particle_type} using Condor ParticleSpheroid with flattening={flattening}")
        return particle
    
    # 对于自定义形状，使用ParticleMap
    num_voxels = int(np.ceil(diameter / dx))
    if num_voxels % 2 == 1:
        num_voxels += 1
    
    # 生成几何形状
    if particle_type == 'cube':
        geometry_array = create_cube(diameter, dx, num_voxels)
    elif particle_type == 'cylinder':
        height = diameter * aspect_ratio
        geometry_array = create_cylinder(diameter, height, dx, num_voxels)
    elif particle_type == 'cone':
        height = diameter * aspect_ratio
        geometry_array = create_cone(diameter, height, dx, num_voxels)
    elif particle_type == 'cuboid':
        length = diameter
        width = diameter * 0.8
        height = diameter * aspect_ratio
        geometry_array = create_cuboid(length, width, height, dx, num_voxels)
    elif particle_type == 'hollow_sphere':
        thickness = wall_thickness if wall_thickness else diameter * 0.1
        geometry_array = create_hollow_sphere(diameter, thickness, dx, num_voxels)
    elif particle_type == 'bipyramid':
        height = diameter * aspect_ratio
        geometry_array = create_bipyramid(diameter, height, dx, num_voxels)
    elif particle_type == 'truncated_cone':
        height = diameter * aspect_ratio
        top_diameter = diameter * 0.5
        geometry_array = create_truncated_cone(diameter, top_diameter, height, dx, num_voxels)
    elif particle_type == 'octahedron':
        geometry_array = create_octahedron(diameter, dx, num_voxels)
    elif particle_type == 'tetrahedron':
        edge_length = diameter * np.sqrt(6) / 2
        geometry_array = create_tetrahedron(edge_length, dx, num_voxels)
    elif particle_type == 'torus':
        major_radius = diameter * 0.35
        minor_radius = diameter * 0.15
        geometry_array = create_torus(major_radius, minor_radius, dx, num_voxels)
    elif particle_type == 'prism':
        height = diameter * aspect_ratio
        geometry_array = create_prism(diameter, height, sides, dx, num_voxels)
    elif particle_type == 'rounded_cube':
        radius = corner_radius if corner_radius else diameter * 0.1
        geometry_array = create_rounded_cube(diameter, radius, dx, num_voxels)
    elif particle_type == 'nanobox':
        thickness = wall_thickness if wall_thickness else diameter * 0.1
        geometry_array = create_nanobox(diameter, thickness, dx, num_voxels)
    elif particle_type == 'rod':
        length = diameter * aspect_ratio
        geometry_array = create_rod(diameter, length, dx, num_voxels)
    
    # 生物样品
    elif particle_type == 'bacteria_rod':
        length = diameter * aspect_ratio
        geometry_array = create_bacteria_rod(length, diameter, dx, num_voxels)
    elif particle_type == 'bacteria_sphere':
        geometry_array = create_bacteria_sphere(diameter, dx, num_voxels)
    elif particle_type == 'virus_capsid':
        spike_len = spike_length if spike_length else diameter * 0.2
        geometry_array = create_virus_capsid(diameter, spike_len, dx, num_voxels)
    elif particle_type == 'cell_nucleus':
        thickness = wall_thickness if wall_thickness else diameter * 0.05
        geometry_array = create_cell_nucleus(diameter, thickness, dx, num_voxels)
    elif particle_type == 'mitochondria':
        length = diameter * aspect_ratio
        geometry_array = create_mitochondria(length, diameter, cristae_density, dx, num_voxels)
    else:
        raise ValueError(f"Unsupported particle type: {particle_type}")
    
    # 设置材料属性
    if material == 'Au':
        atomic_composition = {'Au': 1.0}
        massdensity = 19320
    elif material == 'Ag':
        atomic_composition = {'Ag': 1.0}
        massdensity = 10490
    elif material == 'Pt':
        atomic_composition = {'Pt': 1.0}
        massdensity = 21450
    elif material == 'Pd':
        atomic_composition = {'Pd': 1.0}
        massdensity = 12023
    elif material == 'Cu':
        atomic_composition = {'Cu': 1.0}
        massdensity = 8960
    elif material == 'SiO2':
        atomic_composition = {'Si': 0.33, 'O': 0.67}
        massdensity = 2200
    elif material == 'TiO2':
        atomic_composition = {'Ti': 0.33, 'O': 0.67}
        massdensity = 4230
    # 生物材料
    elif material == 'protein':
        atomic_composition = {'C': 0.5, 'H': 0.07, 'O': 0.24, 'N': 0.16, 'S': 0.03}
        massdensity = 1300
    elif material == 'lipid':
        atomic_composition = {'C': 0.75, 'H': 0.12, 'O': 0.10, 'N': 0.02, 'P': 0.01}
        massdensity = 900
    elif material == 'water':
        atomic_composition = {'H': 0.67, 'O': 0.33}
        massdensity = 1000
    elif material == 'dna':
        atomic_composition = {'C': 0.34, 'H': 0.13, 'O': 0.33, 'N': 0.15, 'P': 0.05}
        massdensity = 1700
    else:
        atomic_composition = {'Au': 1.0}
        massdensity = 19320
    
    # 创建ParticleMap对象
    particle = condor.ParticleMap(
        geometry='custom',
        diameter=diameter,
        dx=dx,
        rotation_formalism='quaternion',
        rotation_mode='extrinsic',
        rotation_values=rotation_values if rotation_values is not None else [[1, 0, 0, 0]],
        number=1,
        arrival='synchronised',
        position=[0, 0, 0],
        atomic_composition=atomic_composition,
        massdensity=massdensity
    )
    
    # 设置自定义几何
    particle.set_custom_geometry_by_array(geometry_array[np.newaxis, ...], dx)
    
    print(f"Created {particle_type} with shape {geometry_array.shape}, size={diameter*1e9:.1f}nm, material={material}")
    
    return particle


def generate_sample(args, fixed_params):
    """生成单个衍射图样"""
    i, seed = args
    det_params, src_params, particle_params = fixed_params
    
    # 初始化随机数生成器
    rng = np.random.RandomState(seed)
    
    try:
        # 监控资源
        if i % 10 == 0:
            print(f"Processing sample {i}...")
        
        # 随机选择粒子类型
        particle_type = rng.choice(particle_params['types'])
        
        # 随机选择尺寸
        diameter = rng.uniform(particle_params['size_range'][0], particle_params['size_range'][1])
        
        # 随机选择材料
        material = rng.choice(particle_params['materials'])
        
        # 设置形状特定参数
        aspect_ratio = None
        corner_radius = None
        wall_thickness = None
        sides = 6
        spike_length = None
        cristae_density = 0.3
        
        if particle_type in ['cylinder', 'cone', 'cuboid', 'bipyramid', 'truncated_cone', 'prism', 'rod']:
            aspect_ratio = rng.uniform(1.5, 3.0)
        
        if particle_type == 'rounded_cube':
            corner_radius = diameter * rng.uniform(0.05, 0.15)
        
        if particle_type in ['hollow_sphere', 'nanobox']:
            wall_thickness = diameter * rng.uniform(0.1, 0.2)
        
        if particle_type == 'prism':
            sides = rng.choice([5, 6, 8])
        
        # 生物样品特殊参数
        if particle_type in ['bacteria_rod', 'mitochondria']:
            aspect_ratio = rng.uniform(2.0, 3.5)
        
        if particle_type == 'virus_capsid':
            spike_length = diameter * rng.uniform(0.1, 0.25)
        
        if particle_type == 'cell_nucleus':
            wall_thickness = diameter * rng.uniform(0.05, 0.1)
        
        if particle_type == 'mitochondria':
            cristae_density = rng.uniform(0.2, 0.4)
        
        # 生成随机旋转
        alpha, beta, gamma = rng.uniform(0, 2*np.pi, 3)
        q0 = np.cos(alpha/2) * np.cos(beta/2) * np.cos(gamma/2) + np.sin(alpha/2) * np.sin(beta/2) * np.sin(gamma/2)
        q1 = np.sin(alpha/2) * np.cos(beta/2) * np.cos(gamma/2) - np.cos(alpha/2) * np.sin(beta/2) * np.sin(gamma/2)
        q2 = np.cos(alpha/2) * np.sin(beta/2) * np.cos(gamma/2) + np.sin(alpha/2) * np.cos(beta/2) * np.sin(gamma/2)
        q3 = np.cos(alpha/2) * np.cos(beta/2) * np.sin(gamma/2) - np.sin(alpha/2) * np.sin(beta/2) * np.cos(gamma/2)
        rotation_values = np.array([[q0, q1, q2, q3]])
        
        # 设置dx
        dx = 10e-9 if diameter < 1e-6 else 15e-9
        
        # 创建粒子
        par = create_particle(
            particle_type=particle_type,
            diameter=diameter,
            aspect_ratio=aspect_ratio,
            corner_radius=corner_radius,
            wall_thickness=wall_thickness,
            sides=sides,
            spike_length=spike_length,
            cristae_density=cristae_density,
            dx=dx,
            rotation_values=rotation_values,
            material=material
        )
        
        # 创建实验装置
        src = condor.Source(**src_params)
        det = condor.Detector(**det_params)
        
        # 运行模拟
        if particle_type in ['sphere', 'spheroid']:
            E = condor.Experiment(src, {f"particle_{particle_type}": par}, det)
        else:
            E = condor.Experiment(src, {"particle_map": par}, det)
        res = E.propagate()
        
        # 提取强度数据
        intensity = res["entry_1"]["data_1"]["data"]
        
        # 记录参数
        params = {
            "particle_type": particle_type,
            "material": material,
            "diameter": diameter,
            "dx": dx,
            "wavelength": src_params['wavelength'],
            "pulse_energy": src_params['pulse_energy'],
            "detector_distance": det_params['distance']
        }
        
        if aspect_ratio:
            params["aspect_ratio"] = aspect_ratio
        if corner_radius:
            params["corner_radius"] = corner_radius
        if wall_thickness:
            params["wall_thickness"] = wall_thickness
        if particle_type == 'prism':
            params["sides"] = sides
        
        return {
            "index": i,
            "intensity": intensity,
            "rotation": rotation_values[0],
            "params": params
        }
    
    except Exception as e:
        print(f"Error in sample {i}: {str(e)}")
        return {"index": i, "error": str(e)}


def generate_sample_wrapper(args):
    """多进程包装函数"""
    task_args, fixed_params = args
    return generate_sample(task_args, fixed_params)


def create_diffraction_dataset(n_samples, det_params, src_params, particle_params, max_processes=2):
    """创建衍射数据集"""
    print(f"Generating {n_samples} diverse diffraction patterns...")
    print(f"Particle types: {particle_params['types']}")
    print(f"Size range: {particle_params['size_range']}")
    print(f"Materials: {particle_params['materials']}")
    
    # 准备任务
    base_seed = 42
    tasks = [(i, base_seed + i) for i in range(n_samples)]
    fixed_args = (det_params, src_params, particle_params)
    task_args = [((i, base_seed + i), fixed_args) for i in range(n_samples)]
    
    # 并行处理
    results = []
    with multiprocessing.Pool(processes=max_processes) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample_wrapper, task_args), 
                          total=n_samples, desc="Generating patterns"):
            results.append(result)
        pool.close()
        pool.join()
    
    # 筛选成功结果
    successful_results = [r for r in results if "error" not in r]
    failed_results = [r for r in results if "error" in r]
    
    print(f"\nSuccessful: {len(successful_results)}/{n_samples}")
    if failed_results:
        print(f"Failed: {len(failed_results)}/{n_samples}")
        for i, failure in enumerate(failed_results[:3]):
            print(f"  Sample {i+1}: {failure['error']}")
    
    if not successful_results:
        print("No successful samples generated!")
        return None
    
    # 按索引排序
    successful_results.sort(key=lambda x: x["index"])
    
    # 统计信息
    stats = {"particle_types": {}, "materials": {}}
    for res in successful_results:
        ptype = res["params"]["particle_type"]
        mat = res["params"]["material"]
        stats["particle_types"][ptype] = stats["particle_types"].get(ptype, 0) + 1
        stats["materials"][mat] = stats["materials"].get(mat, 0) + 1
    
    return {
        "samples": successful_results,
        "stats": stats,
        "params": {"detector": det_params, "source": src_params, "particle": particle_params}
    }


def save_dataset(dataset, output_file):
    """保存数据集到HDF5文件"""
    if not dataset or not dataset["samples"]:
        print(f"Warning: No samples to save to {output_file}")
        return
    
    successful_results = dataset["samples"]
    stats = dataset["stats"]
    params = dataset["params"]
    
    num_successful = len(successful_results)
    
    # 保存到HDF5
    with h5py.File(output_file, 'w') as f:
        train_group = f.create_group("train")
        validation_group = f.create_group("validation")
        test_group = f.create_group("test")
        
        # 全局参数
        f.attrs["wavelength_base"] = params["source"]["wavelength"]
        f.attrs["pulse_energy_base"] = params["source"]["pulse_energy"]
        f.attrs["detector_distance_base"] = params["detector"]["distance"]
        f.attrs["successful_samples"] = num_successful
        
        # 分割数据集
        train_size = int(num_successful * 0.7)
        val_size = int(num_successful * 0.15)
        
        for i, res in enumerate(successful_results):
            idx = res["index"]
            if i < train_size:
                group = train_group
            elif i < train_size + val_size:
                group = validation_group
            else:
                group = test_group
            
            sub_group = group.create_group(f"pattern_{idx}")
            sub_group.create_dataset("intensity", data=res["intensity"])
            sub_group.create_dataset("rotation", data=res["rotation"])
            
            for name, value in res["params"].items():
                if isinstance(value, str):
                    sub_group.attrs[name] = np.bytes_(value)
                else:
                    sub_group.attrs[name] = value
    
    # 打印统计
    print("\nDataset Statistics:")
    print(f"Total samples: {num_successful}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {num_successful - train_size - val_size}")
    
    print("\nParticle type distribution:")
    for ptype, count in stats["particle_types"].items():
        print(f"  {ptype}: {count} ({count/num_successful*100:.1f}%)")
    
    print("\nMaterial distribution:")
    for mat, count in stats["materials"].items():
        print(f"  {mat}: {count} ({count/num_successful*100:.1f}%)")
    
    print(f"\nData saved to {output_file}")


def main():
    import sys
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    print(f"Generating {n_samples} samples with diverse particle shapes...")
    
    # 探测器参数
    det_params = {
        'distance': 0.5,
        'pixel_size': 120e-6,
        'nx': 512,
        'ny': 512
    }
    
    # 光源参数
    src_params = {
        'wavelength': 2.7e-9,
        'pulse_energy': 5e-6,
        'focus_diameter': 10e-6
    }
    
    # 粒子参数 - 包含更多多样化的形状
    # 几何形状 (16种)
    geometric_shapes = [
        'sphere', 'cube', 'spheroid', 'cylinder', 'cone', 
        'cuboid', 'hollow_sphere', 'bipyramid', 'truncated_cone',
        'octahedron', 'tetrahedron', 'torus', 'prism',
        'rounded_cube', 'nanobox', 'rod'
    ]
    
    # 生物样品 (5种)
    biological_shapes = [
        'bacteria_rod', 'bacteria_sphere', 'virus_capsid',
        'cell_nucleus', 'mitochondria'
    ]
    
    # 合并所有形状 (21种)
    all_shapes = geometric_shapes + biological_shapes
    
    # 材料列表 - 无机材料 + 生物材料
    all_materials = ['Au', 'Ag', 'Pt', 'Pd', 'Cu', 'SiO2', 'TiO2',
                     'protein', 'lipid', 'water', 'dna']
    
    particle_params = {
        'types': all_shapes,
        'materials': all_materials,
        'size_range': [500e-9, 1000e-9]  # 500nm - 1um
    }
    
    try:
        dataset = create_diffraction_dataset(
            n_samples=n_samples,
            det_params=det_params,
            src_params=src_params,
            particle_params=particle_params,
            max_processes=2
        )
        
        if dataset:
            save_dataset(dataset, "diffraction_data_diverse_v4.h5")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("="*60)
    print("pro_diverse_4.py - Enhanced Diverse Particle Shape Generator")
    print("="*60)
    print("\n🔬 支持的粒子形状 (21种):\n")
    
    print("【几何形状 - 16种】")
    geometric_shapes = [
        'sphere', 'cube', 'spheroid', 'cylinder', 'cone',
        'cuboid', 'hollow_sphere', 'bipyramid', 'truncated_cone',
        'octahedron', 'tetrahedron', 'torus', 'prism',
        'rounded_cube', 'nanobox', 'rod'
    ]
    for i, shape in enumerate(geometric_shapes, 1):
        print(f"  {i:2d}. {shape}")
    
    print("\n【生物样品 - 5种】")
    biological_shapes = [
        'bacteria_rod', 'bacteria_sphere', 'virus_capsid',
        'cell_nucleus', 'mitochondria'
    ]
    for i, shape in enumerate(biological_shapes, 17):
        print(f"  {i:2d}. {shape}")
    
    print("\n🧪 支持的材料 (11种):")
    print("  无机材料: Au, Ag, Pt, Pd, Cu, SiO2, TiO2")
    print("  生物材料: protein, lipid, water, dna")
    
    print("="*60)
    print("\nRun with: python pro_diverse_4.py <num_samples>")
    print("Example: python pro_diverse_4.py 100\n")
    
    # 如果直接运行，执行main
    if len(__import__('sys').argv) > 1:
        main()
