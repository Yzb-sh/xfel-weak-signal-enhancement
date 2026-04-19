"""
Backend abstraction for CPU/GPU computation selection.

Provides a unified interface to switch between numpy (CPU) and CuPy (GPU)
for array and scipy.ndimage operations.
"""

import numpy as np


def check_gpu_available():
    """Verify CuPy + CUDA GPU is available.

    Returns:
        Tuple of (success: bool, message: str).
    """
    try:
        import cupy
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count == 0:
            return False, "No CUDA GPU detected"
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props.get('name', b'Unknown GPU')
        if isinstance(name, bytes):
            name = name.decode()
        free_mem, total_mem = cupy.cuda.runtime.memGetInfo()
        free_gb = free_mem / 1024**3
        total_gb = total_mem / 1024**3
        return True, f"GPU: {name}, {free_gb:.1f}/{total_gb:.1f} GB free"
    except ImportError:
        return False, "CuPy not installed. Run: pip install cupy-cuda12x"
    except Exception as e:
        return False, f"GPU check failed: {e}"


def get_xp(use_gpu=False):
    """Return cupy (GPU) or numpy (CPU) module.

    Args:
        use_gpu: If True, return cupy module. Otherwise return numpy.

    Returns:
        numpy or cupy module.
    """
    if use_gpu:
        import cupy as xp
        return xp
    return np


def get_ndimage(use_gpu=False):
    """Return cupyx.scipy.ndimage (GPU) or scipy.ndimage (CPU).

    Args:
        use_gpu: If True, return cupyx.scipy.ndimage. Otherwise return scipy.ndimage.

    Returns:
        scipy.ndimage or cupyx.scipy.ndimage module.
    """
    if use_gpu:
        import cupyx.scipy.ndimage as ndimage
        return ndimage
    import scipy.ndimage as ndimage
    return ndimage


def to_gpu(arr, xp):
    """Transfer numpy array to compute device.

    No-op if xp is numpy (CPU mode) or array is already on device.

    Args:
        arr: Input array (numpy or cupy).
        xp: Target backend module (numpy or cupy).

    Returns:
        Array on target device.
    """
    if xp is np:
        return arr
    return xp.asarray(arr)


def to_cpu(arr):
    """Transfer array back to CPU as numpy array.

    No-op if array is already a numpy array.

    Args:
        arr: Input array (numpy or cupy).

    Returns:
        numpy array on CPU.
    """
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)
