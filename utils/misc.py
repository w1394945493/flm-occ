
import logging
from contextlib import contextmanager
import numpy as np

import torch



def map_level(level):
    """
    Map a string or integer level to a logging level.
    
    Args:
        level (str or int): e.g. 'info', 'DEBUG', 20, etc.

    Returns:
        int: Corresponding logging level.

    Raises:
        ValueError: If the level is not recognized.
    """
    if isinstance(level, int):
        if level in logging._levelToName:
            return level
        else:
            raise ValueError(f"Unknown log level: {level}")
    
    level = level.upper()
    if level in logging._nameToLevel:
        return logging._nameToLevel[level]
    
    raise ValueError(f"Unknown log level: {level}")


def create_logger(name, log_level='info', path_save="debug.log", mode='w', file_level='debug'):
    log_level = map_level(log_level)
    file_level = map_level(file_level)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    # formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    file_handler = logging.FileHandler(path_save, mode=mode)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


@contextmanager
def gpu_timing(description='GPU time: ', func_output=None, skip=False):
    """
    准确测量GPU操作的执行时间
    """
    if skip:
        yield
        return
    # 确保之前的所有GPU操作完成
    torch.cuda.synchronize()
    
    # 创建CUDA事件用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 记录起始时间
    start_event.record()
    
    yield
    
    # 记录结束时间
    end_event.record()
    
    # 等待所有GPU操作完成
    torch.cuda.synchronize()
    
    # 计算耗时（毫秒）
    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
    
    if func_output is None:
        print(f"{description}{elapsed_time:.6f} s")
    else:
        func_output(elapsed_time)
        
        
def gpu_timing_repeat(func, *fargs, repeat=10, **fkwargs):
    """
    重复测量GPU操作的执行时间，返回平均耗时
    """
    times = []
    for _ in range(repeat):
        with gpu_timing(func_output=times.append):
            output = func(*fargs, **fkwargs)
    return sum(times) / len(times), output


@contextmanager
def gpu_memory_usage(description='GPU memory usage: ', func_print=None, func_output=None, skip=False):
    """
    准确测量GPU内存使用情况
    """
    if skip:
        yield
        return
    
    # 确保之前的所有GPU操作完成
    torch.cuda.synchronize()
    
    # 获取当前GPU内存使用情况
    initial_memory = torch.cuda.memory_allocated()
    
    yield
    
    # 获取操作后的GPU内存使用情况
    final_memory = torch.cuda.memory_allocated()
    
    # 计算内存变化（字节）
    memory_change = final_memory - initial_memory
    
    if func_output is None:
        msg = f"{description}{memory_change / (1024 ** 3):.3f} GB"
        if func_print is not None:
            func_print(msg)
        else:
            print(msg)
    else:
        func_output(memory_change / (1024 ** 3))


def save_gs_ply_binary(filename, means, scales, wxyz, e1e2=None, colors=None):
    """
    Save Gaussian Splatting data to binary PLY format for Blender Geometry Nodes.

    Args:
        filename (str): Output .ply file path.
        means (np.ndarray): (N, 3) float32, positions (x, y, z)
        scales (np.ndarray): (N, 3) float32, scales (sx, sy, sz)
        wxyz (np.ndarray): (N, 4) float32, quaternions (w, x, y, z)
        e1e2 (np.ndarray or None): (N, 2) float32, optional parameters (e1, e2) for superquadrics
        colors (np.ndarray or None): (N, 3) uint8, RGB colors in [0,1]
    """
    N = means.shape[0]

    # 验证输入形状
    assert means.shape == (N, 3), f"means shape {means.shape} != ({N}, 3)"
    assert scales.shape == (N, 3), f"scales shape {scales.shape} != ({N}, 3)"
    assert wxyz.shape == (N, 4), f"wxyz shape {wxyz.shape} != ({N}, 4)"
    if e1e2 is not None:
        assert e1e2.shape == (N, 2), f"e1e2 shape {e1e2.shape} != ({N}, 2)"
    if colors is not None:
        assert colors.shape == (N, 3), f"colors shape {colors.shape} != ({N}, 3)"

    # 转为 float32 确保兼容
    means = means.astype(np.float32)
    scales = scales.astype(np.float32)
    wxyz = wxyz.astype(np.float32)
    if colors is not None:
        assert colors.dtype == np.uint8, f"colors dtype {colors.dtype} != uint8"

    # 构建 header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Generated for Blender Geometry Nodes - VolMix",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
        "property float scale_x",
        "property float scale_y",
        "property float scale_z",
        "property float quat_w",   # 注意：Blender 用 w,x,y,z 顺序，我们按此保存
        "property float quat_x",
        "property float quat_y",
        "property float quat_z",
    ]
    
    if e1e2 is not None:
        header_lines += [
            "property float e1",
            "property float e2",
        ]

    if colors is not None:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    # 写入文件
    with open(filename, "wb") as f:
        f.write(header.encode('ascii'))

        for i in range(N):
            # 位置
            f.write(means[i].tobytes())   # x, y, z

            # 缩放
            f.write(scales[i].tobytes())  # sx, sy, sz

            # 四元数 (注意：输入是 w,x,y,z，我们按 quat_w, quat_x, quat_y, quat_z 顺序写入)
            f.write(wxyz[i].tobytes())    # w, x, y, z → 对应 quat_w, quat_x, quat_y, quat_z
            
            if e1e2 is not None:
                f.write(e1e2[i].tobytes())    # e1, e2

            # 颜色（可选）
            if colors is not None:
                f.write(colors[i].tobytes())

    # print(f"✅ Saved {N} Gaussian Splatting points to {filename}")
