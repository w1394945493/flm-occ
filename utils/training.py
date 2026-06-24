import os; opse = os.path.splitext
from typing import Iterable, Tuple, List, Dict, Optional, Literal, Any

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)



def create_lr_scheduler(
    optimizer,
    stype: str,
    steps_per_epoch: int,
    warmup_epochs: float=1,
    cosine_epochs: float=100,
    f_max=1.0,
    f_min=0.1,
):
    """
    根据step数返回一个学习率倍率因子，
    注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
    """
    
    iter_warmup = warmup_epochs * steps_per_epoch
    T_max = cosine_epochs * steps_per_epoch
    
    if stype == 'warmup_cosine':
        scheduler = lambda t: t / iter_warmup if t < iter_warmup else \
                (f_min + 0.5 * (f_max-f_min) * (1.0+np.cos( np.pi*(min(t, T_max)-iter_warmup)/(T_max-iter_warmup) )) )
    elif stype == 'warmup_constant':
        scheduler = lambda t: t / iter_warmup if t < iter_warmup else 1.0
    elif stype == 'constant':
        scheduler = lambda t: 1.0
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

def compute_f1_score(grid_pd, grid_gt, mask2d):
    """
    计算两个三维体素网格的F1-score.
    
    Args:
        grid_gt (torch.Tensor):
        grid_pd (torch.Tensor):
    
    Returns:
        F1-score 值.
    """
    # 检查两个输入网格的维度是否相同
    assert grid_gt.shape == grid_pd.shape, "两个体素网格的尺寸必须相同"
    mask3d = mask2d.expand([-1, grid_gt.shape[1], -1, -1])
    grid_gt = grid_gt[mask3d.bool()]
    grid_pd = grid_pd[mask3d.bool()]
    
    grid_gt = grid_gt.round()
    grid_pd = grid_pd.round()

    # 计算TP, FP, FN
    true_positive = (grid_gt * grid_pd).sum().float()
    false_positive = ((1 - grid_gt) * grid_pd).sum().float()
    false_negative = (grid_gt * (1 - grid_pd)).sum().float()
    
    # 计算precision和recall
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    
    # 计算F1-score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1_score, precision, recall

def seed_everything(seed: int, determinstic: bool = False):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = determinstic
    torch.use_deterministic_algorithms(determinstic)

def backup_codes(dir_code, dir_target, train_script=None):
    import shutil
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%m%d_%H%M')
    
    def ignore_files(folder, files):
        # copytree递归访问文件夹folder并复制文件files
        # ignore_files用来返回每个文件夹里包含的忽略路径
        # 忽略指定的文件和文件夹
        ignore_list = ['.gitignore', '__pycache__', 'exp', 'build', '.git', '.vscode', 'notebook', 'debug', 'swanlog']
        ignore_exts = ['.log', '.ipynb', '.pyc', '.pth', '.npz', '.npy', '.png', '.jpg', '.md', '.pptx', '.html']
        return [f for f in files if f in ignore_list or opse(f)[1] in ignore_exts]

    dir_save = dir_target+f'/code@{timestamp}'
    shutil.copytree(dir_code, dir_save, ignore=ignore_files, dirs_exist_ok=True)
    if train_script is not None:
        open(dir_save+f'/{train_script}', 'w').close()
        
    return timestamp

def to_device(obj: Any, device='cuda:0', half: Literal[None, 'bf16', 'fp16'] = None) -> Any:
    """
    递归地将 obj 中的所有 Tensor 移动到指定设备并转换精度。
    支持: Tensor, dict, list, tuple, 其他类型原样返回。
    """
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        if obj.dtype in [torch.float32, torch.float64]:
            if half == 'bf16':
                obj = obj.bfloat16()
            elif half == 'fp16':
                obj = obj.half()
        return obj
    elif isinstance(obj, dict):
        return {k: to_device(v, device, half) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_device(x, device, half) for x in obj)
    else:
        return obj  # 其他类型不处理


def to_numpy(obj: Any) -> Any:
    """
    递归地将 obj 中的所有 Tensor 转换为 numpy 数组。
    支持: Tensor, dict, list, tuple, 其他类型原样返回。
    """
    if isinstance(obj, torch.Tensor):
        if obj.requires_grad:
            obj = obj.detach()
        if obj.dtype in (torch.bfloat16, torch.float16):
            obj = obj.float()
        return obj.cpu().numpy()
    elif isinstance(obj, dict):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_numpy(x) for x in obj)
    else:
        return obj  # 其他类型不处理


@torch.no_grad()
def get_total_norm(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    norm_type: float = 2.0,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    
    
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    
    
    if isinstance(grads, torch.Tensor):
        grads = [grads]
    else:
        grads = list(grads)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device
    grouped_tensors: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [grads]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    return total_norm


def check_grad_nan_inf(model: nn.Module):
    valid_grad = True
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_has_nan = torch.isnan(param.grad).any()
            grad_has_inf = torch.isinf(param.grad).any()
            if grad_has_nan or grad_has_inf:
                valid_grad = False
                break
            
    return valid_grad
