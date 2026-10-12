"""
GPU 版本的几何计算函数
使用 PyTorch CUDA tensors 替代 NumPy CPU 计算
避免 GPU→CPU→GPU 数据传输，全部在 GPU 上完成
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def build_navigation_curve_gpu(
    nav_mask: torch.Tensor,
    water_mask: Optional[torch.Tensor] = None,
    n_rows: int = 50,
    min_row_pixels: int = 20,
    smooth_k: int = 13,
) -> torch.Tensor:
    """
    GPU 版本：从导航 mask 计算导航曲线

    Args:
        nav_mask: shape [H, W], dtype=torch.uint8, on GPU
        water_mask: shape [H, W], dtype=torch.uint8, on GPU (可选)
        n_rows: 曲线采样行数
        min_row_pixels: 有效行的最小像素数
        smooth_k: 平滑核大小

    Returns:
        shape [n_rows, 2], dtype=torch.int32, on GPU  或 shape [0, 2] 空
    """
    if nav_mask is None or not nav_mask.any():
        return torch.empty((0, 2), dtype=torch.int32, device=nav_mask.device)

    device = nav_mask.device
    mask = nav_mask > 0  # [H, W] bool
    h, w = mask.shape

    # 1. 水平投影
    row_counts = mask.sum(dim=1)  # [H]
    valid_rows = torch.where(row_counts >= 1)[0]

    if valid_rows.numel() < 10:
        return torch.empty((0, 2), dtype=torch.int32, device=device)

    # 2. 确定水面顶部（向量化，无 .item() 同步）
    if water_mask is not None and water_mask.any():
        water_any = (water_mask > 0).any(dim=1)  # [H] bool
        water_rows = torch.where(water_any)[0]
        water_top_src = water_rows[0] if water_rows.numel() > 0 else valid_rows[0]
    else:
        water_top_src = valid_rows[0]

    top_row = water_top_src - 1
    bot_row = h - 1

    span = bot_row - top_row
    top_row = top_row + (span * 0.20).to(torch.int32)

    # 3. 采样行（从下到上，与 CPU 版本 geometry.py 顺序一致：ys[0]=底部近处, ys[-1]=顶部远处）
    # bot_row 是 python int，top_row 是 tensor；用 float() 避免同步
    ys = torch.linspace(
        float(bot_row), top_row.float(), n_rows,
        device=device, dtype=torch.float32,
    )
    ys = ys.clamp(0, h - 1).to(torch.int32)

    rows = mask[ys].float()  # [n_rows, W]
    counts = rows.sum(dim=1)  # [n_rows]

    xs = torch.arange(w, dtype=torch.float32, device=device)  # [W]
    # 加权平均 x 坐标
    cx = (rows @ xs) / counts.clamp(min=1.0)  # [n_rows]

    valid = counts >= min_row_pixels  # [n_rows] bool

    # 4. 前向填充无效行（向量化 cummax，无 .item() 同步）
    # 用 cummax 找到每个位置左侧最近的有效值索引
    valid_indices = torch.where(
        valid,
        torch.arange(n_rows, device=device, dtype=torch.long),
        torch.full((n_rows,), -1, device=device, dtype=torch.long),
    )
    fill_indices, _ = torch.cummax(valid_indices, dim=0)
    # 开头连续无效行的 fill_indices 为 -1，用默认值 w*0.5
    has_valid_before = fill_indices >= 0
    cx_filled = torch.where(
        has_valid_before,
        cx[fill_indices.clamp(min=0)],
        torch.full_like(cx, w * 0.5),
    )
    cx = cx_filled

    # 5. 最大步长裁剪（GPU 循环，避免 CPU↔GPU 传输）
    # 递推依赖无法完全向量化，但 GPU 上 torch.clamp 避免了 .cpu().numpy() 同步
    max_step = float(max(8, int(w * 0.035)))
    for i in range(1, n_rows):
        cx[i] = torch.clamp(cx[i], cx[i - 1] - max_step, cx[i - 1] + max_step)

    # 6. 卷积平滑
    if smooth_k > 1 and n_rows >= smooth_k:
        if smooth_k % 2 == 0:
            smooth_k += 1
        # conv1d 需要 [B, C, L] 格式
        kernel = torch.ones(1, 1, smooth_k, dtype=torch.float32, device=device) / smooth_k
        cx_padded = F.pad(cx.view(1, 1, -1), (smooth_k // 2, smooth_k // 2), mode='replicate')
        cx = F.conv1d(cx_padded, kernel).view(-1)

    # 7. 组装结果
    result = torch.empty((n_rows, 2), dtype=torch.int32, device=device)
    result[:, 0] = cx.clamp(0, w - 1).to(torch.int32)
    result[:, 1] = ys.clamp(0, h - 1).to(torch.int32)

    return result


def scale_curve_xy_gpu(
    curve: torch.Tensor,
    sx: float,
    sy: float,
) -> np.ndarray:
    """
    GPU 版本的曲线缩放，返回 CPU numpy 数组

    Args:
        curve: shape [N, 2], dtype=torch.int32, on GPU
        sx, sy: 缩放因子

    Returns:
        shape [N, 2], dtype=np.int32, on CPU
    """
    if curve is None or curve.numel() == 0:
        return np.empty((0, 2), dtype=np.int32)

    out = curve.float()
    out[:, 0] *= sx
    out[:, 1] *= sy

    return out.cpu().numpy().astype(np.int32)


def tensor_to_numpy_int32(t: torch.Tensor) -> np.ndarray:
    """将 GPU tensor 转为 CPU numpy int32"""
    if t is None or t.numel() == 0:
        return np.empty((0, 2), dtype=np.int32)
    return t.cpu().numpy().astype(np.int32)