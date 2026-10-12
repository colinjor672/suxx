import torch
import torch.nn.functional as F


def refine_water_mask(mask_gpu: torch.Tensor) -> torch.Tensor:
    """
    水面mask优化
    水面优先级最高
    """
    m = mask_gpu.float().unsqueeze(0).unsqueeze(0)
    # 闭运算：填补小孔洞
    m = F.max_pool2d(m, 5, stride=1, padding=2)
    # 开运算：去除孤立噪声
    m = -F.max_pool2d(-m, 5, stride=1, padding=2)
    # 更大尺度平滑
    m = -F.max_pool2d(-m, 7, stride=1, padding=3)
    m = F.max_pool2d(m, 7, stride=1, padding=3)
    return m.squeeze(0).squeeze(0).to(torch.uint8)


def _is_night_scene(frame_gray_gpu: torch.Tensor,
                    night_brightness_threshold: float = 0.25) -> torch.Tensor:
    """
    判断是否为夜晚场景（基于整体亮度）

    参数
    ----------
    frame_gray_gpu : torch.Tensor [H, W] float32, 范围 [0, 1]
        原图的灰度图（GPU tensor）
    night_brightness_threshold : float
        平均亮度低于此值判定为夜晚

    返回
    ----------
    torch.Tensor (标量 bool tensor)
    """
    if frame_gray_gpu is None:
        return torch.tensor(False, device='cuda')

    mean_brightness = frame_gray_gpu.float().mean()
    return mean_brightness < night_brightness_threshold


def _filter_bridge_by_brightness(
    bridge_mask: torch.Tensor,
    frame_gray_gpu: torch.Tensor,
    water_gpu: torch.Tensor,
    bridge_min_brightness_ratio: float = 1.15,
) -> torch.Tensor:
    """
    基于亮度过滤桥梁误识别

    核心思路：夜晚场景下，真正的桥梁通常有灯光照明，比水面亮
    如果"桥梁"区域的平均亮度低于水面，很可能是水面误识别

    参数
    ----------
    bridge_mask : torch.Tensor [H, W] uint8
        桥梁掩码
    frame_gray_gpu : torch.Tensor [H, W] float32 [0, 1]
        原图灰度图
    water_gpu : torch.Tensor [H, W] uint8
        水面掩码
    bridge_min_brightness_ratio : float
        桥梁/水面亮度比的最小阈值，低于此值则判定为误识别

    返回
    ----------
    torch.Tensor [H, W] uint8 过滤后的桥梁掩码
    """
    if frame_gray_gpu is None:
        return bridge_mask

    bridge_float = bridge_mask.float()
    water_float = water_gpu.float()

    # 计算桥梁区域平均亮度
    bridge_area = bridge_float.sum()
    water_area = water_float.sum()

    # 如果桥梁或水面区域太小，不做亮度过滤
    if bridge_area < 100 or water_area < 100:
        return bridge_mask

    bridge_brightness = (frame_gray_gpu * bridge_float).sum() / bridge_area
    water_brightness = (frame_gray_gpu * water_float).sum() / water_area

    # 桥梁亮度 / 水面亮度
    brightness_ratio = bridge_brightness / (water_brightness + 1e-6)

    # 如果桥梁比水面还暗，很可能是误识别 → 清空
    # 正常情况：桥梁有灯光 > 水面反光，brightness_ratio > 1.0
    is_misclass = brightness_ratio < bridge_min_brightness_ratio

    # 用 GPU tensor 条件清空，避免 .item() 同步
    result = torch.where(
        is_misclass,
        torch.zeros_like(bridge_mask),
        bridge_mask,
    )
    return result


def refine_bridge_mask(
    bridge_gpu,
    water_gpu,
    mask_h,
    mask_w,
    margin_ratio=0.3, #水面上沿向上允许延伸搜索桥梁的安全余量
    max_height_ratio=0.45, #允许桥梁占据的最大像素高度
    min_aspect_ratio=1.2, #桥梁最小宽高比
    max_area_ratio=0.4, #桥梁掩码总像素的上限比例
    frame_gray_gpu=None,  # 新增：原图灰度图（GPU tensor [H,W] float32 [0,1]）
    night_brightness_threshold=0.25,  # 夜晚亮度阈值
    bridge_min_brightness_ratio=1.15,  # 桥梁/水面亮度比最小值
):
    h, w = bridge_gpu.shape[:2]

    # ── 夜晚场景检测 ──
    is_night = _is_night_scene(frame_gray_gpu, night_brightness_threshold)

    # 夜晚场景下收紧参数，减少误识别
    if is_night:
        # 夜晚桥梁不应占据太大面积（误识别常表现为大面积）
        max_height_ratio = min(max_height_ratio, 0.30)
        max_area_ratio = min(max_area_ratio, 0.25)
        min_aspect_ratio = max(min_aspect_ratio, 1.5)  # 夜晚要求更宽更扁

    # 根据水面范围限制桥梁检测区域

    water_row_sum = water_gpu.float().sum(dim=1)

    water_has = water_row_sum > 0
    water_indices = torch.nonzero(
        water_has,
        as_tuple=False
    )

    if water_indices.numel() == 0:
        bridge_gpu.zero_()
        return bridge_gpu


    # 用 GPU tensor 索引替代 .item()，避免同步
    water_top_t = water_indices[0]
    water_bottom_t = water_indices[-1]

    margin_t = torch.tensor(int(h * margin_ratio), device=bridge_gpu.device)

    bridge_top_limit_t = torch.clamp(water_top_t - margin_t, min=0)

    bridge_gpu = bridge_gpu.clone()
    # 用 boolean mask 替代切片 .item()，完全避免 GPU→CPU 同步
    row_idx = torch.arange(h, device=bridge_gpu.device)  # [H]
    keep_row = (row_idx >= bridge_top_limit_t.view(-1)) & (row_idx < water_bottom_t.view(-1))
    bridge_gpu = bridge_gpu * keep_row.unsqueeze(1).to(bridge_gpu.dtype)

    # 桥梁形态学优化
    m = bridge_gpu.float().unsqueeze(0).unsqueeze(0)
    # 去小孔
    m = -F.max_pool2d(-m, 5,stride=1,padding=2)
    m = F.max_pool2d(m,5,stride=1,padding=2)

    bridge_mask = m.squeeze(0).squeeze(0)
    col_sum = bridge_mask.sum(dim=0)
    max_bridge_h = torch.tensor(h * max_height_ratio, device=bridge_gpu.device)
    bad_cols = (col_sum > max_bridge_h).float()
    bad_ratio = bad_cols.sum() / float(w)
    if bad_ratio > 0.3:
        bridge_mask.zero_()
        return bridge_mask.to(torch.uint8)

    row_sum = bridge_mask.sum(dim=1)
    active_rows = row_sum > 0
    active_cols_mask = col_sum > 0
    if active_rows.any() and active_cols_mask.any():
        row_indices = torch.nonzero(active_rows,as_tuple=False)
        col_indices = torch.nonzero(active_cols_mask,as_tuple=False)
        bbox_h_t = row_indices[-1] - row_indices[0] + 1
        bbox_w_t = col_indices[-1] - col_indices[0] + 1
        # 用 GPU tensor 比较替代 .item()
        aspect_t = bbox_w_t.float() / bbox_h_t.float()
        if aspect_t < min_aspect_ratio:
            bridge_mask.zero_()
            return bridge_mask.to(torch.uint8)
        area_ratio_t = bridge_mask.sum().float() / float(h * w)
        if area_ratio_t > max_area_ratio:
            bridge_mask.zero_()
            return bridge_mask.to(torch.uint8)

    # 水面优先级 > 桥梁
    bridge_mask = bridge_mask * (
        1 - water_gpu.float()
    )

    # ── 夜晚场景：基于亮度过滤误识别 ──
    # 夜晚水面常被误识别为桥梁，通过亮度对比过滤
    if is_night and frame_gray_gpu is not None:
        bridge_mask = _filter_bridge_by_brightness(
            bridge_mask,
            frame_gray_gpu,
            water_gpu,
            bridge_min_brightness_ratio=bridge_min_brightness_ratio,
        )

    return bridge_mask.to(torch.uint8)

