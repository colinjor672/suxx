import math

import cv2
import numpy as np


def scale_curve_xy(curve: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if curve is None or len(curve) == 0:
        return np.empty((0, 2), dtype=np.int32)

    out = curve.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy

    return out.astype(np.int32)


def scale_ships_for_display(ships: list, sx: float, sy: float) -> list:
    if not ships or (sx == 1.0 and sy == 1.0):
        return ships

    result = []

    for s in ships:
        ss = dict(s)

        x1, y1, x2, y2 = s["bbox"]
        ss["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        ss["center"] = [s["center"][0] * sx, s["center"][1] * sy]

        result.append(ss)

    return result


def scale_ships_for_send(
    ships: list,
    sx: float,
    sy: float,
    dst_w: int = None,
    dst_h: int = None,
) -> list:
    if not ships:
        return []

    result = []

    for s in ships:
        if "bbox" not in s or s["bbox"] is None or len(s["bbox"]) < 4:
            continue

        ss = dict(s)

        x1, y1, x2, y2 = [float(v) for v in s["bbox"]]

        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        if dst_w is not None:
            x1 = float(np.clip(x1, 0, dst_w - 1))
            x2 = float(np.clip(x2, 0, dst_w - 1))

        if dst_h is not None:
            y1 = float(np.clip(y1, 0, dst_h - 1))
            y2 = float(np.clip(y2, 0, dst_h - 1))

        ss["bbox"] = [x1, y1, x2, y2]

        if "center" in s and s["center"] is not None and len(s["center"]) >= 2:
            cx = float(s["center"][0]) * sx
            cy = float(s["center"][1]) * sy

            if dst_w is not None:
                cx = float(np.clip(cx, 0, dst_w - 1))

            if dst_h is not None:
                cy = float(np.clip(cy, 0, dst_h - 1))

            ss["center"] = [cx, cy]
        else:
            ss["center"] = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

        result.append(ss)

    return result


def extract_polygons_json(
    water_mask,
    bridge_mask,
    epsilon_ratio=0.005,
    min_area_px=200,
    max_polys=4,
):
    h, w = water_mask.shape
    epsilon = max(1.0, epsilon_ratio * max(w, h))

    def _extract(mask):
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        polys = []

        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:max_polys]:
            if cv2.contourArea(cnt) < min_area_px:
                continue

            approx = cv2.approxPolyDP(cnt, epsilon, closed=True)

            pts = [[int(p[0]), int(p[1])] for p in approx.reshape(-1, 2)]

            if len(pts) >= 3:
                polys.append(pts)

        return polys

    return _extract(water_mask), _extract(bridge_mask)


def should_draw_nav_band(ships, external_ships, max_distance):
    distances = []

    for s in ships:
        d = float(s.get("distance", -1))
        if d > 0:
            distances.append(d)

    for s in external_ships:
        d = float(s.get("bridge_pier_distance", -1))
        if d > 0:
            distances.append(d)

    if not distances:
        return True

    return min(distances) <= max_distance


def _ship_center(ship):
    if not isinstance(ship, dict):
        return None

    try:
        center = ship.get("center")
        if center is not None and len(center) >= 2:
            x, y = float(center[0]), float(center[1])
        else:
            bbox = ship.get("bbox")
            if bbox is None or len(bbox) < 4:
                return None
            x = (float(bbox[0]) + float(bbox[2])) * 0.5
            y = (float(bbox[1]) + float(bbox[3])) * 0.5
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def _fusion_values(ship):
    if not isinstance(ship, dict):
        return None

    try:
        values = {
            key: float(ship[key])
            for key in ("north_vel", "east_vel", "distance", "yaw")
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return None

    if not all(math.isfinite(value) for value in values.values()):
        return None
    return values


def _optional_float(ship, key, default):
    try:
        value = float(ship.get(key, default))
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def merge_ship_data(yolo_ships, external_ships, max_match_distance=80.0):
    if not yolo_ships:
        return []

    # YOLO results can be reused between video frames. Clear fusion fields first
    # so expired or unmatched external data is never displayed as current data.
    for ship in yolo_ships:
        ship["speed"] = 0.0
        ship["bearing"] = 0.0
        ship["distance"] = 99999.0
        ship["north_vel"] = 0.0
        ship["east_vel"] = 0.0
        ship["yaw"] = 0.0
        ship["bridge_pier_distance"] = -1.0
        ship["hasSpeedBearing"] = False
        ship["has_fusion_data"] = False

    if not external_ships or not isinstance(external_ships, (list, tuple)):
        return yolo_ships

    try:
        match_limit = float(max_match_distance)
    except (TypeError, ValueError, OverflowError):
        return yolo_ships
    if not math.isfinite(match_limit) or match_limit < 0.0:
        return yolo_ships

    valid_external = []
    for external_ship in external_ships:
        center = _ship_center(external_ship)
        values = _fusion_values(external_ship)
        if center is not None and values is not None:
            valid_external.append((external_ship, center, values))

    candidates = []
    for yolo_index, yolo_ship in enumerate(yolo_ships):
        center = _ship_center(yolo_ship)
        if center is None:
            continue
        cx, cy = center

        for external_index, (_, external_center, _) in enumerate(valid_external):
            ex, ey = external_center
            distance = math.hypot(cx - ex, cy - ey)
            if distance <= match_limit:
                candidates.append((distance, yolo_index, external_index))

    matched_yolo = set()
    matched_external = set()

    # Assign the globally closest available pair first to enforce one-to-one
    # matching when several delayed fusion boxes are near the same YOLO box.
    for _, yolo_index, external_index in sorted(candidates):
        if yolo_index in matched_yolo or external_index in matched_external:
            continue

        yolo_ship = yolo_ships[yolo_index]
        external_ship, _, fusion_values = valid_external[external_index]
        yolo_ship["speed"] = _optional_float(external_ship, "speed", 0.0)
        yolo_ship["bearing"] = _optional_float(external_ship, "bearing", 0.0)
        yolo_ship.update(fusion_values)
        yolo_ship["bridge_pier_distance"] = _optional_float(
            external_ship, "bridge_pier_distance", -1.0
        )
        yolo_ship["hasSpeedBearing"] = True
        yolo_ship["has_fusion_data"] = True

        matched_yolo.add(yolo_index)
        matched_external.add(external_index)

    return yolo_ships


def scale_pixel_bbox(bbox, src_size, dst_size) -> tuple | None:
    """把像素框从一个分辨率空间换算到另一个。

    回传像素框的参考分辨率与当帧 YOLO 框所在分辨率不同，直接比会整体偏移。
    bbox 缺失或参考尺寸非法时返回 None —— 宁可不做吸附，也不要在错误的
    坐标空间里吸附出一个看似合理的位置。
    """
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        src_w, src_h = (float(v) for v in src_size[:2])
        dst_w, dst_h = (float(v) for v in dst_size[:2])
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError, IndexError):
        return None

    if not all(math.isfinite(v) and v > 0.0 for v in (src_w, src_h, dst_w, dst_h)):
        return None

    sx = dst_w / src_w
    sy = dst_h / src_h
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def attach_dcpa_target(
    ships: list,
    pixel_bbox,
    alert,
    frame_width: float = 640.0,
    base_px_640: float = 12.0,
    diag_ratio: float = 1.0,
    stats: dict = None,
) -> dict | None:
    """把避障指定的单一目标信息吸附到当前帧最接近的 YOLO 框上。

    阈值取 max(base, 框对角线 × diag_ratio)：base 按当帧宽度归一
    （base_px_640 × frame_width / 640），所以 640 空间是 12px、1280 空间是 24px，
    代表的物理宽度一致；对角线项让近处大框自动放宽、远处小框收紧。
    pixel_bbox 必须与 ships 的 bbox 同一坐标空间（AI 解码帧），
    即在缩放发送给 Godot 之前调用。

    命中多个框且它们中心距小于半个阈值时判为歧义，本帧不挂面板 ——
    吸错船的代价比漏一帧高。stats 若传入，会累加 hit/miss/ambiguous 计数。

    命中的框是唯一显示信息面板的框；返回该框，未命中返回 None。
    """
    for ship in ships or ():
        ship["is_dcpa_target"] = False

    if not ships or not alert or not isinstance(pixel_bbox, (list, tuple)):
        return None

    try:
        px1, py1, px2, py2 = (float(v) for v in pixel_bbox[:4])
    except (TypeError, ValueError):
        return None

    target_cx = (px1 + px2) * 0.5
    target_cy = (py1 + py2) * 0.5

    try:
        base_px = base_px_640 * float(frame_width) / 640.0
    except (TypeError, ValueError):
        base_px = base_px_640

    def _count(key: str):
        if stats is not None:
            stats[key] = stats.get(key, 0) + 1

    candidates = []
    for ship in ships:
        bbox = ship.get("bbox")
        if bbox is None or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        limit = max(base_px, math.hypot(x2 - x1, y2 - y1) * diag_ratio)
        distance = math.hypot(cx - target_cx, cy - target_cy)
        if distance > limit:
            continue
        candidates.append({"ship": ship, "cx": cx, "cy": cy, "limit": limit,
                           "distance": distance})

    if not candidates:
        _count("miss")
        return None

    best = min(candidates, key=lambda c: c["distance"])
    for other in candidates:
        if other is best:
            continue
        gap = math.hypot(best["cx"] - other["cx"], best["cy"] - other["cy"])
        if gap < 0.5 * max(best["limit"], other["limit"]):
            _count("ambiguous")
            return None

    _count("hit")
    ship = best["ship"]
    ship["is_dcpa_target"] = True
    ship["dcpa"] = float(alert["dcpa"])
    ship["tcpa"] = float(alert["tcpa"])
    # 框色仍按 distance 分档，这里改用避障回传的距离
    ship["distance"] = float(alert["distance"])
    ship["dcpa_alert_level"] = str(alert.get("alert_level", ""))
    ship["dcpa_alert_info"] = str(alert.get("alert_info", ""))
    return ship
