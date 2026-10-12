import math
import cv2
import numpy as np


def overlay_masks(
    frame_bgr,
    water_mask,
    bridge_mask,
    cfg,
    curve=None,
    frame_idx=0,
    fps_text="",
    draw_curve=True,
    draw_masks=True,
    ships=None,
):
    out = frame_bgr.copy()

    if ships:
        h_img, w_img = out.shape[:2]

        for s in ships:
            x1, y1, x2, y2 = [int(v) for v in s["bbox"]]

            spd = s.get("speed", 0)
            brg = s.get("bearing", 0)
            dist = s.get("distance", 0)
            threat = s.get("threat_level", 0)

            if threat >= 2:
                box_color, tag = (0, 0, 255), "DANGER"
            elif threat >= 1:
                box_color, tag = (0, 165, 255), "CAUTION"
            else:
                box_color, tag = (0, 255, 0), "SAFE"

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

            lines = [
                f"{tag}  DIST:{dist:.1f}m",
                f"SPD:{spd:.1f}m/s  BRG:{brg:.0f}",
            ]

            line_h = 22
            bg_h = line_h * len(lines) + 6
            bg_y = max(0, y1 - bg_h)

            dx1 = max(0, min(x1, w_img - 1))
            dx2 = max(dx1 + 1, min(x2, w_img))
            dy1 = max(0, min(bg_y, h_img - 1))
            dy2 = max(dy1 + 1, min(bg_y + bg_h, h_img))

            if dy2 > dy1 and dx2 > dx1:
                roi_copy = out[dy1:dy2, dx1:dx2].copy()
                cv2.rectangle(out, (dx1, dy1), (dx2, dy2), box_color, -1)

                blended = cv2.addWeighted(
                    out[dy1:dy2, dx1:dx2],
                    0.7,
                    roi_copy,
                    0.3,
                    0,
                )

                if blended is not None:
                    out[dy1:dy2, dx1:dx2] = blended

            for li, line in enumerate(lines):
                ty = bg_y + 16 + li * line_h

                if 0 < ty < h_img and 0 < x1 < w_img:
                    cv2.putText(
                        out,
                        line,
                        (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        out,
                        line,
                        (x1 + 4, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

            if spd > 0.5:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                arrow_len = min(60, int(spd * 8))

                rad = math.radians(brg)
                ex = int(cx + arrow_len * math.sin(rad))
                ey = int(cy - arrow_len * math.cos(rad))

                cv2.arrowedLine(
                    out,
                    (cx, cy),
                    (ex, ey),
                    (0, 255, 255),
                    2,
                    tipLength=0.3,
                )

    if draw_masks:
        def overlay_fast(mask, color_bgr, alpha):
            if mask is None or not mask.any():
                return

            color_layer = np.zeros_like(out, dtype=np.uint8)
            color_layer[mask == 1] = color_bgr

            cv2.addWeighted(
                color_layer,
                alpha,
                out,
                1.0,
                0,
                dst=out,
            )

        overlay_fast(water_mask, cfg.water_color, cfg.water_alpha)
        overlay_fast(bridge_mask, cfg.bridge_color, cfg.bridge_alpha)

    if draw_curve and curve is not None and len(curve) >= 2:
        pts = curve.reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, (0, 255, 128), 2, cv2.LINE_AA)

    if fps_text:
        cv2.putText(
            out,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out