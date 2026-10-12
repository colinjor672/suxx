"""独立 ENU 俯视轨迹小图查看器。

单独开一个终端运行，订阅 MQTT 的 GNSS + prediction 主题，
用 RealtimeNavigationState 重建避障航线，并在 OpenCV 窗口里画俯视小图。
不依赖主渲染管线。

用法（默认本机 broker）:
    python enu_minimap_viewer.py
    python enu_minimap_viewer.py --mqtt-host 192.168.3.125 --mqtt-port 1890
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from network.mqtt_navigation import MqttNavigationClient, MqttNavigationConfig
from utils.realtime_navigation import RealtimeNavigationState


def draw_enu_minimap(canvas, east, north, cur_e, cur_n) -> None:
    """在整张画布上铺满绘制俯视 ENU 轨迹（北朝上、等比例）。"""
    if east is None or north is None or len(east) < 2:
        return

    h, w = canvas.shape[:2]
    pad = 40

    xs = np.asarray(east, dtype=np.float64)
    ys = np.asarray(north, dtype=np.float64)
    xs_all = np.append(xs, float(cur_e))
    ys_all = np.append(ys, float(cur_n))
    min_x, max_x = float(xs_all.min()), float(xs_all.max())
    min_y, max_y = float(ys_all.min()), float(ys_all.max())
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x < 1e-6 and span_y < 1e-6:
        return

    inner_w = w - 2 * pad
    inner_h = h - 2 * pad
    scale = min(inner_w / max(span_x, 1e-6), inner_h / max(span_y, 1e-6))
    off_x = (inner_w - span_x * scale) / 2.0
    off_y = (inner_h - span_y * scale) / 2.0

    def to_px(e, n):
        px = int(round(pad + off_x + (e - min_x) * scale))
        py = int(round(pad + off_y + (max_y - n) * scale))  # 北朝上
        return px, py

    # 轨迹折线
    pts = np.array([to_px(e, n) for e, n in zip(xs, ys)], dtype=np.int32)
    cv2.polylines(canvas, [pts], False, (0, 255, 128), 2, cv2.LINE_AA)

    # 起点（触发船位）与当前船位
    cv2.circle(canvas, to_px(float(xs[0]), float(ys[0])), 4, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, to_px(float(cur_e), float(cur_n)), 6, (0, 215, 255), -1, cv2.LINE_AA)

    # 北向箭头
    ax, ay = w - 50, 50
    cv2.arrowedLine(canvas, (ax, ay + 34), (ax, ay), (255, 255, 255), 2, tipLength=0.35, line_type=cv2.LINE_AA)
    cv2.putText(canvas, "N", (ax - 14, ay - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description="独立 ENU 俯视轨迹小图查看器")
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--prediction-topic", default="v1/11/prediction/result")
    ap.add_argument("--gnss-topic", default="v1/11/sensor/gnss/gnss_01")
    ap.add_argument("--mqtt-username", default=None)
    ap.add_argument("--mqtt-password", default=None)
    ap.add_argument("--width", type=int, default=440)
    ap.add_argument("--height", type=int, default=400)
    args = ap.parse_args()

    state = RealtimeNavigationState()
    config = MqttNavigationConfig(
        host=args.mqtt_host,
        port=args.mqtt_port,
        prediction_topic=args.prediction_topic,
        gnss_topic=args.gnss_topic,
        username=args.mqtt_username,
        password=args.mqtt_password,
    )
    client = MqttNavigationClient(state, config)

    state.start()
    client.start()
    print(f"[ENU Viewer] 已启动 broker={args.mqtt_host}:{args.mqtt_port}，按 q / ESC 退出")

    try:
        while True:
            canvas = np.full((args.height, args.width, 3), 18, dtype=np.uint8)

            data = state.enu_minimap()
            if data is not None:
                east, north, cur_e, cur_n = data
                draw_enu_minimap(canvas, east, north, cur_e, cur_n)
                cv2.putText(
                    canvas,
                    f"east={cur_e:6.2f}m  north={cur_n:6.2f}m",
                    (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas,
                    "Waiting for avoidance route...",
                    (14, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("ENU Minimap", canvas)
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        client.stop()
        state.stop()


if __name__ == "__main__":
    main()