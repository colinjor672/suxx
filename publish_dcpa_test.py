"""向 v1/11/prediction/result 回放一条真实轨迹的 dcpaResults 消息，用于验证信息面板。

独立调试脚本，不 import 主管线的任何模块：
    python3 publish_dcpa_test.py --log yolo_boxes.log

日志数值是 yolo_sender 发送空间里的像素，MQTT pixel 字段的参考分辨率是 1280x720，
两者不一致时吸附会落到错误位置，所以默认按 --src-size -> --dst-size 等比换算。
日志本身已经是 1280x720 的话用 --src-size 1280x720，换算系数即为 1。

只认 class=0 的船只框（--class-id 0 为默认，桥墩 class=1 一律跳过）；
一帧只有一条船时无需再加过滤条件。src_id 在帧与帧之间并不可靠（实测同一条船
会从 7 跳到 5），所以不要依赖 --src-id 追踪目标，只在多船帧里临时用它挑一帧。
"""

import argparse
import json
import time

import paho.mqtt.client as mqtt

TOPIC = "v1/11/prediction/result"

# 与真实报文一致的假数据；数值本身没有物理意义，只为让解析通过
TRAJECTORY = [
    {"latitude": 22.745720212166667, "longitude": 113.59328764366668},
    {"latitude": 22.745720212166667, "longitude": 113.59328764366668},
    {"latitude": 22.745730123456789, "longitude": 113.59339764366668},
]


def parse_size(text) -> tuple:
    w, h = str(text).lower().replace(":", "x").split("x")
    return (int(w), int(h))


def read_log_frames(path: str, src_id=None, box_id=None, class_id=None) -> list:
    """按 frame_id 聚合成 [{frame_id, ts, rows: [{ts, bbox}, ...]}, ...]。

    未指定 src_id/box_id 时一帧可能有多条框，交由 pick_bbox 按 --pick 选一条。
    """
    frames = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                frame_id = int(float(parts[0]))
                ts = float(parts[1])
                row_box_id = int(float(parts[2]))
                row_class = int(float(parts[3]))
                row_src_id = int(float(parts[5]))
                x1, y1, x2, y2 = (float(v) for v in parts[6:10])
            except ValueError:
                continue

            if class_id is not None and row_class != class_id:
                continue
            if src_id is not None and row_src_id != src_id:
                continue
            if box_id is not None and row_box_id != box_id:
                continue

            row = {"ts": ts, "bbox": (x1, y1, x2, y2)}
            if frames and frames[-1]["frame_id"] == frame_id:
                frames[-1]["rows"].append(row)
            else:
                frames.append({"frame_id": frame_id, "ts": ts, "rows": [row]})
    return frames


def scale_bbox(bbox, src_size, dst_size) -> tuple:
    sx = dst_size[0] / float(src_size[0])
    sy = dst_size[1] / float(src_size[1])
    x1, y1, x2, y2 = bbox
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def wait_published(mid, timeout: float = 3.0) -> bool:
    """发送确认，兼容 paho-mqtt 1.x（is_published）与 2.x（published）。"""
    if mid is None:
        return False
    if getattr(mid, "rc", 0) != 0:
        return False
    try:
        mid.wait_for_publish(timeout=timeout)
    except (TypeError, ValueError):
        pass
    published = getattr(mid, "published", None)
    if published is not None:
        return bool(published)
    is_published = getattr(mid, "is_published", None)
    if is_published is not None:
        return bool(is_published())
    return getattr(mid, "rc", 0) == 0


def build_payload(args, bbox) -> dict:
    x1, y1, x2, y2 = bbox
    entry = {
        "zone_id": args.zone_id,
        "distance": args.distance,
        "dcpa": args.dcpa,
        "tcpa": args.tcpa,
        "bearing_local": 358.2251584390757,
        "cpa_lon": 113.59355142947837,
        "cpa_lat": 22.74572777438024,
        "alertLevel": args.alert_level,
        "alertInfo": args.alert_info,
        "dx_ship": None,
        "dy_ship": None,
        "type": "obstacle",
        "latitude": 0.39698797575063155,
        "longitude": 1.9825815426506488,
        "obstacle_heading": 1.5707964,
        "length": 0.75,
        "width": 0.75,
        "pixel_x1": x1,
        "pixel_y1": y1,
        "pixel_x2": x2,
        "pixel_y2": y2,
    }
    return {
        "code": 200,
        "data": {
            "collisionAlert": args.collision_alert,
            "dcpaResults": [entry],
            "avoidanceTrajectory": TRAJECTORY,
        },
    }


def parse_args():
    p = argparse.ArgumentParser(description="按 yolo 框日志回放 dcpaResults 报文")
    p.add_argument("--host", default="127.0.0.1", help="broker 地址（主管线默认 127.0.0.1）")
    p.add_argument("--port", type=int, default=1890, help="broker 端口（主管线默认 1890）")
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--topic", default=TOPIC)

    p.add_argument("--log", default="yolo_boxes.log", help="yolo_box_logger 生成的日志")
    p.add_argument("--src-id", type=int, default=None, help="临时按 src_id 过滤（该列逐帧会变，别用来追船）")
    p.add_argument("--box-id", type=int, default=None, help="临时按 box_id 过滤（同样是当帧编号，不跨帧）")
    p.add_argument("--class-id", type=int, default=0, help="0=只跟船只（默认），负数=不过滤类别")
    p.add_argument("--pick", choices=["first", "last", "leftmost", "rightmost"], default="first",
                   help="帧内有多条框且未指定 src_id/box_id 时怎么选")
    p.add_argument("--frame", type=int, default=None, help="只发这一帧，不回放整段")

    p.add_argument("--src-size", default="640x360", help="日志像素所在分辨率")
    p.add_argument("--dst-size", default="1280x720", help="MQTT pixel 参考分辨率")

    p.add_argument("--dcpa", type=float, default=0.0)
    p.add_argument("--tcpa", type=float, default=0.0)
    p.add_argument("--distance", type=float, default=0.0)
    p.add_argument("--zone-id", default="obs_69838")
    p.add_argument("--alert-level", default="danger")
    p.add_argument("--alert-info", default="存在碰撞风险，请按避障建议路径航行")
    p.add_argument("--collision-alert", type=int, default=0,
                   help="0 也照样带目标，面板不看这个字段")

    p.add_argument("--realtime", action="store_true",
                   help="按日志时间差回放；否则用 --interval 匀速发")
    p.add_argument("--interval", type=float, default=0.067,
                   help="非 realtime 时的发送间隔秒；<=0 表示只发一帧")
    p.add_argument("--count", type=int, default=0, help="最多发送多少帧，0=全部")
    p.add_argument("--loop", action="store_true", help="回放完从头继续")
    p.add_argument("--qos", type=int, default=1)
    return p.parse_args()


def pick_bbox(frame_rows, pick: str):
    if pick == "last":
        return frame_rows[-1]
    if pick == "leftmost":
        return min(frame_rows, key=lambda r: r["bbox"][0])
    if pick == "rightmost":
        return max(frame_rows, key=lambda r: r["bbox"][0])
    return frame_rows[0]


def main():
    args = parse_args()

    src_size = parse_size(args.src_size)
    dst_size = parse_size(args.dst_size)
    class_id = None if args.class_id < 0 else args.class_id

    frames = read_log_frames(args.log, src_id=args.src_id,
                             box_id=args.box_id, class_id=class_id)
    if not frames:
        print(f"[FAIL] {args.log} 里没有匹配的框"
              f"（src_id={args.src_id} box_id={args.box_id} class={class_id}）")
        return 1

    if args.frame is not None:
        frames = [r for r in frames if r["frame_id"] == args.frame]
        if not frames:
            print(f"[FAIL] 日志里没有 frame_id={args.frame}")
            return 1

    print(f"[info] {len(frames)} 帧  "
          f"{src_size[0]}x{src_size[1]} -> {dst_size[0]}x{dst_size[1]}"
          f"  (sx={dst_size[0] / src_size[0]:.2f})")

    api = getattr(mqtt, "CallbackAPIVersion", None)
    client_id = f"dcpa_test_{int(time.time()) % 100000:05d}"
    if api is not None:
        client = mqtt.Client(api.VERSION2, client_id=client_id)
    else:
        client = mqtt.Client(client_id=client_id)
    if args.username:
        client.username_pw_set(args.username, args.password)

    client.connect(args.host, args.port, keepalive=30)
    client.loop_start()

    sent = 0
    try:
        while True:
            for idx, frame in enumerate(frames):
                raw = pick_bbox(frame["rows"], args.pick)["bbox"]
                bbox = scale_bbox(raw, src_size, dst_size)
                payload = json.dumps(build_payload(args, bbox), ensure_ascii=False)

                mid = client.publish(args.topic, payload, qos=args.qos)
                if not wait_published(mid):
                    print(f"[FAIL] frame={frame['frame_id']} 未确认，检查 broker/账号/端口")
                    return 1

                sent += 1
                print(f"[OK] #{sent} frame={frame['frame_id']} "
                      f"log=({raw[0]:.0f},{raw[1]:.0f})"
                      f"-({raw[2]:.0f},{raw[3]:.0f}) "
                      f"pixel=({bbox[0]:.0f},{bbox[1]:.0f})"
                      f"-({bbox[2]:.0f},{bbox[3]:.0f})")

                if args.count and sent >= args.count:
                    return 0

                if args.realtime and idx + 1 < len(frames):
                    delay = frames[idx + 1]["ts"] - frame["ts"]
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
                elif args.interval > 0:
                    time.sleep(args.interval)
                else:
                    return 0

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[stop] 手动中断")
    finally:
        client.loop_stop()
        client.disconnect()
        print(f"[done] 共发送 {sent} 帧")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
