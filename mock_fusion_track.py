import argparse
import json
import math
import random
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path


MAGIC = 0x4655534E
VERSION = 1
MSG_FUSION_OBSTACLES = 3
HEADER = struct.Struct(">IHHIQII")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "mock_data"
    / "front_ship_150s.jsonl"
)


@dataclass(frozen=True)
class TrackConfig:
    duration_s: float = 150.0
    sample_hz: float = 10.0
    seed: int = 20260817
    frame_width: int = 1280
    frame_height: int = 720
    center_x: float = 640.0
    center_y: float = 310.0
    box_width: float = 96.0
    box_height: float = 54.0


def _smooth_noise(rng, count, persistence, innovation):
    state = 0.0
    values = []
    for _ in range(count):
        state = persistence * state + rng.gauss(0.0, innovation)
        values.append(state)
    return values


def generate_track(config=TrackConfig()):
    count = int(round(config.duration_s * config.sample_hz)) + 1
    rng = random.Random(config.seed)

    east_noise = _smooth_noise(rng, count, 0.92, 0.006)
    north_noise = _smooth_noise(rng, count, 0.91, 0.006)
    range_noise = _smooth_noise(rng, count, 0.94, 0.025)

    # Remove endpoint drift so the measured range starts at 386 m and finishes
    # close to 371 m while retaining smooth, locally non-monotonic error.
    first_noise = range_noise[0]
    last_noise = range_noise[-1]

    packets = []
    for index in range(count):
        elapsed = index / config.sample_hz
        progress = elapsed / config.duration_s

        east_vel = (
            -2.06
            + 0.050 * math.sin(2.0 * math.pi * elapsed / 31.0 + 0.35)
            + 0.022 * math.sin(2.0 * math.pi * elapsed / 8.3)
            + east_noise[index]
        )
        north_vel = (
            -1.98
            + 0.047 * math.sin(2.0 * math.pi * elapsed / 37.0 + 1.10)
            + 0.020 * math.sin(2.0 * math.pi * elapsed / 9.7 + 0.20)
            + north_noise[index]
        )
        yaw = math.degrees(math.atan2(east_vel, north_vel)) % 360.0

        nonlinear_progress = 0.58 * progress + 0.42 * progress ** 1.45
        corrected_noise = range_noise[index] - (
            first_noise + (last_noise - first_noise) * progress
        )
        measurement_error = (
            0.16 * math.sin(2.0 * math.pi * elapsed / 13.7)
            + 0.07 * math.sin(2.0 * math.pi * elapsed / 4.3 + 0.40)
            + corrected_noise
            - 0.07 * math.sin(0.40)
        )
        # Remove the remaining deterministic endpoint offset.
        final_wave = (
            0.16 * math.sin(2.0 * math.pi * config.duration_s / 13.7)
            + 0.07 * math.sin(
                2.0 * math.pi * config.duration_s / 4.3 + 0.40
            )
            - 0.07 * math.sin(0.40)
        )
        measurement_error -= final_wave * progress
        distance = 386.0 - 15.05 * nonlinear_progress + measurement_error

        pixel_center_x = (
            config.center_x
            + 7.0 * math.sin(2.0 * math.pi * elapsed / 42.0)
            + 2.0 * math.sin(2.0 * math.pi * elapsed / 11.0 + 0.60)
        )
        pixel_center_y = (
            config.center_y
            + 4.0 * progress
            + 2.5 * math.sin(2.0 * math.pi * elapsed / 29.0 + 0.30)
        )
        apparent_scale = 1.0 + 0.041 * progress
        box_width = config.box_width * apparent_scale
        box_height = config.box_height * apparent_scale
        x1 = int(round(pixel_center_x - box_width * 0.5))
        y1 = int(round(pixel_center_y - box_height * 0.5))
        x2 = int(round(pixel_center_x + box_width * 0.5))
        y2 = int(round(pixel_center_y + box_height * 0.5))
        x1 = max(0, min(config.frame_width - 2, x1))
        y1 = max(0, min(config.frame_height - 2, y1))
        x2 = max(x1 + 1, min(config.frame_width - 1, x2))
        y2 = max(y1 + 1, min(config.frame_height - 1, y2))

        confidence = 0.92 + 0.018 * math.sin(
            2.0 * math.pi * elapsed / 17.0 + 0.80
        )
        obstacle = {
            "id": 0,
            "length": 18.4,
            "width": 5.2,
            "height": 4.6,
            "center_x": round(distance, 4),
            "center_y": round(-8.0 + 0.6 * math.sin(elapsed / 18.0), 4),
            "center_z": 0.0,
            "yaw": round(yaw, 4),
            "score": round(confidence, 4),
            "confidence": round(confidence, 4),
            "class_id": 0,
            "timestamp": round(elapsed, 4),
            "pixel_x1": x1,
            "pixel_y1": y1,
            "pixel_x2": x2,
            "pixel_y2": y2,
            "north_vel": round(north_vel, 4),
            "east_vel": round(east_vel, 4),
            "distance": round(distance, 4),
            "bridge_pier_distance": -1.0,
        }
        packets.append(
            {
                "timestamp": round(elapsed, 4),
                "obstacles": [obstacle],
            }
        )

    return packets


def write_jsonl(packets, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for packet in packets:
            stream.write(
                json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
            )
            stream.write("\n")


def read_jsonl(input_path):
    with Path(input_path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encode_packet(packet, sequence):
    payload = json.dumps(
        packet,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    timestamp = float(packet["timestamp"])
    header = HEADER.pack(
        MAGIC,
        VERSION,
        MSG_FUSION_OBSTACLES,
        len(payload),
        int(timestamp * 1_000_000),
        int(sequence) & 0xFFFFFFFF,
        0,
    )
    return header + payload


def packet_with_bbox(packet, bbox):
    updated = dict(packet)
    obstacles = packet.get("obstacles", [])
    if not obstacles or bbox is None:
        updated["obstacles"] = []
        return updated

    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
        updated["obstacles"] = []
        return updated

    obstacle = dict(obstacles[0])
    obstacle.update(
        {
            "pixel_x1": x1,
            "pixel_y1": y1,
            "pixel_x2": x2,
            "pixel_y2": y2,
        }
    )
    updated["obstacles"] = [obstacle]
    return updated


class YoloBoxTracker:
    def __init__(
        self,
        host="127.0.0.1",
        port=9000,
        bbox_ttl=1.0,
        reconnect_interval=1.0,
        clock=time.monotonic,
    ):
        self.host = str(host)
        self.port = int(port)
        self.bbox_ttl = float(bbox_ttl)
        self.reconnect_interval = float(reconnect_interval)
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._socket = None
        self._bbox = None
        self._updated_at = 0.0
        self._frame_id = -1
        self._last_log_at = 0.0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="MockFusionYoloReader",
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_bbox(self):
        now = self._clock()
        with self._lock:
            if self._bbox is None or now - self._updated_at > self.bbox_ttl:
                return None
            return tuple(self._bbox)

    def update_from_message(self, message):
        if not isinstance(message, dict):
            return

        detections = message.get(
            "yolo_detections",
            message.get("detections", []),
        )
        candidates = []
        for detection in detections if isinstance(detections, list) else []:
            if not isinstance(detection, dict):
                continue
            bbox = detection.get("bbox")
            if bbox is None:
                bbox = [
                    detection.get("x1"),
                    detection.get("y1"),
                    detection.get("x2"),
                    detection.get("y2"),
                ]
            if len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
                confidence = float(
                    detection.get("score", detection.get("confidence", 0.0))
                )
            except (TypeError, ValueError, OverflowError):
                continue
            values = (x1, y1, x2, y2, confidence)
            if not all(math.isfinite(value) for value in values):
                continue
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
                continue
            candidates.append((confidence, (x1, y1, x2, y2)))

        now = self._clock()
        with self._lock:
            self._frame_id = int(message.get("frame_id", -1))
            self._updated_at = now
            self._bbox = max(candidates, key=lambda item: item[0])[1] if candidates else None
            bbox = self._bbox

        if now - self._last_log_at >= 2.0:
            if bbox is None:
                print(
                    f"[MockFusion] YOLO frame={self._frame_id} detections=0"
                )
            else:
                center_x = (bbox[0] + bbox[2]) * 0.5
                center_y = (bbox[1] + bbox[3]) * 0.5
                print(
                    f"[MockFusion] YOLO frame={self._frame_id} "
                    f"detections={len(candidates)} "
                    f"bbox={[round(value, 1) for value in bbox]} "
                    f"center=({center_x:.1f},{center_y:.1f})"
                )
            self._last_log_at = now

    def _run(self):
        while not self._stop_event.is_set():
            try:
                with socket.create_connection(
                    (self.host, self.port),
                    timeout=2.0,
                ) as sock:
                    self._socket = sock
                    sock.settimeout(0.5)
                    print(
                        f"[MockFusion] connected to YOLO "
                        f"{self.host}:{self.port}"
                    )
                    self._receive_messages(sock)
            except (ConnectionError, OSError) as e:
                now = self._clock()
                if now - self._last_log_at >= 2.0:
                    print(
                        f"[MockFusion] YOLO连接失败 "
                        f"{self.host}:{self.port}: {e}"
                    )
                    self._last_log_at = now
            finally:
                self._socket = None

            if not self._stop_event.is_set():
                self._stop_event.wait(self.reconnect_interval)

    def _receive_messages(self, sock):
        buffer = bytearray()
        magic_bytes = struct.pack(">I", MAGIC)
        while not self._stop_event.is_set():
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                return
            buffer.extend(chunk)

            while True:
                while buffer and buffer[0] in b" \r\n\t":
                    del buffer[0]
                if not buffer:
                    break

                # Legacy ar_navigation_video1 used newline-delimited JSON.
                if buffer[0] == ord("{"):
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        break
                    raw = bytes(buffer[:newline]).strip()
                    del buffer[: newline + 1]
                    if not raw:
                        continue
                    try:
                        message = json.loads(raw.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    self.update_from_message(message)
                    continue

                if len(buffer) < HEADER.size:
                    break
                if buffer[:4] != magic_bytes:
                    next_magic = buffer.find(magic_bytes, 1)
                    if next_magic < 0:
                        del buffer[:-3]
                        break
                    del buffer[:next_magic]
                    continue

                try:
                    (
                        _magic,
                        _version,
                        message_type,
                        payload_len,
                        _timestamp_us,
                        sequence,
                        _reserved,
                    ) = HEADER.unpack_from(buffer)
                except struct.error:
                    break

                if payload_len > 10 * 1024 * 1024:
                    del buffer[:4]
                    continue
                total_size = HEADER.size + payload_len
                if len(buffer) < total_size:
                    break

                raw = bytes(buffer[HEADER.size:total_size])
                del buffer[:total_size]
                if message_type != 2:
                    continue
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                message.setdefault("frame_id", int(sequence))
                self.update_from_message(message)


def replay(packets, host, port, speed=1.0, bbox_provider=None):
    if speed <= 0.0:
        raise ValueError("replay speed must be greater than zero")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(f"[MockFusion] listening on {host}:{port}")
        client, address = server.accept()
        print(f"[MockFusion] receiver connected: {address}")

        with client:
            first_timestamp = float(packets[0]["timestamp"])
            started_at = time.monotonic()
            for sequence, packet in enumerate(packets):
                elapsed = float(packet["timestamp"]) - first_timestamp
                deadline = started_at + elapsed / speed
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                outgoing_packet = packet
                if bbox_provider is not None:
                    outgoing_packet = packet_with_bbox(packet, bbox_provider())
                try:
                    client.sendall(encode_packet(outgoing_packet, sequence))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    print("[MockFusion] receiver disconnected; replay stopped")
                    return

    print("[MockFusion] replay complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and optionally replay a 150-second mock ship track."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=150.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--center-x", type=float, default=640.0)
    parser.add_argument("--center-y", type=float, default=310.0)
    parser.add_argument("--box-width", type=float, default=96.0)
    parser.add_argument("--box-height", type=float, default=54.0)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=55103)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--yolo-host", default="127.0.0.1")
    parser.add_argument("--yolo-port", type=int, default=9000)
    parser.add_argument("--yolo-ttl", type=float, default=1.0)
    parser.add_argument(
        "--fixed-pixels",
        action="store_true",
        help="Use generated pixel coordinates instead of following port 9000.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrackConfig(
        duration_s=args.duration,
        sample_hz=args.sample_hz,
        seed=args.seed,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        center_x=args.center_x,
        center_y=args.center_y,
        box_width=args.box_width,
        box_height=args.box_height,
    )
    packets = generate_track(config)
    write_jsonl(packets, args.output)

    first = packets[0]["obstacles"][0]
    last = packets[-1]["obstacles"][0]
    print(
        f"[MockFusion] wrote {len(packets)} samples to {args.output}\n"
        f"[MockFusion] distance {first['distance']:.2f} -> "
        f"{last['distance']:.2f} m; initial yaw {first['yaw']:.2f} deg"
    )
    if args.replay:
        tracker = None
        if not args.fixed_pixels:
            tracker = YoloBoxTracker(
                host=args.yolo_host,
                port=args.yolo_port,
                bbox_ttl=args.yolo_ttl,
            )
            tracker.start()
            print(
                f"[MockFusion] following live YOLO boxes from "
                f"{args.yolo_host}:{args.yolo_port}"
            )
        try:
            replay(
                packets,
                args.host,
                args.port,
                args.speed,
                bbox_provider=tracker.get_bbox if tracker is not None else None,
            )
        finally:
            if tracker is not None:
                tracker.stop()


if __name__ == "__main__":
    main()
