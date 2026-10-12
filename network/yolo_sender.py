import json
import socket
import struct
import threading
import time

from utils.thread_utils import set_thread_name
from yolo_box_logger import YoloBoxLogger


class YoloSender:
    """主动连接 transport_tcp 的 YOLO 检测结果发送客户端。

    transport_tcp 是 TCP 服务端，本模块作为客户端连接它，
    使用与 transport_tcp 一致的二进制协议发送 YOLO 检测框。
    """
    MAGIC = 0x4655534E
    VERSION = 1
    MSG_YOLO_DETECTIONS = 2
    HEADER_STRUCT = struct.Struct(">IHHIQII")
    SHIP_CLASS_ID = 0
    BRIDGE_PIER_CLASS_ID = 1
    SEND_CLASS_IDS = frozenset((SHIP_CLASS_ID, BRIDGE_PIER_CLASS_ID))

    def __init__(
        self,
        host="127.0.0.1",       # transport_tcp 所在地址
        port=9000,              # transport_tcp 的 YOLO 输入端口
        # 低延迟链路：陈旧框会污染融合结果，积压时丢帧优先于排队
        max_lag_frames=2,
        reconnect_interval=1.0,
        connect_timeout=2.0,
        send_timeout=1.0,
        box_log_path="",        # 非空则把实际发出的船只框记入该文件
    ):
        self.host = str(host)
        self.port = int(port)
        self.max_lag_frames = int(max_lag_frames)
        self.reconnect_interval = float(reconnect_interval)
        self.connect_timeout = float(connect_timeout)
        self.send_timeout = float(send_timeout)
        # 日志只跟船只；桥墩框仍照常发给融合
        self._box_logger = (
            YoloBoxLogger(box_log_path, class_ids=(self.SHIP_CLASS_ID,))
            if box_log_path else None
        )

        self._lock = threading.Lock()
        self._sock = None               # 到 transport_tcp 的 TCP 连接
        self._slot = None               # 待发送快照: (detections, frame_idx, width, height, timestamp)
        self._event = threading.Event()
        self._send_thread = None
        self._seq = 0

        self._running = False
        self.sent = 0
        self.dropped = 0
        self._latest_frame = -1

    def start(self):
        """启动后台线程：连接 transport_tcp 并持续发送数据"""
        if self._running:
            return

        self._running = True

        self._send_thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name="YoloSend",
        )
        self._send_thread.start()

        print(f"[YoloSender] 客户端模式，目标: {self.host}:{self.port}")

    def stop(self):
        """停止发送，关闭连接和线程"""
        self._running = False
        self._event.set()

        self._close_socket()

        if self._box_logger:
            self._box_logger.close()

        if self._send_thread and self._send_thread.is_alive() and \
           threading.current_thread() is not self._send_thread:
            self._send_thread.join(timeout=self.send_timeout + 2.0)

        print(
            f"[YoloSender] stopped | sent={self.sent} "
            f"dropped={self.dropped}"
        )

    def send_ships(self, ships, frame_idx, width, height, timestamp=None):
        """外部接口：放入待发送数据（线程安全）

        调用方（YOLO 推理线程）的 ship dict 之后会被主循环就地补写融合字段，
        所以这里立即抽出快照，发送线程只读快照。
        """
        if not self._running:
            return

        frame_timestamp = time.time() if timestamp is None else float(timestamp)
        detections = self._snapshot(ships, frame_timestamp)

        with self._lock:
            self._latest_frame = max(self._latest_frame, int(frame_idx))

            if self._slot is not None:
                self.dropped += 1

            self._slot = (
                detections,
                int(frame_idx),
                int(width),
                int(height),
                frame_timestamp,
            )

        self._event.set()

    @classmethod
    def _snapshot(cls, ships, frame_timestamp):
        """抽出待发送字段，过滤掉不参与融合的 class。"""
        detections = []
        for target in ships or ():
            class_id = int(target.get("class_id", cls.SHIP_CLASS_ID))
            if class_id not in cls.SEND_CLASS_IDS:
                continue

            x1, y1, x2, y2 = target["bbox"]
            detections.append({
                "id": int(target.get("ship_id", target.get("id", 0))),
                "class_id": class_id,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "score": float(target.get("conf", target.get("score", 0.0))),
                "timestamp": frame_timestamp,
            })
        return detections

    # ---------- 连接与发送线程 ----------

    def _connect(self):
        """连接到 transport_tcp，返回 socket 或 None"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.connect_timeout)
            sock.connect((self.host, self.port))
            sock.settimeout(self.send_timeout)
            print(f"[YoloSender] 已连接 transport_tcp {self.host}:{self.port}")
            return sock
        except (socket.error, OSError) as e:
            return None

    def _send_loop(self):
        """主循环：连接 transport_tcp → 发送数据 → 断线重连"""
        set_thread_name("YoloSend")

        while self._running:
            # 连接 transport_tcp
            sock = self._connect()
            if sock is None:
                if not self._running:
                    break
                time.sleep(self.reconnect_interval)
                continue

            with self._lock:
                self._sock = sock

            # 连接后连续发送数据
            while self._running:
                self._event.wait(timeout=0.1)
                self._event.clear()

                if not self._running:
                    break

                # 取数据
                with self._lock:
                    item = self._slot
                    self._slot = None
                    latest = self._latest_frame

                if item is None:
                    continue

                detections, frame_idx, _width, _height, frame_timestamp = item

                # 丢弃过时帧
                if latest - frame_idx > self.max_lag_frames:
                    self.dropped += 1
                    continue

                try:
                    packet = self._build_packet(
                        detections,
                        frame_idx,
                        _width,
                        _height,
                        frame_timestamp,
                        self._seq,
                    )
                    sock.sendall(packet)
                except (socket.error, OSError) as exc:
                    self.dropped += 1
                    print(f"[YoloSender] 发送失败: {exc}，将重连")
                    self._close_socket()
                    break

                self._seq = (self._seq + 1) & 0xFFFFFFFF
                self.sent += 1

                if self._box_logger:
                    self._box_logger.log(frame_idx, frame_timestamp, detections)

            # 断线后关闭，等待重连
            self._close_socket()
            if self._running:
                time.sleep(self.reconnect_interval)

    # ---------- 辅助方法 ----------

    def _close_socket(self):
        """安全关闭当前 socket"""
        with self._lock:
            sock = self._sock
            self._sock = None
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    @classmethod
    def _build_packet(cls, detections, frame_idx, width, height, timestamp, seq):
        """构建与 transport_tcp 兼容的 YOLO 检测数据包（二进制协议）。

        Header: >IHHIQII (28 bytes)
          - MAGIC (4B) | VERSION (2B) | MSG_TYPE (2B) | PAYLOAD_LEN (4B)
          - TIMESTAMP_US (8B) | SEQ (4B) | RESERVED (4B)

        Payload: JSON {"timestamp", "frame_id", "width", "height",
                       "yolo_detections": [...]}
          frame_id / width / height 供下游把框对齐到采集帧并校验坐标空间。
        """
        frame_timestamp = float(timestamp)

        payload_obj = {
            "timestamp": frame_timestamp,
            "frame_id": int(frame_idx),
            "width": int(width),
            "height": int(height),
            "yolo_detections": [
                {
                    "id": det["id"],
                    "x1": det["bbox"][0],
                    "y1": det["bbox"][1],
                    "x2": det["bbox"][2],
                    "y2": det["bbox"][3],
                    "score": det["score"],
                    "class_id": det["class_id"],
                    "timestamp": det["timestamp"],
                }
                for det in detections
            ],
        }
        payload = json.dumps(
            payload_obj,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        header = cls.HEADER_STRUCT.pack(
            cls.MAGIC,
            cls.VERSION,
            cls.MSG_YOLO_DETECTIONS,
            len(payload),
            int(frame_timestamp * 1_000_000),
            int(seq) & 0xFFFFFFFF,
            0,
        )
        return header + payload