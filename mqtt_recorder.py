"""
MQTT 传感器数据录制工具

订阅 MQTT 上的所有传感器数据（LiDAR、毫米波雷达、GNSS），
按传感器分别记录到 JSONL 文件，每行一条 JSON 记录。

文件输出:
  recordings/{timestamp}/
    ├── lidar_01.jsonl
    ├── mmwave_01.jsonl
    ├── mmwave_02.jsonl
    ├── ...
    └── gnss_01.jsonl

每行格式:
  {"recv_ts": 1234567890.123, "topic": "v1/.../mmwave_02", "payload": {原始JSON}}

用法:
    python scripts/mqtt_recorder.py
    python scripts/mqtt_recorder.py --host 192.168.1.100 --port 1883
    python scripts/mqtt_recorder.py --output ./my_recordings
    python scripts/mqtt_recorder.py --username admin --password 123456
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import paho.mqtt.client as mqtt

# 脚本独立运行时，需要能 import pyfusion 的 config
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    from pyfusion.config import load_transport_config
except ImportError:
    load_transport_config = None  # 退化为手动参数


# ============================================================================
# 默认配置 —— 当 transport.ini 不存在或未安装 pyfusion 时使用
# 直接修改下方常量即可自定义默认配置
# ============================================================================
DEFAULT_MQTT_HOST = "127.0.0.1"
DEFAULT_MQTT_PORT = 1890
DEFAULT_MQTT_USERNAME = ""        # 留空则不认证
DEFAULT_MQTT_PASSWORD = ""

# 默认订阅话题（格式: v1/{device_id}/sensor/{sensor_type}/{sensor_id}）
DEFAULT_TOPICS = [
    "v1/+/sensor/lidar/lidar_01",
    "v1/+/sensor/mmwave/+",
    "v1/+/sensor/gnss/gnss_01",
]


class SensorRecorder:
    """按传感器分文件录制 MQTT 数据。"""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._files: Dict[str, object] = {}
        self._counts: Dict[str, int] = {}
        self._byte_counts: Dict[str, int] = {}
        self._start_time = time.time()
        self._last_stats_time = time.time()

    def _get_sensor_key(self, topic: str) -> str:
        """从 topic 提取传感器标识作为文件名。

        topic 格式: v1/{device_id}/sensor/{sensor_type}/{sensor_id}
        返回: {sensor_type}_{sensor_id}，如 lidar_01, mmwave_02, gnss_01
        """
        parts = topic.split("/")
        if len(parts) >= 5 and parts[2] == "sensor":
            sensor_type = parts[3]
            sensor_id = parts[4]
            return f"{sensor_type}_{sensor_id}"
        # 退化：用 topic 末尾段
        return parts[-1].replace("/", "_") if parts else "unknown"

    def _get_file(self, sensor_key: str):
        """获取或创建传感器对应的文件句柄。"""
        if sensor_key not in self._files:
            filepath = self._output_dir / f"{sensor_key}.jsonl"
            fh = open(filepath, "a", encoding="utf-8", buffering=1)  # 行缓冲
            self._files[sensor_key] = fh
            self._counts[sensor_key] = 0
            self._byte_counts[sensor_key] = 0
            print(f"  [录制] {sensor_key} → {filepath.name}")
        return self._files[sensor_key]

    def record(self, topic: str, payload: bytes) -> None:
        """记录一条 MQTT 消息。"""
        import orjson

        sensor_key = self._get_sensor_key(topic)
        recv_ts = time.time()

        # 尝试解析 payload 为 JSON
        try:
            payload_obj = orjson.loads(payload)
        except Exception:
            # 非 JSON payload，用 base64 保存原始字节
            import base64
            payload_obj = {"_raw_base64": base64.b64encode(payload).decode("ascii")}

        # 构建记录行
        record = {
            "recv_ts": recv_ts,
            "topic": topic,
            "payload": payload_obj,
        }

        line = orjson.dumps(record).decode("utf-8")

        fh = self._get_file(sensor_key)
        fh.write(line + "\n")

        self._counts[sensor_key] = self._counts.get(sensor_key, 0) + 1
        self._byte_counts[sensor_key] = self._byte_counts.get(sensor_key, 0) + len(line)

        # 定期打印统计
        now = time.time()
        if now - self._last_stats_time >= 5.0:
            self._print_stats()
            self._last_stats_time = now

    def _print_stats(self) -> None:
        """打印各传感器录制统计。"""
        elapsed = time.time() - self._start_time
        print(f"\n--- 录制统计 ({elapsed:.0f}s) ---")
        for key in sorted(self._counts.keys()):
            count = self._counts[key]
            size_mb = self._byte_counts[key] / (1024 * 1024)
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  {key:20s}  {count:8d} 条  {size_mb:8.2f} MB  {rate:6.1f} msg/s")
        print("-" * 60)

    def close(self) -> None:
        """关闭所有文件，打印最终统计。"""
        self._print_stats()
        for key, fh in self._files.items():
            try:
                fh.close()
            except Exception:
                pass
        self._files.clear()
        print("录制结束。")


def create_client(
    host: str,
    port: int,
    username: Optional[str],
    password: Optional[str],
    topics: list,
    recorder: SensorRecorder,
) -> mqtt.Client:
    """创建并配置 MQTT 客户端。"""
    client_id = f"pyfusion_recorder_{os.getpid()}"

    # paho-mqtt v2.x 需要 CallbackAPIVersion
    try:
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            clean_session=True,
        )
    except AttributeError:
        # paho-mqtt v1.x 退路
        client = mqtt.Client(client_id=client_id, clean_session=True)

    if username:
        client.username_pw_set(username, password or "")

    connected_flag = {"done": False}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        if reason_code == 0:
            print(f"  [MQTT] 已连接 {host}:{port}")
            for topic in topics:
                client.subscribe(topic, qos=0)
                print(f"  [MQTT] 订阅: {topic}")
            connected_flag["done"] = True
        else:
            print(f"  [MQTT] 连接失败，reason_code={reason_code}")

    def on_message(client, userdata, msg):
        try:
            recorder.record(msg.topic, msg.payload)
        except Exception as exc:
            print(f"  [录制错误] topic={msg.topic}: {exc}")

    def on_disconnect(client, userdata, *args, **kwargs):
        print("  [MQTT] 连接断开，尝试重连...")

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    return client, connected_flag


def main() -> None:
    parser = argparse.ArgumentParser(
        description="录制 MQTT 传感器数据到 JSONL 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--host", default=None, help="MQTT Broker 地址（默认读 transport.ini）")
    parser.add_argument("--port", type=int, default=None, help="MQTT Broker 端口")
    parser.add_argument("--username", default=None, help="MQTT 用户名")
    parser.add_argument("--password", default=None, help="MQTT 密码")
    parser.add_argument("--output", default=None, help="输出目录（默认 recordings/{时间戳}）")
    parser.add_argument(
        "--topics", nargs="+", default=None,
        help="自定义订阅 topic 列表（默认 LiDAR+mmWave+GNSS）",
    )
    args = parser.parse_args()

    # 配置优先级：命令行参数 > transport.ini > 文件顶部默认配置
    host = DEFAULT_MQTT_HOST
    port = DEFAULT_MQTT_PORT
    username = DEFAULT_MQTT_USERNAME
    password = DEFAULT_MQTT_PASSWORD
    topics = args.topics or DEFAULT_TOPICS

    # 尝试从 transport.ini 读取（覆盖默认值）
    if load_transport_config is not None:
        ini_path = _PROJECT_ROOT / "transport.ini"
        if ini_path.is_file():
            try:
                cfg = load_transport_config(str(ini_path))
                host = cfg.mqtt.host
                port = cfg.mqtt.port
                username = cfg.mqtt.username or ""
                password = cfg.mqtt.password or ""
                print(f"  [配置] 已读取 {ini_path.name}")
            except Exception as exc:
                print(f"  [配置] 读取 transport.ini 失败: {exc}，使用默认配置")
        else:
            print(f"  [配置] 未找到 transport.ini，使用文件顶部默认配置")
    else:
        print(f"  [配置] 未安装 pyfusion，使用文件顶部默认配置")

    # 命令行参数优先级最高
    if args.host is not None:
        host = args.host
    if args.port is not None:
        port = args.port
    if args.username is not None:
        username = args.username
    if args.password is not None:
        password = args.password

    # 创建输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = _PROJECT_ROOT / "recordings" / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MQTT 传感器数据录制工具")
    print("=" * 60)
    print(f"  Broker:    {host}:{port}")
    print(f"  认证:      {'有' if username else '无'}")
    print(f"  输出目录:  {output_dir}")
    print(f"  订阅话题:  {len(topics)} 个")
    for t in topics:
        print(f"    - {t}")
    print("=" * 60)
    print("  按 Ctrl+C 停止录制\n")

    recorder = SensorRecorder(output_dir)
    client, connected_flag = create_client(
        host, port, username, password, topics, recorder,
    )

    # 优雅退出：SIGINT(Ctrl+C) + SIGTERM(Linux systemd/Docker)
    running = {"stop": False}

    def signal_handler(sig, frame):
        sig_name = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
        print(f"\n\n收到 {sig_name} 信号，正在停止录制...")
        running["stop"] = True

    signal.signal(signal.SIGINT, signal_handler)
    # SIGTERM 在 Linux 上由 systemd/Docker 发送；Windows 上注册也无害
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    # 连接并启动循环
    try:
        client.connect(host, port, keepalive=60)
        client.loop_start()

        # 等待连接
        for _ in range(50):
            if connected_flag["done"]:
                break
            time.sleep(0.1)

        if not connected_flag["done"]:
            print("[警告] 连接超时，继续等待重连...")

        # 主循环：检测退出信号
        while not running["stop"]:
            time.sleep(0.2)

    except ConnectionRefusedError:
        print(f"\n[错误] 无法连接 {host}:{port}，请检查 MQTT Broker 是否运行。")
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"\n[错误] {exc}")
        sys.exit(1)
    finally:
        client.loop_stop()
        recorder.close()


if __name__ == "__main__":
    main()
