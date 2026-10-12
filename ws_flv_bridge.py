#!/usr/bin/env python3
"""ws -> HTTP-FLV 转发器

前端播放器只认 ws:// 时，把 EasyDarwin 的 http-flv 原样转成 WebSocket 二进制流。

用法（Jetson 上）:
    pip install websockets
    python3 ws_flv_bridge.py [源URL] [监听端口]

前端播放地址:
    ws://<jetson-ip>:9001/live/stream_1.flv   （路径部分任意，转发器忽略）
"""
import asyncio
import sys
import urllib.request

import websockets

SOURCE = "http://127.0.0.1:8080/live/stream_1.flv"
HOST = "0.0.0.0"
PORT = 9001

READ_CHUNK = 65536


async def handle(ws, src_url):
    loop = asyncio.get_running_loop()
    try:
        src = await loop.run_in_executor(
            None, lambda: urllib.request.urlopen(src_url, timeout=10)
        )
    except Exception as e:
        print(f"[bridge] 打开源失败: {e}", flush=True)
        await ws.close(code=1011, reason=str(e)[:100])
        return

    print(f"[bridge] 观众接入 {ws.remote_address}", flush=True)
    try:
        while True:
            chunk = await loop.run_in_executor(None, src.read, READ_CHUNK)
            if not chunk:
                break
            await ws.send(chunk)
    except websockets.ConnectionClosed:
        pass
    finally:
        src.close()
        print(f"[bridge] 观众离开 {ws.remote_address}", flush=True)


async def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SOURCE
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    async with websockets.serve(
        lambda ws: handle(ws, src),
        HOST, port, max_size=None, ping_interval=None,
    ):
        print(f"[bridge] ws://{HOST}:{port}  ->  {src}", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
