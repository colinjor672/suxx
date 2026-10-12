"""
RTSP 摄像头画面抓帧工具
- 连接 rtsp://admin:gzsia2011@192.168.1.164:554/Streaming/Channels/101
- 实时显示画面
- 按 空格键 保存当前帧到 out/ 目录
- 按 ESC 或 q 退出
"""

import cv2
import os
import datetime

RTSP_URL = "rtsp://admin:gzsia2011@192.168.1.164:554/Streaming/Channels/101"
OUTPUT_DIR = "out"
WINDOW_NAME = "RTSP Capture - 空格截图 | ESC/q 退出"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 针对 Jetson Orin Nano 的 GStreamer 后端（硬件加速解码）
    # 如果 OpenCV 不是 GStreamer 编译的，回退到默认后端
    cap = None
    for backend, url in [
        # 方案1: GStreamer 硬件加速 (Jetson 推荐)
        (cv2.CAP_GSTREAMER,
         f"rtspsrc location={RTSP_URL} latency=300 ! "
         "rtph264depay ! h264parse ! nvv4l2decoder ! "
         "nvvidconv ! video/x-raw,format=BGRx ! "
         "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"),
        # 方案2: GStreamer 软件解码
        (cv2.CAP_GSTREAMER,
         f"rtspsrc location={RTSP_URL} latency=300 ! "
         "rtph264depay ! h264parse ! avdec_h264 ! "
         "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2"),
        # 方案3: FFmpeg 后端
        (cv2.CAP_FFMPEG, RTSP_URL),
        # 方案4: 默认后端
        (cv2.CAP_ANY, RTSP_URL),
    ]:
        print(f"尝试后端: {backend}, URL: {url[:80]}...")
        cap = cv2.VideoCapture(url, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ 连接成功！分辨率: {frame.shape[1]}x{frame.shape[0]}")
                break
            cap.release()
            cap = None
        else:
            cap.release()
            cap = None

    if cap is None or not cap.isOpened():
        print("❌ 所有后端连接失败，请检查:")
        print("   1. 摄像头是否在线")
        print("   2. RTSP 地址是否正确")
        print("   3. 网络是否连通 (ping 192.168.1.164)")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)

    frame_count = 0
    print(f"\n操作: 空格=截图保存 | ESC/q=退出")
    print(f"截图保存到: {os.path.abspath(OUTPUT_DIR)}/\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️ 读取帧失败，重试...")
            continue

        frame_count += 1
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # 空格键
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"frame_{ts}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"📸 已保存: {filepath}")

        elif key == 27 or key == ord('q'):  # ESC 或 q
            print(f"\n退出，共捕获 {frame_count} 帧")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()