import os
import time
import threading
from threading import Thread

import cv2
import numpy as np
import torch
# 在 ultralytics 导入前设置：关闭全局 logger / checks 输出，减少每次调用的环境检查开销
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

from config import (
    RenderConfig,
    SHIP_ENGINE_PATH,
    SHIP_CONF,
    SHIP_IOU,
    SHIP_CLASS_NAMES,
    SHIP_INPUT_SIZE,
    SHIP_CLASS_IDS,
)
from network.gst_utils import (
    print_capabilities,
    create_hw_decode_capture,
    create_x264enc_writer,
    H264Encoder,
    cuda_resize,
    nvjpeg_encode,
    HAS_CUDA_RESIZE,
    HAS_NVJPEG,
    IS_JETSON,
)
from infer.trt_inferencer import TRTInferencer
from infer.preprocess import PreprocessGPU
from infer.async_seg import AsyncSegInferencer
from infer.yolo_detector import AsyncYOLODetector

from network.shm_video_sender import ShmVideoSender
from network.nav_sender import AsyncNavDataSender
from network.yolo_sender import YoloSender
from network.fusion_receiver import FusionReceiver
from network.mqtt_navigation import MqttNavigationClient, MqttNavigationConfig
import network.server as nav_server_module

from utils.async_writer import AsyncVideoWriter
from utils.geometry import (
    scale_curve_xy,
    scale_pixel_bbox,
    scale_ships_for_display,
    merge_ship_data,
    attach_dcpa_target,
    should_draw_nav_band,
)
from utils.gnss_projection import RealtimeGnssProjectionEngine
from utils.realtime_navigation import RealtimeNavigationState
from utils.bridge_warning import BridgeWarningEngine, BridgeAlertClient, RtkReplay
from utils.frame_pacer import FramePacer

from render.overlay import overlay_masks

def process_video(input_video, output_video,
                  seg_weights,
                  device_str, enable_streaming=True,
                  codec="mp4v",
                  enable_video_stream=True,
                  show_window=False,
                  show_scale=0.5,
                  playback_fps=0.0,
                  blur_kernel=5,
                  draw_curve_on_python=False,
                  draw_masks_on_python=False,
                  ship_host="127.0.0.1",
                  ship_port=0,
                  yolo_send_port=9000,
                  save_video=False,
                  nav_send_every=3,
                  mask_send_scale=0.5,
                  nav_draw_distance=30.0,
                  nav_scale=0.5,
                  nav_projection_hz=10.0,
                  calib_ini="calibration.ini",
                  lidar_height=1.6,
                  nav_heading_offset=0.0,
                  mqtt_host="127.0.0.1",
                  mqtt_port=1883,
                  prediction_topic="v1/11/prediction/result",
                  gnss_topic="v1/11/sensor/gnss/gnss_01",
                  mqtt_username=None,
                  mqtt_password=None,
                  rtk_replay="",
                  rtk_t0_time=32.0,
                  rtk_time_scale=1.0,
                  ship_id=11,
                  redis_host="127.0.0.1",
                  redis_port=6379,
                  redis_password=None,
                  redis_db=1,
                  redis_refresh=1.0):
    # 各组件推理频率（硬编码，各自独立）
    DEPTV3_EVERY = 5   # DeepLabV3 分割：每5帧
    # YOLO 频率由 yolo_detector.py 的 detect_every=7 控制

    cfg = RenderConfig()

    MASK_H, MASK_W = 432, 768

    # 避障经 MQTT 回传的 pixel 以此分辨率为参考；吸附前换算到当帧框所在空间
    DCPA_PIXEL_SRC_SIZE = (1280, 720)

    need_render = show_window or save_video
    show_scale  = float(np.clip(show_scale, 0.25, 1.0))

    print(f"\n推理尺寸  : {MASK_W}×{MASK_H}")
    print(f"掩码策略  : 每 {DEPTV3_EVERY} 帧推理一次，其余复用")
    print(f"本地渲染  : {'开' if need_render else '关(最高性能)'}")
    if show_window:
        print(f"显示缩放  : {show_scale}  (show_scale)")
    print(f"Python绘制: 曲线={'开' if draw_curve_on_python else '关'} "
          f"掩码={'开' if draw_masks_on_python else '关'}")
    print(f"视频流    : 单路RTSP→GStreamer解码→分叉 (AI推理+ShmVideoSender→Godot)")

    # TCP 服务
    if enable_streaming:
        Thread(target=nav_server_module.start_tcp_server, daemon=True).start()
        time.sleep(1.5)
        print(f"TCP 服务器已启动:")
        print(f"  导航数据: tcp://0.0.0.0:8765")

    # 单预处理器
    preprocess = PreprocessGPU(target_h=MASK_H, target_w=MASK_W)

    # TRT 推理器
    inferencer = TRTInferencer(
        engine_path=seg_weights,
        input_h=MASK_H,
        input_w=MASK_W,
    )

    #YOLO 异步检测器
    SHIP_CLASS_IDS = [0, 1]
    SHIP_MODEL = YOLO(SHIP_ENGINE_PATH, task="detect")

    yolo_detector = AsyncYOLODetector(
        model=SHIP_MODEL,
        imgsz=SHIP_INPUT_SIZE,
        conf=SHIP_CONF,
        iou=SHIP_IOU,
        device="cuda:0",
        class_names=SHIP_CLASS_NAMES,
        allowed_class_ids=SHIP_CLASS_IDS,
    )


    # YoloSender（客户端模式：主动连接 transport_tcp）
    yolo_sender = None
    if yolo_send_port > 0 and ship_host and ship_host.strip():
        yolo_sender = YoloSender(
            host=ship_host,
            port=yolo_send_port,
            box_log_path="yolo_boxes.log",
        )
        yolo_sender.start()
        print(f"[YoloSender] → transport_tcp {ship_host}:{yolo_send_port}")

    #  FusionReceiver（55103 默认已停用，DCPA 目标框改走 MQTT）
    fusion_receiver = None
    if ship_port > 0 and ship_host and ship_host.strip():
        fusion_receiver = FusionReceiver(ship_host, ship_port)
        fusion_receiver.start()

    # 视频源 
    #cap = cv2.VideoCapture(input_video)
    #if not cap.isOpened():
        #raise RuntimeError(f"Cannot open: {input_video}")
    #fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    #width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_rtsp = input_video.startswith("rtsp://") or input_video.startswith("rtmp://")

    #if is_rtsp:
        # RTSP 用 ffmpeg backend，低延迟
        #cap = cv2.VideoCapture(input_video, cv2.CAP_FFMPEG)
        #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # 最小缓冲，降低延迟
        #cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    #else:
        #cap = cv2.VideoCapture(input_video)
    
    #if not cap.isOpened():
    print_capabilities()
    cap = create_hw_decode_capture(input_video)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Cannot open: {input_video}")
    
    source_fps   = float(cap.get(cv2.CAP_PROP_FPS))
    fps          = source_fps if np.isfinite(source_fps) and source_fps > 0 else 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # RTSP 没有 total_frames，设为 0 避免除零
    if is_rtsp:
        total_frames = 0
        print(f"RTSP 流: {width}×{height} @ {fps:.1f}fps")

    # YOLO 框由推理线程产出即发送：不等掩码分割，也不占主循环节拍
    def publish_yolo_detections(ships, frame_id, frame_timestamp):
        if yolo_sender is None:
            return
        yolo_sender.send_ships(
            ships,
            frame_id,
            width,
            height,
            timestamp=frame_timestamp,
        )

    yolo_detector.on_result = publish_yolo_detections

    is_file_source = os.path.isfile(input_video)
    frame_pacer = None
    has_realtime_consumer = enable_video_stream or enable_streaming or show_window
    if is_file_source and has_realtime_consumer:
        target_fps = min(float(playback_fps), float(fps)) if playback_fps > 0 else float(fps)
        frame_pacer = FramePacer(target_fps)
        print(
            f"本地视频限帧: {target_fps:.3f} FPS "
            f"({'手动上限' if playback_fps > 0 else '源视频帧率'})"
        )
    elif is_file_source:
        print("本地视频限帧: 关闭（无实时显示或数据消费者）")
    elif is_rtsp and playback_fps > 0:
        target_fps = float(playback_fps)
        frame_pacer = FramePacer(target_fps)
        print(f"RTSP 处理帧率上限: {target_fps:.3f} FPS（始终取最新帧）")

    timeline_fps = frame_pacer.fps if is_rtsp and frame_pacer is not None else fps
    
    #视频写出
    #writer       = None
    #async_writer = None
    #if save_video:
    #    fourcc       = cv2.VideoWriter_fourcc(*codec)
    #    writer       = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    #    async_writer = AsyncVideoWriter(writer, maxsize=6)
    #    print(f"视频保存: 开 → {output_video}")
    #else:
    #    print("视频保存: 关")

    shm_video_sender = None
    if enable_video_stream:
        shm_video_sender = ShmVideoSender()
        shm_video_sender.start()

    # 导航数据异步发送器
    nav_sender = None
    if enable_streaming:
        nav_sender = AsyncNavDataSender(
            nav_server=nav_server_module.nav_server,
            mask_send_scale=1.0,
        )

    # 预热
    dummy = torch.randn(1, 3, MASK_H, MASK_W, dtype=torch.float16, device='cuda')
    for _ in range(3):
        inferencer.infer(dummy)
    print("✓ TRT 预热完成")

    # 发送尺寸（提前计算，传给 async_seg 在 GPU 上 resize）
    send_scale = float(np.clip(mask_send_scale, 0.2, 1.0))
    send_w     = max(1, int(width  * send_scale))
    send_h     = max(1, int(height * send_scale))

    navigation_state = RealtimeNavigationState(
        lookahead_m=nav_draw_distance,
        sample_spacing_m=0.25,
        speed_window_points=15,
        speed_update_hz=10.0,
        moving_threshold_mps=1.0,
        timestamp_tolerance_s=1.0,
        heading_offset_deg=nav_heading_offset,
    )
    gnss_projector = RealtimeGnssProjectionEngine(
        navigation_state=navigation_state,
        calibration_path=calib_ini,
        width=width,
        height=height,
        lidar_height_m=lidar_height,
        projection_scale=nav_scale,
        update_hz=nav_projection_hz,
    )

    # 桥梁警戒：预存桥梁库 + 分割桥面线 → 只产出警戒线/信息面板的像素
    # 坐标，随 nav 包 bridge_warning 字段发给 Godot 渲染（Python 不画线）。
    # 坐标必须与 nav 包 width/height 同空间（发送尺寸，掩码所在空间），
    # 否则 Godot 按包尺寸缩放会整体偏移；投影内参按同比例缩放，几何不变。
    # 船位来源：rtk_replay 给定时按视频时间回放 Excel（离线验证），
    # 否则取实时 GNSS（navigation_state）。
    rtk_replayer = None
    if rtk_replay:
        rtk_replayer = RtkReplay(rtk_replay)
        camera_h = (rtk_replayer.elev_mean
                    if rtk_replayer.elev_mean > 0.0 else None)
        print(f"[BridgeWarning] RTK 回放: {rtk_replay} "
              f"{len(rtk_replayer.time)} 点, 时长 {rtk_replayer.t_end:.1f}s, "
              f"航向源 {rtk_replayer.heading_src}, "
              f"t_rtk=(t_video−{rtk_t0_time:g})×{rtk_time_scale:g}")
    else:
        camera_h = None
    alert_client = None
    if redis_host:
        alert_client = BridgeAlertClient(
            ship_id, host=redis_host, port=redis_port, db=redis_db,
            password=redis_password, refresh_interval=redis_refresh)
    else:
        print("[BridgeAlert] 未设置 --redis-host → 仅用预存桥梁库")
    bridge_warning_engine = BridgeWarningEngine(
        width=send_w, height=send_h, calib_path=calib_ini,
        camera_height=camera_h,
        coord_scale=send_w / float(max(1, width)),
        alert_client=alert_client)
    print(f"[BridgeWarning] ✓ 引擎就绪：坐标空间 {send_w}×{send_h}（与 nav 包"
          "一致），预存桥梁库 8 座，警戒线经 8765 下发 Godot")
    mqtt_config = MqttNavigationConfig(
        host=mqtt_host,
        port=mqtt_port,
        prediction_topic=prediction_topic,
        gnss_topic=gnss_topic,
        username=mqtt_username,
        password=mqtt_password,
    )
    mqtt_navigation = MqttNavigationClient(
        navigation_state,
        mqtt_config,
        clock=time.time,
        monotonic_clock=time.monotonic,
    )

    # 异步分割推理器（在 GPU 上完成 mask resize，避免下游 cuda_resize 往返）
    async_seg = AsyncSegInferencer(
        inferencer = inferencer,
        preprocess = preprocess,
        mask_h     = MASK_H,
        mask_w     = MASK_W,
        send_h     = send_h,
        send_w     = send_w,
    )

    # CPU 端缓存（每帧使用的"当前可用结果"）
    water_small_np      = None
    bridge_small_np     = None
    gnss_curve          = np.empty((0, 2), dtype=np.int32)
    gnss_curve_send     = np.empty((0, 2), dtype=np.int32)
    cached_water_send   = None
    cached_bridge_send  = None
    cached_water_large  = None
    cached_bridge_large = None

    frame_idx   = 0
    frame_times = []
    gst_decode_times = []
    last_processed_seg_fid = -1
    last_sent_nav_seg_fid = -1
    bridge_warning_data = None
    dcpa_snap_stats = {"hit": 0, "miss": 0, "ambiguous": 0}
    dcpa_snap_log_ts = 0.0

    print(f"视频: {width}×{height} @ {fps:.1f}fps, 共 {total_frames} 帧")
    print(f"[AsyncSeg] ✓ 推理完全异步 | 主循环零等待")
    print(f"开始处理...\n")
    t_start = time.time()

    navigation_state.start()
    try:
        mqtt_navigation.start()
    except Exception:
        mqtt_navigation.stop()
        navigation_state.stop()
        raise

    try:
        _loop_started = False
        while True:
            t_frame = time.time()
            # Wait before capture so an RTSP source is sampled at the newest
            # available frame after the pacing interval.
            if frame_pacer is not None:
                frame_pacer.wait()
            ret, result = cap.read()

            # GstTeeCapture 用 PTS 估算了 RTSP 源"编码端出图 → 解码出帧"的耗时，
            # 把帧时间戳向前回拨这段时间，下游按时间戳算出的延迟即覆盖解码前的全链路。
            # 本地文件或 CPU 软解没有该值，取 0，时间戳保持不变。
            gst_decode_s = getattr(cap, "last_pts_latency_s", None) or 0.0
            if gst_decode_s > 0.0:
                gst_decode_times.append(gst_decode_s)
                if len(gst_decode_times) > 120:
                    gst_decode_times.pop(0)

            frame_timestamp = time.time() - gst_decode_s
            if not ret:
                print(f"[Pipeline] cap.read() 返回 False，主循环退出 (frame_idx={frame_idx})", flush=True)
                break

            # 兼容两种捕获模式
            if isinstance(result, tuple) and len(result) == 2:
                # GstTeeCapture: (bgr, rgba)
                bgr_frame, rgba_frame = result
                frame = bgr_frame
            else:
                # cv2.VideoCapture / GstVideoCapture: 单帧 BGR
                frame = result
                bgr_frame = frame
                import cv2 as _cv2
                bgr_small = _cv2.resize(frame, (640, 360))
                rgba_frame = _cv2.cvtColor(bgr_small, _cv2.COLOR_BGR2RGBA)
                if frame_idx == 0:
                    print("[Pipeline] ⚠ 使用 CPU 回退: cv2.resize + cvtColor", flush=True)

            frame_idx += 1

            if not _loop_started:
                _loop_started = True
                print(f"[Pipeline] 主循环首帧 frame_idx=1 AI={bgr_frame.shape} Godot={rgba_frame.shape}",
                      flush=True)

            video_time = float(frame_idx - 1) / max(1e-6, float(timeline_fps))

            # GNSS 地面点经相机标定投映到完整视频坐标。
            # 下游 TCP 协议和 Godot 渲染器仍只接收原有二维 curve。
            gnss_curve = gnss_projector.project()
            if len(gnss_curve) > 0:
                gnss_curve_send = scale_curve_xy(
                    gnss_curve,
                    send_w / float(width),
                    send_h / float(height),
                )
            else:
                gnss_curve_send = np.empty((0, 2), dtype=np.int32)

            # ★ 步骤 0：RGBA 640×360 直接写入 /dev/shm（nvvidconv GPU 已转换，零 CPU）
            if shm_video_sender:
                shm_video_sender.send(rgba_frame)

            # ★ 步骤 1：提交推理（非阻塞，丢给后台线程）
            if frame_idx % DEPTV3_EVERY == 0:
                async_seg.submit(frame, frame_idx)

            yolo_detector.submit(frame, frame_idx, frame_timestamp)

            # ★ 步骤 2：取最新分割结果（非阻塞，有新的就更新缓存）
            seg_result = async_seg.get_result()

            if seg_result is not None:
                if len(seg_result) < 3:
                    raise RuntimeError(
                        f"AsyncSeg 返回字段不足: 期望至少 3 项，实际 {len(seg_result)} 项"
                    )
                # 兼容旧版四项返回值；第四项旧 mask 曲线始终丢弃。
                new_water_np, new_bridge_np, seg_fid = seg_result[:3]

                # 只有拿到新分割结果时，才处理
                if seg_fid != last_processed_seg_fid:
                    last_processed_seg_fid = seg_fid

                    water_small_np = new_water_np
                    bridge_small_np = new_bridge_np

                    # ★ 桥梁警戒：新分割帧上更新（桥面线/面板锚点只在推理帧
                    #   喂一次）；回放模式按视频时间取 Excel 船位，否则取最新 GNSS
                    if rtk_replayer is not None:
                        vessel_fix = rtk_replayer.fix_at(
                            video_time, rtk_t0_time, rtk_time_scale)
                    else:
                        vessel_fix = navigation_state.latest_vessel_fix()
                    bridge_warning_data = (
                        bridge_warning_engine.update(
                            bridge_small_np, *vessel_fix)
                        if vessel_fix is not None else None)

                    # 诊断由引擎内部按秒节流打印（[BridgeWarning] 空间…）

                    # ★ mask 已在 GPU 上 resize 到 send 尺寸，直接使用，无需 cuda_resize
                    cached_water_send = water_small_np
                    cached_bridge_send = bridge_small_np

                    if need_render and save_video:
                        cached_water_large = cuda_resize(
                            water_small_np,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )

                        cached_bridge_large = cuda_resize(
                            bridge_small_np,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )

            # ★ 步骤 3：船只数据汇总（桥墩 class 1 仅发送，不参与显示/融合）
            yolo_detections = yolo_detector.get_result()
            yolo_ships = [
                target
                for target in yolo_detections
                if int(target.get("class_id", 0)) == 0
            ]
            external_ships = fusion_receiver.get_ships() if fusion_receiver else []
            ships_final    = merge_ship_data(yolo_ships, external_ships)

            # 避障只指定单一目标：回传的 1280×720 像素框先换算到当帧 YOLO 框
            # 所在的解码帧空间再就近吸附，命中框是唯一显示信息面板的框
            # （5 秒无更新即视为无目标）
            dcpa_alert = navigation_state.dcpa_alert(max_age_s=5.0)
            dcpa_pixel_bbox = scale_pixel_bbox(
                dcpa_alert.get("pixel_bbox") if dcpa_alert else None,
                DCPA_PIXEL_SRC_SIZE,
                (width, height),
            )
            attach_dcpa_target(
                ships_final,
                dcpa_pixel_bbox,
                dcpa_alert,
                frame_width=width,
                stats=dcpa_snap_stats,
            )

            # 吸附计数按 5 秒打印一次，用于判断阈值是否过松/过紧
            _snap_now = time.time()
            if _snap_now - dcpa_snap_log_ts >= 5.0:
                total = sum(dcpa_snap_stats.values())
                print(
                    f"[DCPA snap] hit={dcpa_snap_stats['hit']} "
                    f"miss={dcpa_snap_stats['miss']} "
                    f"ambiguous={dcpa_snap_stats['ambiguous']} "
                    f"/ {total} 次 (base={12.0 * width / 640.0:.1f}px "
                    f"@{width}x{height})"
                )
                dcpa_snap_log_ts = _snap_now

            # ★ 步骤 4：TCP 导航数据发送
            # 只在新的分割帧出现后发送一次，避免重复提取同一张 mask 的 polygon
            has_new_nav_mask = (
                last_processed_seg_fid > 0
                and last_processed_seg_fid != last_sent_nav_seg_fid
            )

            if (nav_sender
                    and cached_water_send is not None
                    and cached_bridge_send is not None
                    and has_new_nav_mask):

                mask_frame_id = int(last_processed_seg_fid)
                mask_video_time = float(mask_frame_id - 1) / max(1e-6, float(timeline_fps))

                nav_sender.submit(
                    width       = send_w,
                    height      = send_h,
                    curve       = gnss_curve_send,
                    frame_id    = frame_idx,
                    timestamp   = video_time,

                    water_mask  = cached_water_send,
                    bridge_mask = cached_bridge_send,
                    ships_data  = ships_final,
                    ship_src_w  = width,
                    ship_src_h  = height,

                    video_time      = video_time,
                    sync_mode       = "file_time",
                    source_fps      = timeline_fps,
                    mask_frame_id   = mask_frame_id,
                    mask_video_time = mask_video_time,
                    bridge_warning  = bridge_warning_data,
                )

                last_sent_nav_seg_fid = last_processed_seg_fid

            # ★ 步骤 5：本地渲染与窗口显示
            if need_render:
                elapsed  = time.time() - t_start
                cur_fps  = frame_idx / elapsed if elapsed > 0 else 0.0
                fps_text = (
                    f"FPS:{cur_fps:.1f} Frame:{frame_idx}{f'/{total_frames}' if total_frames > 0 else ''} "
                    f"Ships:{len(ships_final)}"
                )

                if show_window:
                    disp_w = max(1, int(width  * show_scale))
                    disp_h = max(1, int(height * show_scale))

                    frame_small = cuda_resize(
                        frame, (disp_w, disp_h),
                        interpolation=cv2.INTER_AREA)

                    if water_small_np is not None:
                        wm_small = cuda_resize(
                            water_small_np, (disp_w, disp_h),
                            interpolation=cv2.INTER_NEAREST)
                        bm_small = cuda_resize(
                            bridge_small_np, (disp_w, disp_h),
                            interpolation=cv2.INTER_NEAREST)
                    else:
                        wm_small = np.zeros((disp_h, disp_w), dtype=np.uint8)
                        bm_small = np.zeros((disp_h, disp_w), dtype=np.uint8)

                    if len(gnss_curve) > 0:
                        curve_disp = scale_curve_xy(
                            gnss_curve,
                            disp_w / float(width),
                            disp_h / float(height),
                        )
                    else:
                        curve_disp = np.empty((0, 2), dtype=np.int32)

                    ships_disp = scale_ships_for_display(
                        ships_final,
                        disp_w / float(width),
                        disp_h / float(height),
                    )

                    vis = overlay_masks(
                        frame_small,
                        water_mask  = wm_small,
                        bridge_mask = bm_small,
                        cfg         = cfg,
                        curve       = curve_disp,
                        frame_idx   = frame_idx,
                        fps_text    = fps_text,
                        draw_curve  = draw_curve_on_python,
                        draw_masks  = draw_masks_on_python,
                        ships       = ships_disp,
                    )
                    #cv2.imshow("Water Navigation", vis)
                    #key = cv2.waitKey(1) & 0xFF
                    #if key == ord('q') or key == 27:
                        #print("\n用户按 Q/ESC 退出")
                        #break

                if save_video and async_writer:
                    wm_disp = (cached_water_large  if cached_water_large  is not None
                               else np.zeros((height, width), dtype=np.uint8))
                    bm_disp = (cached_bridge_large if cached_bridge_large is not None
                               else np.zeros((height, width), dtype=np.uint8))

                    vis_full = overlay_masks(
                        frame,
                        water_mask  = wm_disp,
                        bridge_mask = bm_disp,
                        cfg         = cfg,
                        curve       = gnss_curve,
                        frame_idx   = frame_idx,
                        fps_text    = fps_text,
                        draw_curve  = draw_curve_on_python,
                        draw_masks  = draw_masks_on_python,
                        ships       = ships_final,
                    )
                    async_writer.submit(vis_full)

            # ══════════════════════════════════════════════════════════════
            # ★ 步骤 7：帧耗时统计 + 帧率控制
            # ══════════════════════════════════════════════════════════════
            t_cost = time.time() - t_frame
            frame_times.append(t_cost)
            if len(frame_times) > 120:
                frame_times.pop(0)

            if frame_idx % 100 == 0 and frame_idx > 0:
                avg_ms   = np.mean(frame_times) * 1000
                real_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                gst_text = (
                    f"gst解码 {np.mean(gst_decode_times) * 1000:.1f}ms | "
                    if gst_decode_times else ""
                )
                print(
                    f"  帧 {frame_idx:5d}/{total_frames} | "
                    f"avg {avg_ms:.1f}ms ({real_fps:.1f}fps) | "
                    f"{gst_text}"
                    f"ships={len(ships_final)}"
                )
            # 本地文件由 FramePacer 统一限帧，RTSP 摄像头由实时源控制帧率。

    except KeyboardInterrupt:
        print("\n用户中断 (Ctrl+C)")
    finally:
        print("\n正在释放资源...")
        cap.release()
        if async_seg:
            async_seg.shutdown()
        if shm_video_sender:
            shm_video_sender.stop()
        if nav_sender:
            nav_sender.shutdown()
        mqtt_navigation.stop()
        navigation_state.stop()
        if yolo_detector:
            yolo_detector.shutdown()
        if yolo_sender:
            yolo_sender.stop()
        if fusion_receiver:
            fusion_receiver.stop()
        if inferencer:
            inferencer.shutdown()
        if show_window:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\n处理完成:")
        print(f"  总帧数  : {frame_idx}")
        print(f"  总耗时  : {elapsed:.1f}s")
        print(f"  平均FPS : {avg_fps:.1f}")
        if frame_times:
            print(f"  帧耗时  : avg={np.mean(frame_times)*1000:.1f}ms "
                  f"min={np.min(frame_times)*1000:.1f}ms "
                  f"max={np.max(frame_times)*1000:.1f}ms")
