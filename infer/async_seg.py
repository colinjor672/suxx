import threading
from typing import Optional, Tuple

import numpy as np
import torch
import sys
sys.path.append("/home/jetson/code/Visualization")
print(sys.path)
from config import WATER_CLASS_ID, BRIDGE_CLASS_ID
from infer.mask_refine import (
    refine_water_mask,
    refine_bridge_mask,
)
from utils.smoother import TemporalMaskSmootherGPU
import torch.nn.functional as F
from utils.thread_utils import set_thread_name


def get_raw_masks(frame_bgr, preprocess, inferencer) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回 (water_gpu, bridge_gpu, frame_gray_gpu)

    frame_gray_gpu: [H, W] float32 [0,1] 灰度图（用于夜晚亮度过滤）
    """
    # ★ 一次性上传到 GPU，preprocess 和 frame_gray 复用同一 tensor，避免重复上传
    if isinstance(frame_bgr, torch.Tensor):
        frame_t = frame_bgr
    else:
        if not frame_bgr.flags['C_CONTIGUOUS']:
            frame_bgr = np.ascontiguousarray(frame_bgr)
        frame_t = torch.from_numpy(frame_bgr).to(device='cuda', non_blocking=True)

    # 用已上传的 tensor 计算灰度图（BGR → Gray: 0.114*B + 0.587*G + 0.299*R）
    frame_gray_gpu = (
        0.114 * frame_t[:, :, 0].float() +
        0.587 * frame_t[:, :, 1].float() +
        0.299 * frame_t[:, :, 2].float()
    ) / 255.0  # [H, W] float32 [0, 1]

    # preprocess 直接复用已上传的 tensor
    x = preprocess(frame_t)
    logits = inferencer.infer(x)

    if logits.dim() == 3:
        logits = logits.unsqueeze(0)

    pred = torch.argmax(logits.float(), dim=1)[0]

    water_gpu = (pred == WATER_CLASS_ID).to(torch.uint8)
    bridge_gpu = (pred == BRIDGE_CLASS_ID).to(torch.uint8)

    # resize 灰度图到 mask 尺寸
    h, w = pred.shape
    if frame_gray_gpu.shape[0] != h or frame_gray_gpu.shape[1] != w:
        frame_gray_gpu = F.interpolate(
            frame_gray_gpu.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).squeeze(0)

    return water_gpu, bridge_gpu, frame_gray_gpu


class AsyncSegInferencer:
    def __init__(self, inferencer, preprocess, mask_h: int = 384, mask_w: int = 640,
                 send_h: int = 384, send_w: int = 640):
        self.inferencer = inferencer
        self.preprocess = preprocess
        self.mask_h = mask_h
        self.mask_w = mask_w
        self.send_h = send_h
        self.send_w = send_w

        self._water_smoother = TemporalMaskSmootherGPU(alpha=0.55, threshold=0.40)
        # ★ 降低 bridge alpha（0.85→0.65）：减少历史权重，加快误识别消除
        # ★ 提高 threshold（0.40→0.50）：需要更高置信度才保留桥梁，减少误触发
        self._bridge_smoother = TemporalMaskSmootherGPU(alpha=0.65, threshold=0.50)
        self._input_lock = threading.Lock()
        self._input_frame = None
        self._input_frame_id = -1
        self._new_frame = threading.Event()

        self._output_lock = threading.Lock()
        self._water_np: Optional[np.ndarray] = None
        self._bridge_np: Optional[np.ndarray] = None
        self._output_frame_id = -1

        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="AsyncSeg",
        )
        self._thread.start()

        print("[AsyncSeg] ✓ 分割推理后台线程已启动")

    def submit(self, frame_bgr: np.ndarray, frame_id: int):
        with self._input_lock:
            self._input_frame = frame_bgr
            self._input_frame_id = frame_id

        self._new_frame.set()

    def get_result(self):
        with self._output_lock:
            if self._water_np is None:
                return None

            return (
                self._water_np,
                self._bridge_np,
                self._output_frame_id,
            )

    def shutdown(self):
        self._running = False
        self._new_frame.set()
        self._thread.join(timeout=3)

    def _loop(self):
        set_thread_name("AsyncSeg")
        self._stream = torch.cuda.Stream()

        while self._running:
            self._new_frame.wait(timeout=0.1)
            self._new_frame.clear()

            if not self._running:
                break

            with self._input_lock:
                frame = self._input_frame
                fid = self._input_frame_id
                self._input_frame = None

            if frame is None:
                continue

            try:
                with torch.cuda.stream(self._stream):
                    water_gpu, bridge_gpu, frame_gray_gpu = get_raw_masks(
                        frame,
                        self.preprocess,
                        self.inferencer,
                    )

                    water_gpu = self._water_smoother.update(water_gpu)
                    bridge_gpu = self._bridge_smoother.update(bridge_gpu)

                    water_gpu = refine_water_mask(water_gpu)

                    bridge_gpu = refine_bridge_mask(
                        bridge_gpu,
                        water_gpu,
                        self.mask_h,
                        self.mask_w,
                        margin_ratio=0.75,
                        max_height_ratio=0.45,
                        min_aspect_ratio=1.2,
                        max_area_ratio=0.4,
                        frame_gray_gpu=frame_gray_gpu,  # ★ 传入灰度图用于夜晚过滤
                        night_brightness_threshold=0.25,
                        bridge_min_brightness_ratio=1.15,
                    )

                    # ★ GPU 上直接 resize 到发送尺寸，避免下游 cuda_resize 的 GPU↔CPU 往返
                    if self.send_h != self.mask_h or self.send_w != self.mask_w:
                        water_send = F.interpolate(
                            water_gpu.unsqueeze(0).unsqueeze(0).float(),
                            size=(self.send_h, self.send_w),
                            mode='nearest',
                        ).squeeze(0).squeeze(0).to(torch.uint8)
                        bridge_send = F.interpolate(
                            bridge_gpu.unsqueeze(0).unsqueeze(0).float(),
                            size=(self.send_h, self.send_w),
                            mode='nearest',
                        ).squeeze(0).squeeze(0).to(torch.uint8)
                    else:
                        water_send = water_gpu
                        bridge_send = bridge_gpu

                self._stream.synchronize()

                # 只下载两张发送尺寸的 mask；导航曲线由 GNSS 投映模块生成。
                masks_gpu = torch.stack(
                    [water_send, bridge_send],
                    dim=0,
                ).contiguous()

                masks_np = masks_gpu.cpu().numpy()
                w_np = masks_np[0]
                b_np = masks_np[1]

                with self._output_lock:
                    self._water_np = w_np
                    self._bridge_np = b_np
                    self._output_frame_id = fid

            except Exception as e:
                print(f"[AsyncSeg] 推理异常: {e}")
