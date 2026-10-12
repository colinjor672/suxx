import torch
import torch.nn.functional as F
import numpy as np


class PreprocessGPU:
    def __init__(self, target_h=384, target_w=640):
        self.target_h = target_h
        self.target_w = target_w

        self.buf = torch.empty(
            (1, 3, target_h, target_w),
            dtype=torch.float16,
            device="cuda",
        )

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float16,
            device="cuda",
        ).view(1, 3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float16,
            device="cuda",
        ).view(1, 3, 1, 1)

    def __call__(self, frame_bgr) -> torch.Tensor:
        # ★ 接受 np.ndarray 或已上传的 GPU tensor，避免重复上传
        # frame_bgr: [H, W, 3] uint8 (BGR)
        if isinstance(frame_bgr, torch.Tensor):
            # 已上传到 GPU，直接复用
            t = frame_bgr
        else:
            # torch.from_numpy 要求 C 连续，GStreamer 回调的 bgr_view 是非连续的
            if not frame_bgr.flags['C_CONTIGUOUS']:
                frame_bgr = np.ascontiguousarray(frame_bgr)
            t = torch.from_numpy(frame_bgr).to(
                device="cuda",
                non_blocking=True,
            )  # [H, W, 3] uint8

        # HWC → CHW
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W] uint8

        # uint8 → float16
        t = t.to(torch.float16)

        # GPU resize (bilinear)
        t = F.interpolate(
            t,
            size=(self.target_h, self.target_w),
            mode='bilinear',
            align_corners=False,
        )

        # BGR → RGB (在 CHW 上 flip dim=1)
        t = t.flip(1)

        # 归一化
        t.mul_(1.0 / 255.0)
        t.sub_(self.mean).div_(self.std)

        self.buf.copy_(t)
        return self.buf
