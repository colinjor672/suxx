import torch


class TemporalMaskSmootherGPU:
    def __init__(self, alpha=0.7, threshold=0.5):
        self.alpha = float(alpha)
        self.threshold = float(threshold)
        self._ema = None

    def update(self, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=torch.float32)

        if self._ema is None or self._ema.shape != m.shape:
            self._ema = m.clone()
        else:
            self._ema.mul_(self.alpha).add_(m, alpha=(1.0 - self.alpha))

        if mask.dtype == torch.uint8:
            return (self._ema >= self.threshold).to(dtype=torch.uint8)
        elif mask.dtype == torch.int32:
            return self._ema.round().to(dtype=torch.int32)
        else:
            return (self._ema >= self.threshold).to(dtype=mask.dtype)
