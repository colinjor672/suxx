from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ======================
# 模型路径
# ======================

SHIP_ENGINE_PATH = (
    r"./ship_best1.engine"
)

SHIP_CONF = 0.45
SHIP_IOU = 0.45
SHIP_CLASS_NAMES = ["ship"]
SHIP_INPUT_SIZE = 960
SHIP_CLASS_IDS = [0]


# ======================
# 分割类别
# ======================

NUM_CLASSES = 3
BRIDGE_CLASS_ID = 0
WATER_CLASS_ID = 1
BACKGROUND_CLASS_ID = 2

NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


# ======================
# 渲染配置
# ======================

@dataclass
class RenderConfig:
    water_color: Tuple[int, int, int] = (255, 220, 120)
    bridge_color: Tuple[int, int, int] = (0, 0, 255)
    ar_near_color: Tuple[int, int, int] = (255, 230, 80)
    ar_far_color: Tuple[int, int, int] = (255, 120, 0)
    water_alpha: float = 0.14
    bridge_alpha: float = 0.25
