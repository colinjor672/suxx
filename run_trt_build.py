import os
from trt_builder import build_trt_engine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

onnx_path = os.path.join(BASE_DIR, "models", "best1_fp16.onnx")
engine_path = os.path.join(BASE_DIR, "models", "seg_best.engine")

SEG_SHAPE = (1, 3, 432, 768)

build_trt_engine(
    onnx_path="./best1_fp16.onnx",
    engine_path="./best_fp16.engine",
    input_shape=SEG_SHAPE,
)

print("\n✅ 分割 TensorRT Engine 转换完成！")
print(f"   ONNX   : {onnx_path}")
print(f"   Engine : {engine_path}")
print(f"   输入固定: {SEG_SHAPE}")