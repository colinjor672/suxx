import tensorrt as trt
import numpy as np


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 2,
    input_shape: tuple = (1, 3, 384, 640),
):
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"ONNX解析失败: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_gb * (1 << 30)
    )

    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # ⚠️ 关键：动态 shape profile
    profile = builder.create_optimization_profile()

    in_name = network.get_input(0).name
    print(f"[TRT] input name = {in_name}")

    profile.set_shape(in_name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)

    print(f"🚀 构建 TensorRT Engine: {onnx_path}")

    serialized = builder.build_serialized_network(network, config)

    if serialized is None:
        raise RuntimeError("Engine 构建失败（检查 ONNX / FP16 / shape）")

    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))

    size_mb = serialized.nbytes / (1024 * 1024)
    print(f"✅ 完成: {engine_path} ({size_mb:.1f} MB)")