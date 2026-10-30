"""把训练好的 .pth 权重转成 Jetson 可用的 ONNX / TensorRT engine。

engine 与 GPU 架构 + TensorRT 版本强绑定，不能跨机拷贝，所以分两步：

  # 1) 在训练服务器（有 smp/torch、可能没装 TensorRT）导 ONNX
  python export_trt.py onnx --weights runs/nano_seg1/phase2_best.pth --out seg_432x768.onnx

  # 2) 把 onnx 拷到 Jetson 上建 engine（同时出 fp32 和 fp16 两份）
  python export_trt.py engine --onnx seg_432x768.onnx \
      --out seg.engine --fp16-out seg_fp16.engine

  # 3) 在 Jetson 上校验精度（强烈建议，尤其 fp16）
  python export_trt.py check --onnx seg_432x768.onnx --engine seg_fp16.engine --image one_frame.jpg

  # 4) 比 fp32 / fp16 的单帧延迟，决定用哪份
  python export_trt.py bench --engines seg.engine seg_fp16.engine --image one_frame.jpg

  # 5) 查 fp16 engine 里每层实际精度（确认 FP16 标记真的生效）
  python export_trt.py inspect --engine seg_fp16.engine
"""
import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch

# 必须与训练/create_models.py 完全一致，否则 state_dict 载入会 key 不匹配
NUM_CLASSES = 3
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ════════════════════ stage 1: ONNX ════════════════════
def load_model(weights: str, device: str):
    """延迟 import smp：Jetson 上只跑 engine/check 阶段时不需要装它。"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from create_models import create_model

    model = create_model()
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # 兼容 DDP / 别的封装留下的前缀
        for prefix in ("module.", "model."):
            stripped = {k[len(prefix):]: v for k, v in state.items()
                        if k.startswith(prefix)}
            if stripped:
                model.load_state_dict(stripped, strict=True)
                print(f"已剥离 state_dict 前缀 '{prefix}'")
                break
        else:
            raise RuntimeError(f"权重与模型结构不匹配：{e}")

    model.to(device).eval()
    print(f"✓ 权重载入完成: {weights}  ({sum(p.numel() for p in model.parameters())/1e6:.2f}M)")
    return model


def export_onnx(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)

    dummy = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert tuple(out.shape) == (1, NUM_CLASSES, args.height, args.width), \
        f"前向输出形状异常: {tuple(out.shape)}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["images"],
        output_names=["logits"],
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=None,          # 固定 1×3×H×W：TRT 能挑最优 kernel，和管线尺寸一致
    )
    mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"✅ ONNX 已导出: {args.out}  ({mb:.1f} MB, opset {args.opset}, "
          f"输入 1×3×{args.height}×{args.width})")
    print("   下一步：把该文件拷到 Jetson，跑 `python export_trt.py engine --onnx ...`")


# ════════════════════ stage 2: engine ════════════════════
def _import_trt():
    try:
        import tensorrt as trt
    except ImportError:
        raise SystemExit(
            "未装 tensorrt python 包。Jetson 上它随 JetPack 提供，"
            "若用 venv 需带 --system-site-packages 重建。")
    return trt


def _create_network(builder, trt):
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:          # TRT 10+：explicit batch 已是唯一模式
        flag = 0
    return builder.create_network(flag)


def _set_workspace(config, trt, gib: int):
    limit = gib * (1 << 30)
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, limit)
    else:                           # 旧版 API
        config.max_workspace_size = limit


def _fp16_supported(builder, trt) -> bool:
    try:
        return bool(builder.platform_has_fast_fp16)
    except AttributeError:
        pass
    try:
        return bool(builder.fp16_supported())
    except AttributeError:
        return False


def _first_input_name(network) -> str:
    """跨 TRT 版本取第一个输入的名字：8.x 无 get_tensor，7.x 用 .inputs 列表。"""
    for getter in (lambda: network.get_input(0), lambda: network.inputs[0]):
        try:
            return getter().name
        except (AttributeError, TypeError, IndexError):
            continue
    return "images"


def _first_input_shape(network):
    for getter in (lambda: network.get_input(0), lambda: network.inputs[0]):
        try:
            return tuple(getter().shape)
        except (AttributeError, TypeError, IndexError):
            continue
    return (1, 3, -1, -1)


def build_engine(onnx_path: str, engine_path: str, trt,
                 fp16: bool, height: int, width: int, verbose: bool) -> float:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = _create_network(builder, trt)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX 解析错误: {parser.get_error(i)}")
            raise RuntimeError(f"ONNX 解析失败: {onnx_path}")

    input_name = _first_input_name(network)
    in_shape = _first_input_shape(network)
    out_shape = tuple(network.get_output(0).shape)
    print(f"  网络输入: {input_name} {tuple(in_shape)}  |  输出: {out_shape}")

    config = builder.create_builder_config()
    _set_workspace(config, trt, 2)
    if fp16:
        if not _fp16_supported(builder, trt):
            print("  ⚠ 该平台不支持快速 FP16，仍按 FP16 构建（TRT 会自行退化）")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    shape = (1, 3, height, width)
    profile.set_shape(input_name, shape, shape, shape)   # 固定尺寸
    config.add_optimization_profile(profile)

    print(f"  开始构建 {'FP16' if fp16 else 'FP32'} engine（首次构建可能要几分钟）...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine 构建失败（多为显存/workspace 不足，试调小 workspace）")
    dt = time.time() - t0

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"  ✅ {engine_path}  ({os.path.getsize(engine_path)/1024/1024:.1f} MB, "
          f"耗时 {dt:.1f}s)")
    return dt


def stage_engine(args):
    trt = _import_trt()
    if not args.out and not args.fp16_out:
        raise SystemExit("至少指定 --out 或 --fp16-out")
    if args.out:
        build_engine(args.onnx, args.out, trt, False,
                     args.height, args.width, args.verbose)
    if args.fp16_out:
        build_engine(args.onnx, args.fp16_out, trt, True,
                     args.height, args.width, args.verbose)
    print("\n部署时把 video_pipeline 的 seg_weights 指向 fp16 engine 即可")


# ════════════════════ stage 3: check ════════════════════
class TrtRunner:
    """与 infer/trt_inferencer.py 同一套 API（execute_async_v3），保证校验路径一致。"""

    def __init__(self, engine_path: str):
        trt = _import_trt()
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.buffers: List[Tuple[str, torch.Tensor]] = []
        self.out_tensor = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            t = torch.empty(shape, dtype=torch.from_numpy(
                np.zeros(1, dtype=np_dtype)).dtype, device="cuda")
            self.buffers.append((name, t))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.out_tensor = t

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.buffers[0][1].copy_(x, non_blocking=True)
        for name, t in self.buffers:
            self.ctx.set_tensor_address(name, t.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("execute_async_v3 失败")
        torch.cuda.synchronize()
        return self.out_tensor


def load_frame(path: str, height: int, width: int) -> np.ndarray:
    """复刻 infer/preprocess.py 的预处理：BGR→RGB、resize、/255、ImageNet 归一化。"""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"图片读不到: {path}")
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


def stage_check(args):
    device = "cuda"
    in_buf = torch.empty((1, 3, args.height, args.width), dtype=torch.float32, device=device)

    if args.image:
        hwc = load_frame(args.image, args.height, args.width)
        in_buf[0] = torch.from_numpy(hwc.transpose(2, 0, 1)).to(device)
        print(f"  用真实帧校验: {args.image}")
    else:
        in_buf.normal_()
        print("  ⚠ 未给 --image，用随机张量校验：只能验证数值一致，看不出真实精度")

    model = load_model(args.weights, device) if args.weights else None
    runner = TrtRunner(args.engine)
    with torch.no_grad():
        eng_logits = runner(in_buf).float()
    eng_pred = eng_logits.argmax(1)

    print(f"\nengine 输出: {tuple(eng_logits.shape)} | "
          f"桥梁像素 {(eng_pred == 0).sum().item()} | "
          f"水面像素 {(eng_pred == 1).sum().item()}")

    refs = []
    if model is not None:
        refs.append(("pth(torch)", model(in_buf).float()))
    if args.onnx and has_ort():
        refs.append(("onnx(ort)", run_onnx(args.onnx, in_buf)))

    for name, ref in refs:
        agree = (ref.argmax(1) == eng_pred).float().mean().item()
        maxdiff = (ref - eng_logits).abs().max().item()
        ok = agree >= args.min_agree
        print(f"  vs {name:12s} argmax 一致率={agree*100:.2f}%  logit 最大偏差={maxdiff:.4f} "
              f"{'✅' if ok else '❌ 低于阈值，精度不可信'}")
        if not ok:
            raise SystemExit(1)


def _walk_keys(obj):
    """递归产出 (key, value)，用于在 EngineInspector 的 JSON 里找精度字段。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, v
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_keys(item)


def stage_inspect(args):
    """查 engine 每层实际选中的计算精度，确认 fp16 到底有没有生效。
    不依赖 build 日志措辞（TRT 8 写 HALF、TRT 10 写 FP16，grep 很容易漏）。"""
    import json as _json

    trt = _import_trt()
    logger = trt.Logger(trt.Logger.ERROR)
    with open(args.engine, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

    insp = None
    for maker in (lambda: engine.create_engine_inspector(),
                  lambda: engine.create_execution_context().create_engine_inspector()):
        try:
            insp = maker()
            break
        except Exception:
            continue
    if insp is None:
        raise SystemExit("该 TRT 版本没有 EngineInspector。改用构建日志判断：\n"
                         "  python export_trt.py engine ... --verbose > build.log 2>&1\n"
                         "  grep -icE 'half|fp16' build.log")

    texts = []
    ver_enum = getattr(trt, "EngineInspectVersion", None)
    for attr in ("KERNEL_ANALYTICS", "TENSORS_ENGINE_ATTRIBUTES", "TENSOR_ENGINE_ATTRIBUTES"):
        if ver_enum is not None and hasattr(ver_enum, attr):
            try:
                texts.append(insp.get_engine_json(version=getattr(ver_enum, attr)))
            except Exception:
                pass
    if not texts:
        try:
            texts.append(insp.get_engine_json())
        except Exception as e:
            raise SystemExit(f"EngineInspector 取 JSON 失败: {e}")

    data = _json.loads(max(texts, key=len))
    layers = data.get("Layers") or data.get("layers") or []
    if not layers:
        raise SystemExit("JSON 里没找到 Layers 字段，TRT 版本格式未知。可用 bench 的延迟差判断")

    tally, fp32_layers = {}, []
    for lyr in layers:
        prec = None
        for k, v in _walk_keys(lyr):
            if "PRECISION" in str(k).upper() and isinstance(v, str):
                prec = v.upper()
                break
        if prec is None:                       # 没有显式字段时从文本里认 token
            blob = _json.dumps(lyr).upper()
            for tok in ("INT8", "HALF", "FP16", "BF16", "FLOAT", "FP32"):
                if tok in blob:
                    prec = tok
                    break
        prec = prec or "UNKNOWN"
        tally[prec] = tally.get(prec, 0) + 1
        if prec in ("FLOAT", "FP32"):
            fp32_layers.append(str(lyr.get("Name") or lyr.get("name") or "?"))

    total = len(layers)
    print(f"engine: {args.engine}  共 {total} 层")
    for prec, cnt in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {prec:8s} {cnt:4d} 层  ({cnt / max(1, total) * 100:.0f}%)")

    half = tally.get("HALF", 0) + tally.get("FP16", 0)
    if half == 0:
        print("⚠ 一层都没跑在 fp16：这份 engine 的 FP16 标记没起作用，速度和 fp32 会基本一致")
    elif half / max(1, total) < 0.5:
        print(f"⚠ 只有 {half}/{total} 层走 fp16，加速有限（SE/hardswish 层常回退 fp32）")
    else:
        print(f"✅ {half}/{total} 层走 fp16，构建生效")
    if fp32_layers and args.list_fp32:
        print("仍为 fp32 的层（前 15）: " + ", ".join(fp32_layers[:15]))


def stage_bench(args):
    """量化 engine 的单帧延迟，用来决定 fp16 值不值得用。"""
    if args.image:
        hwc = load_frame(args.image, args.height, args.width)
        x = torch.from_numpy(hwc.transpose(2, 0, 1)).unsqueeze(0).to("cuda")
    else:
        x = torch.randn((1, 3, args.height, args.width), device="cuda")

    for path in args.engines:
        runner = TrtRunner(path)
        for _ in range(args.warmup):          # 预热：cudnn/kernel 首次加载不算数
            runner(x)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            runner(x)
        end.record()
        torch.cuda.synchronize()

        ms = start.elapsed_time(end) / args.iters
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  {os.path.basename(path):24s} {ms:6.2f} ms/帧  "
              f"≈{1000/max(ms,1e-6):5.1f} fps  峰值显存 {mem:.0f} MiB")
        del runner
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def has_ort() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def run_onnx(onnx_path: str, x: torch.Tensor) -> torch.Tensor:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: x.cpu().numpy()})[0]
    return torch.from_numpy(out).to(x.device)


# ════════════════════ CLI ════════════════════
def add_size_args(p):
    p.add_argument("--height", type=int, default=432, help="必须等于管线的 MASK_H")
    p.add_argument("--width", type=int, default=768, help="必须等于管线的 MASK_W")


def main():
    ap = argparse.ArgumentParser(description="pth → onnx → engine / fp16.engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("onnx", help="pth → onnx（在训练服务器跑）")
    p1.add_argument("--weights", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--opset", type=int, default=12)
    add_size_args(p1)
    p1.set_defaults(func=export_onnx)

    p2 = sub.add_parser("engine", help="onnx → engine（必须在 Jetson 上跑）")
    p2.add_argument("--onnx", required=True)
    p2.add_argument("--out", default=None, help="fp32 engine 输出路径，可不填")
    p2.add_argument("--fp16-out", default=None, help="fp16 engine 输出路径")
    p2.add_argument("--verbose", action="store_true")
    add_size_args(p2)
    p2.set_defaults(func=stage_engine)

    p3 = sub.add_parser("check", help="engine 与 pth/onnx 精度一致性校验")
    p3.add_argument("--engine", required=True)
    p3.add_argument("--weights", default=None)
    p3.add_argument("--onnx", default=None)
    p3.add_argument("--image", default=None)
    p3.add_argument("--min-agree", type=float, default=0.98,
                    help="argmax 一致率下限；fp16 通常 99%+，低于此值判不通过")
    add_size_args(p3)
    p3.set_defaults(func=stage_check)

    p4 = sub.add_parser("bench", help="测 engine 单帧延迟，判断 fp16 是否划算")
    p4.add_argument("--engines", nargs="+", required=True,
                    help="可一次给多个，逐个测：bench --engines seg.engine seg_fp16.engine")
    p4.add_argument("--image", default=None)
    p4.add_argument("--warmup", type=int, default=20)
    p4.add_argument("--iters", type=int, default=200)
    add_size_args(p4)
    p4.set_defaults(func=stage_bench)

    p5 = sub.add_parser("inspect", help="查 engine 每层实际精度（fp16 是否真生效）")
    p5.add_argument("--engine", required=True)
    p5.add_argument("--list-fp32", action="store_true", help="列出仍为 fp32 的层名")
    p5.set_defaults(func=stage_inspect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
