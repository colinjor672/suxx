from typing import List, Tuple

import numpy as np
import torch
import tensorrt as trt


class TRTInferencer:
    def __init__(self, engine_path: str, input_h: int = 432, input_w: int = 768):
        self._logger = trt.Logger(trt.Logger.WARNING)
        self.input_h = input_h
        self.input_w = input_w

        print(f"加载 TRT 分割模型: {engine_path} ({input_h}×{input_w})")
        self._engine = self._load(engine_path)
        self._ctx = self._engine.create_execution_context()
        self._alloc()
        print("✓ TRTInferencer 初始化完成")

    def _load(self, path: str):
        runtime = trt.Runtime(self._logger)
        with open(path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _alloc(self):
        self._bufs: List[Tuple[str, torch.Tensor]] = []
        self._out_tensors: List[torch.Tensor] = []

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))

            tdtype = torch.float16 if dtype == np.float16 else torch.float32
            tensor = torch.empty(shape, dtype=tdtype, device="cuda")

            self._bufs.append((name, tensor))

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self._out_tensors.append(tensor)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        stream = torch.cuda.current_stream()

        in_name, in_buf = self._bufs[0]
        in_buf.copy_(x, non_blocking=True)

        for name, tensor in self._bufs:
            self._ctx.set_tensor_address(name, tensor.data_ptr())

        ok = self._ctx.execute_async_v3(stream.cuda_stream)

        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 执行失败")

        return self._out_tensors[0]

    def shutdown(self):
        pass