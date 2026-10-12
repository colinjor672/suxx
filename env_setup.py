import os
import sys


TRT_LIB = r"D:/BaiduNetdiskDownload/TensorRT-8.6.1.6/lib"
TORCH_LIB = r"C:/Users/admin/miniconda4/envs/yolo11/Lib/site-packages/torch/lib"
CUDNN_LIB = r"C:/Users/admin/miniconda4/envs/yolo11/Library/bin"


def add_dll_path(path: str):
    if not path:
        return

    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")

    try:
        os.add_dll_directory(path)
    except Exception:
        pass


def setup_environment():
    add_dll_path(TRT_LIB)
    add_dll_path(TORCH_LIB)
    add_dll_path(CUDNN_LIB)

    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_onnxruntime_cuda():
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            add_dll_path(torch_lib)
            print(f"✓ 已注入 PyTorch CUDA/cuDNN 库: {torch_lib}")
            return torch_lib

    except Exception as e:
        print(f"⚠️ 注入 CUDA 路径失败: {e}")
        

    return None
def _cleanup(signum=None, frame=None):
    import traceback
    print(f"\n[main] 触发清理，信号={signum}")
    traceback.print_stack()
    stop_godot()
    stop_mediamtx()
    sys.exit(0)