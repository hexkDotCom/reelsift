"""Device selection and freeing GPU memory between stages."""

import gc
import os
import sys
from pathlib import Path


def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


_dll_dirs_added = False


def add_cuda12_dlls() -> None:
    """CTranslate2 (faster-whisper) is built for CUDA 12; on Windows its cuBLAS and cuDNN come
    from NVIDIA's pip wheels, whose folders aren't on the DLL search path by default."""
    global _dll_dirs_added
    if _dll_dirs_added or sys.platform != "win32":
        return
    for lib in ("cublas", "cudnn"):
        bin_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / lib / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    _dll_dirs_added = True
