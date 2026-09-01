"""Report the local Jetson inference environment without changing it."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def version_of(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # Environment diagnostics must keep going.
        return f"unavailable ({type(error).__name__}: {error})"
    return str(getattr(module, "__version__", "installed (version unavailable)"))


def main() -> int:
    print("Jetson inference environment")
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {cuda_available}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0) if cuda_available else 'unavailable'}")
    except Exception as error:
        print(f"PyTorch: unavailable ({type(error).__name__}: {error})")

    print(f"Ultralytics: {version_of('ultralytics')}")
    print(f"OpenCV: {version_of('cv2')}")
    print(f"TensorRT: {version_of('tensorrt')}")
    print("Camera device nodes:")
    for index in range(4):
        device = Path(f"/dev/video{index}")
        print(f"  {device}: {'present' if device.exists() else 'missing'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
