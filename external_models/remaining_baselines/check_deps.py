#!/usr/bin/env python3
"""Read-only dependency check for isolated ViTPose/PoseBH interpreters.

Usage from repo root:
  /path/to/vitpose-venv/bin/python -m external_models.remaining_baselines.check_deps
  /path/to/posebh-venv/bin/python -m external_models.remaining_baselines.check_deps
Do not install packages in an unrelated training environment.
"""
from __future__ import annotations

import importlib
import shutil
import sys


def main() -> int:
    print("Python:", sys.version.split()[0])
    print("Interpreter:", sys.executable)
    print("Environment prefix:", sys.prefix)
    print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "not set"))
    problems = []
    if sys.version_info < (3, 10):
        problems.append("Python >=3.10 is required")
    for name, label in (
        ("torch", "PyTorch"), ("numpy", "NumPy"),
        ("cv2", "OpenCV"), ("PIL", "Pillow"),
    ):
        try:
            package = importlib.import_module(name)
            print(f"{label}: {getattr(package, '__version__', 'installed')}")
        except (ImportError, OSError) as exc:
            problems.append(f"{label} unavailable: {exc}")
    if shutil.which("git") is None:
        problems.append("Git not found in PATH")
    if shutil.which("nvidia-smi") is None:
        problems.append("nvidia-smi not found in PATH")
    try:
        import torch
        print("Torch CUDA build:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            print("Visible GPU count:", count)
            if count == 1:
                print("GPU:", torch.cuda.get_device_name(0))
            else:
                problems.append("Exactly one GPU must be visible for these training launchers")
        else:
            problems.append("No usable CUDA GPU detected by PyTorch")
        if not hasattr(torch, "amp") or not hasattr(torch.amp, "GradScaler"):
            problems.append("PyTorch is too old for torch.amp.GradScaler('cuda')")
    except (ImportError, RuntimeError, OSError) as exc:
        problems.append(f"Cannot inspect PyTorch CUDA: {exc}")
    if problems:
        print("\nDEPENDENCY CHECK FAILED:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("\nDEPENDENCIES OK. This does not validate full model training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
