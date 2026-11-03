#!/usr/bin/env bash
# Isolated RTMPose-M / MMPose 1.x environment for Ampere/Hopper NVIDIA GPUs.
# Run with bash, not source: failures never exit your interactive terminal.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${RTMPOSE_ENV_PREFIX:-/home/students/conda-envs/rtmpose-cu121}"
UPSTREAM="${ROOT}/.external_pose_sources/rtmpose"
PIN="759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
MMCV_WHEEL="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl"

verify() {
  "${PREFIX}/bin/python" - <<'PY'
import sys
import numpy
import torch
import torchvision
import mmcv
import mmengine
import mmdet
import mmpose
from mmcv.ops import nms
from mmdet.models.backbones import CSPNeXt
from mmpose.models.heads import RTMCCHead
print("Python:", sys.version.split()[0], sys.executable)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("MMCV:", mmcv.__version__, "MMEngine:", mmengine.__version__)
print("MMDetection:", mmdet.__version__, "MMPose:", mmpose.__version__)
print("NumPy:", numpy.__version__, "MMCV CUDA operations: import PASS")
assert torch.__version__.split("+")[0] == "2.1.0", "Expected torch 2.1.0"
assert torch.version.cuda == "12.1", "Expected PyTorch CUDA 12.1 wheel"
assert mmcv.__version__ == "2.1.0", "Expected mmcv 2.1.0"
assert mmdet.__version__ == "3.2.0", "Expected mmdet 3.2.0"
assert torch.cuda.is_available(), "NVIDIA GPU unavailable"
for i in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}, compute {major}.{minor}")
    assert major < 10, ("Torch 2.1 cannot execute Blackwell kernels; "
                        "use an Ampere/Hopper GPU for this environment")
print("RTMPose isolated environment: PASS")
PY
  "${PREFIX}/bin/python" -m pip check
}

if [[ "${1:-}" == "--verify" ]]; then
  verify
  exit 0
fi
if [[ "$#" != 0 ]]; then
  echo "Usage: bash setup_rtmpose_env.sh [--verify]" >&2
  exit 2
fi

if [[ ! -f "${UPSTREAM}/tools/train.py" ]]; then
  echo "Missing pinned MMPose checkout: bash setup_external_pose_sources.sh" >&2
  exit 1
fi
ACTUAL="$(git -C "${UPSTREAM}" rev-parse HEAD)"
if [[ "${ACTUAL}" != "${PIN}" ]]; then
  echo "Wrong MMPose revision: expected ${PIN}; got ${ACTUAL}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found on PATH" >&2
  exit 1
fi

# Check compute capability before downloading. This fixed binary wheel
# targets CUDA 12.1 / PyTorch 2.1 and cannot run Blackwell devices.
if command -v nvidia-smi >/dev/null 2>&1; then
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  if [[ -n "${caps}" ]]; then
    while IFS= read -r cap; do
      major="${cap%%.*}"
      if [[ "${major}" =~ ^[0-9]+$ ]] && (( major >= 10 )); then
        echo "GPU compute capability ${cap} needs a newer PyTorch/MMCV build." >&2
        echo "This CUDA 12.1 / PyTorch 2.1 recipe is for Ampere/Hopper." >&2
        echo "No packages were modified." >&2
        exit 1
      fi
    done <<< "${caps}"
  fi
fi

if [[ ! -x "${PREFIX}/bin/python" ]]; then
  echo "Creating isolated environment at: ${PREFIX}"
  conda create --prefix "${PREFIX}" python=3.10 -y
fi
PY="${PREFIX}/bin/python"

"${PY}" -m pip install --upgrade "pip<25" "setuptools<75" wheel
"${PY}" -m pip install "numpy==1.26.4" "opencv-python==4.8.1.78"
"${PY}" -m pip install \
  "torch==2.1.0+cu121" "torchvision==0.16.0+cu121" \
  --index-url https://download.pytorch.org/whl/cu121
# Install the known *binary* CUDA-ops MMCV wheel; no source compilation.
"${PY}" -m pip install "${MMCV_WHEEL}"
"${PY}" -m pip install "mmengine==0.10.7" "mmdet==3.2.0" \
  "pycocotools" "xtcocotools" "json-tricks" "scipy<1.14" \
  "pyyaml" "pillow" "timm==0.9.16"
# Pinned MMPose project checkout, installed only in the isolated env.
"${PY}" -m pip install --no-build-isolation -e "${UPSTREAM}"

verify
echo
echo "Environment: ${PREFIX}"
echo "Train FD with: ${PREFIX}/bin/python -m external_models.rtmpose.run_birdgaze --scenario FD"
