#!/usr/bin/env bash
# Isolated RTMPose overlay using Titan's existing PyTorch 2.7/CUDA 12.8.
# bash setup_rtmpose_torch27.sh --inspect | --build | --repair | --verify
# Never installs into the original Bird_HKE conda environment.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="/home/students/conda-envs/bird_hke-cu128/bin/python"
VENV="/home/students/conda-envs/rtmpose-torch27-venv"
MMPOSE="$ROOT/.external_pose_sources/rtmpose"
MMCV="$ROOT/.external_pose_sources/mmcv-2.2.0"
PIN="759b39c13fea6ba094afc1fa932f51dc1b11cbf9"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"

inspect() {
  "$BASE" - <<'PY'
import torch, torchvision, sys
from torch.utils.cpp_extension import CUDA_HOME
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("CUDA_HOME detected:", CUDA_HOME)
assert torch.__version__ == "2.7.0+cu128"
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (8, 6)
PY
  nvcc --version | tail -n 4
  gcc --version | head -n 1
  g++ --version | head -n 1
  df -h "$ROOT" "$(dirname "$VENV")"
}
verify() {
  "$VENV/bin/python" - <<'PY'
import sys, torch, mmcv, mmengine, mmpose
from mmcv.ops import nms
from mmpose.models.backbones import CSPNeXt
from mmpose.models.heads import RTMCCHead
from pathlib import Path
binary_extensions = list(Path(mmcv.__file__).parent.glob("_ext*.so"))
if not binary_extensions:
    raise RuntimeError(
        "MMCV installed without compiled mmcv._ext extension; "
        "inspect the earlier build log before recompiling.")
print("MMCV binary extension:", binary_extensions[0])
assert sys.prefix != sys.base_prefix
assert torch.__version__ == "2.7.0+cu128"
assert mmcv.__version__ == "2.2.0"
boxes = torch.tensor([[0.,0.,20.,20.],[1.,1.,21.,21.]], device="cuda")
scores = torch.tensor([.9,.8], device="cuda")
_, keep = nms(boxes, scores, .5)
torch.cuda.synchronize()
assert len(keep) == 1
print("RTMPose overlay:", sys.executable)
print("Torch:", torch.__version__, "MMCV:", mmcv.__version__)
print("MMEngine:", mmengine.__version__, "MMPose:", mmpose.__version__)
print("MMCV CUDA operators: PASS")
PY
}
repair() {
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "RTMPose virtualenv does not exist: $VENV" >&2
    exit 1
  fi
  # yapf imports importlib_metadata on Python 3.10. This small repair
  # changes only the RTMPose overlay, never the shared base torch environment.
  "$VENV/bin/python" -m pip install --no-deps     'importlib-metadata==8.7.0' 'zipp==3.23.0'
  verify
}
build() {
  inspect
  if [[ ! -x "$VENV/bin/python" ]]; then
    "$BASE" -m venv --system-site-packages "$VENV"
  fi
  "$VENV/bin/python" - <<'PY'
import sys, torch
assert sys.prefix != sys.base_prefix, "Not a virtualenv"
assert torch.__version__ == "2.7.0+cu128", "Incorrect shared torch version"
print("Using inherited torch from:", torch.__file__)
PY
  mkdir -p "$ROOT/.external_pose_sources"
  if [[ ! -e "$MMPOSE" ]]; then
    git init -q "$MMPOSE"
    git -C "$MMPOSE" remote add origin https://github.com/open-mmlab/mmpose.git
    git -C "$MMPOSE" fetch -q --depth=1 origin "$PIN"
    git -C "$MMPOSE" -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
  fi
  [[ "$(git -C "$MMPOSE" rev-parse HEAD)" == "$PIN" ]] || {
    echo "MMPose checkout does not match pinned commit: $MMPOSE" >&2; exit 1;
  }
  if [[ ! -e "$MMCV" ]]; then
    git clone -q --depth=1 --branch v2.2.0 https://github.com/open-mmlab/mmcv.git "$MMCV"
  fi
  [[ "$(git -C "$MMCV" describe --tags --exact-match)" == "v2.2.0" ]] || {
    echo "MMCV checkout does not match v2.2.0" >&2; exit 1;
  }
  # --no-deps prevents pip from replacing shared torch/CUDA/NumPy.
  "$VENV/bin/python" -m pip install --no-deps \
    ninja psutil 'mmengine==0.10.7' json-tricks 'xtcocotools>=1.12' \
    chumpy munkres rich markdown-it-py mdurl termcolor addict 'yapf==0.40.1'     'importlib-metadata==8.7.0' 'zipp==3.23.0'
  export TORCH_CUDA_ARCH_LIST=8.6
  export MMCV_WITH_OPS=1
  export FORCE_CUDA=1
  export MAX_JOBS="${MAX_JOBS:-2}"
  echo "Building full MMCV CUDA extension locally for exact PyTorch 2.7.0+cu128."
  "$VENV/bin/python" -m pip install --no-deps --no-build-isolation -v "$MMCV"
  "$VENV/bin/python" -m pip install --no-deps --no-build-isolation -e "$MMPOSE"
  verify
}
case "${1:---inspect}" in
  --inspect) inspect; echo "Inspection only; no downloads or installs.";;
  --build) build;;
  --repair) repair;;
  --verify) verify;;
  *) echo "Usage: bash setup_rtmpose_torch27.sh --inspect|--build|--repair|--verify" >&2; exit 2;;
esac
