#!/usr/bin/env bash
# PoseBH-B ONLY. Does not touch existing Bird_HKE, RTMPose or ViTPose envs.
# From repository root: bash setup_posebh_b.sh --inspect|--sources|--create|--verify
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIN="8f03438f913b559edb0a2beed54e77d1560edba3"
URL="https://github.com/uyoung-jeong/PoseBH.git"
UPSTREAM="$ROOT/.external_pose_sources/posebh"
PREFIX="${POSEBH_ENV_PREFIX:-/home/students/conda-envs/posebh-b-cu128-venv}"
BASE_PY="${POSEBH_BASE_PY:-python}"

case "${1:-}" in
  --inspect)
    echo "PoseBH-B env target: $PREFIX"
    echo "Base Python: $BASE_PY"
    "$BASE_PY" - <<'PY'
import sys
print("Base interpreter:", sys.executable)
for name in ("torch", "numpy", "cv2"):
    module = __import__(name)
    print(name, getattr(module, "__version__", "(unknown)"))
import torch
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
    command -v nvidia-smi && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    echo "Inspect only: no packages/files changed."
    ;;
  --sources)
    mkdir -p "$ROOT/.external_pose_sources"
    if [[ -e "$UPSTREAM" && ! -d "$UPSTREAM/.git" ]]; then
      echo "Refusing non-Git upstream path: $UPSTREAM" >&2
      exit 1
    fi
    if [[ ! -d "$UPSTREAM/.git" ]]; then
      git clone "$URL" "$UPSTREAM"
    fi
    if [[ -n "$(git -C "$UPSTREAM" status --porcelain)" ]]; then
      echo "Refusing dirty upstream checkout: $UPSTREAM" >&2
      exit 1
    fi
    if ! git -C "$UPSTREAM" cat-file -e "$PIN^{commit}" 2>/dev/null; then
      git -C "$UPSTREAM" fetch origin "$PIN"
    fi
    git -C "$UPSTREAM" -c advice.detachedHead=false checkout --quiet --detach "$PIN"
    [[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$PIN" ]]
    echo "Pinned standard PoseBH source at $PIN; other source checkouts untouched."
    ;;
  --create)
    if [[ -e "$PREFIX" ]]; then
      echo "Environment path exists; refusing to modify/reinstall: $PREFIX" >&2
      exit 1
    fi
    mkdir -p "$(dirname "$PREFIX")"
    # Share *read-only* installed torch/CUDA wheels from base, but create
    # separately writable site-packages for future PoseBH-specific packages.
    "$BASE_PY" -m venv --system-site-packages "$PREFIX"
    "$PREFIX/bin/python" - <<'PY'
import sys
print("Isolated environment:", sys.executable)
print("Environment prefix:", sys.prefix)
print("Now check dependencies and CUDA on the target GPU node.")
PY
    echo "Only the isolated environment was created; no pip installation performed."
    ;;
  --verify)
    [[ -x "$PREFIX/bin/python" ]] || { echo "Missing isolated venv" >&2; exit 1; }
    [[ "$(git -C "$UPSTREAM" rev-parse HEAD)" == "$PIN" ]] ||
       { echo "Pinned PoseBH upstream missing/wrong revision" >&2; exit 1; }
    "$PREFIX/bin/python" - <<'PY'
import sys
import torch
import cv2
import numpy
assert "posebh" in sys.prefix.lower()
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, "Use CUDA_VISIBLE_DEVICES to expose one GPU"
print("Verified isolated interpreter:", sys.executable)
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "cv2:", cv2.__version__, "numpy:", numpy.__version__)
PY
    ;;
  *)
    echo "Usage: bash comparison_models/scripts/setup_posebh_b.sh --inspect|--sources|--create|--verify" >&2
    exit 2
    ;;
esac
