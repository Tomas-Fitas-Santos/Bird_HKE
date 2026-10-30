#!/usr/bin/env bash
# Install Bird_HKE's pinned Linux/CUDA training dependencies in a new Conda prefix.
# Run from the repository root. Existing environments and training files are not changed.
set -Eeuo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

mode="${1:-install}"
case "$mode" in
    install|--verify|--verify-gpu) ;;
    *) die "Usage: bash setup_linux_training_env.sh [--verify | --verify-gpu]" ;;
esac
[[ $# -le 1 ]] || die 'Only one mode argument is accepted.'

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
    die 'The pinned CUDA wheels require Linux x86_64.'

repo_root="${BIRD_HKE_REPO:-$PWD}"
[[ -f "$repo_root/requirements.txt" && -f "$repo_root/Bird_HKE/tools/train.py" ]] ||
    die 'Run from the Bird_HKE repository root, or set BIRD_HKE_REPO to it.'

env_prefix="${BIRD_HKE_ENV_PREFIX:-$HOME/conda-envs/bird_hke-cu128}"
[[ "$env_prefix" = /* ]] || die 'BIRD_HKE_ENV_PREFIX must be an absolute path.'
[[ "$env_prefix" != / && "$env_prefix" != "$HOME" ]] || die 'Choose a dedicated environment directory.'

export PYTHONNOUSERSITE=1
if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    conda_bin="$CONDA_EXE"
elif command -v conda >/dev/null 2>&1; then
    conda_bin=conda
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    conda_bin="$HOME/miniconda3/bin/conda"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    conda_bin="$HOME/miniforge3/bin/conda"
else
    die 'Conda was not found. Install Miniconda/Miniforge under your account, or set CONDA_EXE.'
fi

env_python() { "$env_prefix/bin/python" "$@"; }

verify_packages() {
    env_python - "$repo_root" <<'PY'
import importlib.metadata as metadata
import pathlib
import sys

import cv2
import numpy
import torch
import torchvision
import yacs
import causal_conv1d_cuda
import selective_scan_cuda
from mamba_ssm import Mamba

root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "MambaVision"))
from mambavision.models.mamba_vision import MambaVision

assert torch.__version__ == "2.7.0+cu128", torch.__version__
assert torchvision.__version__ == "0.22.0+cu128", torchvision.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert metadata.version("mamba-ssm").startswith("2.2.6.post3"), metadata.version("mamba-ssm")
assert metadata.version("causal-conv1d").startswith("1.5.3.post1"), metadata.version("causal-conv1d")
print("Package/import verification: PASS")
print("Python:", sys.executable)
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Mamba:", metadata.version("mamba-ssm"), "Causal Conv1d:", metadata.version("causal-conv1d"))
PY
    env_python -m pip check
}

verify_gpu() {
    env_python - <<'PY'
import torch
from mamba_ssm import Mamba

assert torch.cuda.is_available(), "No CUDA GPU is visible. Run this check in a GPU allocation."
model = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(1, 8, 16, device="cuda", requires_grad=True)
model(x).square().mean().backward()
torch.cuda.synchronize()
assert x.grad is not None and torch.isfinite(x.grad).all()
print("Mamba CUDA forward/backward: PASS on", torch.cuda.get_device_name(0))
PY
}

if [[ "$mode" == --verify || "$mode" == --verify-gpu ]]; then
    [[ -x "$env_prefix/bin/python" ]] || die "Environment not found: $env_prefix"
    verify_packages
    if [[ "$mode" == --verify-gpu ]]; then verify_gpu; fi
    exit 0
fi

if [[ -e "$env_prefix" ]]; then
    [[ -f "$env_prefix/.bird_hke_install_in_progress" ]] ||
        die "Environment already exists: $env_prefix. Choose a new BIRD_HKE_ENV_PREFIX."
    printf 'Resuming an interrupted Bird_HKE installation in %s\n' "$env_prefix"
else
    mkdir -p "$(dirname "$env_prefix")"
    "$conda_bin" create --yes --prefix "$env_prefix" python=3.10 pip
    touch "$env_prefix/.bird_hke_install_in_progress"
fi

printf 'Installing PyTorch 2.7.0 with CUDA 12.8 in %s\n' "$env_prefix"
env_python -m pip install 'torch==2.7.0+cu128' 'torchvision==0.22.0+cu128' \
    'torchaudio==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128

# requirements.txt pins the rest of the project and the same PyTorch wheels.
env_python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$repo_root/requirements.txt"

# These releases provide matching CUDA 12 / torch 2.7 / CPython 3.10 wheels.
# Match the PyTorch C++ ABI; selecting the wrong ABI fails at import/runtime.
abi="$(env_python -c \
    'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
abi="${abi//$'\r'/}"
abi="${abi//$'\n'/}"
[[ "$abi" == TRUE || "$abi" == FALSE ]] || die "Could not determine PyTorch C++ ABI: $abi"

conv_name="causal_conv1d-1.5.3.post1+cu12torch2.7cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"
mamba_name="mamba_ssm-2.2.6.post3+cu12torch2.7cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"
if [[ -n "${BIRD_HKE_WHEEL_DIR:-}" ]]; then
    [[ -f "$BIRD_HKE_WHEEL_DIR/$conv_name" && -f "$BIRD_HKE_WHEEL_DIR/$mamba_name" ]] ||
        die "Missing matching CUDA wheels in BIRD_HKE_WHEEL_DIR=$BIRD_HKE_WHEEL_DIR"
    conv_source="$BIRD_HKE_WHEEL_DIR/$conv_name"
    mamba_source="$BIRD_HKE_WHEEL_DIR/$mamba_name"
else
    conv_source="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/${conv_name/+/%2B}"
    mamba_source="https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/${mamba_name/+/%2B}"
fi

env_python -m pip install "$conv_source" "$mamba_source"
verify_packages

# Keep a package inventory outside the Git working tree for reporting.
env_python -m pip freeze > "$env_prefix/bird_hke_pip_freeze.txt"
"$conda_bin" list -p "$env_prefix" --explicit > "$env_prefix/bird_hke_conda_explicit.txt"
rm "$env_prefix/.bird_hke_install_in_progress"
printf '\nEnvironment ready: %s\n' "$env_prefix"
printf 'In a GPU allocation run: bash setup_linux_training_env.sh --verify-gpu\n'
printf 'To activate in an interactive shell: source "%s/bin/activate"\n' "$env_prefix"
printf 'Then run the training protocol audit and a smoke test for each model.\n'
