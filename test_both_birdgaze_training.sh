#!/usr/bin/env bash
# Safe qualification of both baselines BEFORE definitive 100-epoch training.
# Independent venv dependency checks; CPU tests; sequential full-model GPU
# smoke; isolated 2-epoch mini-training, validation, checkpoint/resume probe.
# Never launches --train or writes into the real model run directories.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SCENARIO="${1:-FD}"
case "$SCENARIO" in FD|OS|CS) ;; *)
  echo "Usage: bash test_both_birdgaze_training.sh FD|OS|CS" >&2
  exit 2
esac
VITPOSE_PY="${VITPOSE_PY:-/home/students/conda-envs/vitpose-b-cu128-venv/bin/python}"
POSEBH_PY="${POSEBH_PY:-/home/students/conda-envs/posebh-b-cu128-venv/bin/python}"

for python in "$VITPOSE_PY" "$POSEBH_PY"; do
  [[ -x "$python" ]] || {
    echo "Missing isolated interpreter: $python" >&2
    echo "Run the corresponding setup script --inspect/--create first" >&2
    exit 1
  }
done

echo "=== STEP 1/4: ViTPose-B isolated library/CUDA check ==="
"$VITPOSE_PY" -m external_models.remaining_baselines.check_deps
echo "=== STEP 2/4: PoseBH-B isolated library/CUDA check ==="
"$POSEBH_PY" -m external_models.remaining_baselines.check_deps
echo "=== STEP 3/4: both CPU regression suites and GPU smoke tests ==="
bash smoke_both_birdgaze_baselines.sh "$SCENARIO"
echo "=== STEP 4/4: isolated real-data optimizer/validation/save/resume checks ==="
echo "  ViTPose-B (12 train, 8 val; two short epochs)"
"$VITPOSE_PY" -m external_models.remaining_baselines.test_training \
  --model vitpose-b --scenario "$SCENARIO"
echo "  PoseBH-B (12 train, 8 val; two short epochs)"
"$POSEBH_PY" -m external_models.remaining_baselines.test_training \
  --model posebh-b --scenario "$SCENARIO"

echo "=== BOTH MODELS PASS DEPENDENCIES, CPU, CUDA SMOKE, 2-EPOCH TRAINING AND RESUME ==="
echo "=== Definitive 100-epoch run folders were NOT used ==="
