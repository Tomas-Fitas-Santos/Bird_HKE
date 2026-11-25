#!/usr/bin/env bash
# Safe qualification of the comparison models BEFORE any definitive 100-epoch
# training: isolated dependency checks, CPU regressions, sequential full-model
# GPU smoke, and an isolated 2-epoch train/validate/save/resume probe.
#
# Usage (from the repository root):
#   bash comparison_models/scripts/test_comparison_models.sh FD|OS|CS [model]
#
# Never launches --train and never writes into a definitive run directory. The
# probe writes only under the ignored _Training_Probes scratch directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SCENARIO="${1:-FD}"
MODEL="${2:-all}"
case "$SCENARIO" in FD|OS|CS) ;; *)
  echo "Usage: bash comparison_models/scripts/test_comparison_models.sh FD|OS|CS [all|rtmpose-m|vitpose-b|posebh-b]" >&2
  exit 2
esac

PYTHON="${COMPARISON_CLI_PYTHON:-python3}"

echo "=== STEP 1/4: dependency and CUDA check inside each isolated environment ==="
"$PYTHON" -m comparison_models.cli preflight --model "$MODEL" --scenario "$SCENARIO"

echo "=== STEP 2/4: CPU regression suites ==="
"$PYTHON" -m unittest discover -s comparison_models/tests -t . -p 'test_*.py' -v

echo "=== STEP 3/4: real-data GPU smoke tests, run sequentially ==="
"$PYTHON" -m comparison_models.cli test --model "$MODEL" --scenario "$SCENARIO"

echo "=== STEP 4/4: isolated 2-epoch train/validate/save/resume probes ==="
if [[ "$MODEL" == "all" || "$MODEL" == "vitpose-b" ]]; then
  echo "  ViTPose-B (12 train, 8 val; two short epochs)"
  "$PYTHON" -m comparison_models.common.training_probe \
    --model vitpose-b --scenario "$SCENARIO"
fi
if [[ "$MODEL" == "all" || "$MODEL" == "posebh-b" ]]; then
  echo "  PoseBH-B (12 train, 8 val; two short epochs)"
  "$PYTHON" -m comparison_models.common.training_probe \
    --model posebh-b --scenario "$SCENARIO"
fi
if [[ "$MODEL" == "all" || "$MODEL" == "rtmpose-m" ]]; then
  echo "  RTMPose-M: no disposable optimizer/resume probe exists yet."
  echo "  Its GPU data/model preflight ran in step 3; the 210-epoch MMEngine"
  echo "  trainer has no isolated mini-run harness. Reported as an open"
  echo "  runtime-qualification gap - do not claim it as fully probed."
fi

echo "=== QUALIFICATION COMPLETE: dependencies, CPU, GPU smoke and probes ==="
echo "=== Definitive 100-epoch run folders were NOT used ==="