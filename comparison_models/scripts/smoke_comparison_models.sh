#!/usr/bin/env bash
# CPU regressions and real-data GPU smoke checks for the comparison models,
# run SEQUENTIALLY. No training and no checkpoints are created; each model
# still re-checks free GPU memory inside its own trainer.
#
# Usage (from the repository root):
#   bash comparison_models/scripts/smoke_comparison_models.sh FD|OS|CS [model]
#
# `model` defaults to all; pass rtmpose-m, vitpose-b or posebh-b to check one.
# Each model runs inside its own isolated interpreter, located by the unified
# CLI from VITPOSE_ENV_PREFIX / POSEBH_ENV_PREFIX / RTMPOSE_ENV_PREFIX.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
SCENARIO="${1:-FD}"
MODEL="${2:-all}"
case "$SCENARIO" in FD|OS|CS) ;; *)
  echo "Usage: bash comparison_models/scripts/smoke_comparison_models.sh FD|OS|CS [all|rtmpose-m|vitpose-b|posebh-b]" >&2
  exit 2
esac

PYTHON="${COMPARISON_CLI_PYTHON:-python3}"

echo "===== Layout/isolation and read-only preflight tests (CPU) ====="
"$PYTHON" -m unittest discover -s comparison_models/tests -t . \
  -p 'test_consolidation.py' -v
"$PYTHON" -m unittest discover -s comparison_models/tests -t . \
  -p 'test_remaining_baseline_preflight.py' -v

echo "===== Environment, data and import preflight (no GPU, no writes) ====="
"$PYTHON" -m comparison_models.cli preflight --model "$MODEL" --scenario "$SCENARIO"

echo "===== Real-data GPU smoke, one model at a time: $SCENARIO ====="
"$PYTHON" -m comparison_models.cli test --model "$MODEL" --scenario "$SCENARIO"

echo "===== SMOKE COMPLETE: no training started, no run folder created ====="