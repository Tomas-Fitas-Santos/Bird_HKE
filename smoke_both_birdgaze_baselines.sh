#!/usr/bin/env bash
# Run both baseline CPU regressions and real-data GPU smoke tests SEQUENTIALLY.
# No training/checkpoints are created. Each trainer still checks GPU idleness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SCENARIO="${1:-FD}"
case "$SCENARIO" in FD|OS|CS) ;; *)
  echo "Usage: bash smoke_both_birdgaze_baselines.sh FD|OS|CS" >&2
  exit 2
esac

VITPOSE_PY="${VITPOSE_PY:-/home/students/conda-envs/vitpose-b-cu128-venv/bin/python}"
POSEBH_PY="${POSEBH_PY:-/home/students/conda-envs/posebh-b-cu128-venv/bin/python}"

for python in "$VITPOSE_PY" "$POSEBH_PY"; do
  [[ -x "$python" ]] || { echo "Missing isolated interpreter: $python" >&2; exit 1; }
done

echo "===== Shared source/crop/checkpoint preflight tests ====="
"$VITPOSE_PY" -m unittest discover -s tests -p 'test_remaining_baseline_preflight.py' -v

echo "===== ViTPose-B CPU/unit checks ====="
"$VITPOSE_PY" -m unittest discover -s tests -p 'test_vitpose_b.py' -v

echo "===== PoseBH-B CPU/unit checks ====="
"$POSEBH_PY" -m unittest discover -s tests -p 'test_posebh_b.py' -v

echo "===== ViTPose-B real-data GPU smoke: $SCENARIO ====="
"$VITPOSE_PY" -m external_models.vitpose.train --scenario "$SCENARIO" --smoke-test

echo "===== PoseBH-B real-data GPU smoke: $SCENARIO ====="
"$POSEBH_PY" -m external_models.posebh.train --scenario "$SCENARIO" --smoke-test

echo "===== BOTH GPU SMOKE TESTS PASS: no training was started ====="
