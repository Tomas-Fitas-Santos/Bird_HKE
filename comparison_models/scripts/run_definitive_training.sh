#!/usr/bin/env bash
# Launches the definitive 100-epoch comparison-model runs across GPUs.
#
# The Full Dataset scenario is the wall-clock bottleneck (77,740 train images,
# each one a full-resolution decode + affine warp), so the intended topology is
# one lane holding the fastest card for FD alone, with the smaller OS/CS
# scenarios running concurrently on another card.
#
# Usage (from the repository root):
#   bash comparison_models/scripts/run_definitive_training.sh [options] LANE [LANE...]
#
# A LANE is  GPU=SPEC [SPEC...]  where SPEC is  model:scenario. Everything in
# one lane runs sequentially in its own tmux window; lanes run concurrently.
# Example -- FD alone on the H100, OS+CS sharing the second card:
#
#   bash comparison_models/scripts/run_definitive_training.sh --workers 8 \
#       1=vitpose-b:FD posebh-b:FD \
#       0=vitpose-b:OS posebh-b:OS vitpose-b:CS posebh-b:CS
#
# Options:
#   --workers N    decode workers per run (default 8). Runs sharing a machine
#                  compete for CPU, so keep concurrent_lanes*workers below the
#                  core count; the script warns when it does not.
#   --session NAME tmux session name (default birdhke-train)
#   --no-tmux      run in the foreground, lanes backgrounded with &
#   --dry-run      print the lane scripts without starting anything
#   --status       report which definitive runs exist and stop
#
# Nothing here overwrites a finished run: each trainer reserves a brand-new
# seed_2026 directory and refuses if final_model.pth already exists, so
# re-running this script is safe. Resume an interrupted run with
#   python -m comparison_models.cli resume --model M --scenario S
#
# Requires VITPOSE_ENV_PREFIX / POSEBH_ENV_PREFIX (and RTMPOSE_ENV_PREFIX when
# rtmpose-m appears in a lane) to point at the isolated interpreters.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKERS=8
SESSION="birdhke-train"
DRY_RUN=0
USE_TMUX=1
STATUS_ONLY=0
GPUS=()
SPECS=()
LAST_GPU=""

die() { echo "LAUNCHER REFUSED: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --workers) [ $# -ge 2 ] || die "--workers needs a value"; WORKERS="$2"; shift 2 ;;
    --session) [ $# -ge 2 ] || die "--session needs a value";    SESSION="$2"; shift 2 ;;
    --no-tmux) USE_TMUX=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --status)  STATUS_ONLY=1; shift ;;
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *=*)
      gpu="${1%%=*}"; rest="${1#*=}"
      case "$gpu" in ''|*[!0-9]*) die "lane GPU must be a number, got '$gpu'" ;; esac
      LAST_GPU="$gpu"
      for spec in $rest; do
        case "$spec" in *:*) ;; *) die "lane entry '$spec' is not model:scenario" ;; esac
        GPUS+=("$gpu"); SPECS+=("$spec")
      done
      shift ;;
    *:*)
      # Continuation of the previous lane: "1=vitpose-b:FD posebh-b:FD".
      [ -n "$LAST_GPU" ] || die "first argument must name a lane GPU, e.g. 1=vitpose-b:FD"
      GPUS+=("$LAST_GPU"); SPECS+=("$1"); shift ;;
    *) die "unknown argument '$1' (lanes look like 1=vitpose-b:FD)" ;;
  esac
done

PYTHON="${COMPARISON_CLI_PYTHON:-python3}"
OUT_ROOT="$ROOT/Bird_HKE/trained_models/External_Baselines"

run_dir_for() {
  # Mirrors preflight.prepare(): External_Baselines/<Run_Folder>/<SC>/seed_2026
  local model="${1%%:*}" scenario="${1##*:}" folder
  case "$model" in
    vitpose-b) folder=ViTPose_B ;;
    posebh-b)  folder=PoseBH_B ;;
    rtmpose-m) folder=RTMPose_M ;;
    *)         die "unknown model '$model'" ;;
  esac
  printf '%s/%s/%s/seed_2026' "$OUT_ROOT" "$folder" "$scenario"
}

if [ "$STATUS_ONLY" -eq 1 ]; then
  for model in vitpose-b posebh-b rtmpose-m; do
    for scenario in FD OS CS; do
      case "$model:$scenario" in rtmpose-m:OS|rtmpose-m:CS) continue ;; esac
      dir="$(run_dir_for "$model:$scenario")"
      if   [ -f "$dir/final_model.pth" ]; then echo "DONE     $model  $scenario  $dir"
      elif [ -d "$dir" ];                  then echo "PARTIAL  $model  $scenario  $dir"
      else                                      echo "PENDING  $model  $scenario"
      fi
    done
  done
  echo "----- end of status report -----"
  exit 0
fi

[ ${#SPECS[@]} -gt 0 ] || die "no lanes given; pass --help"

# --- pre-flight: refuse before tmux exists, not halfway through a lane ------
QUEUED=()
for spec in "${SPECS[@]}"; do
  model="${spec%%:*}"; scenario="${spec##*:}"
  case "$spec" in rtmpose-m:OS|rtmpose-m:CS) die "rtmpose-m is only validated for FD" ;; esac
  if [ "$DRY_RUN" -eq 0 ]; then
    case "$model" in
      rtmpose-m) [ -n "${RTMPOSE_ENV_PREFIX:-}" ] || die "RTMPOSE_ENV_PREFIX is not set" ;;
      vitpose-b) [ -n "${VITPOSE_ENV_PREFIX:-}" ] || die "VITPOSE_ENV_PREFIX is not set" ;;
      posebh-b)  [ -n "${POSEBH_ENV_PREFIX:-}" ]  || die "POSEBH_ENV_PREFIX is not set" ;;
    esac
    if [ -f "$(run_dir_for "$spec")/final_model.pth" ]; then
      echo "SKIP     $spec -- final_model.pth already exists"
      continue
    fi
    echo "--- preflight $spec ---"
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON" -m comparison_models.cli preflight --model "$model" --scenario "$scenario" \
        || die "preflight failed for $spec"
  fi
  QUEUED+=("$spec")
done

[ ${#QUEUED[@]} -gt 0 ] || { echo "----- nothing left to run -----"; exit 0; }

CORES="$(nproc 2>/dev/null || echo 0)"
LANES="$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l | tr -d ' ')"
if [ "$CORES" -gt 0 ] && [ $((LANES * WORKERS)) -gt "$CORES" ]; then
  echo "WARNING: $LANES concurrent lanes x $WORKERS workers > $CORES cores." >&2
  echo "         Lower --workers; oversubscribed decoders slow every run down." >&2
fi

if [ "$DRY_RUN" -eq 1 ]; then
  for gpu in $(printf '%s\n' "${GPUS[@]}" | sort -u); do
    echo "--- lane GPU=$gpu ---"
    for i in "${!GPUS[@]}"; do
      [ "${GPUS[$i]}" = "$gpu" ] || continue
      spec="${SPECS[$i]}"
      echo "CUDA_VISIBLE_DEVICES=$gpu BIRD_HKE_NUM_WORKERS=$WORKERS \\"
      echo "  $PYTHON -m comparison_models.cli train --model ${spec%%:*} --scenario ${spec##*:}"
    done
  done
  echo "----- dry run only, nothing was launched -----"
  exit 0
fi

mkdir -p "$OUT_ROOT"
for gpu in $(printf '%s\n' "${GPUS[@]}" | sort -u); do
  lane="$(mktemp "${TMPDIR:-/tmp}/birdhke_lane_gpu${gpu}_XXXXXX")"
  : > "$lane"
  for i in "${!GPUS[@]}"; do
    [ "${GPUS[$i]}" = "$gpu" ] || continue
    spec="${SPECS[$i]}"; model="${spec%%:*}"; scenario="${spec##*:}"
    [ -f "$(run_dir_for "$spec")/final_model.pth" ] && continue
    log="$OUT_ROOT/launcher_${model}_${scenario}_gpu${gpu}.log"
    {
      echo "echo '=== $spec on GPU $gpu workers=$WORKERS start='\$(date -Is)"
      echo "CUDA_VISIBLE_DEVICES=$gpu BIRD_HKE_NUM_WORKERS=$WORKERS \\"
      echo "  $PYTHON -m comparison_models.cli train --model $model --scenario $scenario"
      echo "echo '=== $spec rc='\$?' end='\$(date -Is)"
    } >> "$lane"
  done
  [ -s "$lane" ] || { rm -f "$lane"; continue; }
  if [ "$USE_TMUX" -eq 1 ]; then
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux new-window -t "$SESSION" -n "gpu$gpu" || die "tmux new-window failed"
    else
      tmux new-session -d -s "$SESSION" -c "$ROOT" -n "gpu$gpu" || die "tmux failed"
    fi
    tmux send-keys -t "$SESSION:gpu$gpu" "bash '$lane' 2>&1 | tee -a '$OUT_ROOT/launcher_gpu${gpu}.log'" Enter
    echo "lane GPU=$gpu -> tmux $SESSION:gpu$gpu"
  else
    bash "$lane" > "$OUT_ROOT/launcher_gpu${gpu}.log" 2>&1 &
    echo "lane GPU=$gpu -> pid $!"
  fi
done

echo "----- lanes launched -----"
echo "Logs:      $OUT_ROOT/launcher_*.log"
if [ "$USE_TMUX" -eq 1 ]; then
  echo "Watch:     tmux attach -t $SESSION   (Ctrl-b n cycles lanes, Ctrl-b d detaches)"
fi
echo "Progress:  bash comparison_models/scripts/run_definitive_training.sh --status"