#!/usr/bin/env bash
# Launches the definitive 100-epoch comparison-model runs in parallel lanes.
#
# FD is the wall-clock bottleneck (77,740 train images, each one a
# full-resolution decode + affine warp), so the intended layout keeps the two
# FD runs on the fastest card and puts the ~4x smaller OS/CS runs on another.
#
# Usage (from the repository root, on one machine):
#   bash comparison_models/scripts/run_definitive_training.sh [options] LANE [LANE...]
#
# A LANE is  GPU=SPEC [SPEC...]  where SPEC is  model:scenario. Runs inside a
# lane are sequential; lanes are concurrent, and several lanes may name the
# same GPU (both models peak near 2 GiB, so they comfortably share a card).
# Each lane gets its own tmux window.
#
#   # both FD runs sharing the H100, one lane each so they overlap
#   bash comparison_models/scripts/run_definitive_training.sh --workers 8 \
#       1=vitpose-b:FD 1=posebh-b:FD
#
#   # on the other machine: OS then CS, sharing one card
#   bash comparison_models/scripts/run_definitive_training.sh --workers 4 \
#       1=vitpose-b:OS vitpose-b:CS 0=posebh-b:OS posebh-b:CS
#
# Options:
#   --workers N    decode workers per run (default 8). Lanes sharing a machine
#                  compete for CPU, so keep lanes*workers below the core count;
#                  the script warns when it does not.
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
LANE_GPU=()      # one entry per lane: the CUDA_VISIBLE_DEVICES value
LANE_SPECS=()   # one entry per lane: space-separated model:scenario list

die() { echo "LAUNCHER REFUSED: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --workers) [ $# -ge 2 ] || die "--workers needs a value"; WORKERS="$2"; shift 2 ;;
    --session) [ $# -ge 2 ] || die "--session needs a value";    SESSION="$2"; shift 2 ;;
    --no-tmux) USE_TMUX=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --status)  STATUS_ONLY=1; shift ;;
    -h|--help) sed -n '2,37p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *=*)
      gpu="${1%%=*}"; rest="${1#*=}"
      case "$gpu" in ''|*[!0-9]*) die "lane GPU must be a number, got '$gpu'" ;; esac
      for spec in $rest; do
        case "$spec" in *:*) ;; *) die "lane entry '$spec' is not model:scenario" ;; esac
      done
      [ -n "$rest" ] || die "lane '$1' names no runs"
      LANE_GPU+=("$gpu"); LANE_SPECS+=("$rest"); shift ;;
    *:*)
      # Continuation of the previous lane: "1=vitpose-b:FD posebh-b:FD".
      [ ${#LANE_GPU[@]} -gt 0 ] || die "first argument must name a lane GPU, e.g. 1=vitpose-b:FD"
      LANE_SPECS[$(( ${#LANE_GPU[@]} - 1 ))]="${LANE_SPECS[$(( ${#LANE_GPU[@]} - 1 ))]} $1"
      shift ;;
    *) die "unknown argument '$1' (lanes look like 1=vitpose-b:FD)" ;;
  esac
done

PYTHON="${COMPARISON_CLI_PYTHON:-python3}"
OUT_ROOT="$ROOT/Bird_HKE/trained_models/External_Baselines"

# On some cluster logins LD_LIBRARY_PATH points at a CUDA directory whose
# libtinfo/libevent shadows tmux's own bundled libraries, and tmux then dies
# on startup. Dropping it for the tmux invocation only is enough.
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  TMUX_CMD=(env -u LD_LIBRARY_PATH tmux)
else
  TMUX_CMD=(tmux)
fi

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

[ ${#LANE_GPU[@]} -gt 0 ] || die "no lanes given; pass --help"

# Every run must be unique: two concurrent trainers on one seed_2026 folder
# would fight over the same checkpoint even though each refuses to overwrite.
ALL_SPECS="$(printf '%s\n' "${LANE_SPECS[@]}" | tr ' ' '\n' | grep -v '^$')"
DUPES="$(printf '%s\n' "$ALL_SPECS" | sort | uniq -d)"
[ -z "$DUPES" ] || die "duplicate run(s) requested: $(echo $DUPES)"

# --- pre-flight: refuse before tmux exists, not halfway through a lane ------
for spec in $ALL_SPECS; do
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
done

CORES="$(nproc 2>/dev/null || echo 0)"
# Each lane runs one training at a time, so lanes are the concurrency, not runs.
if [ "$CORES" -gt 0 ] && [ $(( ${#LANE_GPU[@]} * WORKERS)) -gt "$CORES" ]; then
  echo "WARNING: ${#LANE_GPU[@]} concurrent lanes x $WORKERS workers > $CORES cores." >&2
  echo "         Lower --workers; oversubscribed decoders slow every run down." >&2
fi

if [ "$DRY_RUN" -eq 1 ]; then
  for i in "${!LANE_GPU[@]}"; do
    echo "--- lane $((i + 1)) on GPU ${LANE_GPU[$i]} ---"
    for spec in ${LANE_SPECS[$i]}; do
      echo "CUDA_VISIBLE_DEVICES=${LANE_GPU[$i]} BIRD_HKE_NUM_WORKERS=$WORKERS \\"
      echo "  $PYTHON -m comparison_models.cli train --model ${spec%%:*} --scenario ${spec##*:}"
    done
  done
  echo "----- dry run only, nothing was launched -----"
  exit 0
fi

mkdir -p "$OUT_ROOT"
for i in "${!LANE_GPU[@]}"; do
  gpu="${LANE_GPU[$i]}"
  lane="$(mktemp "${TMPDIR:-/tmp}/birdhke_lane$((i + 1))_gpu${gpu}_XXXXXX")"
  : > "$lane"
  # The tmux server was started without LD_LIBRARY_PATH (see above), so put it
  # back for the training processes themselves.
  [ -z "${LD_LIBRARY_PATH:-}" ] || printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH" >> "$lane"
  for spec in ${LANE_SPECS[$i]}; do
    model="${spec%%:*}"; scenario="${spec##*:}"
    [ -f "$(run_dir_for "$spec")/final_model.pth" ] && continue
    {
      echo "echo '=== $spec on GPU $gpu workers=$WORKERS start='\$(date -Is)"
      echo "CUDA_VISIBLE_DEVICES=$gpu BIRD_HKE_NUM_WORKERS=$WORKERS \\"
      echo "  $PYTHON -m comparison_models.cli train --model $model --scenario $scenario"
      echo "echo '=== $spec rc='\$?' end='\$(date -Is)"
    } >> "$lane"
  done
  [ -s "$lane" ] || { rm -f "$lane"; continue; }
  log="$OUT_ROOT/launcher_lane$((i + 1))_gpu${gpu}.log"
  win="lane$((i + 1))-gpu$gpu"
  if [ "$USE_TMUX" -eq 1 ]; then
    if "${TMUX_CMD[@]}" has-session -t "$SESSION" 2>/dev/null; then
      "${TMUX_CMD[@]}" new-window -t "$SESSION" -n "$win" || die "tmux new-window failed"
    else
      "${TMUX_CMD[@]}" new-session -d -s "$SESSION" -c "$ROOT" -n "$win" || die "tmux failed"
    fi
    "${TMUX_CMD[@]}" send-keys -t "$SESSION:$win" "bash '$lane' 2>&1 | tee -a '$log'" Enter
    echo "lane $((i + 1)) (${LANE_SPECS[$i]}) GPU=$gpu -> tmux $SESSION:$win"
  else
    bash "$lane" > "$log" 2>&1 &
    echo "lane $((i + 1)) (${LANE_SPECS[$i]}) GPU=$gpu -> pid $!"
  fi
done

echo "----- lanes launched -----"
echo "Logs:      $OUT_ROOT/launcher_lane*.log"
if [ "$USE_TMUX" -eq 1 ]; then
  echo "Watch:     ${TMUX_CMD[*]} attach -t $SESSION   (Ctrl-b n cycles lanes, Ctrl-b d detaches)"
fi
echo "Progress:  bash comparison_models/scripts/run_definitive_training.sh --status"