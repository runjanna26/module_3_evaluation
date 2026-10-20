#!/usr/bin/env bash
# Run the V100 experiment schedule on one booked GPU, one run after another.
#
# For the department machines that are booked one card at a time and reached
# over SSH (no SLURM). Start it inside tmux so it survives a dropped
# connection:
#
#   tmux new -s ifeval
#   ./scripts/run_schedule.sh 1                  # everything, on GPU 1
#   ./scripts/run_schedule.sh 1 repeats          # only the baseline repeats
#   ./scripts/run_schedule.sh 1 sysprompt scale  # two groups
#   (detach: Ctrl-b then d    come back: tmux attach -t ifeval)
#
# Groups (9 runs in total):
#   repeats    qwen05b.yaml x3                       spread, and laptop vs V100
#   sysprompt  qwen05b_sys_{generic,strict,casual}   prompt sensitivity
#   scale      qwen15b.yaml x3                       Qwen2.5-1.5B comparison
#
# Rough time on a V100: 10-20 min per 0.5B run, 20-30 min per 1.5B run,
# so about 3 hours for everything.
#
# BACKEND=dummy runs the same schedule without a model, to test this script.
#
# A failed run is logged and the schedule moves on to the next one; the
# summary at the end lists every run with its exit code.

set -uo pipefail     # deliberately no -e: one failed run must not cancel the rest

GPU="${1:?usage: scripts/run_schedule.sh <gpu-index> [repeats|sysprompt|scale ...]}"
shift
# Not "GROUPS": that is a bash built-in and silently ignores assignment.
STEPS=("$@")
if [ ${#STEPS[@]} -eq 0 ]; then
  STEPS=(repeats sysprompt scale)
fi
BACKEND="${BACKEND:-hf}"

cd "$(dirname "$0")/.."

# Number GPUs the way nvidia-smi and the booking list do (by PCI bus). CUDA's
# default is fastest-first, so without this "GPU 1" can be a different
# physical card from the one you booked - possibly someone else's.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

# Model cache inside the project, not $HOME, where quotas are usually small.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON - create it first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# Build the run list first, so a typo in a group name fails before anything runs.
CONFIGS=()
for step in "${STEPS[@]}"; do
  case "$step" in
    repeats)   CONFIGS+=(configs/qwen05b.yaml configs/qwen05b.yaml configs/qwen05b.yaml) ;;
    sysprompt) CONFIGS+=(configs/qwen05b_sys_generic.yaml configs/qwen05b_sys_strict.yaml
                         configs/qwen05b_sys_casual.yaml) ;;
    scale)     CONFIGS+=(configs/qwen15b.yaml configs/qwen15b.yaml configs/qwen15b.yaml) ;;
    *) echo "unknown group '$step' - use: repeats sysprompt scale" >&2; exit 2 ;;
  esac
done

mkdir -p logs
LOG="logs/schedule_$(hostname -s)_gpu${GPU}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "host     $(hostname)"
echo "gpu      $GPU (CUDA_DEVICE_ORDER=PCI_BUS_ID)"
echo "backend  $BACKEND"
echo "runs     ${#CONFIGS[@]}"
echo "log      $LOG"
echo

# Confirm the card exists, and warn if another job is already on it.
if ! nvidia-smi -i "$GPU" --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
       --format=csv,noheader; then
  echo "nvidia-smi cannot see GPU $GPU - check the index you booked" >&2
  exit 1
fi
used_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [ "${used_mib:-0}" -gt 1000 ]; then
  echo "WARNING: GPU $GPU already has ${used_mib} MiB in use - is someone else on it?"
fi

# Preflight every distinct config once, before the first run, so a problem
# with the last group cannot surface three hours in.
for c in $(printf '%s\n' "${CONFIGS[@]}" | sort -u); do
  echo
  echo "--- preflight: $c"
  if ! "$PYTHON" scripts/preflight.py --config "$c"; then
    echo "preflight failed for $c - nothing was run" >&2
    exit 1
  fi
done

RESULTS=()
failed=0
i=0
for c in "${CONFIGS[@]}"; do
  i=$((i + 1))
  echo
  echo "=== [$i/${#CONFIGS[@]}] $c   $(date '+%F %T')"
  # The index keeps run directories unique even if a run fails within a second.
  RUN_SUFFIX="$(hostname -s)-gpu${GPU}-${i}" PYTHON="$PYTHON" ./run.sh "$c" "$BACKEND"
  rc=$?
  [ "$rc" -ne 0 ] && failed=$((failed + 1))
  RESULTS+=("$(printf '[%d] %-36s exit=%d' "$i" "$c" "$rc")")
done

echo
echo "=== summary   $(date '+%F %T')"
printf '  %s\n' "${RESULTS[@]}"
echo "  $failed of ${#CONFIGS[@]} failed"
echo "  log: $LOG"
[ "$failed" -eq 0 ]
