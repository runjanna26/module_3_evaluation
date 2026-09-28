#!/usr/bin/env bash
# One-command entry point. This is what the "a stranger can re-run it" mark
# is assessed on.
#
#   ./run.sh configs/qwen05b.yaml dummy      # laptop, no GPU, no model
#   ./run.sh configs/qwen05b.yaml hf         # real run on the GPU
#
# Every invocation creates its own timestamped run directory. Nothing is
# ever overwritten, so repeats are just repeated calls.

set -euo pipefail

CONFIG_ARG="${1:?usage: ./run.sh <config.yaml> [dummy|hf]}"
BACKEND="${2:-hf}"
PYTHON="${PYTHON:-python3}"     # override with PYTHON=/path/to/python ./run.sh ...

case "$BACKEND" in
  dummy|hf) ;;
  *) echo "backend must be 'dummy' or 'hf', got '$BACKEND'" >&2; exit 2 ;;
esac

# Resolve the config before changing directory, so a relative path given by
# the caller still works. Then work from the repo root, because the paths
# inside the config (data/, external/) are relative to it.
CONFIG="$(cd "$(dirname "$CONFIG_ARG")" && pwd)/$(basename "$CONFIG_ARG")"
cd "$(dirname "$0")"

# Fail before creating a run directory if setup was never done - otherwise a
# stranger gets a half-empty run folder and a confusing traceback.
if [ ! -f data/input_data.jsonl ] || [ ! -d external/instruction_following_eval ]; then
  echo "missing data or scorer - run ./scripts/setup.sh first" >&2
  exit 1
fi

# RUN_SUFFIX lets a scheduler distinguish array tasks that start in the
# same second (scripts/v100.sbatch sets it to the SLURM job id).
RUN="runs/$(basename "${CONFIG%.yaml}")_${BACKEND}_$(date +%Y%m%d_%H%M%S)${RUN_SUFFIX:+_$RUN_SUFFIX}"
mkdir -p runs
# No -p here: if two runs start in the same second, the second one fails
# instead of silently writing into the first one's directory.
mkdir "$RUN"

# Snapshot the environment BEFORE anything runs. Required by the brief and
# useless if collected afterwards.
cp "$CONFIG" "$RUN/config.yaml"
{
  echo "backend: $BACKEND"
  echo "config_source: $CONFIG_ARG"
  if git rev-parse --git-dir >/dev/null 2>&1; then
    echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo none)"
    # Uncommitted changes mean git_commit alone does not describe the code.
    # runs/ is excluded: earlier run folders are results, not code.
    echo "git_uncommitted_files: $(git status --porcelain -- . ':!runs' | wc -l | tr -d ' ')"
  else
    echo "git_commit: none (not a git repository)"
  fi
  echo "scorer_commit: $(cat external/SCORER_COMMIT 2>/dev/null || echo unknown)"
  echo "python: $("$PYTHON" --version 2>&1)"
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "packages:"
  "$PYTHON" -m pip freeze 2>/dev/null | sed 's/^/  /'
  echo "gpu:"
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
             --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  none"
} > "$RUN/environment.txt"

echo "[1/4] building prompts"
"$PYTHON" src/build_prompts.py --config "$CONFIG" --out "$RUN" --backend "$BACKEND"

echo "[2/4] generating  (backend: $BACKEND)"
"$PYTHON" src/generate.py --config "$CONFIG" --out "$RUN" --backend "$BACKEND"

echo "[3/4] scoring"
"$PYTHON" src/score.py --config "$CONFIG" --out "$RUN"

echo "[4/4] analysing"
"$PYTHON" src/analyze.py --out "$RUN"

echo "done -> $RUN"
