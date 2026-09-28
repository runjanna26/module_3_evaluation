#!/usr/bin/env bash
# SLURM submit script for one IFEval run on the department V100.
#
#   sbatch scripts/v100.sbatch configs/qwen05b.yaml              # one run
#   sbatch --array=1-3 scripts/v100.sbatch configs/qwen05b.yaml  # three repeats
#
# Each array task makes its own timestamped run directory (the job id is added
# to the name, so two tasks starting in the same second cannot collide).
#
# CHECK THESE THREE LINES FIRST - partition and GPU names differ per cluster:
#   --partition, --gres, --time
# `sinfo -o "%P %G"` lists the partitions and their GPUs.

#SBATCH --job-name=ifeval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

CONFIG="${1:?usage: sbatch scripts/v100.sbatch <config.yaml>}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Keep the model cache inside the project, not in $HOME, where quotas are
# small and a 1 GB download fails halfway through.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON - create it first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# Distinguishes array tasks that start in the same second.
export RUN_SUFFIX="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}${SLURM_ARRAY_TASK_ID:+.$SLURM_ARRAY_TASK_ID}"

echo "host:    $(hostname)"
echo "config:  $CONFIG"
echo "suffix:  $RUN_SUFFIX"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader

# Fail here rather than three minutes into generation.
"$PYTHON" scripts/preflight.py --config "$CONFIG"

PYTHON="$PYTHON" ./run.sh "$CONFIG" hf
