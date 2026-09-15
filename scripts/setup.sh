#!/usr/bin/env bash
# Fetch the benchmark data, the official scorer and the NLTK data the scorer
# needs. Run once from anywhere; safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root, so the paths below are stable

PYTHON="${PYTHON:-python3}"

# Pinned google-research commit. The prompts and the scorer both come from
# this one commit, so the benchmark and its checker cannot drift apart.
# "master" moves; a score computed against master is not reproducible.
GR_COMMIT=ff4d1a67e07ddd242032d3c391bc57c4db05bcc2
GR_REPO=https://github.com/google-research/google-research.git

mkdir -p data external

# The authors' scorer. We call it, we do not modify it.
if [ "$(cat external/SCORER_COMMIT 2>/dev/null || true)" != "$GR_COMMIT" ]; then
  # Clear leftovers from an interrupted or differently-pinned fetch.
  rm -rf external/gr external/instruction_following_eval external/SCORER_COMMIT
  git init -q external/gr
  git -C external/gr remote add origin "$GR_REPO"
  # sparse + blob:none: download one folder, not the whole monorepo.
  git -C external/gr sparse-checkout set instruction_following_eval
  git -C external/gr fetch -q --depth 1 --filter=blob:none origin "$GR_COMMIT"
  git -C external/gr checkout -q FETCH_HEAD
  mv external/gr/instruction_following_eval external/
  rm -rf external/gr
  echo "$GR_COMMIT" > external/SCORER_COMMIT
fi

# The 541 prompts, with their instruction-type annotations. Copied from the
# same pinned commit rather than downloaded from master.
cp external/instruction_following_eval/data/input_data.jsonl data/input_data.jsonl

# Sentence tokenizer used by the scorer's sentence-count checks. Kept inside
# the repo (not ~/nltk_data) so the run does not depend on the user's home dir.
"$PYTHON" -m nltk.downloader -q -d data/nltk_data punkt

echo "scorer commit: $(cat external/SCORER_COMMIT)"
echo "prompts: $(wc -l < data/input_data.jsonl | tr -d ' ') lines (expect 541)"
