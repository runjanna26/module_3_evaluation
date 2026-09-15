"""Hand our generations to the official IFEval scorer.

We do not reimplement the scoring logic. The brief requires the authors'
code, and 25 hand-written constraint checkers are exactly the kind of thing
that goes subtly wrong when rewritten.

Gotcha: the scorer matches responses to prompts by the ORIGINAL prompt
string, not the chat-templated one. Writing the rendered string here breaks
scoring. We check for this before calling the scorer.

Known scorer behaviour (documented, not patched): scoring is not fully
deterministic, because the scorer uses unseeded randomness.
  - Confirmed: `keywords:letter_frequency` replaces a non-letter target with a
    random letter. Items 1122 ('#') and 1129 ('!') therefore pass or fail at
    random. Re-scoring identical responses moved prompt-level accuracy by up
    to 0.19 points (1 prompt in 541).
  - Possible: the language and all-caps/all-lowercase checks call
    `langdetect.detect`, which is also unseeded. Not observed so far.

Usage:
    python src/score.py --config configs/qwen05b.yaml --out runs/<run>
"""

import argparse
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys

import yaml

from manifest import update_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    # Absolute paths: the scorer runs from a different working directory, so
    # relative paths would point at the wrong place.
    out = pathlib.Path(args.out).resolve()
    ifeval_path = pathlib.Path(cfg["data"]["ifeval_path"]).resolve()
    scorer_dir = pathlib.Path(cfg["scoring"]["scorer_repo"]).resolve()
    nltk_data = pathlib.Path(cfg["scoring"]["nltk_data"]).resolve()

    # The scorer wants exactly two fields: prompt and response.
    src = [json.loads(l) for l in
           open(out / "generations.jsonl", encoding="utf-8")]
    scorer_input = out / "scorer_input.jsonl"
    with open(scorer_input, "w", encoding="utf-8") as f:
        for r in src:
            f.write(json.dumps({"prompt": r["prompt"],       # original, not rendered
                                "response": r["response"]},
                               ensure_ascii=False) + "\n")

    # Pre-flight for the gotcha above. The scorer looks responses up in a
    # dict keyed by prompt text, so every benchmark prompt must appear here
    # exactly as written in the IFEval file.
    ifeval_prompts = [json.loads(l)["prompt"]
                      for l in open(ifeval_path, encoding="utf-8") if l.strip()]
    missing = set(ifeval_prompts) - {r["prompt"] for r in src}
    if missing:
        sys.exit(f"{len(missing)} of {len(ifeval_prompts)} IFEval prompts have "
                 "no response. Is the rendered (chat-templated) prompt being "
                 "written instead of the original?")

    # Run as a module from the folder that CONTAINS the scorer package: its
    # own imports are `from instruction_following_eval import ...`.
    env = dict(os.environ, NLTK_DATA=str(nltk_data))
    result = subprocess.run(
        [sys.executable, "-m",
         "instruction_following_eval.evaluation_main",
         f"--input_data={ifeval_path}",
         f"--input_response_data={scorer_input}",
         f"--output_dir={out}"],
        cwd=scorer_dir.parent,
        env=env,
        capture_output=True, text=True,
    )

    # The scorer prints its headline accuracies to stdout. Keep them.
    (out / "scorer_stdout.txt").write_text(
        result.stdout + result.stderr, encoding="utf-8")

    if result.returncode != 0:
        print(result.stdout + result.stderr)
        sys.exit("scorer failed - see scorer_stdout.txt")

    for mode in ("strict", "loose"):
        if not (out / f"eval_results_{mode}.jsonl").exists():
            sys.exit(f"scorer exited cleanly but wrote no eval_results_{mode}.jsonl")

    commit_file = scorer_dir.parent / "SCORER_COMMIT"
    update_manifest(out, "scoring", {
        "scorer_dir": str(scorer_dir),
        "scorer_commit": (commit_file.read_text().strip()
                          if commit_file.exists() else "unknown"),
        "nltk_version": importlib.metadata.version("nltk"),
        "langdetect_version": importlib.metadata.version("langdetect"),
        "scorer_is_deterministic": False,   # see module docstring
    })

    print(result.stdout.strip() or "scorer finished")


if __name__ == "__main__":
    main()
