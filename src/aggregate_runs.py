"""Report the spread of the IFEval metrics across several runs.

A single run gives one number; the brief asks for repeats with a reported
spread. This reads finished run directories and summarises how much the four
metrics move between them - across identical repeats (is the setup
deterministic?) or across deliberate variants (how sensitive is the score?).

Usage:
    python src/aggregate_runs.py runs/<run1> runs/<run2> [runs/<run3> ...]

Writes aggregate.json to the current directory.
"""

import argparse
import hashlib
import json
import pathlib
import sys

from analyze import metrics_for, read_jsonl

METRIC_KEYS = [
    ("prompt_level", "strict"),
    ("instruction_level", "strict"),
    ("prompt_level", "loose"),
    ("instruction_level", "loose"),
]


def run_metrics(run):
    """Exact (passed, total) counts for the four metrics of one run.

    Read from the scorer's per-item files rather than report.json, whose
    accuracies are rounded: a range between repeats can be a single prompt,
    and rounding would blur it.
    """
    out = {}
    for mode in ("strict", "loose"):
        path = run / f"eval_results_{mode}.jsonl"
        if not path.exists():
            sys.exit(f"{path} not found - is {run} a finished run?")
        for level, d in metrics_for(read_jsonl(path)).items():
            out[f"{level}_{mode}"] = (d["k"], d["n"])
    return out


def generations_hash(run):
    """Fingerprint of the raw model output.

    If every run has the same hash, any spread in the scores cannot come from
    the model - it must come from the scorer - so this is reported alongside
    the ranges.
    """
    return hashlib.sha256((run / "generations.jsonl").read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories to compare")
    args = ap.parse_args()

    runs = [pathlib.Path(r) for r in args.runs]
    per_run = {run.name: run_metrics(run) for run in runs}
    hashes = {run.name: generations_hash(run) for run in runs}

    summary = {}
    for level, mode in METRIC_KEYS:
        key = f"{level}_{mode}"
        # Percentages, so the range reads directly as "points" in the report.
        values = [100 * k / n for k, n in (per_run[name][key] for name in per_run)]
        summary[key] = {
            "mean": round(sum(values) / len(values), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "range": round(max(values) - min(values), 2),
            "values": [round(v, 2) for v in values],
        }

    identical = len(set(hashes.values())) == 1

    print(f"\n--- {len(runs)} runs ---")
    print(f"{'metric':28s} {'mean':>7s} {'min':>7s} {'max':>7s} {'range':>7s}")
    for key, s in summary.items():
        print(f"{key:28s} {s['mean']:7.2f} {s['min']:7.2f} {s['max']:7.2f} {s['range']:7.2f}")

    print("\nprompt-level strict per run:")
    for name, m in per_run.items():
        k, n = m["prompt_level_strict"]
        print(f"  {name:60s} {100 * k / n:6.2f}%  ({k}/{n})")

    print(f"\ngenerations identical across runs: {'yes' if identical else 'no'}")
    if identical and any(s["range"] > 0 for s in summary.values()):
        print("  -> the model produced the same text every time; the remaining "
              "spread comes from the scorer, not the model")

    (pathlib.Path.cwd() / "aggregate.json").write_text(json.dumps({
        "runs": [str(r) for r in runs],
        "metrics_percent": summary,
        "counts": {name: {k: list(v) for k, v in m.items()} for name, m in per_run.items()},
        "generations_identical": identical,
        "generations_sha256": hashes,
    }, indent=2), encoding="utf-8")
    print("wrote aggregate.json")


if __name__ == "__main__":
    main()
