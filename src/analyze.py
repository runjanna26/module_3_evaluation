"""Turn the scorer's per-item output into every figure the report needs.

Produces: four metrics with confidence intervals, a per-instruction-type
breakdown, a response-length distribution and a truncation rate.

Usage:
    python src/analyze.py --out runs/<run>
"""

import argparse
import json
import math
import pathlib
import sys
from collections import Counter


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion.

    Wilson rather than the textbook normal interval: it stays inside [0, 1]
    and behaves sensibly when the pass rate is near 0 or 1, which it will be
    for several instruction types at small scale.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def metrics_for(records):
    """Prompt-level and instruction-level accuracy from one scorer output file.

    prompt-level      : the prompt counts only if EVERY instruction passed
    instruction-level : each instruction counted separately
    """
    prompt_ok = sum(1 for r in records if r["follow_all_instructions"])
    flags = [f for r in records for f in r["follow_instruction_list"]]
    return {
        "prompt_level": {"k": prompt_ok, "n": len(records)},
        "instruction_level": {"k": sum(flags), "n": len(flags)},
    }


def per_type(records):
    """Pass rate for each of the 25 verifiable instruction types."""
    passed, total = Counter(), Counter()
    for r in records:
        for name, ok in zip(r["instruction_id_list"],
                            r["follow_instruction_list"]):
            total[name] += 1
            passed[name] += bool(ok)
    return {name: {"k": passed[name], "n": total[name],
                   "rate": passed[name] / total[name]}
            for name in sorted(total, key=lambda x: passed[x] / total[x])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)

    report = {}

    # --- the four headline metrics -------------------------------------
    # Field names come from the official scorer. If a future version of the
    # repo renames them, this is the place that breaks - check first.
    for mode in ("strict", "loose"):
        f = out / f"eval_results_{mode}.jsonl"
        if not f.exists():
            # A report with two of the four metrics silently missing is worse
            # than no report.
            sys.exit(f"{f} not found - did score.py run?")
        recs = read_jsonl(f)
        for level, d in metrics_for(recs).items():
            lo, hi = wilson(d["k"], d["n"])
            report[f"{level}_{mode}"] = {
                "accuracy": round(d["k"] / d["n"], 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "n": d["n"],
            }
        if mode == "strict":
            report["per_instruction_type"] = per_type(recs)

    # --- length and truncation ------------------------------------------
    gens = read_jsonl(out / "generations.jsonl")
    lengths = sorted(g["n_new_tokens"] for g in gens)
    report["response_length"] = {
        "median": lengths[len(lengths) // 2],
        "p90": lengths[int(len(lengths) * 0.9)],
        "max": lengths[-1],
        "histogram_bins": dict(sorted(Counter(l // 100 * 100 for l in lengths).items())),
    }
    report["truncation_rate"] = round(
        sum(g["truncated"] for g in gens) / len(gens), 4)

    report["timing"] = json.loads((out / "timing.json").read_text())

    (out / "report.json").write_text(json.dumps(report, indent=2),
                                     encoding="utf-8")

    # Console summary - what you paste into the writeup.
    print(f"\n--- {out.name} ---")
    for k, v in report.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"{k:32s} {v['accuracy']:.3f}  "
                  f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
    print(f"{'truncation_rate':32s} {report['truncation_rate']:.3f}")
    print("\nworst 5 instruction types:")
    for name, d in list(report.get("per_instruction_type", {}).items())[:5]:
        print(f"  {name:40s} {d['rate']:.2f}  (n={d['n']})")


if __name__ == "__main__":
    main()
