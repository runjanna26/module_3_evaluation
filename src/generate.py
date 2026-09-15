"""Run the model over every prompt and save the raw text it produced.

This is the only step that needs a GPU. The `dummy` backend fakes it so the
rest of the pipeline can be developed and tested on a laptop.

Truncation is recorded here rather than inferred later: a response that hit
the token ceiling was cut off, and that is a property of the run, not of the
text.

Usage:
    python src/generate.py --config configs/qwen05b.yaml --out runs/<run> --backend hf
"""

import argparse
import copy
import json
import pathlib
import platform
import random
import time

import yaml

from manifest import refuse_overwrite, update_manifest


def generate_dummy(rendered, seed):
    """Fake responses, for testing the pipeline without a model.

    Deliberately includes a few that satisfy common IFEval constraints, so a
    dummy run scores above zero and proves the scorer is actually wired up.
    """
    rng = random.Random(seed)
    canned = [
        "this is a lowercase answer with no commas at all",
        "* first point\n* second point\n* third point",
        '"A quoted answer that sits inside double quotation marks."',
        "SECTION 1\nSome text here.\n\nSECTION 2\nMore text here.",
    ]
    out = []
    for _ in rendered:
        text = rng.choice(canned)
        # token count is approximated for the dummy backend only
        out.append({"response": text, "n_new_tokens": len(text.split()),
                    "truncated": False})
    return out, {"note": "dummy backend - canned responses, no model"}


def generate_hf(rendered, cfg):
    """Real generation with Hugging Face transformers on a single GPU."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m, d = cfg["model"], cfg["decoding"]

    if m["revision"] == "main":
        print("  WARNING: model.revision is 'main' - this run is not "
              "reproducible. Pin a commit hash in the config.")

    # No effect on greedy decoding, but required for the result to be
    # repeatable as soon as do_sample is switched on.
    transformers.set_seed(cfg["runtime"]["seed"])

    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
    tok.padding_side = "left"          # required for batched decoder-only gen
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        m["id"],
        revision=m["revision"],        # pinned: "main" is a moving target
        torch_dtype=torch.float16,     # explicit; V100 has no bf16
        attn_implementation="sdpa",    # FlashAttention-2 needs Ampere+
        device_map="cuda",
    )
    model.eval()

    # The settings we pass explicitly. Anything NOT listed here is taken from
    # the model's own generation_config.json (for Qwen2.5 that includes
    # repetition_penalty), so we also compute and record the merged result -
    # the same merge model.generate() does internally.
    gen_kwargs = dict(
        max_new_tokens=d["max_new_tokens"],
        do_sample=d["do_sample"],      # False = greedy baseline
        temperature=d.get("temperature"),
        top_p=d.get("top_p"),
        pad_token_id=tok.pad_token_id,
    )
    effective = copy.deepcopy(model.generation_config)
    effective.update(**gen_kwargs)

    # Any of these tokens means the model chose to stop.
    eos = effective.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    results = []
    bs = cfg["runtime"]["batch_size"]
    for i in range(0, len(rendered), bs):
        chunk = rendered[i:i + bs]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")

        with torch.no_grad():
            ids = model.generate(**enc, **gen_kwargs)

        # Strip the prompt tokens; keep only what the model added.
        new_ids = ids[:, enc["input_ids"].shape[1]:]
        for row in new_ids:
            row = row.tolist()
            # Cut at the first end-of-sequence token. Everything after it is
            # padding added because other prompts in the batch were longer.
            stop = next((j for j, t in enumerate(row) if t in eos_ids), None)
            content = row if stop is None else row[:stop]
            results.append({
                "response": tok.decode(content, skip_special_tokens=True),
                "n_new_tokens": len(content),
                # No end token means generation only stopped because it hit
                # max_new_tokens: the answer was cut off mid-thought.
                "truncated": stop is None,
            })
        print(f"  {min(i+bs, len(rendered))}/{len(rendered)}", flush=True)

    info = {
        "model_id": m["id"],
        "revision_requested": m["revision"],
        # The commit actually loaded. Differs from the request if the
        # request was a branch name like "main".
        "revision_resolved": getattr(model.config, "_commit_hash", None),
        "torch_dtype": str(model.dtype),
        "attn_implementation": model.config._attn_implementation,
        "seed": cfg["runtime"]["seed"],
        "batch_size": bs,
        "padding_side": tok.padding_side,
        "generate_kwargs": gen_kwargs,
        "effective_generation_config": effective.to_dict(),
        "eos_token_ids": sorted(eos_ids),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(
                map(str, torch.cuda.get_device_capability(0))),
        },
    }
    return results, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["dummy", "hf"], default="hf")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = pathlib.Path(args.out)
    # Raw generations are the evidence for every number. Never replace them.
    refuse_overwrite(out / "generations.jsonl")

    items = [json.loads(l) for l in
             open(out / "prompts_rendered.jsonl", encoding="utf-8")]
    rendered = [i["rendered"] for i in items]

    t0 = time.time()
    if args.backend == "dummy":
        gen, info = generate_dummy(rendered, cfg["runtime"]["seed"])
    else:
        gen, info = generate_hf(rendered, cfg)
    elapsed = time.time() - t0

    with open(out / "generations.jsonl", "w", encoding="utf-8") as f:
        for item, g in zip(items, gen):
            f.write(json.dumps({"key": item["key"],
                                "prompt": item["prompt"], **g},
                               ensure_ascii=False) + "\n")

    update_manifest(out, "generation", {"backend": args.backend,
                                        "decoding_config": cfg["decoding"],
                                        **info})

    # GPU-hours is a required figure in the report.
    (out / "timing.json").write_text(json.dumps({
        "backend": args.backend,
        "wall_clock_seconds": round(elapsed, 1),
        "gpu_hours": round(elapsed / 3600, 4),
        "n_prompts": len(rendered),
    }, indent=2), encoding="utf-8")

    print(f"generated {len(gen)} responses in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
