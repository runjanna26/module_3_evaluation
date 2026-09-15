"""Turn raw IFEval prompts into the exact strings the model will receive.

This is the single largest source of spurious differences between
evaluations, so we render the strings once, write them to disk, and never
re-render them during generation. What the model saw is then an artifact,
not an assumption.

Usage:
    python src/build_prompts.py --config configs/qwen05b.yaml --out runs/<run> [--backend dummy]
"""

import argparse
import json
import pathlib

import yaml

from manifest import file_sha256, refuse_overwrite, update_manifest


def load_ifeval(path):
    """Read the IFEval JSONL file. One record per prompt, 541 total."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def render(records, model_id, revision, system_prompt, template_mode):
    """Wrap each prompt in the model's chat format.

    template_mode:
        "chat" - use the model's own chat template (correct for instruct models)
        "raw"  - pass the bare prompt through, no wrapping and no system
                 prompt. Only for dummy runs and for a deliberate template
                 ablation.
    """
    if template_mode == "raw":
        return [r["prompt"] for r in records]

    # Imported here so that dummy runs need no transformers install.
    from transformers import AutoTokenizer

    # Same pinned revision as generate.py: the chat template lives in the
    # tokenizer files, and it can change between model commits.
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    rendered = []
    for r in records:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": r["prompt"]})
        rendered.append(
            tok.apply_chat_template(
                messages,
                tokenize=False,          # we want the string, not token ids
                add_generation_prompt=True,  # append the "assistant:" cue
            )
        )
    return rendered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--backend", choices=["dummy", "hf"], default="hf")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    refuse_overwrite(out / "prompts_rendered.jsonl")

    template_mode = cfg["prompt"]["template_mode"]
    system_prompt = cfg["prompt"].get("system_prompt", "")
    # The chat template comes from the model's tokenizer, which means a
    # download. The dummy backend must work offline with no model, so it
    # always uses raw prompts. This is recorded in the manifest, because
    # config.yaml alone would say "chat".
    template_mode_used = template_mode
    if args.backend == "dummy" and template_mode == "chat":
        template_mode_used = "raw"
        print("  dummy backend: using raw prompts (no tokenizer download)")

    ifeval_path = cfg["data"]["ifeval_path"]
    records = load_ifeval(ifeval_path)
    rendered = render(
        records,
        cfg["model"]["id"],
        cfg["model"]["revision"],
        system_prompt,
        template_mode_used,
    )

    # Keep the original prompt alongside the rendered one. The scorer needs
    # the original; the model needs the rendered one.
    with open(out / "prompts_rendered.jsonl", "w", encoding="utf-8") as f:
        for r, text in zip(records, rendered):
            f.write(json.dumps({
                "key": r["key"],
                "prompt": r["prompt"],
                "rendered": text,
            }, ensure_ascii=False) + "\n")

    # A human-readable sample. Paste this into the report.
    (out / "sample_prompt.txt").write_text(
        "=== EXACT STRING SENT TO THE MODEL (item 0) ===\n" + rendered[0],
        encoding="utf-8",
    )

    transformers_version = None
    if template_mode_used == "chat":
        import transformers
        transformers_version = transformers.__version__

    update_manifest(out, "prompts", {
        "backend": args.backend,
        "ifeval_path": ifeval_path,
        "ifeval_sha256": file_sha256(ifeval_path),
        "n_prompts": len(records),
        "template_mode_config": template_mode,
        "template_mode_used": template_mode_used,
        "system_prompt": system_prompt,
        "tokenizer": (None if template_mode_used == "raw" else
                      {"id": cfg["model"]["id"],
                       "revision": cfg["model"]["revision"]}),
        "transformers_version": transformers_version,
    })

    print(f"rendered {len(rendered)} prompts -> {out/'prompts_rendered.jsonl'}")


if __name__ == "__main__":
    main()
