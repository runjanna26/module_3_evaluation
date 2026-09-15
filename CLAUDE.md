# CLAUDE.md — project context for Claude Code

Read this before doing anything. It describes what this repo is for and the
rules that must not be broken.

## What this is

A class project. We build an **evaluation harness** that measures how well
small open-weight language models follow instructions, using the IFEval
benchmark, then compare our numbers against the numbers in the IFEval paper
(Zhou et al. 2023).

Marking: 60% on the harness being re-runnable by a stranger, 40% on reading
the gap to the paper honestly. Matching the paper's number earns nothing.

The author is a robotics/embedded engineer, not an ML researcher. Prefer
plain, explicit code over clever abstractions. Comment the *why*, not the
*what*.

## Hard rules

1. **Never train, fine-tune, or modify a model.** Inference only.
2. **Never reimplement the scorer.** We call the official Google scorer.
   If a scorer bug is found, document it — do not silently patch it.
3. **Every run writes a config snapshot.** Model revision hash, precision,
   seed, decoding settings, library versions. No exceptions.
4. **Every run saves raw generations verbatim.** Never overwrite a run
   directory; each run gets its own timestamped folder under `runs/`.
5. **The pipeline must run end-to-end with `--backend dummy`** on a laptop
   with no GPU and no model download. This is how we test before deploying.
6. **Target hardware is an NVIDIA V100 (Volta, compute capability 7.0).**
   fp16 only — no bf16, no FlashAttention-2. Use
   `attn_implementation="sdpa"`, never `"flash_attention_2"`.

## Pipeline

    IFEval prompts (541)
      -> build_prompts.py   applies chat template, saves exact final string
      -> generate.py        model produces one response per prompt   [GPU]
      -> score.py           official scorer -> per-item pass/fail
      -> analyze.py         4 metrics + CI, per-type, length, truncation

Only `generate.py` needs a GPU. Everything else is CPU text processing.

## Known gotcha

The official scorer matches responses to prompts by the **original** IFEval
prompt string, not the chat-templated one. `score.py` must write the original
prompt into the scorer input file. Getting this wrong silently scores 0%.

## When asked to add a feature

Ask first whether it changes a measured number. If it does, it needs a
before/after comparison in the report — the brief calls an unmeasured
improvement "a preference".
