# IFEval harness — small models on one GPU

An evaluation instrument for measuring instruction-following in small
open-weight models, and for reading the gap to the IFEval paper honestly.

## Quick start

```bash
pip install -r requirements-scoring.txt   # laptop: no torch needed
./scripts/setup.sh                        # fetch prompts + official scorer (pinned commit)

./run.sh configs/qwen05b.yaml dummy       # laptop test, no GPU needed
```

On the GPU machine install the full `requirements.txt` instead and use
`./run.sh configs/qwen05b.yaml hf`. Set `PYTHON=/path/to/python` if `python3`
is not the interpreter you installed into.

Each invocation writes a fresh timestamped directory under `runs/`:

| File | What it is |
|---|---|
| `config.yaml` | copy of the config the run was started with |
| `environment.txt` | git commit, scorer commit, Python, `pip freeze`, GPU — captured before anything runs |
| `run_manifest.json` | what each stage actually used: resolved model commit, dtype, merged decoding settings, library versions, data sha256 |
| `prompts_rendered.jsonl`, `sample_prompt.txt` | the exact strings sent to the model |
| `generations.jsonl` | raw model output, verbatim, with token count and truncation flag |
| `scorer_input.jsonl`, `eval_results_{strict,loose}.jsonl`, `scorer_stdout.txt` | official scorer input and output |
| `timing.json` | wall clock and GPU-hours |
| `report.json` | four metrics with 95% Wilson intervals, per-type, length, truncation |

## Layout

```
CLAUDE.md                project rules — Claude Code reads this first
run.sh                   one-command entry point
requirements.txt         full pinned install (GPU machine)
requirements-scoring.txt everything except torch/transformers (laptop)
scripts/setup.sh         fetches prompts, scorer and NLTK data at a pinned commit
configs/*.yaml           every knob that can change a number
src/
  build_prompts.py  applies the chat template, saves the exact string
  generate.py       runs the model          <- the only GPU step
  score.py          calls the official scorer
  analyze.py        metrics, intervals, per-type, length, truncation
  manifest.py       writes run_manifest.json (what each stage really used)
data/, external/         created by setup.sh, not committed
runs/                    one directory per run, never overwritten
report.md                the write-up
evaluation_figures.ipynb every chart in the report, computed from runs/
figures/                 the charts as 200 dpi PNGs, written by the notebook
reference/gpt4_paper/    the paper's GPT-4 responses, scored by this harness
```

### Figures

```bash
pip install -r requirements-notebook.txt
jupyter lab evaluation_figures.ipynb        # or: jupyter nbconvert --execute --inplace ...
```

The notebook reads `runs/` directly and rewrites `figures/*.png`. The only
hand-entered numbers in it are the published reference values, each cited.

### PDF

```bash
python3 scripts/make_pdf.py        # report.md + figures/ -> report.pdf
```

Renders the report through headless Chrome, the only PDF engine on this
machine. Intermediate HTML lands in `build/` and is not committed.

## Before the department GPU

Everything except `generate.py --backend hf` runs on a laptop. Get a clean
dummy run first — if the dummy scores above zero, the scorer is correctly
wired and the GPU session only has to work once.

The dummy backend always sends **raw** prompts (the chat template needs a
tokenizer download). `run_manifest.json` records this as
`template_mode_used: raw`, so a dummy run is never mistaken for a chat run.

### Running on a small laptop GPU

`configs/qwen05b_laptop.yaml` is identical to `qwen05b.yaml` except
`batch_size: 4`. On a 4 GB RTX 3050 Ti, batch 8 runs out of memory on the
longest prompts; batch 4 completes 541 prompts in about 19 minutes. The
specified V100 was unavailable for this project, so all reported runs use
the laptop configs (see report.md, Section 8).

## Things that can move the number without showing up in the config

Found while building the harness. None of them is changed silently; each
is recorded per run so the report can discuss it.

- **Qwen's default system prompt.** With `system_prompt: ""`, Qwen2.5's chat
  template inserts *"You are Qwen, created by Alibaba Cloud. You are a
  helpful assistant."* The "empty" baseline is not empty — see
  `sample_prompt.txt`.
- **Hidden decoding defaults.** Settings not in the config come from the
  model's `generation_config.json`. For Qwen2.5-0.5B-Instruct that includes
  `repetition_penalty: 1.1`, which still applies under greedy decoding. See
  `run_manifest.json -> generation.effective_generation_config`.
- **Batch size.** Batched generation left-pads prompts; in fp16 that can
  change greedy outputs slightly. Lowering `batch_size` to fix an OOM is a
  config change, and it is recorded as one.
- **Scorer non-determinism (scorer bug, not patched).** Two prompts (keys
  1122 and 1129) ask for the frequency of `#` and `!`. The
  `letter_frequency` checker treats these as invalid and substitutes a
  *random* letter each time it scores. Re-scoring identical responses can
  move prompt-level accuracy by about 0.2 points. `langdetect` is also
  unseeded, but no flips from it have been observed.
- **Benchmark file drift.** At the pinned commit, prompt 2785 says
  "at least 3 placeholders"; the GPT-4 responses published with the paper
  were answered to "at least one placeholder". The paper's GPT-4 response
  file therefore cannot be scored as-is.
- **Scorer needs `nltk==3.8.1`.** It loads the legacy `punkt` pickle, which
  newer nltk refuses to load.

## Deploying to the department GPU

```bash
git clone <your-repo> && cd ifeval-harness
pip install -r requirements.txt
./scripts/setup.sh
./run.sh configs/qwen05b.yaml hf
```

The model revision in `configs/qwen05b.yaml` is pinned to a commit hash.
`run_manifest.json` records the commit that was actually loaded.

Check `nvidia-smi` first. If generation runs out of memory, lower
`runtime.batch_size` before touching anything else.

## Experiment schedule

| Runs | Config change | Purpose |
|---|---|---|
| 3 | none | within-config spread |
| 3 | vary `prompt.system_prompt` | prompt sensitivity |
| 3 | `model.id` -> Qwen2.5-1.5B-Instruct | scale comparison |
| 1 | `template_mode: raw` | template ablation (optional credit) |

---

# Using Claude Code on this repo

## Setup

```bash
npm install -g @anthropic-ai/claude-code
cd ifeval-harness
claude
```

`CLAUDE.md` is loaded automatically at the start of every session. It carries
the hard rules — fp16 only, never reimplement the scorer, dummy backend must
keep working — so you don't have to restate them each time.

## How to ask

Claude Code works on whole files, so point at them and state the goal and
the constraint.

Weak: *"make the analysis better"*
Strong: *"In src/analyze.py, add a bootstrap interval across multiple run
directories, alongside the existing Wilson interval. Take run paths as
positional arguments. Keep the console summary format unchanged."*

Useful patterns for this project:

- *"Run `./run.sh configs/qwen05b.yaml dummy` and fix whatever breaks."*
  It executes the command, reads the traceback and iterates.
- *"Read runs/<dir>/report.json and draft the findings paragraph using the
  sentence template in the assignment brief."*
- *"Add a config for Qwen2.5-1.5B-Instruct. Change nothing else."*
- *"Explain what src/score.py does, line by line. I am not an ML person."*

## Two habits worth keeping

Commit before each session, so you can always see exactly what changed:

```bash
git add -A && git commit -m "before: bootstrap intervals"
```

And when Claude proposes a change that affects a measured number, ask for
the before/after comparison in the same breath. The brief calls an
unmeasured improvement "a preference" — the repo rule exists so that Claude
Code enforces it on your behalf.

## What not to delegate

The findings section. Claude Code can compute your numbers, tabulate them
and check your arithmetic, but the confounds and the residual are the
argument you are being marked on.
