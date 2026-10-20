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
scripts/
  setup.sh          fetches prompts, scorer and NLTK data at a pinned commit
  preflight.py      checks a machine before a run (GPU, data, project rules)
  v100.sbatch       SLURM submit script for the department V100
  make_pdf.py       renders report.md to report.pdf
configs/*.yaml           every knob that can change a number
src/
  build_prompts.py  applies the chat template, saves the exact string
  generate.py       runs the model          <- the only GPU step
  score.py          calls the official scorer
  analyze.py        metrics, intervals, per-type, length, truncation
  manifest.py       writes run_manifest.json (what each stage really used)
data/, external/         created by setup.sh, not committed
runs/                    one directory per run, never overwritten
reference/gpt4_paper/    the paper's GPT-4 responses, scored by this harness
report/
  report.md         the write-up
  report.pdf        the same, rendered for handing in
  evaluation_figures.ipynb  every chart, computed from runs/
  figures/          the charts as 200 dpi PNGs, written by the notebook
```

### Figures

```bash
pip install -r requirements-notebook.txt
jupyter lab report/evaluation_figures.ipynb   # or: jupyter nbconvert --execute --inplace ...
```

The notebook reads `runs/` directly and rewrites `report/figures/*.png`. The
only hand-entered numbers in it are the published reference values, each cited.

### PDF

```bash
python3 scripts/make_pdf.py        # report/report.md + figures -> report/report.pdf
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
`*_laptop*` configs produced the first round of results (report/report.md);
the V100 runs use the batch-8 configs below.

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

## Deploying to the department V100

The department GPUs are booked one card at a time (for example
`10.204.100.192 [DGX-GPU 1]`) and used over SSH. The hosts have internet
access, and Python is a plain `python3 -m venv`.

**1. Once, on the GPU host**

The DGX's system Python is 3.8, but the official scorer needs **3.9+** (it
annotates fields as `list[str]`, which 3.8 cannot load). `uv` installs the same
Python the laptop runs, 3.10.14, into your home directory with no admin rights:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/runjanna26/module_3_evaluation.git
cd module_3_evaluation
uv venv --python 3.10.14 --seed .venv      # --seed adds pip, which run.sh's snapshot uses
.venv/bin/pip install -r requirements.txt
./scripts/setup.sh                          # uses .venv automatically
```

**2. Run the whole schedule on your booked card, inside tmux**

```bash
tmux new -s ifeval
./scripts/run_schedule.sh 1          # 1 = the GPU number you booked
# detach with Ctrl-b then d; come back with: tmux attach -t ifeval
```

`run_schedule.sh` pins the job to that one physical card (numbered by PCI bus,
the same way the booking list and `nvidia-smi` number them), warns if another
job is already on it, runs preflight for every config, then the 9 runs one
after another. Expect about 3 hours. Everything is logged to `logs/`, and a
failed run is recorded without cancelling the rest. To run only part of it:
`./scripts/run_schedule.sh 1 repeats` (or `sysprompt`, `scale`).

### If the cluster uses SLURM instead

`scripts/v100.sbatch` is for clusters that queue jobs. Create the log
directory first (`mkdir -p logs`; SLURM will not), then:

**Edit the three cluster-specific lines** at the top of
`scripts/v100.sbatch`: `--partition`, `--gres` and `--time`.
`sinfo -o "%P %G"` prints the partitions and their GPUs.

**Check a GPU node before queueing real work**

```bash
srun --partition=gpu --gres=gpu:v100:1 --pty \
  .venv/bin/python scripts/preflight.py --config configs/qwen05b.yaml
```

It checks the GPU is a V100 (compute capability 7.0), that `generate.py` is
fp16 + SDPA with no FlashAttention-2, that prompts, scorer and NLTK data are
in place, and that the model revision is pinned rather than `main`. It exits
non-zero on anything that would waste the slot.

**Submit**

```bash
sbatch scripts/v100.sbatch configs/qwen05b.yaml                  # one run
sbatch --array=1-3 scripts/v100.sbatch configs/qwen05b.yaml      # three repeats
```

Each task runs preflight first, then `run.sh`. The SLURM job id is appended to
the run directory name (`RUN_SUFFIX`), so array tasks starting in the same
second cannot collide. Job output lands in `logs/`.

**Worth knowing**

- **Model cache:** the sbatch script points `HF_HOME` at `.hf_cache/` inside the
  project, because `$HOME` quotas are usually small. Override it if your
  cluster has a scratch area.
- **Time budget:** 541 prompts took 18.6 minutes on a 4 GB laptop GPU at
  batch 4. A V100 at batch 8 should be that or faster; `--time=02:00:00`
  leaves room.
- **Out of memory:** lower `runtime.batch_size`. That is a config change that
  can move the numbers, and it is recorded as one.
- **Provenance:** `run_manifest.json` records the model commit actually
  loaded, the GPU name and compute capability, and the merged decoding
  settings, so a V100 run is distinguishable from a laptop run afterwards.

## Experiment schedule

On the booked DGX card, `./scripts/run_schedule.sh 1` runs the first three
rows. The group name runs one row on its own.

| Runs | Group | Config | Purpose |
|---|---|---|---|
| 3 | `repeats` | `qwen05b.yaml` | within-config spread, and the laptop-vs-V100 comparison |
| 3 | `sysprompt` | `qwen05b_sys_{generic,strict,casual}.yaml` | prompt sensitivity |
| 3 | `scale` | `qwen15b.yaml` | scale comparison (Qwen2.5-1.5B) |
| 1 | — | `template_mode: raw` | template ablation (optional credit) |

The V100 configs mirror the laptop ones at `batch_size: 8`, so each V100 run
has a laptop counterpart to compare against. On the laptop, 4 identical runs
produced byte-identical generations, so any difference the repeats show on the
V100 is hardware or batch size, not run-to-run noise.

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
