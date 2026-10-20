#!/usr/bin/env python3
"""Check the machine before spending a GPU slot on a real run.

A queued job that dies two minutes in because NLTK data is missing costs more
than the thirty seconds this takes. Run it on the GPU node, inside the venv:

    .venv/bin/python scripts/preflight.py --config configs/qwen05b.yaml

Every check prints one line. FAIL means the run would break or would break a
project rule; WARN means it would still run, but not as specified.
Exit code 0 = safe to launch, 1 = something FAILed.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

failures = 0
warnings = 0


def ok(msg):
    print(f"  [ok]   {msg}")


def warn(msg):
    global warnings
    warnings += 1
    print(f"  [WARN] {msg}")


def fail(msg):
    global failures
    failures += 1
    print(f"  [FAIL] {msg}")


def section(name):
    print(f"\n{name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/qwen05b.yaml")
    args = ap.parse_args()

    config_path = (ROOT / args.config) if not pathlib.Path(args.config).is_absolute() \
        else pathlib.Path(args.config)

    section("Python")
    version = ".".join(map(str, sys.version_info[:3]))
    # The official scorer annotates dataclass fields as `list[str]`, which only
    # Python 3.9+ can evaluate; on 3.8 it crashes when it is imported, after
    # the GPU has already spent ~15 minutes generating.
    if sys.version_info < (3, 9):
        fail(f"Python {version} - the official scorer needs 3.9+. Make the venv with:"
             " uv venv --python 3.10.14 --seed .venv")
    else:
        ok(f"Python {version} ({sys.executable})")

    section("Python packages")
    try:
        import yaml
        ok(f"PyYAML {yaml.__version__}")
    except ImportError:
        fail("PyYAML missing - pip install -r requirements.txt")
        print("\nStopping: cannot read the config without PyYAML.")
        return 1

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))

    for module, label in [("torch", "torch"), ("transformers", "transformers"),
                          ("absl", "absl-py"), ("langdetect", "langdetect"),
                          ("nltk", "nltk"), ("immutabledict", "immutabledict")]:
        try:
            m = __import__(module)
            ok(f"{label} {getattr(m, '__version__', 'installed')}")
        except ImportError:
            fail(f"{label} missing - pip install -r requirements.txt")

    import nltk
    if nltk.__version__ != "3.8.1":
        warn(f"nltk is {nltk.__version__}; the scorer needs the legacy punkt "
             "pickle that only 3.8.1 loads (sentence-count checks will break)")

    section("GPU")
    try:
        import torch
    except ImportError:
        print("\nStopping: torch missing.")
        return 1

    if not torch.cuda.is_available():
        fail("CUDA not available - generate.py --backend hf needs a GPU")
        # Most common cause on the department DGX: a CUDA 11 driver with the
        # default CUDA 12 torch wheel. torch's own warning above says
        # "driver ... too old" when that is the case.
        if torch.version.cuda and torch.version.cuda.startswith("12"):
            print("         torch is built for CUDA 12. If `nvidia-smi` shows CUDA Version 11.x,"
                  " install the CUDA 11.8 build:\n"
                  "         pip uninstall -y torch && pip install torch==2.4.1"
                  " --index-url https://download.pytorch.org/whl/cu118")
    else:
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        ok(f"{name}, compute capability {major}.{minor}, {total_gb:.1f} GiB")
        # Rule 6: the project targets a V100 (Volta, 7.0).
        if (major, minor) != (7, 0):
            warn(f"this is not a V100 (expected compute capability 7.0, got "
                 f"{major}.{minor}) - record the difference in the report")
        if major >= 8:
            ok("bf16 is supported here but the config stays fp16, as specified")
        ok(f"torch {torch.__version__}, CUDA {torch.version.cuda}")

    section("Project rules")
    generate_src = (ROOT / "src" / "generate.py").read_text(encoding="utf-8")
    if 'attn_implementation="sdpa"' in generate_src:
        ok('generate.py uses attn_implementation="sdpa"')
    else:
        fail('generate.py must use attn_implementation="sdpa" (rule 6)')
    if "flash_attention_2" in generate_src:
        fail("generate.py mentions flash_attention_2 - not supported on Volta (rule 6)")
    if "torch.float16" in generate_src:
        ok("generate.py loads the model in fp16")
    else:
        fail("generate.py should load fp16 - the V100 has no bf16 (rule 6)")

    revision = str(cfg["model"]["revision"])
    if revision == "main":
        fail("model.revision is 'main' - pin a commit hash, or the run is not "
             "reproducible (rule 3)")
    else:
        ok(f"model revision pinned to {revision[:12]}")

    section("Data and scorer")
    data_path = ROOT / cfg["data"]["ifeval_path"]
    if not data_path.exists():
        fail(f"{data_path.relative_to(ROOT)} missing - run ./scripts/setup.sh")
    else:
        lines = data_path.read_text(encoding="utf-8").strip().splitlines()
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()[:12]
        if len(lines) == 541:
            ok(f"{len(lines)} IFEval prompts, sha256 {digest}")
        else:
            fail(f"expected 541 prompts, found {len(lines)}")

    scorer_dir = ROOT / cfg["scoring"]["scorer_repo"]
    if (scorer_dir / "evaluation_main.py").exists():
        commit_file = scorer_dir.parent / "SCORER_COMMIT"
        commit = commit_file.read_text().strip()[:12] if commit_file.exists() else "unknown"
        ok(f"official scorer present, commit {commit}")
    else:
        fail(f"scorer missing at {scorer_dir.relative_to(ROOT)} - run ./scripts/setup.sh")

    punkt = ROOT / cfg["scoring"]["nltk_data"] / "tokenizers" / "punkt"
    if punkt.exists():
        ok("NLTK punkt data present")
    else:
        fail(f"{punkt.relative_to(ROOT)} missing - run ./scripts/setup.sh")

    section("Model weights")
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(cfg["model"]["id"], revision=revision,
                                 local_files_only=True)
        ok(f"weights already cached at {path}")
    except Exception:
        warn("weights not in the local cache - the run will download them "
             "first (about 1 GB for the 0.5B model)")

    section("Disk")
    free_gb = shutil.disk_usage(ROOT).free / 2**30
    if free_gb < 5:
        fail(f"only {free_gb:.1f} GiB free - weights plus run outputs need a few GiB")
    else:
        ok(f"{free_gb:.1f} GiB free")

    print(f"\n{'-' * 60}")
    if failures:
        print(f"{failures} FAIL, {warnings} WARN - fix the failures before submitting.")
        return 1
    print(f"0 FAIL, {warnings} WARN - safe to launch "
          f"({cfg['model']['id']}, batch {cfg['runtime']['batch_size']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
