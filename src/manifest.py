"""Record the settings each stage actually used, in runs/<run>/run_manifest.json.

config.yaml records what we ASKED for. This file records what we GOT: the
resolved model commit, the precision, the merged decoding settings, library
versions, and any setting a stage had to override (e.g. the dummy backend
forcing raw prompts). The two can differ, and the difference is the point.
"""

import hashlib
import json
import pathlib


def update_manifest(run_dir, section, data):
    """Write one stage's section into the manifest, keeping the others."""
    path = pathlib.Path(run_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest[section] = data
    # default=str: some library objects (dtypes, versions) are not JSON types.
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def file_sha256(path):
    """Fingerprint of a data file, so two runs can prove they used the same one."""
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def refuse_overwrite(path):
    """Stop if a raw artifact already exists. Runs are never overwritten."""
    if pathlib.Path(path).exists():
        raise SystemExit(f"{path} already exists - refusing to overwrite. "
                         "Start a new run with ./run.sh instead.")
