"""Snapshot the exact research executable before a queued training/evaluation job."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new immutable source snapshot directory")
    paths = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")])
    paths += sorted((ROOT / "benchmarks").glob("seeded-*-seeds.json"))
    paths += [ROOT / "benchmarks/seeded-evaluation-inputs.json", ROOT / "pyproject.toml"]
    files = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    output.mkdir(parents=True)
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    if any(digest(ROOT / p) != value or digest(output / p) != value
           for p, value in files.items()):
        raise RuntimeError("Source changed while copying; this incomplete snapshot must not run")
    manifest = {
        "schema": "processed-research-executable-snapshot-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "origin": str(ROOT),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=normal"], cwd=ROOT, text=True
        ).splitlines(),
        "source_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": files,
        "note": "Includes uncommitted research files. Source data and weights are external.",
    }
    (output / "executable-snapshot.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = freeze(args.output)
    print(json.dumps({"output": args.output, "source_sha256": result["source_sha256"]}))
