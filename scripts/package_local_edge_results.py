"""Preserve trained local pairs, common-graph comparisons and actual replay."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from package_dense_experiments import EXPORTS, ROOT, sha, verify

OUTPUT = ROOT / "benchmarks/seeded-local-edge-results"


def selected_files():
    files = {}
    for directory, prefix in (
        ("local-patch-pairs-v1", "training"),
        ("local-patch-edges-v1", "experiments"),
    ):
        root = EXPORTS / directory
        for p in root.rglob("*"):
            if not p.is_file() or "jamshoro-interrupted-encode" in p.parts:
                continue
            if p.name == "training.npz" or p.suffix not in (
                ".json",
                ".npz",
                ".png",
                ".py",
                ".md",
                ".pt",
            ):
                continue
            if "__pycache__" in p.parts:
                continue
            files[p] = f"{prefix}/{p.relative_to(root).as_posix()}"
    return files


def package(output):
    if output.exists():
        raise ValueError("Preserve previous evidence package")
    output.mkdir(parents=True)
    entries = []
    for original, relative in selected_files().items():
        destination = output / relative
        digest = sha(original)
        compress = original.suffix == ".json" and original.stat().st_size > 200_000
        if compress:
            destination = destination.with_suffix(destination.suffix + ".gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            destination.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
        else:
            shutil.copy2(original, destination)
        assert sha(original) == digest
        entries.append(
            {
                "file": destination.relative_to(output).as_posix(),
                "original": original.relative_to(ROOT).as_posix(),
                "original_sha256": digest,
                "sha256": sha(destination),
                "bytes": destination.stat().st_size,
                "transformation": "gzip" if compress else "copy",
            }
        )
    attributes = output / ".gitattributes"
    attributes.write_bytes(b"* -text whitespace=cr-at-eol\n")
    index = {
        "schema": "seeded-local-edge-results-v1",
        "attributes_sha256": sha(attributes),
        "claim": "Development interpretation; local CNN and tail variant fail promotion",
        "limitations": [
            "More correct and wrong proposals; no useful accepted coverage at 95%",
            "Training pairs, embedding banks and dense edge tensors remain local with hashes",
            "Radar stays local; coordinate/source/model/configuration hashes are recorded",
            "Legacy switch field counts wrong-signed-lobe observations, not physical switch events",
            "Scoped correction replay is measured; no new model enters application dispatch",
        ],
        "files": entries,
    }
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return verify(output, original=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--originals", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.output, args.originals) if args.verify else package(args.output), indent=2
        )
    )
