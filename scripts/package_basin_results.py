"""Create a separate immutable package for the basin and confidence experiments."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from package_dense_experiments import EXPORTS, ROOT, sha, verify

OUTPUT = ROOT / "benchmarks/seeded-basin-results"


def selected_files():
    root = EXPORTS / "dense-basin-emissions-v1"
    files = {
        p: f"basin/{p.relative_to(root).as_posix()}"
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".json", ".npz", ".png")
    }
    for name in ("mandiali", "gujrat", "replay", "gujrat-replay"):
        for p in (root / name / "source").glob("*.py"):
            files[p] = f"basin/{p.relative_to(root).as_posix()}"
    for folder, prefix in (
        ("jamshoro-guide-v1", "guide"),
        ("dense-geometry-confidence-v1/mandiali", "geometry"),
    ):
        for path in (EXPORTS / folder).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".npz", ".png", ".py"):
                files[path] = f"{prefix}/{path.relative_to(EXPORTS / folder).as_posix()}"
    files[root / "readout/executed-readout.py"] = "basin/readout/executed-readout.py"
    return files


def package(output):
    if output.exists():
        raise ValueError("Preserve existing package")
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
        "schema": "seeded-basin-negative-evidence-v1",
        "claim": "Development interpretation; all changes rejected for application promotion",
        "limitations": [
            "Reference observation counts do not represent independent road trials",
            "Corrected legacy switch field means wrong-signed-lobe observations, not events",
            "No raw radar is distributed; input hashes and local source paths are retained",
            "Application source is unchanged from the recorded Git base",
        ],
        "attributes_sha256": sha(attributes),
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
