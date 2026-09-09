"""Preserve completed followups in a separate byte-verified evidence package."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

from package_dense_experiments import EXPORTS, ROOT, sha, verify

OUTPUT = ROOT / "benchmarks/seeded-followup-results"


def files():
    selected = {
        f"dense-followups-v1/{name}": name
        for name in ("summary.json", "executed-summary.py", "intervention-curves.png")
    }
    for folder, prefix in (
        ("dense-replay-v1/mandiali", "replay"),
        ("dense-peak-modes-v1/mandiali", "peak"),
        ("dense-relative-gap-v1/mandiali", "gap"),
    ):
        for path in (EXPORTS / folder).rglob("*"):
            if path.is_file() and path.suffix in (".json", ".npz", ".png", ".py"):
                selected[path.relative_to(EXPORTS).as_posix()] = (
                    f"{prefix}/{path.relative_to(EXPORTS / folder).as_posix()}"
                )
    for name in (
        "contract.json",
        "inputs.json",
        "source.json",
        "sampling-source.json",
        "training-data.json",
        "training-result.json",
        "preprediction-parity.json",
        "model.pt",
        "source/experiment_patchnet_far_negatives.py",
        "source/experiment_patchnet_dense.py",
        "source/executed-prepare.py",
        "source/patchnet_model.py",
        "source/seeded_eval.py",
        "independent-review/audit.py",
        "independent-review/review.json",
        "independent-review/tests.log",
        "diagnosis/diagnosis.json",
        "diagnosis/layer2-comparison.png",
        "diagnosis/layer3-comparison.png",
        "jamshoro-predictions/evaluation.json",
        "jamshoro-predictions/prediction-manifest.json",
        "jamshoro-predictions/classical-layer2.npz",
        "jamshoro-predictions/classical-layer3.npz",
        "jamshoro-predictions/learned-layer2.npz",
        "jamshoro-predictions/learned-layer3.npz",
    ):
        selected[f"patchnet-v3-far-negatives/{name}"] = f"far-negatives/{name}"
    return selected


def package(output):
    if output.exists():
        raise ValueError("Preserve existing evidence package")
    output.mkdir(parents=True)
    entries = []
    for source, target in files().items():
        original, destination = EXPORTS / source, output / target
        digest = sha(original)
        compress = original.suffix == ".json" and original.stat().st_size > 200_000
        if compress:
            destination = destination.with_suffix(destination.suffix + ".gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            destination.write_bytes(gzip.compress(original.read_bytes(), mtime=0))
        else:
            shutil.copy2(original, destination)
        if sha(original) != digest:
            raise ValueError(f"Source changed while copying: {source}")
        entries.append(
            {
                "file": destination.relative_to(output).as_posix(),
                "original": original.relative_to(ROOT).as_posix(),
                "original_sha256": digest,
                "transformation": "gzip" if compress else "copy",
                "sha256": sha(destination),
                "bytes": destination.stat().st_size,
            }
        )
    attributes = output / ".gitattributes"
    attributes.write_bytes(b"* -text whitespace=cr-at-eol\n")
    index = {
        "schema": "seeded-dense-followups-evidence-v1",
        "claim": "Four rejected development experiments; no reliable tracker or thickness claim",
        "limitations": [
            "No original raw radar or training arrays are distributed",
            "Package source is available at recorded Git commit c442b60",
            "Model evidence maps remain local; their hashes are recorded",
            "Copied execution snapshots can retain historical formatting violations",
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
    result = verify(args.output, args.originals) if args.verify else package(args.output)
    print(json.dumps(result, indent=2))
