"""Preserve a separate, byte-exact evidence supplement without touching old packages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports/seeded-tracker"
OUTPUT = ROOT / "benchmarks/seeded-dense-results"
FILES = {
    **{
        f"patchnet-v1/{name}": f"patchnet/{name}"
        for name in (
            "contract.json",
            "inputs.json",
            "source.json",
            "training-data.json",
            "training-result.json",
            "model.pt",
            "mask-fix-real-input-parity.json",
            "source/experiment_patchnet_dense.py",
            "source/patchnet_model.py",
            "independent-review/review.json",
            "independent-review/audit.py",
            "diagnosis-v1/diagnosis.json",
            "diagnosis-v1/executed-diagnosis.py",
            "diagnosis-v1/layer2-comparison.png",
            "diagnosis-v1/layer3-comparison.png",
            "jamshoro-predictions/evaluation.json",
            "jamshoro-predictions/prediction-manifest.json",
            "jamshoro-predictions/classical-layer2.npz",
            "jamshoro-predictions/classical-layer3.npz",
            "jamshoro-predictions/learned-layer2.npz",
            "jamshoro-predictions/learned-layer3.npz",
        )
    },
    **{
        f"patchnet-v2-mask/{name}": f"patchnet/reproduction/{name}"
        for name in (
            "contract.json",
            "source.json",
            "training-parity.json",
            "reproduction-parity.json",
            "source/experiment_patchnet_dense.py",
            "source/patchnet_model.py",
        )
    },
    **{
        f"dense-pairwise-v1/mandiali/{name}": f"pairwise/{name}"
        for name in (
            "contract.json",
            "summary.json",
            "executed-summary.py",
            "classical.json",
            "pairwise.json",
            "classical.npz",
            "pairwise.npz",
            "layer2-accepted-comparison.png",
            "layer3-accepted-comparison.png",
            "source/experiment_dense_pairwise.py",
            "source/dense_pairwise_model.py",
            "source/experiment_seeded_challenger.py",
            "source/seeded_challenger.py",
        )
    },
    "dense-contract-tests-final.log": "contract-tests.log",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(output, original=False):
    index = json.loads((output / "index.json").read_text())
    for entry in index["files"]:
        path = output / entry["file"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Packaged bytes differ: {path}")
        if entry["transformation"] == "gzip":
            digest = hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest()
            if digest != entry["original_sha256"]:
                raise ValueError(f"Decompressed source differs: {path}")
        elif entry["sha256"] != entry["original_sha256"]:
            raise ValueError(f"Copied source differs: {path}")
        if original and sha(ROOT / entry["original"]) != entry["original_sha256"]:
            raise ValueError(f"Original source differs: {entry['original']}")
    if sha(output / ".gitattributes") != index["attributes_sha256"]:
        raise ValueError("Git byte-preservation attributes differ")
    return {
        "verified_files": len(index["files"]),
        "index_sha256": sha(output / "index.json"),
        "original_sources_verified": original,
        "claim": "Byte integrity against the index; inference is not rerun",
    }


def package(output):
    if output.exists():
        raise ValueError("Preserve existing evidence package")
    parity = json.loads((EXPORTS / "patchnet-v2-mask/reproduction-parity.json").read_text())
    if not parity["success"]:
        raise ValueError("Fresh-run parity is required before packaging")
    output.mkdir(parents=True)
    entries = []
    for source, target in FILES.items():
        original, destination = EXPORTS / source, output / target
        digest = sha(original)
        compress = original.name in ("evaluation.json", "classical.json", "pairwise.json")
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
        "schema": "seeded-dense-negative-evidence-v1",
        "claim": "Previously used development interpretation only; both mechanisms rejected",
        "limitations": [
            "No product accuracy or physical thickness claim",
            "Training arrays and dense CNN evidence maps remain local, with saved hashes",
            "The legacy packet experiment retains its original uncalibrated gate",
            "Missing legacy correspondence instrumentation is not measured zero survival",
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
