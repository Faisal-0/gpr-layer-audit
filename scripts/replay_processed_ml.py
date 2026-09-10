"""Explicit processed-data research replay; no application/default dispatch change.

Example (frozen Gujrat observations, same source/model for all policy comparisons)::

    python scripts/replay_processed_ml.py --manifest CACHE/manifest.json \
      --record-id gujrat-087182ae3b6f --checkpoint MODEL_DIRECTORY \
      --case gujrat-second --layer 2 --stride 4 --policy uncertainty \
      --operation local_correction --output RUN/local-uncertainty.json

Run all three policies and both operations at identical model, seeds and stride.
The default 0/1/2/4 budget prefixes are measured in one persisted run. This command
freezes package/script sources before execution and validates cached/native input,
frozen observations, scorer and model fingerprints. Resume only trusted local
checkpoints; the established replay serializer uses pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files():
    paths = list((ROOT / "src").rglob("*.py"))
    paths += [
        ROOT / "scripts" / name
        for name in ("replay_processed_ml.py", "replay_seeded_interaction.py", "seeded_eval.py")
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frozen_sources(args):
    """Snapshot just source and frozen seed manifests, never raw data/checkpoints."""
    files = source_files()
    source_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if args.frozen_source:
        contract = read(ROOT / "source-manifest.json")
        if contract["files"] != files or contract["source_sha256"] != source_hash:
            raise ValueError("Frozen replay source snapshot changed")
        return source_hash
    if args.resume:
        checkpoint = args.output.with_suffix(".checkpoint")
        with checkpoint.open("rb") as stream:
            header = json.loads(stream.readline())
        snapshot = Path(header["contract"]["provenance"]["source_snapshot"])
        if not (snapshot / "scripts/replay_processed_ml.py").is_file():
            raise ValueError("Original replay source snapshot is unavailable")
    else:
        snapshot = args.output.parent / ("replay-source-" + source_hash[:16])
        if not snapshot.exists():
            snapshot.mkdir(parents=True)
            for relative in files:
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, destination)
            for path in (ROOT / "benchmarks").glob("seeded-*-seeds.json"):
                destination = snapshot / "benchmarks" / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            shutil.copy2(
                ROOT / "benchmarks/seeded-evaluation-inputs.json",
                snapshot / "benchmarks/seeded-evaluation-inputs.json",
            )
            (snapshot / "source-manifest.json").write_text(
                json.dumps({"source_sha256": source_hash, "files": files}, indent=2),
                encoding="utf-8",
            )
        if source_files() != files:
            raise ValueError(
                "Package changed while freezing research replay; rerun after edits finish"
            )
    command = [
        sys.executable,
        str(snapshot / "scripts/replay_processed_ml.py"),
        *sys.argv[1:],
        "--frozen-source",
    ]
    completed = subprocess.run(command, check=False, cwd=Path.cwd())
    raise SystemExit(completed.returncode)


def load_anchors(args, record):
    """Only an explicit native seed artifact supplies the initial operating answers."""
    if args.case:
        cases = read(ROOT / "benchmarks/seeded-evaluation-inputs.json")["cases"]
        case = cases[args.case]
        if (
            case["dzt_sha256"] != record["source_sha256"]
            or case["dzx_sha256"] != record["label_sha256"]
        ):
            raise ValueError("Frozen case does not match the processed source/reference")
        path = ROOT / case["seed_source"]
        if digest(path) != case["seed_sha256"]:
            raise ValueError("Frozen seed source fingerprint changed")
    else:
        path = args.seeds
    seeds = read(path)
    if seeds.get("mode") != "processed" or seeds.get("dzt_sha256") != record["source_sha256"]:
        raise ValueError("Seed artifact must identify this exact processed source")
    if seeds.get("dzx_sha256") != record["label_sha256"]:
        raise ValueError("Seed artifact reference fingerprint mismatch")
    points = seeds["observations"].get(str(args.layer), [])
    if not points:
        raise ValueError("No explicit initial observations for the requested layer")
    anchors = {}
    for point in points:
        trace, sample = point["trace"], point["sample"]
        if int(trace) != trace or trace % args.stride:
            raise ValueError("Seed/native grid mismatch; snapping is prohibited")
        if not isinstance(sample, (int, float)) or not 0 <= sample <= record["shape"][1] - 1:
            raise ValueError("Seed sample is outside the native temporal grid")
        if point.get("channel", 0) != 0:
            raise ValueError("Only coordinate-audited channel zero is supported")
        row = int(trace) // args.stride
        if row in anchors:
            raise ValueError("Duplicate initial seed coordinate")
        anchors[row] = sample
    return {args.layer: anchors}, digest(path)


def run(args):
    source_hash = frozen_sources(args)
    # These imports and model loading are confined to this explicit research CLI.
    import numpy as np
    from seeded_eval import FROZEN_HELPER_LF_SHA256

    from gpr_layer_audit.ml.processed_data import (
        load_processed_manifest,
        open_processed_record,
    )
    from gpr_layer_audit.ml.processed_inference import ProcessedPredictor
    from gpr_layer_audit.ml.processed_ml_replay import ReplayConfig, replay_frozen_model
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse

    for name, expected in FROZEN_HELPER_LF_SHA256.items():
        data = (ROOT / "src/gpr_layer_audit" / name).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen scoring helper changed: {name}")
    loaded = open_processed_record(args.manifest, args.record_id, verify_hashes=True)
    record = loaded["record"]
    source_path = Path(record["path"])
    if digest(source_path) != record["source_sha256"]:
        raise ValueError("Native processed source fingerprint changed")
    manifest = load_processed_manifest(args.manifest)
    anchors, seed_hash = load_anchors(args, record)
    measurement = loaded["amplitudes"][:: args.stride]
    valid = loaded["sample_validity"][:: args.stride]
    dt, dx = record["sample_interval_ns"], record["horizontal_step_m"] * args.stride
    predictor_input = {
        key: loaded[key]
        for key in (
            "record",
            "amplitudes",
            "sample_validity",
            "native_trace_indices",
            "native_sample_indices",
            "distances_m",
        )
    }
    predictor = ProcessedPredictor.from_checkpoint(
        args.checkpoint,
        predictor_input,
        device=args.device,
        trace_stride=args.stride,
        acceptance_threshold=args.acceptance_threshold,
    )
    model_hash = digest(predictor.weights_path)
    if model_hash != predictor.provenance["model_sha256"]:
        raise ValueError("Model best-weights fingerprint changed during loading")
    label_values = loaded["labels"][args.layer - 1, :: args.stride]
    reviewed = loaded["label_valid"][args.layer - 1, :: args.stride]
    # Labels remain at this evaluator/answer boundary; never passed to .predict.
    references = {
        args.layer: [
            SimpleNamespace(trace=int(row), sample=float(label_values[row]))
            for row in np.flatnonzero(reviewed)
        ]
    }
    pulse = {
        args.layer: resolve_pulse(
            measurement, valid, anchors[args.layer], {}, dt, ConventionalConfig()
        ).lobe_samples
    }
    preprocess = {
        "dataset_schema": manifest["schema"],
        "dataset_preprocessing_version": manifest["preprocessing_version"],
        "sample_interval_ns": dt,
        "native_horizontal_step_m": record["horizontal_step_m"],
        "time_origin_ns": record["time_origin_ns"],
        "trace_stride": args.stride,
        "sample_validity_sha256": record["array_sha256"]["sample_validity"],
    }
    provenance = {
        "coordinate_mode": "processed",
        "input_sha256": record["source_sha256"],
        "reference_sha256": record["label_sha256"],
        "model_sha256": model_hash,
        "acceptance_threshold": (
            args.acceptance_threshold if np.isfinite(args.acceptance_threshold) else None
        ),
        "acceptance_policy": (
            "explicit research threshold; caller must select on training-road inner validation"
            if np.isfinite(args.acceptance_threshold)
            else "withhold autonomous rows; no validated threshold supplied"
        ),
        "preprocessing_sha256": hashlib.sha256(
            json.dumps(preprocess, sort_keys=True).encode()
        ).hexdigest(),
        "preprocessing": preprocess,
        "predictor": predictor.provenance,
        "source_sha256": source_hash,
        "source_snapshot": str(ROOT.resolve()),
        "trace_stride": args.stride,
        "initial_seeds_sha256": seed_hash,
        "record_id": record["record_id"],
        "physical_road_group": record["physical_road_group"],
        "model_path": str(args.checkpoint.resolve()),
        "manifest_sha256": digest(args.manifest),
        "scoring_helpers": FROZEN_HELPER_LF_SHA256,
    }
    result = replay_frozen_model(
        predictor.predict,
        measurement,
        valid,
        anchors,
        references,
        [
            LayerSpec(
                args.layer,
                ("Asphalt", "Base", "Subbase")[args.layer - 1],
                1,
                measurement.shape[1] - 1,
                1,
            )
        ],
        dt_ns=dt,
        dx_m=dx,
        pulse_samples=pulse,
        provenance=provenance,
        output=args.output,
        config=ReplayConfig(
            policy=args.policy,
            operation=args.operation,
            budgets=tuple(args.budgets),
            fixed_spacing_slots=max(4, max(args.budgets)),
        ),
        resume=args.resume,
    )
    if digest(predictor.weights_path) != model_hash:
        raise ValueError("Model checkpoint changed during frozen replay")
    summary = [
        {
            "requests": step["additional_requests"],
            "actions_per_km": step["actions_per_km"],
            "accepted_agreement": step["layers"][str(args.layer)]["accepted_pick_agreement"],
            "correct_automatic_coverage": step["layers"][str(args.layer)][
                "correct_automatic_coverage"
            ],
            "wrong_automatic_coverage": step["layers"][str(args.layer)]["wrong_automatic_coverage"],
        }
        for step in result["steps"]
        if step["reported_budget"]
    ]
    print(json.dumps({"output": str(args.output), "steps": summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=float("inf"),
        help="Research threshold fixed on inner validation; default withholds automation",
    )
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--case", choices=("gujrat-second", "mandiali-short"))
    seeds.add_argument("--seeds", type=Path)
    parser.add_argument("--layer", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--policy",
        choices=("uncertainty", "fixed_spacing", "largest_interval_midpoint"),
        default="uncertainty",
    )
    parser.add_argument(
        "--operation", choices=("local_correction", "global_model_seed"), default="local_correction"
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume trusted local replay checkpoint"
    )
    parser.add_argument("--frozen-source", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    if args.budgets != sorted(set(args.budgets)) or not args.budgets or args.budgets[0] != 0:
        parser.error("--budgets must be unique ascending prefixes beginning with 0")
    run(args)


if __name__ == "__main__":
    main()
