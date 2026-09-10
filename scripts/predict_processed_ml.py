"""Predict from an explicit processed research checkpoint and native seed artifact.

Example::

    python scripts/predict_processed_ml.py --manifest CACHE/manifest.json \
      --record-id gujrat-087182ae3b6f --checkpoint RUN/model \
      --case gujrat-second --stride 4 --device cpu --output RUN/predictions.npz

The default withholds autonomous measurements until a separately selected
training-validation acceptance threshold is explicitly supplied. Full direct
depth probabilities and provisional dense paths remain available for research.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_seeds(args, record):
    """Resolve only explicit observations; never infer seeds from label arrays."""
    if args.case:
        case = read(ROOT / "benchmarks/seeded-evaluation-inputs.json")["cases"][args.case]
        if (
            case["dzt_sha256"] != record["source_sha256"]
            or case["dzx_sha256"] != record["label_sha256"]
        ):
            raise ValueError("Frozen case does not match this processed acquisition/reference")
        path = ROOT / case["seed_source"]
        if digest(path) != case["seed_sha256"]:
            raise ValueError("Frozen seed artifact fingerprint changed")
    else:
        path = args.seeds
    seeds = read(path)
    if seeds.get("mode") != "processed" or seeds.get("dzt_sha256") != record["source_sha256"]:
        raise ValueError("Seeds must identify this exact native processed source")
    if seeds.get("dzx_sha256") != record["label_sha256"]:
        raise ValueError("Seed reference fingerprint mismatch")
    anchors = {}
    for key, points in seeds["observations"].items():
        layer = int(key)
        if layer not in args.layers:
            continue
        selected = {}
        for point in points:
            trace, sample = point["trace"], float(point["sample"])
            if int(trace) != trace or trace % args.stride:
                raise ValueError(
                    "Seed trace is absent from the retained grid; snapping is prohibited"
                )
            if point.get("channel", 0) != 0:
                raise ValueError("Only native channel-zero seed observations are supported")
            row = int(trace) // args.stride
            if row in selected:
                raise ValueError("Duplicate initial seed coordinate")
            selected[row] = sample
        if selected:
            anchors[layer] = selected
    if not anchors:
        raise ValueError("Seed artifact contains no observations for requested layers")
    return anchors, {"path": str(Path(path).resolve()), "sha256": digest(path)}


def run(args):
    import numpy as np

    from gpr_layer_audit.ml.processed_data import load_processed_manifest, open_processed_record
    from gpr_layer_audit.ml.processed_inference import RADAR_KEYS, ProcessedPredictor

    output = args.output.resolve()
    if output.suffix.lower() != ".npz":
        if output.exists():
            raise FileExistsError("Use a new immutable prediction output directory")
        output = output / "predictions.npz"
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError("Prediction artifact already exists; choose a new immutable output")
    manifest = load_processed_manifest(args.manifest)
    loaded = open_processed_record(args.manifest, args.record_id, verify_hashes=True)
    record = loaded["record"]
    if digest(record["path"]) != record["source_sha256"]:
        raise ValueError("Native processed DZT fingerprint changed")
    anchors, seed_source = load_seeds(args, record)
    predictor = ProcessedPredictor.from_checkpoint(
        args.checkpoint,
        {key: loaded[key] for key in RADAR_KEYS},
        device=args.device,
        trace_stride=args.stride,
        acceptance_threshold=args.acceptance_threshold,
    )
    script_hash = digest(__file__)
    predictions = predictor.predict(anchors)
    arrays = {
        "native_trace_indices": predictor.native_trace_indices,
        "native_sample_indices": loaded["native_sample_indices"],
        "distances_m": predictor.distances_m,
        "sample_validity": predictor.valid,
        "sample_interval_ns": np.asarray(predictor.dt_ns),
        "time_origin_ns": np.asarray(record["time_origin_ns"]),
        "trace_stride": np.asarray(args.stride),
    }
    for layer, path in predictions.items():
        arrays.update(
            {
                f"probabilities_layer{layer}": path.feature,
                f"samples_layer{layer}": path.samples,
                f"proposed_samples_layer{layer}": path.provisional_samples,
                f"accepted_layer{layer}": path.visible,
                f"confidence_layer{layer}": path.confidence,
                f"manual_seed_layer{layer}": path.evidence["manual_seed"],
            }
        )
    source_hashes = predictor.provenance["source_hashes"]
    package = ROOT / "src/gpr_layer_audit/ml"
    if any(digest(package / name) != expected for name, expected in source_hashes.items()):
        raise ValueError(
            "Inference source changed during prediction; freeze source before rerunning"
        )
    if (
        digest(__file__) != script_hash
        or digest(predictor.weights_path) != predictor.weights_sha256
    ):
        raise ValueError("Predictor script or frozen best model weights changed during prediction")
    metadata = {
        "schema": "processed-ml-predictions-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "record_id": record["record_id"],
        "source_sha256": record["source_sha256"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": digest(args.manifest),
        "dataset_preprocessing_version": manifest["preprocessing_version"],
        "native_arrays_sha256": {key: record["array_sha256"][key] for key in RADAR_KEYS[1:]},
        "model_sha256": predictor.weights_sha256,
        "preprocessing_sha256": predictor.provenance["preprocessing_sha256"],
        "inference_source_sha256": predictor.provenance["source_sha256"],
        "script_sha256": script_hash,
        "seed_source": seed_source,
        "layers": {str(layer): path.provenance for layer, path in predictions.items()},
        "device": predictor.device,
        "production_eligible": False,
        "accepted_claim": "research proposals; threshold is not calibrated correctness probability",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as target:
        np.savez_compressed(target, **arrays)
    metadata["prediction_npz_sha256"] = digest(output)
    with metadata_path.open("x", encoding="utf-8") as target:
        json.dump(metadata, target, indent=2, allow_nan=False)
        target.write("\n")
    print(
        json.dumps(
            {
                "predictions": str(output),
                "metadata": str(metadata_path),
                "model_sha256": predictor.weights_sha256,
                "layers": sorted(predictions),
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    seeds = parser.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seeds", type=Path)
    seeds.add_argument("--case", choices=("gujrat-second", "mandiali-short"))
    parser.add_argument("--layers", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3])
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--acceptance-threshold", type=float, default=float("inf"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    run(args)


if __name__ == "__main__":
    main()
