"""Evaluate saved native dense probabilities with unchanged signed-lobe scoring.

Example:
  python scripts/evaluate_processed_ml.py --dataset-manifest MANIFEST
    --record-id gujrat-087182ae3b6f --probabilities evidence.npz --layer 3
    --seed-source benchmarks/seeded-gujrat-second-seeds.json --trace-stride 4
    --output results.json

Evidence must contain ``probabilities`` [retained native trace, native sample].
No weights are fitted here. Labels are loaded only after prediction evidence is
saved. Optional thresholds require an explicit TRAIN-only calibration artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_layer_audit.ml.processed_dense_decoder import DenseDecoderConfig  # noqa: E402
from gpr_layer_audit.ml.processed_dense_eval import (  # noqa: E402
    canonical_sha256,
    evaluate_evidence_stages,
    validate_evidence_provenance,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument(
        "--evidence-metadata",
        type=Path,
        help="Provenance JSON; default same evidence stem with .json",
    )
    parser.add_argument("--layer", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--seed-source", type=Path, required=True)
    parser.add_argument("--trace-stride", type=int, default=1)
    parser.add_argument(
        "--frozen-comparator",
        type=Path,
        help="Pinned original Gujrat/Mandiali control JSON for its exact tolerance",
    )
    parser.add_argument("--decoder-config", type=Path)
    parser.add_argument("--threshold-calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--omit-observations", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evaluation output exists; choose a new immutable artifact")
    if args.trace_stride < 1:
        raise ValueError("Trace stride must be positive")
    from gpr_layer_audit.ml.processed_data import load_processed_manifest, open_processed_record

    manifest = load_processed_manifest(args.dataset_manifest)
    records = manifest["records"]
    record = next(r for r in records if r["record_id"] == args.record_id)
    arrays = open_processed_record(args.dataset_manifest, record, verify_hashes=True)
    seeds_document = json.loads(args.seed_source.read_text(encoding="utf-8"))
    if seeds_document.get("schema") != "conventional-native-seeds-v1":
        raise ValueError("Native frozen seed manifest is required")
    if seeds_document.get("dzt_sha256") != record["source_sha256"]:
        raise ValueError("Seeds belong to another processed acquisition")
    observations = seeds_document["observations"][str(args.layer)]
    if any(p["trace"] % args.trace_stride or p["channel"] != 0 for p in observations):
        raise ValueError("Seed does not lie on retained native grid; snapping prohibited")
    anchors = {p["trace"] // args.trace_stride: p["sample"] for p in observations}
    if len(anchors) != len(observations):
        raise ValueError("Duplicate seed row")
    evidence_metadata_path = args.evidence_metadata or args.probabilities.with_suffix(".json")
    evidence_metadata = json.loads(evidence_metadata_path.read_text(encoding="utf-8"))
    original_evidence_metadata = evidence_metadata
    with np.load(args.probabilities, allow_pickle=False) as evidence:
        evidence_key = (
            f"probabilities_layer{args.layer}"
            if evidence_metadata.get("schema") == "processed-ml-predictions-v1"
            else "probabilities"
        )
        probabilities = np.asarray(evidence[evidence_key])
        if evidence_metadata.get("schema") == "processed-ml-predictions-v1":
            layer_provenance = evidence_metadata["layers"][str(args.layer)]
            weights_path = Path(layer_provenance["model_path"])
            model_path = weights_path.with_name("model.json")
            model_metadata = json.loads(model_path.read_text(encoding="utf-8"))
            if (
                sha(model_path) != layer_provenance["model_metadata_sha256"]
                or sha(weights_path) != evidence_metadata["model_sha256"]
                or model_metadata["weights_sha256"] != evidence_metadata["model_sha256"]
            ):
                raise ValueError("Prediction model/weights no longer match immutable provenance")
            if (
                layer_provenance["training_overlap"]
                or layer_provenance["allow_training_records"]
                or layer_provenance["preprocessing"]["sample_resampling"]
            ):
                raise ValueError("Expected strict grouped native-coordinate inference")
            recorded_anchors = {int(k): v for k, v in layer_provenance["seed_observations"].items()}
            if recorded_anchors != anchors:
                raise ValueError("Evidence layer seed observations differ from evaluation seeds")
            evidence_metadata = {
                "schema": "processed-native-probability-evidence-v1",
                "source_sha256": evidence_metadata["source_sha256"],
                "seed_sha256": evidence_metadata["seed_source"]["sha256"],
                "probabilities_sha256": evidence_metadata["prediction_npz_sha256"],
                "native_trace_indices_sha256": canonical_sha256(
                    evidence["native_trace_indices"].tolist()
                ),
                "native_sample_indices_sha256": canonical_sha256(
                    evidence["native_sample_indices"].tolist()
                ),
                "trace_stride": int(evidence["trace_stride"]),
                "preprocessing_version": evidence_metadata["dataset_preprocessing_version"],
                "layer": args.layer,
                "held_out_group": layer_provenance["held_out_group"],
                "model_sha256": layer_provenance["model_metadata_sha256"],
                "weights_sha256": evidence_metadata["model_sha256"],
                "training_groups": model_metadata["training_road_groups"],
            }
    take = slice(None, None, args.trace_stride)
    validate_evidence_provenance(
        evidence_metadata,
        source_sha256=record["source_sha256"],
        seed_sha256=sha(args.seed_source),
        probabilities_sha256=sha(args.probabilities),
        native_trace_indices=arrays["native_trace_indices"][take],
        native_sample_indices=arrays["native_sample_indices"],
        preprocessing_version=manifest["preprocessing_version"],
        layer=args.layer,
        road_group=record["physical_road_group"],
    )
    if evidence_metadata.get("trace_stride") != args.trace_stride:
        raise ValueError("Evidence native stride differs from evaluation")
    tolerance = None
    if args.frozen_comparator:
        # Existing historical controls are pinned in the established endpoint scorer.
        sys.path.insert(0, str(ROOT / "scripts"))
        from seeded_eval_endpoints import CONTROLS

        if sha(args.frozen_comparator) not in {value[1] for value in CONTROLS.values()}:
            raise ValueError("Comparator is not a pinned original frozen control")
        comparator = json.loads(args.frozen_comparator.read_text(encoding="utf-8"))
        if (
            comparator["audit"]["dzt_sha256"] != record["source_sha256"]
            or comparator["stride"] != args.trace_stride
            or comparator["seed_support"]["observations"] != seeds_document["observations"]
        ):
            raise ValueError("Frozen comparator input, native grid, or initial seeds differ")
        tolerance = comparator["methods"]["seed_hybrid"]["layers"][str(args.layer)][
            "tolerance_samples"
        ]
    config = (
        DenseDecoderConfig(**json.loads(args.decoder_config.read_text(encoding="utf-8")))
        if args.decoder_config
        else DenseDecoderConfig()
    )
    threshold, calibration = None, None
    if args.threshold_calibration:
        calibration = json.loads(args.threshold_calibration.read_text(encoding="utf-8"))
        road = record["physical_road_group"]
        if calibration.get("schema") != "processed-ml-operating-calibration-v1":
            raise ValueError("Use complete per-layer and per-stage operating calibration")
        if (
            calibration["held_out_group"] != road
            or calibration["weights_sha256"] != evidence_metadata["weights_sha256"]
            or calibration["trace_stride"] != args.trace_stride
            or calibration["decoder_config"] != asdict(config)
        ):
            raise ValueError("Threshold provenance does not exclude this evaluation road")
        threshold = {}
        for source_stage, target_stage in (
            ("direct", "direct_evidence"),
            ("dense", "dense_decoder"),
        ):
            gate = calibration["calibrations"][str(args.layer)][source_stage]
            if road in gate["training_groups"] or road not in gate["excluded_outer_groups"]:
                raise ValueError("Layer/stage gate does not exclude this evaluation road")
            value = gate["selected_threshold"]
            threshold[target_stage] = 1.000001 if value is None else value
    report = evaluate_evidence_stages(
        arrays["amplitudes"][take],
        arrays["labels"][args.layer - 1, take],
        probabilities,
        initial_seeds=anchors,
        sample_valid=arrays["sample_validity"][take],
        native_dx_m=record["horizontal_step_m"],
        dt_ns=record["sample_interval_ns"],
        road_group=record["physical_road_group"],
        layer=args.layer,
        native_trace_indices=arrays["native_trace_indices"][take],
        tolerance_samples=tolerance,
        decoder_config=config,
        operating_threshold=threshold,
        include_observations=not args.omit_observations,
    )
    report["provenance"] = {
        "manifest_sha256": sha(args.dataset_manifest),
        "source_sha256": record["source_sha256"],
        "label_sha256": record["label_sha256"],
        "evidence_sha256": sha(args.probabilities),
        "evidence_metadata_sha256": sha(evidence_metadata_path),
        "evidence_metadata": evidence_metadata,
        "original_evidence_metadata": original_evidence_metadata,
        "frozen_comparator_sha256": sha(args.frozen_comparator) if args.frozen_comparator else None,
        "seed_sha256": sha(args.seed_source),
        "script_sha256": sha(__file__),
        "record_id": args.record_id,
        "trace_stride": args.trace_stride,
        "threshold_calibration_sha256": sha(args.threshold_calibration)
        if args.threshold_calibration
        else None,
        "source_files": {
            p.name: sha(p)
            for p in (
                ROOT / "src/gpr_layer_audit/ml/processed_dense_eval.py",
                ROOT / "src/gpr_layer_audit/ml/processed_dense_decoder.py",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    key: result[key]
                    for key in (
                        "fixed_initial_nonseed_N",
                        "accepted_agreement",
                        "correct_coverage",
                        "wrong_coverage",
                    )
                }
                for name, result in report["stages"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
