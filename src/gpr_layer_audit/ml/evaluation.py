"""Held-out hybrid interpretation evaluation; never a claim of physical accuracy."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from gpr_layer_audit.io import DZTFile
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.hybrid_evidence import PREPROCESSING_VERSION
from gpr_layer_audit.processing.preprocessing import preprocess_for_interpretation
from gpr_layer_audit.processing.tracker import pick_interfaces

from .dataset import annotation_reason, transform_samples


def promotion_report(records, baseline_records=(), *, minimum_points=30):
    """Evaluate each physical road separately; missing observations cannot inflate accuracy."""
    result = {}
    for layer in (1, 2, 3):
        rows = [r for r in records if r["layer_order"] == layer and r["expected_visible"]]
        old = [r for r in baseline_records if r["layer_order"] == layer and r["expected_visible"]]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["road_group"]].append(row)
        per_road = {}
        for group, observations in grouped.items():
            accepted = [r for r in observations if r["accepted"]]
            correct = sum(r["correct"] for r in accepted)
            per_road[group] = {
                "observations": len(observations),
                "accepted": len(accepted),
                "accepted_correctness": correct / len(accepted) if accepted else 0,
                "correct_coverage": correct / len(observations),
            }
        accepted = [r for r in rows if r["accepted"]]
        correct = sum(r["correct"] for r in accepted)
        precision = correct / len(accepted) if accepted else 0
        coverage = correct / len(rows) if rows else 0
        baseline_accepted = [r for r in old if r["accepted"]]
        baseline_precision = (
            sum(r["correct"] for r in baseline_accepted) / len(baseline_accepted)
            if baseline_accepted
            else 0
        )
        baseline_coverage = (
            sum(r["accepted"] and r["correct"] for r in old) / len(old) if old else 0
        )
        adequate = len(per_road) >= 3 and all(
            r["observations"] >= minimum_points and r["accepted_correctness"] >= 0.95
            for r in per_road.values()
        )
        passed = (
            adequate
            and precision >= max(0.95, baseline_precision)
            and coverage - baseline_coverage >= 0.10
        )
        invisible = [r for r in records if r["layer_order"] == layer and not r["expected_visible"]]
        false_visible = sum(r["accepted"] for r in invisible)
        passed = passed and false_visible == 0
        result[str(layer)] = {
            "passed": bool(passed),
            "roads": per_road,
            "accepted_correctness": precision,
            "correct_coverage": coverage,
            "baseline_correct_coverage": baseline_coverage,
            "additional_corrections_per_km": None,
            "invisible_observations": len(invisible),
            "false_accepted_invisible": false_visible,
            "reason": "passed"
            if passed
            else "insufficient_precision_coverage_or_independent_roads",
        }
    return result


def evaluate_hybrid(manifest_path, *, model=None, split="test", calibration=None, cancel=None):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation must use validation or test roads")
    dataset = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sources = {s["sha256"]: s for s in dataset["sources"]}
    labels_by_source = defaultdict(list)
    for label in [*dataset["annotations"], *dataset.get("evaluation_annotations", [])]:
        if annotation_reason(label, sources, evaluation=True):
            raise ValueError("Evaluation requires verified, raw-coordinate annotations")
        source = sources[label["source_sha256"]]
        if source["split"] == split:
            labels_by_source[source["sha256"]].append(label)
    records, baseline, runtimes = [], [], []
    model_metadata = None
    if model:
        bundle = Path(model)
        model_metadata = json.loads(
            (bundle / "model.json" if bundle.is_dir() else bundle).read_text()
        )
        forbidden = set(model_metadata.get("training_road_groups", []))
        if split == "test":
            forbidden.update(model_metadata.get("validation_road_groups", []))
        if any(sources[h]["road_group"] in forbidden for h in labels_by_source):
            raise ValueError("Evaluation road was used to train this model")
    if calibration and split == "test":
        settings = json.loads(Path(calibration).read_text())
        tuned_roads = set(settings.get("calibration_road_groups", []))
        if any(sources[h]["road_group"] in tuned_roads for h in labels_by_source):
            raise ValueError("Test road was used for acceptance calibration")
    for digest, labels in labels_by_source.items():
        source = sources[digest]
        if fingerprint_file(source["path"]) != digest:
            raise ValueError("Evaluation acquisition fingerprint changed")
        road = DZTFile(source["path"])
        dx = source["horizontal_step_m"]
        if not dx:
            raise ValueError("Evaluation requires physical trace spacing")
        stack = max(1, int(np.ceil(0.4 / dx)))
        prepared = calibrate(road, None, stack_size=stack, cancel=cancel)
        interpreted = preprocess_for_interpretation(
            prepared.radargram, prepared.reference_surface_sample
        )
        interpreted.feature_branches["hybrid_measurement"] = prepared.measurement_radargram
        anchors, seed_rows = {}, {}
        # Only the three selected input labels are exposed to the tracker.
        for order in sorted({int(r["layer_order"]) for r in labels}):
            visible = sorted(
                (
                    r
                    for r in labels
                    if int(r["layer_order"]) == order
                    and r["visibility"] == "visible"
                    and r.get("training_use") == "allowed"
                    and r.get("origin") in ("manual", "manual_correction")
                ),
                key=lambda r: float(r["trace_index"]),
            )
            anchors[order] = {}
            seed_rows[order] = []
            for index in np.unique(
                np.rint(np.linspace(0, len(visible) - 1, min(3, len(visible)))).astype(int)
            ):
                label = visible[index]
                row = int(np.argmin(abs(prepared.trace_centres - float(label["trace_index"]))))
                sample = int(
                    round(
                        float(
                            transform_samples(
                                float(label["sample_raw"]),
                                prepared.surface_samples[row],
                                prepared.reference_surface_sample,
                            )
                        )
                    )
                )
                anchors[order][row] = sample
                seed_rows[order].append(float(label["trace_index"]))
        layers = LayerSpec.defaults()
        highest = max(anchors, default=1)
        for layer in layers:
            layer.analysis_enabled = layer.order <= highest
        runs = []
        configurations = [("joint_seed_adaptive", False), ("seed_hybrid", False)]
        if model:
            configurations.append(("seed_hybrid", True))
        for method, ml in configurations:
            start = time.perf_counter()
            paths = pick_interfaces(
                interpreted.radargram,
                prepared.reference_surface_sample,
                layers,
                anchor_samples=anchors,
                feature_branches=dict(interpreted.feature_branches),
                method=method,
                horizontal_step_m=stack * dx,
                sample_interval_ns=road.header.sample_interval_ns,
                model_path=str(model) if ml else None,
                ml_policy="require" if ml else "off",
                calibration_path=str(calibration) if calibration and (ml or not model) else None,
                cancel=cancel,
            )
            runtimes.append(
                {
                    "road_group": source["road_group"],
                    "ml": ml,
                    "method": method,
                    "seconds": time.perf_counter() - start,
                }
            )
            observations, last_position = [], {}
            for label in sorted(labels, key=lambda r: float(r["trace_index"])):
                order, trace = int(label["layer_order"]), float(label["trace_index"])
                if any(abs(trace - seed) * dx <= 25 for seed in seed_rows.get(order, [])):
                    continue
                # Adjacent labelled pixels cannot act as independent checkpoints.
                if (trace - last_position.get(order, -1e20)) * dx < 10:
                    continue
                last_position[order] = trace
                row = int(np.argmin(abs(prepared.trace_centres - trace)))
                path = paths[order]
                proposed = (
                    int(path.provisional_samples[row])
                    if path.provisional_samples is not None
                    else int(path.samples[row])
                )
                predicted = (
                    float(
                        transform_samples(
                            proposed,
                            prepared.surface_samples[row],
                            prepared.reference_surface_sample,
                            inverse=True,
                        )
                    )
                    if proposed >= 0
                    else None
                )
                visible = label["visibility"] == "visible"
                tolerance = max(2, float(label.get("pulse_width_samples", 7)) / 4)
                lobe = (
                    "negative_trough"
                    if proposed >= 0 and interpreted.radargram[row, proposed] < 0
                    else "positive_peak"
                )
                correct = bool(
                    visible
                    and predicted is not None
                    and abs(predicted - float(label["sample_raw"])) <= tolerance
                    and lobe == label["selected_lobe"]
                )
                observations.append(
                    {
                        "road_group": source["road_group"],
                        "source_sha256": digest,
                        "layer_order": order,
                        "trace_index": trace,
                        "expected_visible": visible,
                        "predicted_raw": predicted,
                        "accepted": bool(path.visible[row]),
                        "correct": correct,
                        "error_samples": abs(predicted - float(label["sample_raw"]))
                        if visible and predicted is not None
                        else None,
                        "correspondence": float(
                            path.evidence.get("hybrid_correspondence", np.zeros(len(path.samples)))[
                                row
                            ]
                        ),
                        "path_margin": float(
                            path.evidence.get("hybrid_path_margin", np.zeros(len(path.samples)))[
                                row
                            ]
                        ),
                        "margin": float(
                            path.evidence.get("candidate_margin", np.zeros(len(path.samples)))[row]
                        ),
                        "measurement_support": float(
                            path.evidence.get("measurement_support", np.zeros(len(path.samples)))[
                                row
                            ]
                        ),
                    }
                )
            runs.append(observations)
        baseline.extend(runs[-2])
        records.extend(runs[-1])
    report = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "dataset_sha256": fingerprint_file(manifest_path),
        "split": split,
        "model_weights_sha256": model_metadata["weights_sha256"] if model_metadata else None,
        "records": records,
        "baseline_records": baseline,
        "runtimes": runtimes,
        "acceptance_calibration": json.loads(Path(calibration).read_text()) if calibration else {},
        "field_accuracy_established": False,
        "layers": promotion_report(records, baseline),
    }
    if not records:
        report["reason"] = "no_eligible_spatially_separated_non_seed_observations"
    return report


def calibrate_acceptance(report):
    if report.get("split") != "validation":
        raise ValueError("Acceptance calibration cannot use the held-out test set")
    layers = {}
    for order in (2, 3):
        rows = [r for r in report["records"] if r["layer_order"] == order]
        if len(rows) < 30 or len({r["road_group"] for r in rows}) < 2:
            continue
        choices = []
        for threshold in (0.75, 0.80, 0.85, 0.90, 0.95):
            for margin in (0.025, 0.05, 0.075, 0.10, 0.15):
                accepted = [
                    r
                    for r in rows
                    if r["correspondence"] >= threshold
                    and r.get("path_margin", 0) > 1e-8
                    and r["margin"] >= margin
                    and r["measurement_support"] >= 0.015
                    and r["predicted_raw"] is not None
                ]
                accuracy = sum(r["correct"] for r in accepted) / len(accepted) if accepted else 0
                if accuracy >= 0.95:
                    choices.append((len(accepted), accuracy, threshold, margin))
        if choices:
            count, accuracy, threshold, margin = max(choices)
            layers[str(order)] = {
                "minimum_correspondence": threshold,
                "minimum_margin": margin,
                "validation_correctness": accuracy,
                "accepted_observations": count,
            }
    return {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "model_weights_sha256": report.get("model_weights_sha256"),
        "layers": layers,
        "calibration_dataset_sha256": report["dataset_sha256"],
        "calibration_road_groups": sorted({r["road_group"] for r in report["records"]}),
        "validated": bool(layers),
    }
