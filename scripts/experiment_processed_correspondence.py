"""One Mandiali-trained / Gujrat-evaluated processed correspondence experiment.

Usage: .venv/Scripts/python.exe scripts/experiment_processed_correspondence.py
No untouched-road or thickness claim. Predictions are written before evaluation
references are opened. The retained graph and its original score are preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.signal import find_peaks
from scipy.stats import rankdata

CODE_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--workspace-root", type=Path, default=CODE_ROOT)
ROOT = BOOTSTRAP.parse_known_args()[0].workspace_root.resolve()
sys.path.insert(0, str(CODE_ROOT / "src"))

from gpr_layer_audit.conventional import _same_lobe  # noqa: E402
from gpr_layer_audit.conventional_reference import (  # noqa: E402
    audit_reference,
    distributed_seeds,
    road_partition,
)
from gpr_layer_audit.io.dzt import DZTFile  # noqa: E402
from gpr_layer_audit.io.dzx import read_dzx  # noqa: E402
from gpr_layer_audit.ml.processed_correspondence import (  # noqa: E402
    PairLogistic,
    PatchContract,
    extract_patches,
    fit_pair_logistic,
    interval_weights,
    pair_features,
    reviewed_candidate_targets,
    score_candidates,
)
from gpr_layer_audit.processing.conventional_config import (  # noqa: E402
    ConventionalConfig,
    resolve_pulse,
)
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask  # noqa: E402


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def training_examples(manifest, contract, limit):
    features, targets, weights, reports = [], [], [], []
    for record in manifest["reviewed_inventory"]:
        if record["physical_road_group"] != "mandiali":
            continue
        path = ROOT / record["dzx"]
        if sha(path) != record["sha256"] or road_partition(path)[0] != "mandiali":
            raise ValueError("Training source violates frozen physical-road grouping")
        audit = audit_reference(path)
        if audit["issues"] or any(
            layer["issues"]
            or layer["amplitude_match_fraction"] != 1
            or layer["time_mapping_status"] != "verified_header"
            for layer in audit["layers"]
        ):
            raise ValueError(
                f"Training coordinates failed independent amplitude/time audit: {path}"
            )
        radar = DZTFile(path.with_suffix(".DZT"))
        measurement = np.asarray(radar.channel(), np.float32)
        valid = processed_boundary_mask(measurement)
        dt, dx = radar.header.sample_interval_ns, radar.header.distance_per_trace_m
        metadata = read_dzx(path)
        for layer in metadata.layers:
            order = layer.number + 1
            if order not in (2, 3):
                continue
            picks = sorted((p for p in layer.picks if p.channel == 0), key=lambda p: p.trace)
            short = manifest["cases"]["mandiali-short"]
            if str(path.relative_to(ROOT)) == short["dzx"]:
                supplied = read(ROOT / short["seed_source"])["observations"][str(order)]
                anchors = {p["trace"]: p["sample"] for p in supplied}
            else:
                anchors = {p.trace: p.sample for p in distributed_seeds(picks)}
            pulse = resolve_pulse(measurement, valid, anchors, {}, dt, ConventionalConfig())
            tolerance = max(2, pulse.lobe_samples / 4)
            held = [p for p in picks if p.trace not in anchors]
            spacing = max(1, int(np.ceil(len(held) / limit)))
            reviewed = held[::spacing]
            rows, samples, labels, row_weight = [], [], [], []
            no_positive = 0
            for point in reviewed:
                trace = measurement[point.trace]
                extrema = np.r_[find_peaks(trace)[0], find_peaks(-trace)[0]]
                candidates = np.unique(np.concatenate([extrema - 1, extrema, extrema + 1]))
                candidates = candidates[(candidates > 0) & (candidates < len(trace) - 1)]
                candidates = candidates[valid[point.trace, candidates]]
                target = reviewed_candidate_targets(trace, candidates, point.sample, tolerance)
                positive = np.flatnonzero(target == 1)
                if not len(positive):
                    no_positive += 1
                    continue
                negative = np.flatnonzero(
                    (target == 0) & (abs(candidates - point.sample) <= 4 * pulse.lobe_samples)
                )
                # Neighboring wrong lobes only, not unreviewed pixels elsewhere in the image.
                negative = negative[np.argsort(abs(candidates[negative] - point.sample))[:8]]
                if not len(negative):
                    continue
                for indices, label in ((positive, 1), (negative, 0)):
                    for index in indices:
                        rows.append(point.trace)
                        samples.append(candidates[index])
                        labels.append(label)
                        row_weight.append(0.5 / len(indices) / len(reviewed))
            rows, samples, labels = np.asarray(rows), np.asarray(samples), np.asarray(labels)
            candidate, candidate_support, observable = extract_patches(
                measurement, valid, rows, samples, dt, dx, contract
            )
            seed_rows, seed_samples = np.asarray(sorted(anchors.items())).T
            seed_patch, seed_support, seed_valid = extract_patches(
                measurement, valid, seed_rows, seed_samples, dt, dx, contract
            )
            if not np.all(seed_valid):
                raise ValueError("Invalid training seed")
            mix = interval_weights(rows, seed_rows)
            for index in range(len(seed_rows)):
                use = observable & (mix[:, index] > 0)
                features.append(
                    pair_features(
                        candidate[use],
                        candidate_support[use],
                        seed_patch[index],
                        seed_support[index],
                    )
                )
                targets.append(labels[use])
                weights.append(np.asarray(row_weight)[use] * mix[use, index])
            reports.append(
                {
                    "dzt": str(path.with_suffix(".DZT").relative_to(ROOT)),
                    "dzt_sha256": audit["dzt_sha256"],
                    "dzx_sha256": sha(path),
                    "physical_road_group": "mandiali",
                    "layer": order,
                    "coordinate_audit": audit["layers"][order - 1],
                    "dt_ns": dt,
                    "native_dx_m": dx,
                    "header_time_origin_ns": radar.header.position_ns,
                    "anchors": anchors,
                    "reviewed_nonseed_rows_available": len(held),
                    "reviewed_rows_sampled": len(reviewed),
                    "reviewed_rows_without_positive_extremum_candidate": no_positive,
                    "target_candidates": len(labels),
                    "positive_candidates": int(np.sum(labels == 1)),
                    "negative_candidates": int(np.sum(labels == 0)),
                    "support_rows_used_as_targets": sorted(set(rows) & set(anchors)),
                    "tolerance_samples": tolerance,
                    "initial_seed_pulse": pulse.metadata(),
                }
            )
            print(f"Training {path.stem}, layer {order}: {len(labels)} candidates", flush=True)
    if not reports:
        raise ValueError("No eligible Mandiali-only training sources")
    return np.concatenate(features), np.concatenate(targets), np.concatenate(weights), reports


def solve_graph(arrays, metadata, score):
    from gpr_layer_audit.processing.hybrid import ReflectorSegment, solve_complete_intervals

    anchors = dict(arrays["anchors"].tolist())
    table = SimpleNamespace(
        **{name: arrays[name] for name in ("samples", "valid", "polarities")}, component_maps={}
    )
    if "lobes" in arrays:
        table.component_maps["hybrid_lobe_id"] = arrays["lobes"]
    segments = []
    for index, identifier in enumerate(arrays["segment_ids"]):
        nodes = [
            tuple(n)
            for n in arrays["nodes"][arrays["offsets"][index] : arrays["offsets"][index + 1]]
        ]
        seed_rows = {row for row, col in nodes if anchors.get(row) == table.samples[row, col]}
        segments.append(ReflectorSegment(int(identifier), nodes, seed_rows, arrays["costs"][index]))
    edges = {(int(a), int(b)): float(q) for a, b, q in arrays["edges"]}
    diagnostics = {}
    start = time.perf_counter()
    result = solve_complete_intervals(
        segments,
        edges,
        arrays["support"],
        table,
        score,
        anchors,
        metadata["breaks"],
        metadata["pulse_width_samples"],
        metadata["step_m"],
        ConventionalConfig(**metadata["config"]),
        diagnostics,
    )
    return np.asarray(result), diagnostics["path_margin"], time.perf_counter() - start


def auc(target, score):
    target, score = np.asarray(target), np.asarray(score)
    positive, negative = int(np.sum(target == 1)), int(np.sum(target == 0))
    if not positive or not negative:
        return None
    return float(
        (rankdata(score)[target == 1].sum() - positive * (positive + 1) / 2) / positive / negative
    )


def evaluate_prediction(prediction_path, case, graph_path, layer):
    """Scorer-only access occurs after predictions and their hashes are finalized."""
    with np.load(prediction_path) as values:
        prediction = {key: values[key] for key in values.files}
    with np.load(graph_path) as values:
        graph = {key: values[key] for key in values.files}
    metadata = read(graph_path.with_suffix(".json"))
    radar = DZTFile(ROOT / case["dzt"])
    measurement = radar.channel()[:: case["stride"]]
    references = read_dzx(ROOT / case["dzx"])
    anchors = dict(graph["anchors"].tolist())
    points = [
        p
        for group in references.layers
        if group.number + 1 == layer
        for p in group.picks
        if p.channel == 0
        and p.trace % case["stride"] == 0
        and p.trace // case["stride"] not in anchors
    ]
    tolerance = max(2, metadata["pulse_width_samples"] / 4)
    output = {}
    for name in ("classical", "learned"):
        score = prediction[name + "_scores"]
        path = prediction[name + "_paths"][0]
        correct, wrong, missing, top1, retained = 0, 0, 0, 0, 0
        discrimination_targets, discrimination_scores, errors, wrong_rows = [], [], [], []
        hard_targets, hard_scores, pure_scores = [], [], []
        intervals = {"bracketed": [0, 0, 0], "tail": [0, 0, 0]}
        for point in points:
            row = point.trace // case["stride"]
            columns = np.flatnonzero(graph["valid"][row] & (graph["samples"][row] >= 0))
            candidates = graph["samples"][row, columns]
            labels = reviewed_candidate_targets(
                measurement[row], candidates, point.sample, tolerance
            )
            retained += int(np.any(labels == 1))
            top1 += int(len(columns) > 0 and labels[np.argmax(score[row, columns])] == 1)
            use = (labels >= 0) & np.isfinite(score[row, columns])
            discrimination_targets.extend(labels[use].tolist())
            discrimination_scores.extend(score[row, columns][use].tolist())
            hard = use & (abs(candidates - point.sample) <= 4 * metadata["pulse_width_samples"])
            hard_targets.extend(labels[hard].tolist())
            hard_scores.extend(score[row, columns][hard].tolist())
            if name == "learned":
                pure = (
                    prediction["learned_scores"][row, columns]
                    - prediction["classical_scores"][row, columns]
                ) / 0.55 + 0.5
                pure_scores.extend(pure[hard].tolist())
            selected = path[row]
            region = "bracketed" if min(anchors) <= row <= max(anchors) else "tail"
            if selected < 0:
                missing += 1
                intervals[region][2] += 1
            elif abs(selected - point.sample) <= tolerance and _same_lobe(
                measurement[row], selected, point.sample
            ):
                correct += 1
                intervals[region][0] += 1
            else:
                wrong += 1
                intervals[region][1] += 1
                wrong_rows.append(row)
            if selected >= 0:
                errors.append(abs(selected - point.sample) * metadata["dt_ns"])
        spans, span = [], 0
        for index, row in enumerate(wrong_rows):
            span = span + 1 if index > 0 and row == wrong_rows[index - 1] + 1 else 1
            spans.append(span)
        output[name] = {
            "reviewed_nonseed_observations": len(points),
            "reference_candidate_retained": retained,
            "candidate_discrimination_auc": auc(discrimination_targets, discrimination_scores),
            "candidate_discrimination_count": len(discrimination_targets),
            "neighboring_lobe_discrimination_auc": auc(hard_targets, hard_scores),
            "neighboring_lobe_candidate_count": len(hard_targets),
            "learned_evidence_neighboring_lobe_auc": (
                auc(hard_targets, pure_scores) if name == "learned" else None
            ),
            "local_top1_correct": top1,
            "local_top1_correct_fraction": top1 / len(points) if points else None,
            "proposal_correct": correct,
            "proposal_wrong": wrong,
            "proposal_missing": missing,
            "proposal_precision": correct / (correct + wrong) if correct + wrong else None,
            "proposal_correct_coverage": correct / len(points) if points else None,
            "proposal_wrong_coverage": wrong / len(points) if points else None,
            "longest_contiguous_wrong_proposal_span_m": max(spans, default=0) * metadata["step_m"],
            "proposal_timing_error_ns_median": float(np.median(errors)) if errors else None,
            "bracketed_and_tail_correct_wrong_missing": intervals,
            "acceptance_metrics": "not computed: selection-only experiment, unchanged graph",
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "exports/seeded-tracker/learning")
    parser.add_argument("--max-training-rows-per-acquisition-layer", type=int, default=1500)
    parser.add_argument("--skip-graph-solve", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "benchmarks/seeded-evaluation-inputs.json"
    manifest = read(manifest_path)
    contract = PatchContract()
    source = {
        str(path.relative_to(CODE_ROOT)): sha(path)
        for path in sorted((CODE_ROOT / "src/gpr_layer_audit").rglob("*.py"))
    }
    source[str(Path(__file__).relative_to(CODE_ROOT))] = sha(__file__)
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    snapshot = args.output / ("source-" + source_hash[:16])
    for relative in source:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE_ROOT / relative, target)
    report = {
        "schema": "processed-correspondence-experiment-v1",
        "contract": asdict(contract),
        "source_sha256": source_hash,
        "source_files": source,
        "input_manifest_sha256": sha(manifest_path),
        "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
        "training_groups": ["mandiali"],
        "evaluation_groups": ["gujrat"],
        "independence": "Development transfer only; both roads previously used for development",
        "claim": (
            "Candidate evidence and fixed-graph proposal selection; "
            "no acceptance or thickness claim"
        ),
        "score_change": "original_score + 0.55 * (uncalibrated_learned_evidence - 0.5)",
        "predeclared_parameters": True,
    }
    write(args.output / "specification.json", report)
    start = time.perf_counter()
    features, targets, weights, training = training_examples(
        manifest, contract, args.max_training_rows_per_acquisition_layer
    )
    model, optimization = fit_pair_logistic(features, targets, weights, contract)
    model_path = args.output / "mandiali-pair-logistic.npz"
    model.save(model_path)
    report.update(
        training=training,
        training_pair_count=len(targets),
        training_feature_count=features.shape[1],
        optimization=optimization,
        training_runtime_s=time.perf_counter() - start,
        model_sha256=sha(model_path),
    )
    del features, targets, weights
    write(args.output / "training.json", report)
    print(f"Model frozen: {report['model_sha256']}", flush=True)
    case = manifest["cases"]["gujrat-second"]
    if road_partition(case["dzt"])[0] in report["training_groups"]:
        raise ValueError("Physical-road leakage")
    for field, hash_field in (("dzt", "dzt_sha256"), ("seed_source", "seed_sha256")):
        if sha(ROOT / case[field]) != case[hash_field]:
            raise ValueError("Frozen inference input changed")
    radar = DZTFile(ROOT / case["dzt"])
    measurement = np.asarray(radar.channel()[:: case["stride"]], np.float32)
    valid = processed_boundary_mask(measurement)
    seeds = read(ROOT / case["seed_source"])["observations"]
    report["evaluation"] = {}
    for layer in (2, 3):
        graph_path = ROOT / f"exports/seeded-tracker/identity/gujrat-geometry-graph-{layer}.npz"
        metadata = read(graph_path.with_suffix(".json"))
        if sha(graph_path) != metadata["sha256"]:
            raise ValueError("Frozen retained graph changed")
        with np.load(graph_path) as values:
            graph = {key: values[key] for key in values.files}
        anchors = {p["trace"] // case["stride"]: p["sample"] for p in seeds[str(layer)]}
        if any(p["trace"] % case["stride"] for p in seeds[str(layer)]):
            raise ValueError("Operating seed would be snapped")
        if sorted(anchors.items()) != sorted(map(tuple, graph["anchors"].tolist())):
            raise ValueError("Retained graph operating seeds differ")
        rows, columns = np.nonzero(graph["valid"] & (graph["samples"] >= 0))
        evidence = score_candidates(
            PairLogistic.load(model_path),
            measurement,
            valid,
            rows,
            graph["samples"][rows, columns],
            anchors,
            metadata["dt_ns"],
            metadata["step_m"],
        )
        learned = graph["scores"].copy()
        learned[rows, columns] += 0.55 * (np.nan_to_num(evidence, nan=0.5) - 0.5)
        output = {"classical_scores": graph["scores"], "learned_scores": learned}
        timings = {}
        for name, score in (("classical", graph["scores"]), ("learned", learned)):
            if args.skip_graph_solve:
                output[name + "_paths"] = np.full((1, len(measurement)), -1)
                continue
            paths, margins, elapsed = solve_graph(graph, metadata, score)
            output[name + "_paths"], output[name + "_margins"] = paths, margins
            timings[name] = elapsed
            print(f"Gujrat layer {layer} {name} fixed graph: {elapsed:.1f}s", flush=True)
        prediction_path = args.output / f"gujrat-layer-{layer}-predictions.npz"
        np.savez_compressed(prediction_path, **output)
        provenance = {
            "graph_sha256": sha(graph_path),
            "graph_metadata_sha256": sha(graph_path.with_suffix(".json")),
            "model_sha256": sha(model_path),
            "prediction_sha256": sha(prediction_path),
            "dzt_sha256": case["dzt_sha256"],
            "operating_seeds_sha256": case["seed_sha256"],
            "evaluation_reference_used_in_inference": False,
            "same_retained_graph": True,
            "solver_runtime_s": timings,
        }
        write(args.output / f"gujrat-layer-{layer}-prediction-provenance.json", provenance)
        report["evaluation"][str(layer)] = provenance
        write(args.output / "result.json", report)
    # Both interfaces have finished inference before any Gujrat references are opened.
    if sha(ROOT / case["dzx"]) != case["dzx_sha256"]:
        raise ValueError("Frozen scoring reference changed")
    for layer in (2, 3):
        graph_path = ROOT / f"exports/seeded-tracker/identity/gujrat-geometry-graph-{layer}.npz"
        prediction_path = args.output / f"gujrat-layer-{layer}-predictions.npz"
        metrics = evaluate_prediction(prediction_path, case, graph_path, layer)
        report["evaluation"][str(layer)]["metrics"] = metrics
        print(json.dumps({layer: metrics}), flush=True)
    report["recommendation"] = (
        "Selection evidence only; no production integration authorized by AUC"
    )
    write(args.output / "result.json", report)


if __name__ == "__main__":
    main()
