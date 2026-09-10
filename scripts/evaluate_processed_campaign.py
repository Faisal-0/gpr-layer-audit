"""Freeze source, predict processed folds, and separately score dense evidence.

The parent orchestrator owns the single GPU queue. This script starts no work on
import. Default execution copies source into a new output/source directory and
reexecutes that snapshot. --frozen-source is for an already frozen parent snapshot.
All requested probability controls are saved before any target label array opens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ("removed-seeds", "changed-target", "seed-shuffle", "context-off", "conditioning-off")
RADAR_ARRAYS = (
    "amplitudes",
    "sample_validity",
    "native_trace_indices",
    "native_sample_indices",
    "distances_m",
)
RADAR_METADATA = (
    "record_id",
    "source_sha256",
    "input_mode",
    "physical_road_group",
    "shape",
    "sample_interval_ns",
    "time_origin_ns",
    "horizontal_step_m",
    "acquisition_id",
    "processing_variant",
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_hashes(root=ROOT):
    files = sorted((root / "src").rglob("*.py")) + [root / "scripts" / Path(__file__).name]
    return {str(path.relative_to(root)): sha(path) for path in files}


def freeze_and_run(args):
    if args.output.exists():
        raise FileExistsError("Campaign output exists; choose a new immutable artifact directory")
    before = source_hashes()
    snapshot = args.output / "source"
    shutil.copytree(
        ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    (snapshot / "scripts").mkdir()
    shutil.copy2(__file__, snapshot / "scripts" / Path(__file__).name)
    if source_hashes() != before or source_hashes(snapshot) != before:
        raise ValueError("Source changed while freezing; no inference was launched")
    write(args.output / "source-snapshot.json", {"origin": str(ROOT), "files": before})
    command = [
        sys.executable,
        str(snapshot / "scripts" / Path(__file__).name),
        *sys.argv[1:],
        "--frozen-source",
    ]
    return subprocess.call(command)


def open_radar_only(manifest_path, record):
    """Open only radar-derived files; never call the full reference-aware loader."""
    root = Path(manifest_path).parent.resolve()
    result = {"record": {key: record[key] for key in RADAR_METADATA if key in record}}
    for key in RADAR_ARRAYS:
        path = (root / record["arrays"][key]).resolve()
        if not path.is_relative_to(root) or sha(path) != record["array_sha256"][key]:
            raise ValueError(f"Radar cache path/hash invalid: {key}")
        result[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    if sha(record["path"]) != record["source_sha256"]:
        raise ValueError("Processed source changed after coordinate audit")
    return result


def resolve_path(value, registry_path, *, source_root=None):
    path = Path(value)
    if path.is_absolute():
        return path
    choices = [Path(registry_path).parent / path]
    if source_root:
        choices.append(Path(source_root) / path)
    choices.append(ROOT / path)
    for choice in choices:
        if choice.exists():
            return choice.resolve()
    raise FileNotFoundError(f"Registry path is unavailable: {value}")


def registry_cases(registry):
    cases = registry["cases"]
    if isinstance(cases, dict):
        return [{"case_id": key, **value} for key, value in cases.items()]
    return list(cases)


def prepare_case(case, record, registry_path, radar, *, source_root=None):
    source = case.get("dzt_sha256", case.get("source_sha256"))
    if source != record["source_sha256"]:
        raise ValueError("Case registry source fingerprint differs from processed record")
    group = case.get("physical_road_group", case.get("road_group"))
    if group != record["physical_road_group"]:
        raise ValueError("Case registry physical road differs from processed record")
    label_hash = case.get("dzx_sha256", case.get("label_sha256", case.get("reference_sha256")))
    if label_hash is not None and label_hash != record["label_sha256"]:
        raise ValueError("Case registry reference fingerprint differs from processed record")
    stride = case.get("stride", case.get("trace_stride"))
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("Case requires an exact positive native trace stride")
    seed_path = resolve_path(
        case.get("seed_source", case.get("seed_path")),
        registry_path,
        source_root=source_root,
    )
    if sha(seed_path) != case["seed_sha256"]:
        raise ValueError("Frozen case seed fingerprint changed")
    seeds = read(seed_path)
    if (
        seeds.get("schema") != "conventional-native-seeds-v1"
        or seeds.get("mode") != "processed"
        or seeds.get("dzt_sha256") != record["source_sha256"]
        or seeds.get("dzx_sha256") != record["label_sha256"]
    ):
        raise ValueError("Initial observations do not identify this processed coordinate source")
    anchors = {}
    for key, points in seeds["observations"].items():
        layer, chosen = int(key), {}
        for point in points:
            trace, sample = point["trace"], float(point["sample"])
            if (
                int(trace) != trace
                or trace % stride
                or point.get("channel", 0) != 0
                or not 0 <= trace < record["shape"][0]
                or not np.isfinite(sample)
                or not 0 <= sample <= record["shape"][1] - 1
            ):
                raise ValueError("Exact seed coordinate is incompatible; snapping is prohibited")
            row = int(trace) // stride
            if row in chosen:
                raise ValueError("Duplicate initial seed trace")
            if sample == int(sample):
                actual = float(radar["amplitudes"][int(trace), int(sample)])
                if "recorded_amplitude" in point and abs(actual - point["recorded_amplitude"]) > 1:
                    raise ValueError("Stored seed amplitude does not match native radar")
            expected_time = record["time_origin_ns"] + sample * record["sample_interval_ns"]
            if "time_ns" in point and abs(expected_time - point["time_ns"]) > 5.1e-8:
                raise ValueError("Stored seed time does not match native header mapping")
            chosen[row] = sample
        if chosen:
            anchors[layer] = chosen
    return stride, anchors, seed_path


def control_anchors(original, layer, control):
    """Alter authorized conditioning, never reveal another reference coordinate."""
    selected = original[layer]
    metadata = {"control": control, "model_layer_held_fixed": layer}
    if control == "removed-seeds":
        return {}, {**metadata, "authorized_input_seed_count": 0}
    if control == "changed-target":
        others = [key for key in sorted(original) if key != layer]
        if not others:
            return None, {**metadata, "status": "unavailable: no other authorized target seeds"}
        alternate = min(others, key=lambda key: (abs(key - layer), -key))
        return dict(original[alternate]), {
            **metadata,
            "authorized_seed_target_layer": alternate,
            "interpretation": "valid alternate target observations with requested layer held fixed",
        }
    if control == "seed-shuffle":
        return dict(reversed(list(selected.items()))), {
            **metadata,
            "interpretation": "seed ordering permutation; identity should be invariant",
        }
    return dict(selected), metadata


def load_calibrations(paths, metadata, manifest_sha, decoder_config, metadata_sha):
    calibrations = {}
    for path in paths:
        doc = read(path)
        if (
            doc.get("schema") != "processed-ml-operating-calibration-v1"
            or doc.get("held_out_group") != metadata["held_out_group"]
            or doc.get("model_sha256") != metadata_sha
            or doc.get("weights_sha256") != metadata["weights_sha256"]
            or doc.get("dataset_sha256") != manifest_sha
            or doc.get("decoder_config") != asdict(decoder_config)
            or doc.get("training_source_sha256") != metadata["code_hashes"]
            or doc.get("selection_reads_outer_arrays") is not False
        ):
            raise ValueError("TRAIN calibration model/dataset/fold/decoder provenance differs")
        stride = doc["trace_stride"]
        if stride in calibrations:
            raise ValueError("Ambiguous calibration artifacts for one native stride")
        for layers in doc["calibrations"].values():
            for gate in layers.values():
                if (
                    metadata["held_out_group"] in gate["training_groups"]
                    or metadata["held_out_group"] not in gate["excluded_outer_groups"]
                    or not set(gate["training_groups"]) <= set(metadata["training_road_groups"])
                ):
                    raise ValueError("Calibration gate includes an excluded/nontraining road")
        calibrations[stride] = {
            "document": doc,
            "path": str(Path(path).resolve()),
            "sha256": sha(path),
        }
    return calibrations


def stage_thresholds(calibrations, stride, layer, control):
    # Ablation distributions have not been independently calibrated; no operating
    # acceptance is borrowed from the baseline under changed conditioning.
    entry = calibrations.get(stride) if control == "baseline" else None
    thresholds = {"direct": 1.000001, "dense": 1.000001}
    status = {
        "source": None,
        "reason": "no matching TRAIN calibration; automatic acceptance withheld",
    }
    if entry and str(layer) in entry["document"]["calibrations"]:
        status = {"source": entry["path"], "sha256": entry["sha256"], "stages": {}}
        for stage in thresholds:
            calibration = entry["document"]["calibrations"][str(layer)][stage]
            value = calibration["selected_threshold"]
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Invalid TRAIN operating threshold")
            thresholds[stage] = 1.000001 if value is None else float(value)
            status["stages"][stage] = {
                "selected_threshold": value,
                "objective_met": calibration["objective_met"],
                "interpretation": "withheld: training objective not met"
                if value is None
                else "TRAIN-derived gate",
            }
    return thresholds, status


def score_stages(
    radar,
    references,
    probabilities,
    *,
    original_seeds,
    inference_seeds,
    native_traces,
    record,
    layer,
    tolerance,
    decoder_config,
    thresholds,
    label_valid,
    include_observations,
):
    from gpr_layer_audit.ml.processed_dense_decoder import (
        decode_dense_path,
        direct_predictions,
        seed_interpolation,
    )
    from gpr_layer_audit.ml.processed_dense_eval import (
        evaluate_native_predictions,
        precision_coverage_curve,
    )

    stride = int(native_traces[1] - native_traces[0]) if len(native_traces) > 1 else 1
    direct = direct_predictions(probabilities, sample_valid=radar["sample_validity"])
    dense = decode_dense_path(
        probabilities,
        inference_seeds,
        sample_valid=radar["sample_validity"],
        dx_m=record["horizontal_step_m"] * stride,
        dt_ns=record["sample_interval_ns"],
        config=decoder_config,
    )
    kwargs = {
        "initial_seeds": original_seeds,
        "native_dx_m": record["horizontal_step_m"],
        "dt_ns": record["sample_interval_ns"],
        "tolerance_samples": tolerance,
        "road_group": record["physical_road_group"],
        "layer": layer,
        "native_trace_indices": native_traces,
        "trace_step": stride,
        "label_valid": label_valid,
        "include_observations": True,
    }
    # Alternate authorized observations are supplied answers and receive no
    # automatic credit; they stay in the original fixed denominator.
    additional = sorted(set(inference_seeds) - set(original_seeds))
    kwargs["additional_seed_rows"] = additional
    stages, arrays = {}, {}
    for stage, result in (("direct", direct), ("dense", dense)):
        report = evaluate_native_predictions(
            radar["amplitudes"],
            references,
            result.proposed_samples,
            result.accepted,
            accepted_samples=result.samples,
            confidence=result.confidence,
            **kwargs,
        )
        report["precision_coverage_curve"] = precision_coverage_curve(report)
        report["diagnostics"] = result.diagnostics
        gated = evaluate_native_predictions(
            radar["amplitudes"],
            references,
            result.proposed_samples,
            result.accepted & (result.confidence >= thresholds[stage]),
            accepted_samples=result.samples,
            confidence=result.confidence,
            **kwargs,
        )
        gated["operating_threshold"] = thresholds[stage]
        if not include_observations:
            report.pop("evaluation_observations")
            gated.pop("evaluation_observations")
        stages[stage + "_ungated"] = report
        stages[stage + "_gated"] = gated
        arrays.update(
            {
                f"{stage}_proposal": result.proposed_samples,
                f"{stage}_accepted": result.accepted & (result.confidence >= thresholds[stage]),
                f"{stage}_confidence": result.confidence,
                f"{stage}_manual_seed": result.manual,
            }
        )
    interpolated = seed_interpolation(len(references), inference_seeds)
    stages["seed_interpolation_no_radar"] = evaluate_native_predictions(
        radar["amplitudes"],
        references,
        interpolated,
        interpolated >= 0,
        **kwargs,
    )
    if not include_observations:
        stages["seed_interpolation_no_radar"].pop("evaluation_observations")
    arrays["seed_interpolation_no_radar"] = interpolated
    return stages, arrays


def run(args):
    sys.path.insert(0, str(ROOT / "src"))
    from gpr_layer_audit.ml.processed_data import load_processed_manifest
    from gpr_layer_audit.ml.processed_dense_decoder import DenseDecoderConfig
    from gpr_layer_audit.ml.processed_dense_eval import (
        aggregate_reports,
        canonical_sha256,
        frozen_initial_tolerance,
        scorer_provenance,
        validate_evidence_provenance,
    )
    from gpr_layer_audit.ml.processed_inference import ProcessedPredictor

    if (args.output / "run.json").exists():
        raise FileExistsError("Campaign already started in this output; choose a new artifact")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.model / "model.json" if args.model.is_dir() else args.model
    metadata = read(metadata_path)
    if metadata.get("status") != "completed":
        raise ValueError("Evaluate a completed immutable training run")
    weights_path = metadata_path.parent / metadata["weights"]
    manifest = load_processed_manifest(args.dataset_manifest)
    manifest_sha = sha(args.dataset_manifest)
    if (
        manifest_sha != metadata["dataset_sha256"]
        or sha(weights_path) != metadata["weights_sha256"]
    ):
        raise ValueError("Model dataset/weights fingerprint differs from training record")
    for filename, expected in metadata["code_hashes"].items():
        if sha(ROOT / "src/gpr_layer_audit/ml" / filename) != expected:
            raise ValueError(f"Inference must retain exact trained source: {filename}")
    code = source_hashes()
    registry = read(args.cases)
    registry_hash = sha(args.cases)
    if registry.get("dataset_manifest_sha256") != manifest_sha:
        raise ValueError("Frozen case registry belongs to another processed dataset")
    decoder_config = (
        DenseDecoderConfig(**read(args.decoder_config))
        if args.decoder_config
        else DenseDecoderConfig()
    )
    if decoder_config.acceptance_threshold != 0:
        raise ValueError(
            "Base decoder configuration must be ungated; gates come from TRAIN calibration"
        )
    calibrations = load_calibrations(
        args.calibration,
        metadata,
        manifest_sha,
        decoder_config,
        sha(metadata_path),
    )
    candidates = registry_cases(registry)
    requested_cases = {registry.get("case_aliases", {}).get(value, value) for value in args.case}
    records = {r["record_id"]: r for r in manifest["records"]}
    selected = []
    for case in candidates:
        case_id = case.get("case_id", case.get("name"))
        if requested_cases and case_id not in requested_cases:
            continue
        record = records.get(case.get("record_id"))
        if record is None:
            source = case.get("dzt_sha256", case.get("source_sha256"))
            matches = [r for r in records.values() if r["source_sha256"] == source]
            if len(matches) != 1:
                raise ValueError(f"Registry acquisition cannot resolve uniquely: {case_id}")
            record = matches[0]
        if record["physical_road_group"] == metadata["held_out_group"]:
            selected.append((case_id, case, record))
    if not selected:
        raise ValueError("No registered cases match the model's held-out physical road")
    run_info = {
        "schema": "processed-campaign-evaluation-v1",
        "status": "prediction_started",
        "model": str(metadata_path.resolve()),
        "model_sha256": sha(metadata_path),
        "weights_sha256": metadata["weights_sha256"],
        "manifest_sha256": manifest_sha,
        "case_registry_sha256": registry_hash,
        "source_root": str(ROOT),
        "source_hashes": code,
        "scorer": scorer_provenance(),
        "held_out_group": metadata["held_out_group"],
        "cases": [x[0] for x in selected],
        "controls": ["baseline", *args.controls],
        "predictor_receives_label_arrays": False,
        "target_label_arrays_opened": False,
        "fine_native_trace_stride": 1,
        "native_temporal_resampling": False,
        "decoder_config": asdict(decoder_config),
        "production_promotion": False,
        "claim": "grouped development interpretation agreement; not untouched-road validation",
    }
    write(args.output / "run.json", run_info)
    started, jobs = time.perf_counter(), []
    for case_id, case, record in selected:
        radar = open_radar_only(args.dataset_manifest, record)
        stride, all_anchors, seed_path = prepare_case(
            case,
            record,
            args.cases,
            radar,
            source_root=registry.get("source_root"),
        )
        layers = [
            layer
            for layer in metadata["config"]["layers"]
            if layer in all_anchors and (not args.layers or layer in args.layers)
        ]
        if not layers:
            raise ValueError(f"No trained layer has authorized seeds in case {case_id}")
        predictor = ProcessedPredictor.from_checkpoint(
            metadata_path.parent,
            radar,
            device=args.device,
            trace_stride=stride,
            decoder_config=decoder_config,
        )
        for layer in layers:
            supplied = case.get("tolerance_samples", case.get("tolerance_samples_by_layer", {}))
            tolerance = supplied.get(str(layer)) if isinstance(supplied, dict) else supplied
            pulse = {"source": "frozen case registry"}
            if tolerance is None:
                tolerance, pulse = frozen_initial_tolerance(
                    predictor.measurement,
                    predictor.valid,
                    all_anchors[layer],
                    predictor.dt_ns,
                )
            baseline = None
            for control in ["baseline", *args.controls]:
                anchors, control_meta = control_anchors(all_anchors, layer, control)
                if anchors is None:
                    write(
                        args.output / case_id / control / f"layer{layer}-unavailable.json",
                        control_meta,
                    )
                    continue
                progress = {
                    "case": case_id,
                    "layer": layer,
                    "control": control,
                    "stage": "predicting",
                }
                print(json.dumps(progress), flush=True)
                before = time.perf_counter()
                kwargs = {}
                if control == "context-off":
                    kwargs["use_context"] = False
                if control == "conditioning-off":
                    kwargs["conditioning"] = False
                probabilities = predictor.predict_evidence({layer: anchors}, **kwargs)[layer]
                elapsed = time.perf_counter() - before
                destination = args.output / case_id / control / f"layer{layer}"
                destination.mkdir(parents=True)
                evidence_path = destination / "probabilities.npz"
                np.savez(
                    evidence_path,
                    probabilities=probabilities,
                    native_trace_indices=predictor.native_trace_indices,
                    native_sample_indices=radar["native_sample_indices"],
                )
                seed_episode = {
                    "layer": layer,
                    "anchors": [[r, s] for r, s in sorted(anchors.items())],
                    "initial_seed_source_sha256": sha(seed_path),
                    "control": control,
                }
                write(destination / "seed-episode.json", seed_episode)
                conditioning_seed_hash = (
                    sha(seed_path)
                    if control == "baseline"
                    else sha(destination / "seed-episode.json")
                )
                evidence = {
                    "schema": "processed-native-probability-evidence-v1",
                    "source_sha256": record["source_sha256"],
                    "seed_sha256": conditioning_seed_hash,
                    "seed_episode_sha256": sha(destination / "seed-episode.json"),
                    "initial_seed_source_sha256": sha(seed_path),
                    "seed_episode": seed_episode,
                    "probabilities_sha256": sha(evidence_path),
                    "model_sha256": sha(metadata_path),
                    "weights_sha256": metadata["weights_sha256"],
                    "native_trace_indices_sha256": canonical_sha256(
                        predictor.native_trace_indices.tolist()
                    ),
                    "native_sample_indices_sha256": canonical_sha256(
                        radar["native_sample_indices"].tolist()
                    ),
                    "preprocessing_version": manifest["preprocessing_version"],
                    "layer": layer,
                    "held_out_group": record["physical_road_group"],
                    "training_groups": metadata["training_road_groups"],
                    "trace_stride": stride,
                    "predictor": predictor.provenance,
                    "control": control_meta,
                    "prediction_seconds": elapsed,
                    "target_labels_opened_before_prediction": False,
                    "original_initial_seeds": {str(r): s for r, s in all_anchors[layer].items()},
                    "inference_seeds": {str(r): s for r, s in anchors.items()},
                    "fixed_tolerance_samples": tolerance,
                    "initial_scoring_pulse": pulse,
                }
                if baseline is not None:
                    evidence["conditioning_change"] = {
                        "mean_total_variation_depth_distribution": float(
                            np.mean(np.abs(probabilities - baseline).sum(axis=1) / 2)
                        ),
                        "native_depth_argmax_changed_fraction": float(
                            np.mean(probabilities.argmax(axis=1) != baseline.argmax(axis=1))
                        ),
                        "mean_absolute_argmax_shift_samples": float(
                            np.mean(abs(probabilities.argmax(axis=1) - baseline.argmax(axis=1)))
                        ),
                    }
                if control == "baseline":
                    baseline = probabilities.copy()
                write(destination / "evidence.json", evidence)
                if source_hashes() != code or sha(weights_path) != metadata["weights_sha256"]:
                    raise ValueError("Frozen source/model changed during prediction")
                validate_evidence_provenance(
                    evidence,
                    source_sha256=record["source_sha256"],
                    seed_sha256=conditioning_seed_hash,
                    probabilities_sha256=sha(evidence_path),
                    native_trace_indices=predictor.native_trace_indices,
                    native_sample_indices=radar["native_sample_indices"],
                    preprocessing_version=manifest["preprocessing_version"],
                    layer=layer,
                    road_group=record["physical_road_group"],
                )
                jobs.append(
                    {
                        "case_id": case_id,
                        "record_id": record["record_id"],
                        "layer": layer,
                        "control": control,
                        "stride": stride,
                        "directory": str(destination),
                        "tolerance": tolerance,
                        "original": all_anchors[layer],
                        "anchors": anchors,
                    }
                )
                print(
                    json.dumps({**progress, "stage": "evidence_saved", "seconds": elapsed}),
                    flush=True,
                )
        del predictor, radar
    run_info.update(
        status="all_predictions_saved_scoring_started",
        evidence_jobs=len(jobs),
        prediction_seconds=time.perf_counter() - started,
    )
    write(args.output / "run.json", run_info)
    # Only after every requested model inference completes may labels enter scoring.
    from gpr_layer_audit.ml.processed_data import open_processed_record

    grouped, compact = defaultdict(list), []
    for job in jobs:
        record = records[job["record_id"]]
        arrays = open_processed_record(args.dataset_manifest, record, verify_hashes=True)
        stride, layer = job["stride"], job["layer"]
        take = slice(None, None, stride)
        directory = Path(job["directory"])
        with np.load(directory / "probabilities.npz", allow_pickle=False) as artifact:
            probabilities = artifact["probabilities"]
            native = artifact["native_trace_indices"]
        thresholds, calibration_status = stage_thresholds(
            calibrations, stride, layer, job["control"]
        )
        reports, stage_arrays = score_stages(
            {
                "amplitudes": arrays["amplitudes"][take],
                "sample_validity": arrays["sample_validity"][take],
            },
            arrays["labels"][layer - 1, take],
            probabilities,
            original_seeds=job["original"],
            inference_seeds=job["anchors"],
            native_traces=native,
            record=record,
            layer=layer,
            tolerance=job["tolerance"],
            decoder_config=decoder_config,
            thresholds=thresholds,
            label_valid=arrays["label_valid"][layer - 1, take],
            include_observations=not args.omit_observations,
        )
        cohort_hashes = {report["cohort_sha256"] for report in reports.values()}
        if len(cohort_hashes) != 1:
            raise RuntimeError("Stage scoring changed the fixed initial nonseed cohort")
        np.savez(directory / "stage-arrays.npz", native_trace_indices=native, **stage_arrays)
        span = float((len(native) - 1) * stride * record["horizontal_step_m"])
        document = {
            "case_id": job["case_id"],
            "record_id": job["record_id"],
            "control": job["control"],
            "evidence_metadata_sha256": sha(directory / "evidence.json"),
            "layer": layer,
            "physical_road_group": record["physical_road_group"],
            "trace_stride": stride,
            "retained_distance_span_m": span,
            "initial_workload": len(job["original"]),
            "initial_actions_per_km": len(job["original"]) * 1000 / span if span else None,
            "additional_actions": 0,
            "calibration": calibration_status,
            "stages": reports,
            "label_source_sha256": record["label_sha256"],
            "production_promotion": False,
            "ungated_stages_are_proposal_diagnostics": True,
            "control_score_interpretation": (
                "changed target uses valid alternate authorized seeds; "
                "original-target score is diagnostic"
                if job["control"] == "changed-target"
                else "fixed original target and nonseed cohort"
            ),
        }
        write(directory / "report.json", document)
        for stage, report in reports.items():
            report.update(case_id=job["case_id"], record_id=job["record_id"])
            grouped[(job["control"], stage)].append(
                {key: value for key, value in report.items() if key != "evaluation_observations"}
            )
            compact.append(
                {
                    "case_id": job["case_id"],
                    "record_id": job["record_id"],
                    "layer": layer,
                    "control": job["control"],
                    "stage": stage,
                    **{
                        key: report[key]
                        for key in (
                            "fixed_initial_nonseed_N",
                            "correct_accepted",
                            "wrong_accepted",
                            "unresolved",
                            "accepted_agreement",
                            "correct_coverage",
                            "wrong_coverage",
                            "correct_proposals_before_gating",
                        )
                    },
                }
            )
    summaries = {
        f"{control}/{stage}": aggregate_reports(reports)
        for (control, stage), reports in sorted(grouped.items())
    }
    write(
        args.output / "summary.json",
        {"per_case_layer": compact, "per_road_macro_pooled": summaries},
    )
    if source_hashes() != code or sha(metadata_path) != run_info["model_sha256"]:
        raise ValueError("Source or model metadata changed during evaluation")
    run_info.update(
        status="completed",
        target_label_arrays_opened=True,
        total_seconds=time.perf_counter() - started,
        summary_sha256=sha(args.output / "summary.json"),
    )
    write(args.output / "run.json", run_info)
    print(
        json.dumps(
            {
                "status": "completed",
                "cases": len(selected),
                "evidence_jobs": len(jobs),
                "summary": str(args.output / "summary.json"),
            }
        ),
        flush=True,
    )
    return run_info


def self_test():
    """Numerical integration check without loading torch, source data or a GPU."""
    sys.path.insert(0, str(ROOT / "src"))
    from gpr_layer_audit.ml.processed_dense_decoder import DenseDecoderConfig

    radar = np.tile([0.0, 1.0, 3.0, 1.0, 0.0, -1.0, -3.0, -1.0, 0.0], (10, 1))
    p = np.full_like(radar, 0.005)
    p[:, 2] = 0.96
    p /= p.sum(axis=1, keepdims=True)
    original = {0: 2.0, 9: 2.0}
    refs = np.full(10, 2.0)
    refs[4] = np.nan
    arguments = dict(
        radar={"amplitudes": radar, "sample_validity": np.ones_like(radar, bool)},
        references=refs,
        probabilities=p,
        original_seeds=original,
        native_traces=np.arange(10) * 4,
        record={
            "horizontal_step_m": 0.025,
            "sample_interval_ns": 0.03,
            "physical_road_group": "fixture",
        },
        layer=2,
        tolerance=2,
        decoder_config=DenseDecoderConfig(gap_probability=0.001),
        thresholds={"direct": 1.000001, "dense": 1.000001},
        label_valid=np.isfinite(refs),
        include_observations=True,
    )
    baseline, _ = score_stages(inference_seeds=original, **arguments)
    removed, _ = score_stages(inference_seeds={}, **arguments)
    assert baseline["direct_ungated"]["correct_accepted"] == 7
    assert baseline["direct_gated"]["accepted"] == 0
    assert baseline["dense_gated"]["accepted_agreement"] is None
    assert removed["seed_interpolation_no_radar"]["correct_accepted"] == 0
    assert {r["fixed_initial_nonseed_N"] for r in [*baseline.values(), *removed.values()]} == {7}
    assert len({r["cohort_sha256"] for r in [*baseline.values(), *removed.values()]}) == 1
    changed, detail = control_anchors({2: original, 3: {1: 6.0, 8: 6.0}}, 2, "changed-target")
    assert changed == {1: 6.0, 8: 6.0} and detail["model_layer_held_fixed"] == 2
    changed_report, _ = score_stages(inference_seeds=changed, **arguments)
    assert changed_report["dense_ungated"]["fixed_initial_nonseed_N"] == 7
    assert changed_report["dense_ungated"]["manual_answers_in_initial_cohort"] == 2
    print(json.dumps({"self_test": "passed", "fixed_N": 7, "cuda_loaded": "torch" in sys.modules}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--calibration", type=Path, action="append", default=[])
    parser.add_argument("--decoder-config", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--layers", type=int, nargs="+", choices=(1, 2, 3))
    parser.add_argument("--controls", choices=CONTROLS, nargs="*", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--omit-observations", action="store_true")
    parser.add_argument("--frozen-source", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    for key in ("model", "dataset_manifest", "cases", "output"):
        if getattr(args, key) is None:
            parser.error(f"--{key.replace('_', '-')} is required")
        setattr(args, key, getattr(args, key).resolve())
    if len(set(args.controls)) != len(args.controls):
        parser.error("Each conditioning control may be requested only once")
    if not args.frozen_source:
        return freeze_and_run(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
