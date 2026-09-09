"""Auditable labels, physical-road splits and incremental numeric radar caches."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gpr_layer_audit.catalog import discover_survey_catalog
from gpr_layer_audit.io import DZTFile
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.hybrid_evidence import PREPROCESSING_VERSION

ANNOTATION_FIELDS = (
    "source_sha256",
    "trace_index",
    "layer_order",
    "sample_raw",
    "visibility",
    "verified",
    "origin",
    "training_use",
    "selected_lobe",
    "pulse_width_samples",
)


def export_confirmed_annotations(result, stations, path, *, current_station_ids=None):
    """Export only current analyst clicks, mapped through the saved surface transform.

    Imported historical display-space seeds require reconfirmation before export.
    Never export automatic path points or validation checkpoints as training data.
    """
    if not result.source.fingerprint:
        raise ValueError("Run analysis to fingerprint the source before exporting labels")
    current_station_ids = set(current_station_ids or ())
    records = []
    for station in stations:
        row = int(np.argmin(abs(result.chainage_m - station.chainage_m)))
        # The centre trace and its surface are the coordinates used by the display.
        picks = [p for p in result.picks if p.chainage_m == result.chainage_m[row]]
        if not picks:
            continue
        for layer in station.user_confirmed:
            if (station.station_id, layer) not in current_station_ids:
                continue
            if not station.user_confirmed[layer]:
                continue
            visibility = str(station.visibility.get(layer, "visible"))
            sample = station.samples.get(layer)
            raw = (
                float(
                    transform_samples(
                        sample,
                        result.surface_samples_raw[row],
                        result.reference_surface_sample,
                        inverse=True,
                    )
                )
                if visibility == "visible" and sample is not None
                else ""
            )
            records.append(
                {
                    "source_sha256": result.source.fingerprint,
                    "trace_index": picks[0].trace_index,
                    "layer_order": layer,
                    "sample_raw": raw,
                    "visibility": visibility,
                    "verified": "true",
                    "origin": "manual_correction" if station.role == "correction" else "manual",
                    "training_use": "allowed",
                    "selected_lobe": station.selected_lobe.get(layer, ""),
                    "pulse_width_samples": station.pulse_width_samples.get(layer, 7),
                }
            )
    if not records:
        raise ValueError("No newly confirmed training observations; reconfirm imported seeds first")
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def physical_road_group(path: str | Path) -> str:
    """Conservative grouping of the supplied duplicate/portion/repeat acquisitions."""
    name = re.sub(r"[^a-z0-9]+", " ", Path(path).parent.name.lower())
    for city in (
        "talagang",
        "daska",
        "bahawalpur",
        "burewala",
        "jhang",
        "rawalpindi",
        "jamshoro",
        "pattoki",
        "mandiali",
    ):
        if city in name:
            return city
    if "gujrat" in name or "sohal" in name or "sohl" in name:
        return "sohal-gujrat"
    return re.sub(r"\b(back|forward|portion|prj|\d+)\b", "", name).strip()


def assign_splits(records: list[dict]) -> None:
    """Union physical groups and exact duplicates BEFORE producing any crops."""
    groups = sorted({r["road_group"] for r in records})
    parent = {g: g for g in groups}

    def root(g):
        while parent[g] != g:
            g = parent[g]
        return g

    hashes = {}
    for record in records:
        g = root(record["road_group"])
        other = hashes.setdefault(record["sha256"], g)
        parent[root(g)] = root(other)
    for record in records:
        record["road_group"] = root(record["road_group"])
    ordered = sorted(
        {r["road_group"] for r in records},
        key=lambda g: hashlib.sha256(("gpr-split-v1:" + g).encode()).hexdigest(),
    )
    count = len(ordered)
    ntest = max(1, count // 5) if count >= 3 else 0
    nvalidation = max(1, count // 5) if count >= 3 else 0
    splits = {
        g: "test" if i < ntest else "validation" if i < ntest + nvalidation else "train"
        for i, g in enumerate(ordered)
    }
    for record in records:
        record["split"] = splits[record["road_group"]]


def annotation_reason(row: dict, sources: dict[str, dict], *, evaluation=False) -> str | None:
    if row.get("training_use") != "allowed" and not (
        evaluation and row.get("training_use") in ("evaluation_only", "prohibited")
    ):
        return "training_not_authorized_or_evaluation_only"
    if str(row.get("verified", "")).lower() != "true":
        return "unverified"
    origins = (
        ("manual", "manual_correction", "checkpoint")
        if evaluation
        else ("manual", "manual_correction")
    )
    if row.get("origin") not in origins:
        return "not_a_manual_observation"
    source = sources.get(row.get("source_sha256", ""))
    if source is None:
        return "unknown_source_fingerprint"
    try:
        trace, layer = float(row["trace_index"]), int(row["layer_order"])
        if not np.isfinite(trace) or not 0 <= trace < source["traces"] or layer not in (1, 2, 3):
            return "invalid_trace_or_layer"
        if row.get("visibility") not in ("visible", "not_visible", "absent"):
            return "uncertain_visibility"
        if row["visibility"] == "visible":
            sample = float(row["sample_raw"])
            if not np.isfinite(sample) or not 0 <= sample < source["samples"]:
                return "invalid_raw_sample"
            if row.get("selected_lobe") not in ("positive_peak", "negative_trough"):
                return "unresolved_lobe"
        pulse = float(row.get("pulse_width_samples", 7))
        if not np.isfinite(pulse) or pulse <= 0:
            return "invalid_pulse_width"
    except (KeyError, ValueError, TypeError):
        return "missing_raw_coordinates"
    return None


def audit_dataset(data_root, *, seed_root=None, annotations=None, group_overrides=None) -> dict:
    catalog = discover_survey_catalog(Path(data_root))
    records, duplicate_map = [], {}
    for survey in catalog.roads:
        road = DZTFile(survey.dzt_path)
        digest = fingerprint_file(survey.dzt_path)
        if digest in duplicate_map:
            duplicate_map[digest]["aliases"].append(str(survey.dzt_path.resolve()))
            continue
        record = {
            "survey_id": survey.survey_id,
            "path": str(survey.dzt_path.resolve()),
            "sha256": digest,
            "aliases": [],
            "road_group": physical_road_group(survey.dzt_path),
            "traces": road.header.trace_count,
            "samples": road.header.samples_per_trace,
            "sample_interval_ns": road.header.sample_interval_ns,
            "horizontal_step_m": road.header.distance_per_trace_m,
        }
        if group_overrides and survey.survey_id in group_overrides:
            record["road_group"] = group_overrides[survey.survey_id]
        duplicate_map[digest] = record
        records.append(record)
    assign_splits(records)
    sources = {r["sha256"]: r for r in records}
    labels, exclusions, seed_inventory = [], Counter(), []
    evaluation_labels = []
    if annotations:
        with Path(annotations).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                evaluation_only = row.get("training_use") in ("evaluation_only", "prohibited")
                reason = annotation_reason(row, sources, evaluation=evaluation_only)
                if reason:
                    exclusions[reason] += 1
                else:
                    row = {
                        **row,
                        "road_group": sources[row["source_sha256"]]["road_group"],
                        "split": sources[row["source_sha256"]]["split"],
                    }
                    (evaluation_labels if evaluation_only else labels).append(row)
    evaluation_groups = {r["road_group"] for r in evaluation_labels}
    for source in records:
        if source["road_group"] in evaluation_groups:
            source["split"] = "test"
    for row in [*labels, *evaluation_labels]:
        row["split"] = sources[row["source_sha256"]]["split"]
    if seed_root:
        files = (
            [Path(seed_root)]
            if Path(seed_root).is_file()
            else sorted(Path(seed_root).rglob("*.json"))
        )
        for file in files:
            try:
                document = json.loads(file.read_text(encoding="utf-8-sig"))
            except (ValueError, OSError):
                continue
            if not isinstance(document, dict):
                continue
            if document.get("training_use") == "prohibited" or "checkpoints" in document:
                seed_inventory.append({"path": str(file), "status": "evaluation_only"})
                continue
            if "stations" not in document:
                continue
            picks = [
                p for station in document["stations"] for p in station.get("picks", {}).values()
            ]
            seed_inventory.append(
                {
                    "path": str(file),
                    "survey_id": document.get("survey_id"),
                    "confirmed_picks": sum(p.get("user_confirmed") is True for p in picks),
                    "status": "requires_source_fingerprint_and_raw_coordinate_alignment",
                }
            )
    references = []
    from gpr_layer_audit.reference import normalize_reference_workbook

    for file in catalog.reference_files:
        try:
            points = normalize_reference_workbook(file)
            references.append(
                {
                    "path": str(file),
                    "rows": len(points),
                    "origins": dict(Counter(str(p.label_origin) for p in points)),
                    "status": "depth_alignment_unverified_not_training_labels",
                }
            )
        except (ValueError, OSError, KeyError) as exc:
            references.append({"path": str(file), "status": "unreadable", "reason": str(exc)})
    eligibility = {}
    for layer in (1, 2, 3):
        selected = [r for r in labels if int(r["layer_order"]) == layer]
        counts = Counter(r["split"] for r in selected)
        train_groups = {r["road_group"] for r in selected if r["split"] == "train"}
        valid_groups = {r["road_group"] for r in selected if r["split"] == "validation"}
        eligible = len(train_groups) >= 3 and bool(valid_groups) and counts["train"] >= 96
        eligibility[str(layer)] = {
            "eligible_for_pilot": eligible,
            "labels_by_split": dict(counts),
            "training_groups": sorted(train_groups),
            "reason": "ready" if eligible else "insufficient_verified_aligned_training_labels",
        }
    return {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "data_root": str(Path(data_root).resolve()),
        "sources": records,
        "annotations": labels,
        "evaluation_annotations": evaluation_labels,
        "exclusions": dict(exclusions),
        "seed_inventory": seed_inventory,
        "reference_inventory": references,
        "eligibility": eligibility,
        "field_accuracy_established": False,
        "annotation_fields": list(ANNOTATION_FIELDS),
    }


def transform_samples(samples, surface_samples, reference_surface, *, inverse=False):
    shift = reference_surface - np.asarray(surface_samples)
    return np.asarray(samples) - shift if inverse else np.asarray(samples) + shift


def build_dataset(manifest_path, output_dir, *, horizontal_step_m=0.4) -> dict:
    """Materialize fixed-grid chunks, keeping raw/aligned transforms per row."""
    manifest_path, output = Path(manifest_path), Path(output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")
    output.mkdir(parents=True, exist_ok=True)
    sources = {r["sha256"]: r for r in manifest["sources"]}
    by_source = defaultdict(list)
    for annotation in manifest["annotations"]:
        reason = annotation_reason(annotation, sources)
        if reason:
            raise ValueError(f"Dataset annotation is ineligible: {reason}")
        source = sources[annotation["source_sha256"]]
        if (
            annotation.get("split") != source["split"]
            or annotation.get("road_group") != source["road_group"]
        ):
            raise ValueError("Annotation split does not match its physical road")
        by_source[annotation["source_sha256"]].append(annotation)
    chunks = []
    for digest, labels in by_source.items():
        source = sources[digest]
        if fingerprint_file(source["path"]) != digest:
            raise ValueError("Source changed after dataset audit")
        road = DZTFile(source["path"])
        step = source["horizontal_step_m"]
        if step is None or step <= 0:
            raise ValueError("Physical horizontal spacing is required for training")
        stack = max(1, math.ceil(horizontal_step_m / step))
        reference = None
        for start in range(0, road.header.trace_count, 256 * stack):
            stop = min(road.header.trace_count, start + 256 * stack)
            selected = [r for r in labels if start <= float(r["trace_index"]) < stop]
            if not selected:
                continue
            calibrated = calibrate(road, None, stack_size=stack, start_trace=start, stop_trace=stop)
            if reference is None:
                reference = calibrated.reference_surface_sample
            shift = reference - calibrated.reference_surface_sample
            from gpr_layer_audit.processing.pipeline import _shift_sample_axis

            signal = _shift_sample_axis(calibrated.measurement_radargram, shift)
            name = f"{digest[:16]}-{start}"
            np.save(output / f"{name}.npy", signal)
            np.save(output / f"{name}-surface.npy", calibrated.surface_samples)
            aligned = []
            for label in selected:
                row = int(np.argmin(abs(calibrated.trace_centres - float(label["trace_index"]))))
                sample = (
                    float(
                        transform_samples(
                            float(label["sample_raw"]), calibrated.surface_samples[row], reference
                        )
                    )
                    if label["visibility"] == "visible"
                    else None
                )
                if sample is not None and not 0 <= sample < signal.shape[1]:
                    raise ValueError("Label was moved outside the aligned acquisition")
                aligned.append({**label, "row": row, "sample_aligned": sample})
            chunks.append(
                {
                    "path": str((output / f"{name}.npy").resolve()),
                    "surface_path": str((output / f"{name}-surface.npy").resolve()),
                    "source_sha256": digest,
                    "road_group": source["road_group"],
                    "split": source["split"],
                    "trace_start": start,
                    "stack_size": stack,
                    "reference_surface_sample": reference,
                    "horizontal_step_m": step * stack,
                    "sample_interval_ns": source["sample_interval_ns"],
                    "labels": aligned,
                }
            )
    result = {**manifest, "chunks": chunks, "audit_sha256": fingerprint_file(manifest_path)}
    (output / "dataset.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def make_targets(shape, labels, layer_order):
    """Unknown traces are ignored, not labelled background. Absence is supervised."""
    target, valid = np.zeros(shape, np.float32), np.zeros(shape, bool)
    visible, observed = np.zeros(shape[0], np.float32), np.zeros(shape[0], bool)
    x = np.arange(shape[1])
    for label in labels:
        if int(label["layer_order"]) != layer_order:
            continue
        row = int(label["row"])
        observed[row] = True
        if label["visibility"] == "visible":
            width = max(1, float(label.get("pulse_width_samples", 7)) / 4)
            target[row] = np.exp(-0.5 * ((x - float(label["sample_aligned"])) / width) ** 2)
            valid[row] = True
            visible[row] = 1
        elif label["visibility"] == "absent":
            valid[row] = True
        # Not-visible supervises visibility only: the layer may still exist.
    return target, valid, visible, observed
