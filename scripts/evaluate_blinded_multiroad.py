"""Evaluate frozen radar-only seeds against references revealed afterward.

The seed manifest must be created before this script runs. Each road is first
tracked, including leave-one-station-out fits, before its workbook is opened.
Only manual (not formula/interpolated) workbook rows outside the seed
neighborhoods are validation checkpoints. Physical-thickness metrics are
withheld unless seed identity and local time-to-depth scale both pass their
independent gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.models import LabelOrigin, LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.dielectric import LIGHT_SPEED_M_PER_S
from gpr_layer_audit.processing.pipeline import (
    _chainage,
    _detect_anomalies,
    _effective_stack,
)
from gpr_layer_audit.processing.preprocessing import (
    measurement_packet_support,
    preprocess_for_interpretation,
)
from gpr_layer_audit.processing.tracker import pick_interfaces
from gpr_layer_audit.reference import normalize_reference_workbook

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "benchmarks/blinded-radar-only-seeds-20260829.json"
FROZEN_OUTPUT = ROOT / "exports/blinded-multiroad-20260829"
OUTPUT = ROOT / "exports/blinded-multiroad-strict-scale-20260829"
PULSE_WIDTH_SAMPLES = 7.0
SEED_EXCLUSION_M = 25.0
MAX_SCALE_REFERENCE_DISTANCE_M = 10.0
MIN_LOCAL_SCALE_POINTS = 2
MAX_SCALE_RELATIVE_DEVIATION = 0.15
MIN_RELEASE_ROADS = 3
MIN_RELEASE_CHECKPOINTS_PER_ROAD = 30
MIN_RELEASE_AUTOMATIC_COVERAGE = 0.85
MIN_RELEASE_VISIBLE_ACCURACY = 0.85


def _run_tracker(radar, surface, layers, branches, anomaly_mask, chainage, anchors):
    spacing = float(np.median(np.diff(chainage))) if len(chainage) > 1 else 0.4
    return pick_interfaces(
        radar,
        surface,
        layers,
        anchor_samples=anchors,
        feature_branches=dict(branches),
        search_corridors={},
        design_weight=0.0,
        anomaly_mask=anomaly_mask,
        break_rows=set(),
        max_interpolation_rows=0,
        horizontal_step_m=spacing,
    )


def _observation(paths, order: int, row: int) -> tuple[float | None, bool]:
    graph = float(paths[order]["canonical"][row])
    return (graph if graph >= 0 else None, bool(paths[order]["samples"][row] >= 0))


def _tracking_orders(case: dict) -> tuple[int, ...]:
    validation = {int(value) for value in case.get("validation_layers", (1, 2))}
    if not validation or min(validation) < 1:
        raise ValueError("validation_layers must contain positive layer orders")
    return tuple(range(1, max(validation) + 1))


def _path_arrays(full, orders: tuple[int, ...] | None = None):
    selected_orders = orders or tuple(sorted(full))
    return {
        order: {
            "samples": np.asarray(full[order].samples),
            "alternate": np.asarray(full[order].alternate_samples),
            "graph": np.asarray(full[order].evidence["graph_selected_sample"]),
            "canonical": np.asarray(full[order].evidence["canonical_event_sample"]),
        }
        for order in selected_orders
    }


def _load_frozen_case(
    case_id: str,
    orders: tuple[int, ...],
    output: Path = OUTPUT,
):
    locations = tuple(dict.fromkeys((output, OUTPUT, FROZEN_OUTPUT)))
    source = next(
        (
            location
            for location in locations
            if (location / f"{case_id}-paths.npz").exists()
            and (location / f"{case_id}-summary.json").exists()
        ),
        None,
    )
    if source is None:
        raise FileNotFoundError(
            f"Frozen evidence is incomplete for {case_id}: "
            f"expected {case_id}-paths.npz and {case_id}-summary.json"
        )
    path_file = source / f"{case_id}-paths.npz"
    summary_file = source / f"{case_id}-summary.json"
    with np.load(path_file) as data:
        chainage = np.asarray(data["chainage_m"], dtype=float)
        paths = {
            order: {
                "samples": np.asarray(data[f"layer_{order}_samples"]),
                "alternate": np.asarray(
                    data[f"layer_{order}_alternate"]
                    if f"layer_{order}_alternate" in data
                    else np.full_like(data[f"layer_{order}_samples"], -1)
                ),
                "graph": np.asarray(data[f"layer_{order}_graph"]),
                "canonical": np.asarray(data[f"layer_{order}_canonical"]),
            }
            for order in orders
        }
    prior = json.loads(summary_file.read_text(encoding="utf-8"))
    return chainage, paths, prior["dropout_audit"]


def _nearest_manual(points, order: int, chainage_m: float):
    candidates = [
        item
        for item in points
        if item.layer_order == order and item.label_origin == LabelOrigin.MANUAL
    ]
    return min(candidates, key=lambda item: abs(item.chainage_m - chainage_m), default=None)


def _case_reference_points(case, reference_path: Path):
    points = normalize_reference_workbook(reference_path)
    road_stem = Path(case["road"]).stem.casefold()
    matching = [item for item in points if road_stem in item.road_id.casefold()]
    selected = matching or points
    offset = float(case.get("_reference_chainage_offset_m", 0.0))
    if offset:
        selected = [
            replace(item, chainage_m=float(item.chainage_m - offset))
            for item in selected
        ]
    return selected


def _dropout_assessment(dropout, validation_orders):
    by_layer = {}
    for order in sorted(validation_orders):
        items = [item for item in dropout if int(item["layer_order"]) == order]
        passing = sum(bool(item["within_one_pulse_width"]) for item in items)
        differences = [
            float(item["absolute_sample_difference"])
            for item in items
            if item["absolute_sample_difference"] is not None
        ]
        by_layer[str(order)] = {
            "withheld_seed_count": len(items),
            "within_one_pulse_count": passing,
            "within_one_pulse_percent": 100.0 * passing / len(items) if items else 0.0,
            "maximum_absolute_sample_difference": max(differences, default=None),
            "all_within_one_pulse": bool(items) and passing == len(items),
        }
    return by_layer


def _assess_scale(records):
    local = [item for item in records if item["eligible_local_reference"]]
    if len(local) < MIN_LOCAL_SCALE_POINTS:
        return {
            "status": "rejected",
            "reason": "insufficient_local_manual_scale_references",
            "local_reference_count": len(local),
            "fitted_mm_per_sample": None,
            "maximum_relative_deviation": None,
            "records": records,
        }
    ratios = np.asarray([item["mm_per_sample"] for item in local], dtype=float)
    median = float(np.median(ratios))
    deviations = np.abs(ratios - median) / median
    maximum = float(np.max(deviations))
    implied = [
        float(item["implied_dielectric"])
        for item in local
        if item.get("implied_dielectric") is not None
    ]
    physical = not implied or all(1.0 < value <= 40.0 for value in implied)
    accepted = maximum <= MAX_SCALE_RELATIVE_DEVIATION and physical
    return {
        "status": "accepted" if accepted else "rejected",
        "reason": (
            "local_scale_consistent"
            if accepted
            else (
                "local_scale_ratios_exceed_relative_deviation_limit"
                if maximum > MAX_SCALE_RELATIVE_DEVIATION
                else "implied_dielectric_outside_physical_bounds"
            )
        ),
        "local_reference_count": len(local),
        "fitted_mm_per_sample": median if accepted else None,
        "diagnostic_median_mm_per_sample": median,
        "maximum_relative_deviation": maximum,
        "implied_dielectric_range": (
            [min(implied), max(implied)] if implied else None
        ),
        "records": records,
    }


def _scale_assessments(
    case,
    points,
    surface_sample,
    sample_interval_ns,
    validation_orders,
):
    assessments = {}
    for order in sorted(validation_orders):
        records = []
        for station in case["stations"]:
            seed_chainage = float(station["chainage_m"])
            reference = _nearest_manual(points, order, seed_chainage)
            bottom = float(station["samples"][str(order)])
            top = (
                float(surface_sample)
                if order == 1
                else float(station["samples"][str(order - 1)])
            )
            if reference is None or reference.individual_thickness_mm is None:
                continue
            distance = abs(float(reference.chainage_m) - seed_chainage)
            gap = bottom - top
            if gap <= 0:
                continue
            records.append(
                {
                    "layer_order": order,
                    "seed_chainage_m": seed_chainage,
                    "reference_chainage_m": float(reference.chainage_m),
                    "reference_distance_m": distance,
                    "eligible_local_reference": (
                        distance <= MAX_SCALE_REFERENCE_DISTANCE_M
                    ),
                    "reference_individual_thickness_mm": float(
                        reference.individual_thickness_mm
                    ),
                    "manual_seed_sample_gap": gap,
                    "mm_per_sample": float(reference.individual_thickness_mm / gap),
                    "implied_dielectric": float(
                        (
                            LIGHT_SPEED_M_PER_S
                            * 1e-6
                            * sample_interval_ns
                            / (
                                2.0
                                * (reference.individual_thickness_mm / gap)
                            )
                        )
                        ** 2
                    ),
                }
            )
        assessments[str(order)] = _assess_scale(records)
    return assessments


def _eligible_checkpoint(reference, order, case, chainage):
    return bool(
        int(reference.layer_order) == order
        and reference.label_origin == LabelOrigin.MANUAL
        and reference.individual_thickness_mm is not None
        and chainage[0] <= reference.chainage_m <= chainage[-1]
        and all(
            abs(reference.chainage_m - float(station["chainage_m"]))
            > SEED_EXCLUSION_M
            for station in case["stations"]
        )
    )


def _release_assessment(cases):
    orders = sorted(
        {
            int(order)
            for case in cases
            for order in case.get("thickness_validation_by_layer", {})
        }
    )
    output = {}
    for order in orders:
        qualifying = []
        supplementary = []
        for case in cases:
            result = case.get("thickness_validation_by_layer", {}).get(str(order))
            if not result or result["status"] != "evaluated":
                continue
            item = {
                "case_id": case["case_id"],
                "held_out_manual_checkpoints": result[
                    "eligible_held_out_manual_checkpoints"
                ],
            }
            if not case.get("release_eligible", True):
                item["release_exclusion_reason"] = case.get(
                    "release_exclusion_reason",
                    "declared_development_case",
                )
                supplementary.append(item)
                continue
            if (
                result["eligible_held_out_manual_checkpoints"]
                >= MIN_RELEASE_CHECKPOINTS_PER_ROAD
            ):
                qualifying.append((case, result, item))
            else:
                supplementary.append(item)
        checkpoints = [
            checkpoint
            for _, result, _ in qualifying
            for checkpoint in result["checkpoints"]
        ]
        visible = [item for item in checkpoints if item["automatically_visible"]]
        coverage = len(visible) / len(checkpoints) if checkpoints else 0.0
        visible_accuracy = (
            sum(bool(item["within_release_target"]) for item in visible) / len(visible)
            if visible
            else 0.0
        )
        reasons = []
        if len(qualifying) < MIN_RELEASE_ROADS:
            reasons.append("fewer_than_three_qualifying_roads")
        if coverage < MIN_RELEASE_AUTOMATIC_COVERAGE:
            reasons.append("aggregate_automatic_coverage_below_85_percent")
        if visible_accuracy < MIN_RELEASE_VISIBLE_ACCURACY:
            reasons.append("aggregate_visible_accuracy_below_85_percent")
        output[str(order)] = {
            "field_accuracy_established": not reasons,
            "reasons": reasons,
            "qualifying_roads": [item for _, _, item in qualifying],
            "supplementary_or_excluded_roads": supplementary,
            "qualifying_road_count": len(qualifying),
            "held_out_manual_checkpoints": len(checkpoints),
            "automatically_visible_checkpoints": len(visible),
            "aggregate_automatic_coverage_percent": 100.0 * coverage,
            "aggregate_visible_within_release_target_percent": (
                100.0 * visible_accuracy
            ),
        }
    return output


def _thickness_evaluation(
    case,
    points,
    chainage,
    paths,
    surface_sample,
    order,
    dropout_layer,
    scale_layer,
    identity_chain_stable: bool | None = None,
):
    references = [
        item
        for item in points
        if _eligible_checkpoint(item, order, case, chainage)
    ]
    reasons = []
    if identity_chain_stable is None:
        identity_chain_stable = dropout_layer["all_within_one_pulse"]
    if not dropout_layer["all_within_one_pulse"]:
        reasons.append("seed_identity_dropout_failed")
    elif not identity_chain_stable:
        reasons.append("upstream_interface_seed_identity_dropout_failed")
    if scale_layer["status"] != "accepted":
        reasons.append(scale_layer["reason"])
    if reasons:
        return {
            "status": "not_evaluated",
            "reasons": reasons,
            "eligible_held_out_manual_checkpoints": len(references),
            "metrics": None,
            "checkpoints": [],
        }

    scale = float(scale_layer["fitted_mm_per_sample"])
    checkpoints = []
    for reference in references:
        row = int(np.argmin(np.abs(chainage - reference.chainage_m)))
        bottom, bottom_visible = _observation(paths, order, row)
        if order == 1:
            top = float(surface_sample)
            top_visible = True
        else:
            top, top_visible = _observation(paths, order - 1, row)
        graph_thickness = (
            (bottom - top) * scale
            if bottom is not None and top is not None and bottom > top
            else None
        )
        visible = bool(bottom_visible and top_visible and graph_thickness is not None)
        error = (
            abs(graph_thickness - reference.individual_thickness_mm)
            if graph_thickness is not None
            else None
        )
        target = 12.7 if order == 1 else 25.4
        checkpoints.append(
            {
                "chainage_m": float(reference.chainage_m),
                "reference_individual_thickness_mm": float(
                    reference.individual_thickness_mm
                ),
                "graph_individual_thickness_mm": graph_thickness,
                "graph_top_sample": top,
                "graph_bottom_sample": bottom,
                "graph_alternate_bottom_sample": (
                    float(paths[order]["alternate"][row])
                    if paths[order]["alternate"][row] >= 0
                    else None
                ),
                "automatically_visible": visible,
                "absolute_error_mm": error,
                "within_release_target": bool(error is not None and error <= target),
            }
        )
    graph = [item for item in checkpoints if item["absolute_error_mm"] is not None]
    visible = [item for item in graph if item["automatically_visible"]]
    metrics = {
        "held_out_manual_checkpoints": len(checkpoints),
        "graph_resolved_checkpoints": len(graph),
        "automatically_visible_checkpoints": len(visible),
        "automatic_coverage_percent": (
            100.0 * len(visible) / len(checkpoints) if checkpoints else 0.0
        ),
        "graph_within_release_target_percent": (
            100.0
            * sum(bool(item["within_release_target"]) for item in graph)
            / len(graph)
            if graph
            else 0.0
        ),
        "visible_within_release_target_percent": (
            100.0
            * sum(bool(item["within_release_target"]) for item in visible)
            / len(visible)
            if visible
            else 0.0
        ),
        "visible_median_absolute_error_mm": (
            float(np.median([item["absolute_error_mm"] for item in visible]))
            if visible
            else None
        ),
    }
    return {
        "status": "evaluated",
        "reasons": [],
        "eligible_held_out_manual_checkpoints": len(references),
        "metrics": metrics,
        "checkpoints": checkpoints,
    }


def _evaluate_case(
    case: dict,
    *,
    reuse_frozen_paths: bool,
    output: Path = OUTPUT,
) -> dict[str, object]:
    road_path = ROOT / case["road"]
    plate_path = ROOT / case["plate"]
    road, plate = DZTFile(road_path), DZTFile(plate_path)
    stack = _effective_stack(
        road.header.trace_count, 0, road.header.distance_per_trace_m
    )
    calibrated = calibrate(road, plate, stack_size=stack)
    dzx_path = road_path.with_suffix(".DZX")
    dzx = read_dzx(dzx_path) if dzx_path.exists() else None
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    station_rows = [
        int(np.argmin(np.abs(chainage - float(station["chainage_m"]))))
        for station in case["stations"]
    ]
    tracking_orders = _tracking_orders(case)
    anchors = {
        order: {
            row: int(station["samples"][str(order)])
            for row, station in zip(station_rows, case["stations"], strict=True)
        }
        for order in tracking_orders
    }
    if reuse_frozen_paths:
        frozen_chainage, paths, dropout = _load_frozen_case(
            case["case_id"], tracking_orders, output
        )
        if frozen_chainage.shape != chainage.shape or not np.allclose(
            frozen_chainage, chainage
        ):
            raise ValueError(f"Frozen chainage no longer matches {case['case_id']}")
        chainage = frozen_chainage
    else:
        interpreted = preprocess_for_interpretation(
            calibrated.radargram,
            calibrated.reference_surface_sample,
            calibrated.plate_template,
        )
        branches = {
            **interpreted.feature_branches,
            "measurement_support": measurement_packet_support(
                calibrated.measurement_radargram
            ),
        }
        anomaly_mask, _ = _detect_anomalies(
            chainage, branches.get("anomaly_score")
        )
        layers = LayerSpec.defaults()[: max(tracking_orders)]
        full = _run_tracker(
            interpreted.radargram,
            calibrated.reference_surface_sample,
            layers,
            branches,
            anomaly_mask,
            chainage,
            anchors,
        )
        paths = _path_arrays(full, tracking_orders)
        dropout = []
        for held_index, held_row in enumerate(station_rows):
            reduced = {
                order: {
                    row: sample
                    for row, sample in values.items()
                    if row != held_row
                }
                for order, values in anchors.items()
            }
            withheld = _run_tracker(
                interpreted.radargram,
                calibrated.reference_surface_sample,
                layers,
                branches,
                anomaly_mask,
                chainage,
                reduced,
            )
            for order in tracking_orders:
                expected = anchors[order][held_row]
                graph = float(
                    withheld[order].evidence["graph_selected_sample"][held_row]
                )
                dropout.append(
                    {
                        "station_index": held_index,
                        "chainage_m": float(chainage[held_row]),
                        "layer_order": order,
                        "withheld_sample": expected,
                        "independent_graph_sample": graph if graph >= 0 else None,
                        "absolute_sample_difference": (
                            abs(graph - expected) if graph >= 0 else None
                        ),
                        "within_one_pulse_width": bool(
                            graph >= 0
                            and abs(graph - expected) <= PULSE_WIDTH_SAMPLES
                        ),
                        "automatically_visible": bool(
                            withheld[order].samples[held_row] >= 0
                        ),
                    }
                )
        arrays: dict[str, np.ndarray] = {"chainage_m": chainage}
        for order in tracking_orders:
            arrays.update(
                {
                    f"layer_{order}_samples": paths[order]["samples"],
                    f"layer_{order}_alternate": paths[order]["alternate"],
                    f"layer_{order}_graph": paths[order]["graph"],
                    f"layer_{order}_canonical": paths[order]["canonical"],
                }
            )
        np.savez_compressed(output / f"{case['case_id']}-paths.npz", **arrays)

    # Reference values are opened only after the full and dropout radar fits
    # above are frozen in memory.
    points = _case_reference_points(case, ROOT / case["reference"])
    validation_orders = set(case.get("validation_layers", (1, 2)))
    dropout_by_layer = _dropout_assessment(dropout, validation_orders)
    scale_by_layer = _scale_assessments(
        case,
        points,
        calibrated.reference_surface_sample,
        road.header.sample_interval_ns,
        validation_orders,
    )
    thickness_by_layer = {
        str(order): _thickness_evaluation(
            case,
            points,
            chainage,
            paths,
            calibrated.reference_surface_sample,
            order,
            dropout_by_layer[str(order)],
            scale_by_layer[str(order)],
            all(
                dropout_by_layer[str(upper)]["all_within_one_pulse"]
                for upper in range(1, order + 1)
                if str(upper) in dropout_by_layer
            ),
        )
        for order in sorted(validation_orders)
    }
    return {
        "case_id": case["case_id"],
        "road": str(road_path),
        "reference_revealed_after_tracking": True,
        "tracking_source": "frozen_prior_run" if reuse_frozen_paths else "new_run",
        "release_eligible": bool(case.get("release_eligible", True)),
        "release_exclusion_reason": case.get("release_exclusion_reason"),
        "dropout_audit": dropout,
        "event_identity_by_layer": dropout_by_layer,
        "scale_calibration_by_layer": scale_by_layer,
        "thickness_validation_by_layer": thickness_by_layer,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-frozen-paths",
        action="store_true",
        help="reuse the pre-reveal paths and dropout evidence from the first run",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="evaluate only this frozen case and merge it into the existing summary",
    )
    parser.add_argument(
        "--seed-manifest",
        type=Path,
        default=SEEDS,
        help="seed manifest to evaluate (defaults to the original frozen set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="evidence directory; use a new directory for a new blind cohort",
    )
    parser.add_argument(
        "--reference-alignment",
        type=Path,
        help="post-tracking workbook-to-radar chainage offsets, keyed by case id",
    )
    args = parser.parse_args(argv)
    seed_manifest = (
        args.seed_manifest
        if args.seed_manifest.is_absolute()
        else ROOT / args.seed_manifest
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(seed_manifest.read_text(encoding="utf-8"))
    alignment_path = None
    alignment: dict[str, object] = {}
    if args.reference_alignment is not None:
        alignment_path = (
            args.reference_alignment
            if args.reference_alignment.is_absolute()
            else ROOT / args.reference_alignment
        )
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    offsets = alignment.get("reference_chainage_offset_m_by_case", {})
    for case in manifest["cases"]:
        if case["case_id"] in offsets:
            case["_reference_chainage_offset_m"] = float(offsets[case["case_id"]])
    cases = []
    selected_cases = [
        case
        for case in manifest["cases"]
        if not args.case_ids or case["case_id"] in set(args.case_ids)
    ]
    if args.case_ids and len(selected_cases) != len(set(args.case_ids)):
        known = {case["case_id"] for case in manifest["cases"]}
        missing = sorted(set(args.case_ids) - known)
        parser.error(f"unknown case(s): {', '.join(missing)}")
    for case in selected_cases:
        action = "Loading frozen paths for" if args.reuse_frozen_paths else "Tracking"
        print(f"{action} {case['case_id']}...", flush=True)
        result = _evaluate_case(
            case,
            reuse_frozen_paths=args.reuse_frozen_paths,
            output=output,
        )
        cases.append(result)
        (output / f"{case['case_id']}-summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    summary_path = output / "summary.json"
    if args.case_ids and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        merged = {case["case_id"]: case for case in existing.get("cases", [])}
        merged.update({case["case_id"]: case for case in cases})
        cases = [
            merged[case["case_id"]]
            for case in manifest["cases"]
            if case["case_id"] in merged
        ]
    summary = {
        "purpose": "blinded multi-road radar-only seed validation",
        "seed_manifest": str(seed_manifest),
        "seed_manifest_sha256": hashlib.sha256(seed_manifest.read_bytes()).hexdigest(),
        "reference_alignment": str(alignment_path) if alignment_path else None,
        "reference_alignment_sha256": (
            hashlib.sha256(alignment_path.read_bytes()).hexdigest()
            if alignment_path
            else None
        ),
        "seed_exclusion_radius_m": SEED_EXCLUSION_M,
        "maximum_scale_reference_distance_m": MAX_SCALE_REFERENCE_DISTANCE_M,
        "minimum_local_scale_points": MIN_LOCAL_SCALE_POINTS,
        "maximum_scale_relative_deviation": MAX_SCALE_RELATIVE_DEVIATION,
        "minimum_release_roads": MIN_RELEASE_ROADS,
        "minimum_release_checkpoints_per_road": MIN_RELEASE_CHECKPOINTS_PER_ROAD,
        "minimum_release_automatic_coverage": MIN_RELEASE_AUTOMATIC_COVERAGE,
        "minimum_release_visible_accuracy": MIN_RELEASE_VISIBLE_ACCURACY,
        "tracking_source": (
            "frozen_prior_run" if args.reuse_frozen_paths else "new_run"
        ),
        "field_accuracy_by_layer": _release_assessment(cases),
        "field_accuracy_established": False,
        "cases": cases,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    compact = {
        case["case_id"]: {
            order: {
                "event_identity_stable": case["event_identity_by_layer"][order][
                    "all_within_one_pulse"
                ],
                "scale_status": case["scale_calibration_by_layer"][order]["status"],
                "thickness_status": value["status"],
                "metrics": value["metrics"],
            }
            for order, value in case["thickness_validation_by_layer"].items()
        }
        for case in cases
    }
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
