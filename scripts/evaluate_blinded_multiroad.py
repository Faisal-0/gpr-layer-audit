"""Evaluate frozen radar-only seeds against references revealed afterward.

The seed manifest must be created before this script runs. Each road is first
tracked, including leave-one-station-out fits, before its workbook is opened.
Only manual (not formula/interpolated) workbook rows outside the seed
neighborhoods are validation checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.models import LabelOrigin, LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
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
OUTPUT = ROOT / "exports/blinded-multiroad-20260829"
PULSE_WIDTH_SAMPLES = 7.0
SEED_EXCLUSION_M = 25.0


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


def _observation(path, row: int) -> tuple[float | None, bool]:
    graph = float(path.evidence["canonical_event_sample"][row])
    return (graph if graph >= 0 else None, bool(path.samples[row] >= 0))


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
    return matching or points


def _evaluate_case(case: dict) -> dict[str, object]:
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
    layers = LayerSpec.defaults()[:2]
    station_rows = [
        int(np.argmin(np.abs(chainage - float(station["chainage_m"]))))
        for station in case["stations"]
    ]
    anchors = {
        order: {
            row: int(station["samples"][str(order)])
            for row, station in zip(station_rows, case["stations"], strict=True)
        }
        for order in (1, 2)
    }
    full = _run_tracker(
        interpreted.radargram,
        calibrated.reference_surface_sample,
        layers,
        branches,
        anomaly_mask,
        chainage,
        anchors,
    )
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
        for order in (1, 2):
            expected = anchors[order][held_row]
            graph = float(withheld[order].evidence["graph_selected_sample"][held_row])
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
                        graph >= 0 and abs(graph - expected) <= PULSE_WIDTH_SAMPLES
                    ),
                    "automatically_visible": bool(withheld[order].samples[held_row] >= 0),
                }
            )

    # Reference values are opened only after the full and dropout radar fits
    # above are frozen in memory.
    points = _case_reference_points(case, ROOT / case["reference"])
    validation_orders = set(case.get("validation_layers", (1, 2)))
    scales: dict[int, float] = {}
    calibration_records = []
    for order in sorted(validation_orders):
        ratios = []
        for station, row in zip(case["stations"], station_rows, strict=True):
            reference = _nearest_manual(points, order, float(station["chainage_m"]))
            bottom, _ = _observation(full[order], row)
            top = (
                float(calibrated.reference_surface_sample)
                if order == 1
                else _observation(full[order - 1], row)[0]
            )
            if (
                reference is None
                or reference.individual_thickness_mm is None
                or bottom is None
                or top is None
                or bottom <= top
            ):
                continue
            ratio = float(reference.individual_thickness_mm / (bottom - top))
            ratios.append(ratio)
            calibration_records.append(
                {
                    "layer_order": order,
                    "seed_chainage_m": float(station["chainage_m"]),
                    "reference_chainage_m": float(reference.chainage_m),
                    "reference_individual_thickness_mm": float(
                        reference.individual_thickness_mm
                    ),
                    "sample_gap": float(bottom - top),
                    "mm_per_sample": ratio,
                }
            )
        if ratios:
            scales[order] = float(np.median(ratios))

    checkpoints = []
    for reference in points:
        order = int(reference.layer_order)
        if (
            order not in validation_orders
            or reference.label_origin != LabelOrigin.MANUAL
            or reference.individual_thickness_mm is None
            or order not in scales
            or not chainage[0] <= reference.chainage_m <= chainage[-1]
            or any(
                abs(reference.chainage_m - float(station["chainage_m"]))
                <= SEED_EXCLUSION_M
                for station in case["stations"]
            )
        ):
            continue
        row = int(np.argmin(np.abs(chainage - reference.chainage_m)))
        bottom, bottom_visible = _observation(full[order], row)
        if order == 1:
            top = float(calibrated.reference_surface_sample)
            top_visible = True
        else:
            top, top_visible = _observation(full[order - 1], row)
        graph_thickness = (
            (bottom - top) * scales[order]
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
                "layer_order": order,
                "chainage_m": float(reference.chainage_m),
                "reference_individual_thickness_mm": float(
                    reference.individual_thickness_mm
                ),
                "graph_individual_thickness_mm": graph_thickness,
                "automatically_visible": visible,
                "absolute_error_mm": error,
                "within_release_target": bool(error is not None and error <= target),
            }
        )

    metrics = {}
    for order in sorted(validation_orders):
        layer = [item for item in checkpoints if item["layer_order"] == order]
        graph = [item for item in layer if item["absolute_error_mm"] is not None]
        visible = [item for item in graph if item["automatically_visible"]]
        metrics[str(order)] = {
            "held_out_manual_checkpoints": len(layer),
            "graph_resolved_checkpoints": len(graph),
            "automatically_visible_checkpoints": len(visible),
            "automatic_coverage_percent": (
                100.0 * len(visible) / len(layer) if layer else 0.0
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
    np.savez_compressed(
        OUTPUT / f"{case['case_id']}-paths.npz",
        chainage_m=chainage,
        layer_1_samples=full[1].samples,
        layer_1_graph=full[1].evidence["graph_selected_sample"],
        layer_1_canonical=full[1].evidence["canonical_event_sample"],
        layer_2_samples=full[2].samples,
        layer_2_graph=full[2].evidence["graph_selected_sample"],
        layer_2_canonical=full[2].evidence["canonical_event_sample"],
    )
    return {
        "case_id": case["case_id"],
        "road": str(road_path),
        "reference_revealed_after_tracking": True,
        "reference_points_used_for_scale": calibration_records,
        "fitted_mm_per_sample_by_layer": {
            str(order): value for order, value in scales.items()
        },
        "dropout_audit": dropout,
        "all_seed_dropouts_within_one_pulse": all(
            item["within_one_pulse_width"] for item in dropout
        ),
        "metrics": metrics,
        "checkpoints": checkpoints,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(SEEDS.read_text(encoding="utf-8"))
    cases = []
    for case in manifest["cases"]:
        print(f"Tracking {case['case_id']} before opening its reference...", flush=True)
        result = _evaluate_case(case)
        cases.append(result)
        (OUTPUT / f"{case['case_id']}-summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    summary = {
        "purpose": "blinded multi-road radar-only seed validation",
        "seed_manifest": str(SEEDS),
        "seed_exclusion_radius_m": SEED_EXCLUSION_M,
        "field_accuracy_established": False,
        "cases": cases,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
