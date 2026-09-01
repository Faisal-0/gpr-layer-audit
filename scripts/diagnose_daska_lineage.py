"""Freeze direct seed-lineage diagnostics for one development road.

The tracker runs before the reference workbook is opened.  Reference-derived
candidate comparisons are explicitly marked development-only and are used to
diagnose event-family propagation, never as release evidence or graph input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.models import LayerSpec
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


def _runs(mask: np.ndarray, chainage: np.ndarray) -> list[dict[str, float | int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [
        {
            "start_m": float(chainage[group[0]]),
            "end_m": float(chainage[group[-1]]),
            "row_count": int(len(group)),
        }
        for group in groups
    ]


def _candidate_at_canonical(
    components: dict[str, np.ndarray],
    row: int,
    expected: float,
    *,
    lobe_code: int | None = None,
) -> tuple[int | None, float | None]:
    canonical = np.asarray(components["event_canonical_sample"])[row]
    candidates = np.flatnonzero(np.isfinite(canonical))
    if lobe_code is not None:
        lobes = np.asarray(components["event_lobe_code"])[row]
        matching = candidates[lobes[candidates] == lobe_code]
        if len(matching):
            candidates = matching
    if not len(candidates):
        return None, None
    selected = int(candidates[np.argmin(np.abs(canonical[candidates] - expected))])
    return selected, float(canonical[selected])


def _component(
    components: dict[str, np.ndarray], name: str, row: int, sample: int | None
) -> float | None:
    if sample is None or name not in components:
        return None
    value = float(np.asarray(components[name])[row, sample])
    return value if np.isfinite(value) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "benchmarks/daska-five-seed-development-20260830.json",
    )
    parser.add_argument("--case", default="daska-pasrur-forward-development")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = next(item for item in manifest["cases"] if item["case_id"] == args.case)

    road_path = ROOT / case["road"]
    plate_path = ROOT / case["plate"]
    road, plate = DZTFile(road_path), DZTFile(plate_path)
    stack = _effective_stack(road.header.trace_count, 0, road.header.distance_per_trace_m)
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
        "measurement_support": measurement_packet_support(calibrated.measurement_radargram),
    }
    anomaly_mask, _ = _detect_anomalies(chainage, branches.get("anomaly_score"))
    station_rows = [
        int(np.argmin(np.abs(chainage - float(station["chainage_m"]))))
        for station in case["stations"]
    ]
    orders = tuple(range(1, max(int(value) for value in case.get("validation_layers", (1, 2))) + 1))
    anchors = {
        order: {
            row: int(station["samples"][str(order)])
            for row, station in zip(station_rows, case["stations"], strict=True)
        }
        for order in orders
    }
    layers = LayerSpec.defaults()[: max(orders)]
    paths = pick_interfaces(
        interpreted.radargram,
        calibrated.reference_surface_sample,
        layers,
        anchor_samples=anchors,
        feature_branches=dict(branches),
        search_corridors={},
        design_weight=0.0,
        anomaly_mask=anomaly_mask,
        break_rows=set(),
        max_interpolation_rows=0,
        horizontal_step_m=float(np.median(np.diff(chainage))),
    )

    arrays: dict[str, np.ndarray] = {
        "chainage_m": chainage,
        "anomaly_mask": anomaly_mask,
    }
    summary: dict[str, object] = {
        "purpose": "revealed-reference lineage diagnosis; not release evidence",
        "case_id": case["case_id"],
        "tracker_ran_before_reference_opened": True,
        "plate_subtraction_applied": calibrated.diagnostics.plate_subtraction_applied,
        "stack_size": stack,
        "reference_surface_sample": calibrated.reference_surface_sample,
        "layers": {},
    }
    for order in orders:
        path = paths[order]
        prefix = f"layer_{order}_"
        for name, values in {
            "samples": path.samples,
            "graph": path.evidence["graph_selected_sample"],
            "canonical": path.evidence["canonical_event_sample"],
            "confidence": path.confidence,
        }.items():
            arrays[prefix + name] = np.asarray(values)
        for name in (
            "direct_seed_family_support",
            "deep_identity_support",
            "seed_gap_support",
            "independent_radar_identity_agreement",
            "seed_reachable",
            "tracklet_support",
            "cycle_slip_risk",
            "event_family_index",
        ):
            if name in path.evidence:
                arrays[prefix + name] = np.asarray(path.evidence[name])
        direct = (
            np.asarray(path.evidence.get("direct_seed_family_support", np.zeros(len(chainage))))
            >= 0.5
        )
        reachable = np.asarray(path.evidence["seed_reachable"]) >= 0.5
        tracked = np.asarray(path.evidence["tracklet_support"]) > 0
        summary["layers"][str(order)] = {
            "graph_coverage_percent": 100.0 * float(np.mean(arrays[prefix + "graph"] >= 0)),
            "visible_percent": 100.0 * float(np.mean(path.samples >= 0)),
            "selected_seed_reachable_percent": 100.0 * float(np.mean(reachable)),
            "selected_tracklet_percent": 100.0 * float(np.mean(tracked)),
            "direct_seed_family_percent": 100.0 * float(np.mean(direct)),
            "direct_seed_family_runs": _runs(direct, chainage),
        }
        if order == 2:
            for name in (
                "event_canonical_sample",
                "event_lobe_code",
                "audit_candidate_rank",
                "audit_candidate_score",
                "seed_reachable",
                "tracklet_support",
                "cycle_slip_risk",
                "event_family_index",
                "generic_radar_score",
                "oriented_coherence",
                "signed_seed_correlation",
                "phase_cycle_agreement",
                "polarity_agreement",
            ):
                if name in path.candidate_components:
                    arrays[prefix + "candidate_" + name] = np.asarray(
                        path.candidate_components[name]
                    )

    # The reference is intentionally opened only after the complete radar-only
    # path and direct-lineage diagnostics above have been frozen in memory.
    references = normalize_reference_workbook(ROOT / case["reference"])
    base_scale_mm_per_sample = 1.5391190476190473
    base = paths[2]
    asphalt_canonical = np.asarray(paths[1].evidence["canonical_event_sample"])
    comparisons: list[dict[str, object]] = []
    for point in references:
        if point.layer_order != 2 or point.individual_thickness_mm is None:
            continue
        if point.label_origin.value != "manual":
            continue
        row = int(np.argmin(np.abs(chainage - point.chainage_m)))
        if abs(float(chainage[row]) - float(point.chainage_m)) > 0.25:
            continue
        if asphalt_canonical[row] < 0:
            continue
        expected = float(
            asphalt_canonical[row] + float(point.individual_thickness_mm) / base_scale_mm_per_sample
        )
        candidate, canonical = _candidate_at_canonical(
            base.candidate_components, row, expected, lobe_code=1
        )
        graph_sample = int(round(float(base.evidence["graph_selected_sample"][row])))
        graph_sample = graph_sample if graph_sample >= 0 else None
        comparisons.append(
            {
                "chainage_m": float(point.chainage_m),
                "row": row,
                "expected_canonical_sample": expected,
                "nearest_candidate_lobe": candidate,
                "nearest_candidate_canonical": canonical,
                "nearest_candidate_canonical_error": (
                    abs(float(canonical) - expected) if canonical is not None else None
                ),
                "nearest_candidate_rank": _component(
                    base.candidate_components, "audit_candidate_rank", row, candidate
                ),
                "nearest_candidate_radar_score": _component(
                    base.candidate_components, "audit_candidate_score", row, candidate
                ),
                "nearest_candidate_seed_reachable": _component(
                    base.candidate_components, "seed_reachable", row, candidate
                ),
                "nearest_candidate_tracklet_support": _component(
                    base.candidate_components, "tracklet_support", row, candidate
                ),
                "nearest_candidate_cycle_slip_risk": _component(
                    base.candidate_components, "cycle_slip_risk", row, candidate
                ),
                "nearest_candidate_family": _component(
                    base.candidate_components, "event_family_index", row, candidate
                ),
                "graph_lobe": graph_sample,
                "graph_seed_reachable": _component(
                    base.candidate_components, "seed_reachable", row, graph_sample
                ),
                "graph_tracklet_support": _component(
                    base.candidate_components, "tracklet_support", row, graph_sample
                ),
                "visible": bool(base.samples[row] >= 0),
            }
        )
    retained = [
        item
        for item in comparisons
        if item["nearest_candidate_canonical_error"] is not None
        and float(item["nearest_candidate_canonical_error"]) <= 7.0
    ]
    summary["revealed_reference_candidate_diagnosis"] = {
        "manual_checkpoint_count": len(comparisons),
        "candidate_within_one_pulse_count": len(retained),
        "candidate_within_one_pulse_percent": (
            100.0 * len(retained) / len(comparisons) if comparisons else 0.0
        ),
        "within_pulse_seed_reachable_count": sum(
            float(item["nearest_candidate_seed_reachable"] or 0.0) >= 0.5 for item in retained
        ),
        "within_pulse_tracklet_count": sum(
            float(item["nearest_candidate_tracklet_support"] or 0.0) > 0 for item in retained
        ),
        "checkpoints": comparisons,
    }
    np.savez_compressed(output / "lineage-paths.npz", **arrays)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary["layers"], indent=2), flush=True)
    print(
        json.dumps(
            {
                key: value
                for key, value in summary["revealed_reference_candidate_diagnosis"].items()
                if key != "checkpoints"
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
