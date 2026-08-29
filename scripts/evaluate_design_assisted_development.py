"""Diagnose the bounded design branch on an already revealed development road.

This script never contributes blind release evidence. It asks whether explicit
design thickness can surface a radar-supported alternative to a coherent but
physically inaccurate signal-only family.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from evaluate_blinded_multiroad import (
    OUTPUT as STRICT_OUTPUT,
)
from evaluate_blinded_multiroad import (
    ROOT,
    SEEDS,
    _case_reference_points,
    _dropout_assessment,
    _path_arrays,
    _scale_assessments,
    _thickness_evaluation,
)

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.models import (
    DielectricSource,
    LayerDesign,
    LayerSpec,
    SeedStation,
    VisibilityState,
)
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.corridor import build_search_corridors
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


def _case(case_id: str) -> dict:
    manifest = json.loads(SEEDS.read_text(encoding="utf-8"))
    return next(item for item in manifest["cases"] if item["case_id"] == case_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("--asphalt-mm", type=float, required=True)
    parser.add_argument("--base-mm", type=float, required=True)
    args = parser.parse_args(argv)
    if args.asphalt_mm <= 0 or args.base_mm <= 0:
        parser.error("design thicknesses must be positive")

    case = _case(args.case_id)
    road_path, plate_path = ROOT / case["road"], ROOT / case["plate"]
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
    anomaly_mask, _ = _detect_anomalies(chainage, branches.get("anomaly_score"))
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
    seed_stations = [
        SeedStation(
            station_id=f"frozen-{index}",
            chainage_m=float(station["chainage_m"]),
            samples={order: float(station["samples"][str(order)]) for order in (1, 2)},
            visibility={order: VisibilityState.VISIBLE for order in (1, 2)},
            user_confirmed={order: True for order in (1, 2)},
        )
        for index, station in enumerate(case["stations"], 1)
    ]
    designs = [
        LayerDesign(1, layers[0].name, args.asphalt_mm),
        LayerDesign(2, layers[1].name, args.base_mm),
    ]
    dielectric = {
        order: (7.0, DielectricSource.ASSUMED_SCAN) for order in (1, 2)
    }
    corridors = build_search_corridors(
        chainage,
        calibrated.reference_surface_sample,
        road.header.sample_interval_ns,
        layers,
        designs,
        [],
        dielectric,
        seed_stations,
    )
    spacing = float(np.median(np.diff(chainage)))
    full = pick_interfaces(
        interpreted.radargram,
        calibrated.reference_surface_sample,
        layers,
        anchor_samples=anchors,
        feature_branches=branches,
        search_corridors=corridors,
        design_weight=0.10,
        anomaly_mask=anomaly_mask,
        break_rows=set(),
        max_interpolation_rows=0,
        horizontal_step_m=spacing,
    )
    paths = _path_arrays(full)

    # Open the disclosed reference only after both signal and design passes are
    # frozen. This is a development diagnostic, not a new blind result.
    points = _case_reference_points(case, ROOT / case["reference"])
    validation_orders = set(case.get("validation_layers", (1, 2)))
    prior = json.loads(
        (STRICT_OUTPUT / f"{case['case_id']}-summary.json").read_text(
            encoding="utf-8"
        )
    )
    dropout = _dropout_assessment(prior["dropout_audit"], validation_orders)
    scales = _scale_assessments(
        case,
        points,
        calibrated.reference_surface_sample,
        road.header.sample_interval_ns,
        validation_orders,
    )
    evaluations = {
        str(order): _thickness_evaluation(
            case,
            points,
            chainage,
            paths,
            calibrated.reference_surface_sample,
            order,
            dropout[str(order)],
            scales[str(order)],
            all(
                dropout[str(upper)]["all_within_one_pulse"]
                for upper in range(1, order + 1)
                if str(upper) in dropout
            ),
        )
        for order in sorted(validation_orders)
    }
    result = {
        "purpose": "revealed-reference design-assisted development diagnostic",
        "counts_as_blind_release_evidence": False,
        "case_id": case["case_id"],
        "design_mm": {"1": args.asphalt_mm, "2": args.base_mm},
        "corridors": {
            str(order): {
                "source": corridor.source,
                "median_gap_centre_samples": float(
                    np.nanmedian(corridor.gap_centre_samples)
                ),
                "median_gap_half_width_samples": float(
                    np.nanmedian(
                        0.5
                        * (
                            corridor.gap_upper_samples
                            - corridor.gap_lower_samples
                        )
                    )
                ),
            }
            for order, corridor in corridors.items()
        },
        "design_selected_fraction": {
            str(order): float(
                np.mean(full[order].evidence["selected_design_path"] > 0.5)
            )
            for order in validation_orders
        },
        "signal_design_conflict_fraction": {
            str(order): float(np.mean(full[order].design_conflict))
            for order in validation_orders
        },
        "thickness_validation_by_layer": evaluations,
    }
    output = ROOT / "exports/design-assisted-development-20260829"
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{case['case_id']}-summary.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
