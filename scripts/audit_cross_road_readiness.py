"""Audit acquisition and independent-measurement readiness on every road.

This is a robustness/readiness check, not an accuracy benchmark. It reads a
bounded centre window from each road, records calibration pairing ambiguity,
and verifies that pre-subtraction packet support is finite across acquisitions.
No workbook values are used and no tracker output is promoted to a label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpr_layer_audit.catalog import (
    calibration_candidates_for,
    calibration_pairing_is_ambiguous,
    discover_survey_catalog,
)
from gpr_layer_audit.io import DZTFile
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.preprocessing import measurement_packet_support

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "GPR Data")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exports/cross-road-readiness-20260829.json",
    )
    parser.add_argument("--window-traces", type=int, default=4096)
    parser.add_argument("--stack-size", type=int, default=16)
    args = parser.parse_args()

    catalog = discover_survey_catalog(args.data)
    records: list[dict[str, object]] = []
    for line in catalog.roads:
        road = DZTFile(line.dzt_path)
        candidates = calibration_candidates_for(catalog, line.survey_id)
        best = candidates[0] if candidates else None
        runner_up = candidates[1] if len(candidates) > 1 else None
        pairing_margin = (
            best.compatibility_score - runner_up.compatibility_score
            if best is not None and runner_up is not None
            else None
        )
        unambiguous = bool(best is not None and not calibration_pairing_is_ambiguous(candidates))
        plate_line = next(
            (
                item
                for item in catalog.calibrations
                if best is not None and item.survey_id == best.calibration_survey_id
            ),
            None,
        )
        plate = DZTFile(plate_line.dzt_path) if plate_line is not None else None
        width = min(max(1, args.window_traces), road.header.trace_count)
        start = max(0, (road.header.trace_count - width) // 2)
        stop = min(road.header.trace_count, start + width)
        calibrated = calibrate(
            road,
            plate,
            stack_size=max(1, args.stack_size),
            start_trace=start,
            stop_trace=stop,
        )
        support = measurement_packet_support(calibrated.measurement_radargram)
        sample_start = min(support.shape[1], calibrated.reference_surface_sample + 12)
        below_surface = support[:, sample_start:]
        finite = np.isfinite(below_surface)
        values = below_surface[finite]
        record = {
            "survey_id": line.survey_id,
            "road": str(line.dzt_path),
            "traces": line.trace_count,
            "window": [start, stop],
            "stack_size": max(1, args.stack_size),
            "plate": str(plate_line.dzt_path) if plate_line is not None else None,
            "plate_pairing_score": (
                float(best.compatibility_score) if best is not None else None
            ),
            "plate_pairing_margin": (
                float(pairing_margin) if pairing_margin is not None else None
            ),
            "automatic_plate_selection_safe": unambiguous,
            "gain_compatible": bool(best.gain_compatible) if best is not None else False,
            "waveform_compatible": (
                bool(best.waveform_compatible) if best is not None else False
            ),
            "measurement_support_finite_fraction": float(np.mean(finite)),
            "measurement_support_p10": float(np.percentile(values, 10)),
            "measurement_support_median": float(np.median(values)),
            "measurement_support_p90": float(np.percentile(values, 90)),
            "measurement_gate_coverage_fraction": float(np.mean(values >= 0.015)),
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    summary = {
        "purpose": "cross-road acquisition readiness; not physical layer accuracy",
        "roads": len(records),
        "finite_measurement_support_roads": sum(
            item["measurement_support_finite_fraction"] == 1.0 for item in records
        ),
        "unambiguous_plate_pairings": sum(
            bool(item["automatic_plate_selection_safe"]) for item in records
        ),
        "gain_compatible_plate_pairings": sum(
            bool(item["gain_compatible"]) for item in records
        ),
        "records": records,
        "field_accuracy_established": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
