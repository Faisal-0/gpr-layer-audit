"""Leave one development seed out and test radar-only event recovery.

The development seeds are provisional and were used while building the
tracker, so this is a regression/transfer diagnostic rather than independent
accuracy validation. No workbook or design thickness enters the solver.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.models import LayerSpec, SeedStation
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.pipeline import (
    _anchor_rows,
    _chainage,
    _detect_anomalies,
    _effective_stack,
    _seed_metadata_rows,
    _seed_regime_break_rows,
)
from gpr_layer_audit.processing.preprocessing import (
    measurement_packet_support,
    preprocess_for_interpretation,
)
from gpr_layer_audit.processing.tracker import pick_interfaces
from gpr_layer_audit.seeds import load_seed_file

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    case_id: str
    road: Path
    plate: Path
    seeds: Path


CASES = (
    Case(
        "talagang",
        ROOT / "GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT",
        ROOT
        / "GPR Data/talagang/TALAGANG METAL PLATE.PRJ/TALAGANG METAL PLATE_001.DZT",
        ROOT / "benchmarks/talagang-development-seeds.json",
    ),
    Case(
        "pattoki-jhoru",
        ROOT
        / (
            "GPR Data/CENTRAL ZONE/PATTOKI MEGA ROAD TO JHORU VILLAGE.PRJ/"
            "PATTOKI MEGA ROAD TO JHORU VILLAGE_001.DZT"
        ),
        ROOT
        / (
            "GPR Data/CENTRAL ZONE/METAL PLATE 11 AUGUST 2026.PRJ/"
            "METAL PLATE 11 AUGUST 2026_001.DZT"
        ),
        ROOT / "benchmarks/pattoki-development-seeds.json",
    ),
)


def _confirmed_samples(stations: list[SeedStation]) -> dict[int, list[tuple[float, float]]]:
    output: dict[int, list[tuple[float, float]]] = {}
    for station in stations:
        for order, sample in station.samples.items():
            if station.user_confirmed.get(order, False):
                output.setdefault(order, []).append((station.chainage_m, float(sample)))
    return output


def _run_case(
    case: Case, holdout_station_id: str | None = None
) -> list[dict[str, object]]:
    road = DZTFile(case.road)
    plate = DZTFile(case.plate)
    stack = _effective_stack(road.header.trace_count, 0, road.header.distance_per_trace_m)
    calibrated = calibrate(road, plate, stack_size=stack)
    dzx_path = case.road.with_suffix(".DZX")
    dzx = read_dzx(dzx_path) if dzx_path.exists() else None
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
    )
    measurement_support = measurement_packet_support(calibrated.measurement_radargram)
    _survey_id, stations = load_seed_file(case.seeds)
    layers = LayerSpec.defaults()[:2]
    bin_width = float(np.median(np.diff(chainage)))
    records: list[dict[str, object]] = []
    for holdout in stations:
        if holdout_station_id and holdout.station_id != holdout_station_id:
            continue
        expected_orders = sorted(
            order
            for order in holdout.samples
            if holdout.user_confirmed.get(order, False)
        )
        if not expected_orders:
            continue
        training = [item for item in stations if item.station_id != holdout.station_id]
        anchors = _anchor_rows(_confirmed_samples(training), chainage)
        metadata = _seed_metadata_rows(training, chainage)
        break_rows = _seed_regime_break_rows(training, chainage)
        branches = {
            **interpreted.feature_branches,
            "measurement_support": measurement_support,
        }
        anomaly_mask, _ = _detect_anomalies(
            chainage, branches.get("anomaly_score")
        )
        paths = pick_interfaces(
            interpreted.radargram,
            calibrated.reference_surface_sample,
            layers,
            anchor_samples=anchors,
            seed_metadata=metadata,
            feature_branches=branches,
            search_corridors={},
            design_weight=0.0,
            break_rows=break_rows,
            anomaly_mask=anomaly_mask,
            max_interpolation_rows=max(1, int(round(1.0 / max(bin_width, 1e-6)))),
            horizontal_step_m=bin_width,
        )
        row = int(np.argmin(np.abs(chainage - holdout.chainage_m)))
        for order in expected_orders:
            path = paths[order]
            expected = float(holdout.samples[order])
            graph = float(path.evidence["graph_selected_sample"][row])
            visible = float(path.samples[row]) if path.samples[row] >= 0 else None
            width = float(holdout.pulse_width_samples.get(order, 7.0))
            training_regimes = {
                item.regime_ids.get(order, "default")
                for item in training
                if order in item.samples and item.user_confirmed.get(order, False)
            }
            holdout_regime = holdout.regime_ids.get(order, "default")

            def candidate(
                sample: int,
                candidate_path=path,
                candidate_row=row,
            ) -> dict[str, float | None]:
                output: dict[str, float | None] = {}
                for name in (
                    "audit_candidate_rank",
                    "audit_candidate_score",
                    "event_canonical_sample",
                    "event_prototype_index",
                    "signed_seed_correlation",
                    "phase_cycle_agreement",
                    "polarity_agreement",
                    "oriented_coherence",
                    "measurement_support",
                    "seed_reachable",
                ):
                    values = candidate_path.candidate_components.get(name)
                    value = (
                        float(values[candidate_row, sample])
                        if values is not None
                        else None
                    )
                    output[name] = value if value is None or np.isfinite(value) else None
                return output

            records.append(
                {
                    "case": case.case_id,
                    "holdout_station": holdout.station_id,
                    "layer_order": order,
                    "chainage_m": float(chainage[row]),
                    "expected_sample": expected,
                    "graph_selected_sample": graph if graph >= 0 else None,
                    "visible_sample": visible,
                    "graph_error_samples": abs(graph - expected) if graph >= 0 else None,
                    "visible_error_samples": (
                        abs(visible - expected) if visible is not None else None
                    ),
                    "within_one_pulse_width": bool(
                        graph >= 0 and abs(graph - expected) <= width
                    ),
                    "automatically_visible": visible is not None,
                    "pulse_width_samples": width,
                    "holdout_regime": holdout_regime,
                    "regime_represented_in_training": holdout_regime in training_regimes,
                    "measurement_support": float(
                        path.evidence["measurement_support"][row]
                    ),
                    "expected_candidate": candidate(int(round(expected))),
                    "selected_candidate": (
                        candidate(int(round(graph))) if graph >= 0 else None
                    ),
                    "candidate_neighborhood": [
                        {"sample": sample, **candidate(sample)}
                        for sample in range(
                            max(0, int(round(expected)) - 30),
                            min(
                                interpreted.radargram.shape[1],
                                int(round(expected)) + 31,
                            ),
                        )
                        if (
                            candidate(sample)["audit_candidate_rank"] is not None
                        )
                    ],
                    "training_seed_stations": len(training),
                    "training_seeds_preserved": all(
                        paths[layer_order].samples[anchor_row] == sample
                        for layer_order, layer_anchors in anchors.items()
                        for anchor_row, sample in layer_anchors.items()
                    ),
                    "independent_validation": False,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exports/sparse-seed-recovery-20260829.json",
    )
    parser.add_argument("--case", choices=[item.case_id for item in CASES])
    parser.add_argument("--holdout-station")
    args = parser.parse_args()
    records = [
        record
        for case in CASES
        if args.case is None or case.case_id == args.case
        for record in _run_case(case, args.holdout_station)
    ]
    comparable = [item for item in records if item["regime_represented_in_training"]]
    summary = {
        "purpose": "development-seed leave-one-out regression; not field accuracy",
        "records": records,
        "comparable_holdouts": len(comparable),
        "within_one_pulse_width": sum(
            bool(item["within_one_pulse_width"]) for item in comparable
        ),
        "automatically_visible": sum(
            bool(item["automatically_visible"]) for item in comparable
        ),
        "all_training_seeds_preserved": all(
            bool(item["training_seeds_preserved"]) for item in records
        ),
        "field_accuracy_established": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
