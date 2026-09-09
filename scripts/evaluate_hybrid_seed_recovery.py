"""Real-road development check using frozen radar-only seeds, without workbooks.

Full paths are frozen before each station is withheld. Seed recovery is a
repeatability diagnostic, not field accuracy or eligibility for model training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.pipeline import _chainage, _detect_anomalies, _effective_stack
from gpr_layer_audit.processing.preprocessing import preprocess_for_interpretation
from gpr_layer_audit.processing.tracker import TRACKER_METHODS, pick_interfaces

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=TRACKER_METHODS, default="seed_hybrid")
    parser.add_argument("--full-only", action="store_true")
    args = parser.parse_args()
    case = next(
        c for c in json.loads(args.manifest.read_text())["cases"] if c["case_id"] == args.case
    )
    args.output.mkdir(parents=True, exist_ok=False)
    road_path, plate_path = ROOT / case["road"], ROOT / case["plate"]
    road, plate = DZTFile(road_path), DZTFile(plate_path)
    stack = _effective_stack(road.header.trace_count, 0, road.header.distance_per_trace_m)
    prepared = calibrate(road, plate, stack_size=stack)
    dzx_path = road_path.with_suffix(".DZX")
    chainage = _chainage(
        road.header, prepared.trace_centres, read_dzx(dzx_path) if dzx_path.exists() else None
    )
    interpreted = preprocess_for_interpretation(
        prepared.radargram, prepared.reference_surface_sample, prepared.plate_template
    )
    branches = dict(interpreted.feature_branches)
    branches["hybrid_measurement"] = prepared.measurement_radargram
    anomaly, _ = _detect_anomalies(chainage, branches.get("anomaly_score"))
    station_rows = [int(np.argmin(abs(chainage - s["chainage_m"]))) for s in case["stations"]]
    layers = LayerSpec.defaults()[: max(case["validation_layers"])]
    anchors = {
        layer.order: {
            row: int(station["samples"][str(layer.order)])
            for row, station in zip(station_rows, case["stations"], strict=True)
        }
        for layer in layers
    }
    code_files = list((ROOT / "src/gpr_layer_audit/processing").glob("*.py"))
    report = {
        "case": args.case,
        "method": args.method,
        "ml": "off",
        "field_accuracy": False,
        "seed_manifest_sha256": fingerprint_file(args.manifest),
        "road_sha256": fingerprint_file(road_path),
        "plate_sha256": fingerprint_file(plate_path),
        "code_sha256": {p.name: fingerprint_file(p) for p in code_files},
        "stack_size": stack,
        "reference_surface_sample": prepared.reference_surface_sample,
        "pulse_width_samples": 7,
        "seed_signal_audit": [
            {
                "layer_order": order,
                "row": row,
                "sample": sample,
                "declared_lobe": case.get("lobe_choice", {}).get(str(order)),
                "processed_polarity": int(np.sign(interpreted.radargram[row, sample])),
                "measurement_polarity": int(np.sign(prepared.measurement_radargram[row, sample])),
            }
            for order, values in anchors.items()
            for row, sample in values.items()
        ],
        "anchors": anchors,
        "runs": [],
    }
    (args.output / "configuration.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        args.output / "radar.npz",
        radargram=interpreted.radargram,
        measurement=prepared.measurement_radargram,
        chainage_m=chainage,
        surface_samples=prepared.surface_samples,
    )
    held_out = [None] if args.full_only else [None, *station_rows]
    for withheld in held_out:
        start = time.perf_counter()
        inputs = {
            order: {r: s for r, s in values.items() if r != withheld}
            for order, values in anchors.items()
        }
        paths = pick_interfaces(
            interpreted.radargram,
            prepared.reference_surface_sample,
            layers,
            anchor_samples=inputs,
            feature_branches=dict(branches),
            anomaly_mask=anomaly,
            method=args.method,
            ml_policy="off",
            design_weight=0,
            max_interpolation_rows=0,
            horizontal_step_m=float(np.median(np.diff(chainage))),
            sample_interval_ns=road.header.sample_interval_ns,
        )
        name = "full" if withheld is None else f"withheld-{withheld}"
        record = {"name": name, "seconds": time.perf_counter() - start, "layers": {}}
        arrays = {}
        for order, path in paths.items():
            proposed = path.provisional_samples
            if proposed is None:
                proposed = path.samples
            arrays.update(
                {
                    f"L{order}_samples": path.samples,
                    f"L{order}_proposed": proposed,
                    f"L{order}_visible": path.visible,
                }
            )
            for key in (
                "hybrid_correspondence",
                "candidate_margin",
                "measurement_support",
                "hybrid_path_margin",
                "hybrid_path_alternate",
            ):
                if key in path.evidence:
                    arrays[f"L{order}_{key}"] = path.evidence[key]
            layer = {
                "accepted": int(path.visible.sum()),
                "rows": len(chainage),
                "proposed": int(np.sum(proposed >= 0)),
                "seed_identity_warnings": path.provenance.get("seed_identity_warnings", []),
                "suggested_observations": path.provenance.get("suggested_observations", []),
            }
            if withheld is not None:
                expected, prediction = anchors[order][withheld], int(proposed[withheld])
                layer["withheld"] = {
                    "chainage_m": float(chainage[withheld]),
                    "expected_sample": expected,
                    "proposed_sample": prediction,
                    "accepted": bool(path.visible[withheld]),
                    "absolute_error_samples": abs(expected - prediction)
                    if prediction >= 0
                    else None,
                    "same_measurement_polarity": bool(
                        prediction >= 0
                        and np.sign(prepared.measurement_radargram[withheld, prediction])
                        == np.sign(prepared.measurement_radargram[withheld, expected])
                    ),
                    "same_polarity": bool(
                        prediction >= 0
                        and np.sign(interpreted.radargram[withheld, prediction])
                        == np.sign(interpreted.radargram[withheld, expected])
                    ),
                }
            record["layers"][str(order)] = layer
        np.savez_compressed(args.output / f"{name}.npz", **arrays)
        report["runs"].append(record)
        (args.output / "summary.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
