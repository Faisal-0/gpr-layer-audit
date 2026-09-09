"""Reproducible development comparison, not a verified field-accuracy benchmark.

Three workbook stations supply approximate guides. All other original workbook
stations are comparison-only. The acquisition dielectric and detected surface
define an explicit provisional conversion; no offset is fitted to the workbook.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

from gpr_layer_audit.io import DZTFile, read_dzg, read_dzx
from gpr_layer_audit.io.dzg import interpolate_gps
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence
from gpr_layer_audit.processing.preprocessing import preprocess_for_interpretation
from gpr_layer_audit.processing.tracker import pick_interfaces
from gpr_layer_audit.reference import normalize_reference_workbook
from gpr_layer_audit.seeds import load_seed_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "GPR Data/talagang")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-m", type=float, default=150)
    parser.add_argument("--seed-chainages", type=float, nargs=3, default=[15, 75, 140])
    parser.add_argument("--radar-seeds", type=Path,
                        help="Use saved radar-click seeds; workbook becomes comparison-only")
    parser.add_argument("--whole-trace-registration", action="store_true",
                        help="Experimental registration ablation; no field improvement established")
    parser.add_argument("--method", choices=["seed_hybrid", "joint_seed_adaptive"],
                        default="seed_hybrid")
    args = parser.parse_args()
    road_path = args.data / "TALAGANG.PRJ/TALAGANG_001.DZT"
    plate_path = args.data / "TALAGANG METAL PLATE.PRJ/TALAGANG METAL PLATE_001.DZT"
    workbook = args.data / "Talagang_layer_thickness_.xlsx"
    road, plate = DZTFile(road_path), DZTFile(plate_path)
    metadata = read_dzx(road_path.with_suffix(".DZX"))
    dielectric = metadata.dielectric
    if dielectric is None or not np.isfinite(dielectric) or dielectric <= 0:
        raise ValueError("This comparison requires a recorded acquisition dielectric")
    step = metadata.units_per_scan
    if step is None or step <= 0:
        raise ValueError("This comparison requires recorded metres per trace")
    if not 0 < args.stop_m <= (road.header.trace_count - 1) * step:
        raise ValueError("Comparison end must lie within the acquisition")
    points = normalize_reference_workbook(workbook)
    # Check recorded workbook GPS at its stated chainages; do not optimize an
    # offset or use nearest geographical matches to manufacture alignment.
    book = load_workbook(workbook, data_only=True, read_only=True)
    sheet = book["Interpolated Data"]
    records = list(sheet.values)[1:]
    book.close()
    distances = np.asarray([r[3] for r in records], float)
    lat, lon = interpolate_gps(read_dzg(road_path.with_suffix(".DZG")), distances / step)
    gps_error = 6371000 * np.hypot(
        np.deg2rad(lat - np.asarray([r[4] for r in records], float)),
        np.deg2rad(lon - np.asarray([r[5] for r in records], float))
        * np.cos(np.deg2rad(lat)),
    )
    start = time.perf_counter()
    # Prepare the full acquisition before cropping so surface reference and
    # background statistics match full-road processing.
    prepared = calibrate(road, plate, stack_size=16)
    interpreted = preprocess_for_interpretation(
        prepared.radargram, prepared.reference_surface_sample, prepared.plate_template
    )
    all_x = prepared.trace_centres * step
    select = all_x <= args.stop_m
    x = all_x[select]
    scale = road.header.sample_interval_ns * 299.792458 / (2 * np.sqrt(dielectric))
    surface = prepared.reference_surface_sample
    guides, guide_cells, seed_metadata = {}, set(), {}
    guide_records = []
    pulse_width = 7.0
    if args.radar_seeds:
        survey, stations = load_seed_file(args.radar_seeds)
        if survey != "TALAGANG.PRJ/TALAGANG_001":
            raise ValueError("Saved radar seeds belong to a different acquisition")
        widths = []
        for station in stations:
            if station.role == "correction" or not 0 <= station.chainage_m <= args.stop_m:
                continue
            row = int(np.argmin(abs(x - station.chainage_m)))
            for order, clicked in station.samples.items():
                if order not in (1, 2) or station.visibility[order].value != "visible":
                    continue
                sample = round(clicked)
                guides.setdefault(order, {})[row] = sample
                widths.append(station.pulse_width_samples[order])
                seed_metadata.setdefault(order, {})[row] = {
                    "canonical_sample_index": station.canonical_samples[order],
                    "selected_lobe": station.selected_lobe[order],
                    "regime_id": station.regime_ids[order],
                    "pulse_width_samples": station.pulse_width_samples[order],
                }
                guide_records.append({"layer": order, "chainage_m": station.chainage_m,
                                      "row": row, "sample": sample,
                                      "source_station": station.station_id})
                # Do not count the same location as an independent checkpoint.
                for point in points:
                    if point.layer_order == order and abs(point.chainage_m - x[row]) <= 8 * step:
                        guide_cells.add((point.source_sheet, point.source_cell))
        if not guides or not widths:
            raise ValueError("No visible radar seeds within this window")
        pulse_width = float(np.median(widths))
    for order in (() if args.radar_seeds else (1, 2)):
        originals = [p for p in points if p.layer_order == order
                     and p.label_origin.value == "manual" and p.chainage_m <= args.stop_m]
        guides[order] = {}
        for station in args.seed_chainages:
            if not 0 <= station <= args.stop_m:
                raise ValueError("Guide stations must lie within the comparison window")
            point = min(originals, key=lambda p: abs(p.chainage_m - station))
            row = int(np.argmin(abs(x - point.chainage_m)))
            sample = round(surface + point.cumulative_depth_mm / scale)
            if not 0 <= sample < road.header.samples_per_trace:
                raise ValueError("Projected guide is outside the radar sample range")
            guides[order][row] = sample
            guide_cells.add((point.source_sheet, point.source_cell))
            guide_records.append({"layer": order, "chainage_m": point.chainage_m,
                                  "row": row, "sample": sample,
                                  "source_cell": point.source_cell})
        if len(guides[order]) != 3:
            raise ValueError("Guide stations must select three distinct original observations")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": ("radar-seeded development; workbook comparison-only" if args.radar_seeds
                    else "workbook-assisted development; not held-out road evaluation"),
        "field_accuracy_verified": False,
        "training_labels_created": False,
        "method": args.method,
        "whole_trace_registration": args.whole_trace_registration,
        "ml": "off",
        "source_sha256": {str(p.relative_to(args.data)): fingerprint_file(p) for p in
                          (road_path, plate_path, workbook, road_path.with_suffix('.DZX'),
                           road_path.with_suffix('.DZG'))},
        "code_sha256": {p.name: fingerprint_file(p) for p in
                        (ROOT / "src/gpr_layer_audit/processing").glob("*.py")},
        "script_sha256": fingerprint_file(Path(__file__)),
        "label_counts": {f"layer_{order}_{origin}": n for (order, origin), n in
                         Counter((p.layer_order, p.label_origin.value) for p in points).items()},
        "subbase_labels": 0,
        "gps_alignment": {"compared": int(np.isfinite(gps_error).sum()),
                          "median_error_m": float(np.nanmedian(gps_error)),
                          "maximum_error_m": float(np.nanmax(gps_error)),
                          "chainage_offset_m": 0},
        "conversion": {
            "dielectric": dielectric, "source": "acquisition DZX; workbook setting unverified",
            "depth_semantics": "cumulative interface depth in inches, converted by 25.4",
            "sample_interval_ns": road.header.sample_interval_ns,
            "reference_surface_sample": surface,
            "mm_per_sample": float(scale),
            "workbook_processed_file": sorted({str(r[0]) for r in records}),
            "depth_origin_and_lobe_convention_verified": False,
            "workbook_offset_fitted": False,
        },
        "stack_size": 16, "pulse_width_samples": pulse_width, "stop_m": args.stop_m,
        "plate_subtraction_applied": prepared.diagnostics.plate_subtraction_applied,
        "guides": guide_records,
        "preparation_seconds": time.perf_counter() - start,
    }
    if args.radar_seeds:
        report["seed_file_sha256"] = fingerprint_file(args.radar_seeds)
    for guide in guide_records:
        row, sample = guide["row"], guide["sample"]
        guide["measurement_polarity"] = int(np.sign(prepared.measurement_radargram[row, sample]))
    (args.output / "configuration.json").write_text(json.dumps(report, indent=2))
    start = time.perf_counter()
    paths = pick_interfaces(
        interpreted.radargram[select], surface, LayerSpec.defaults()[:2],
        anchor_samples=guides, seed_metadata=seed_metadata or None,
        pulse_width_samples=pulse_width,
        feature_branches={k: v[select] for k, v in interpreted.feature_branches.items()},
        hybrid_evidence=HybridEvidence(prepared.measurement_radargram[select],
                                       road.header.sample_interval_ns, 16 * step,
                                       provenance={"whole_trace_registration":
                                                   args.whole_trace_registration}),
        method=args.method, ml_policy="off", design_weight=0, max_interpolation_rows=0,
        horizontal_step_m=16 * step, sample_interval_ns=road.header.sample_interval_ns,
    )
    report["tracking_seconds"] = time.perf_counter() - start
    arrays = {"chainage_m": x, "measurement": prepared.measurement_radargram[select]}
    comparisons = []
    for point in points:
        if point.chainage_m > args.stop_m or point.layer_order not in paths:
            continue
        row = int(np.argmin(abs(x - point.chainage_m)))
        path = paths[point.layer_order]
        proposed = path.provisional_samples
        proposed = path.samples if proposed is None else proposed
        sample = int(proposed[row])
        target = surface + point.cumulative_depth_mm / scale
        comparisons.append({
            "layer": point.layer_order, "chainage_m": point.chainage_m,
            "source_cell": point.source_cell, "origin": point.label_origin.value,
            "is_guide": (point.source_sheet, point.source_cell) in guide_cells,
            "projected_sample": float(target), "proposed_sample": sample,
            "accepted": bool(path.visible[row]),
            "absolute_error_samples": abs(sample - target) if sample >= 0 else None,
        })
    report["layers"] = {}
    for order, path in paths.items():
        comparison = [c for c in comparisons if c["layer"] == order
                      and c["origin"] == "manual" and not c["is_guide"]]
        accepted = [c["absolute_error_samples"] for c in comparison if c["accepted"]
                    and c["absolute_error_samples"] is not None]
        report["layers"][order] = {
            "accepted_rows_including_guides": int(path.visible.sum()), "rows": len(x),
            "original_non_guide_checkpoints": len(comparison),
            "accepted_at_checkpoints": len(accepted),
            "accepted_projection_error_median_samples": float(np.median(accepted))
            if accepted else None,
            "accepted_within_2_samples_of_projection": int(sum(e <= 2 for e in accepted)),
            "seed_identity_warnings": path.provenance.get("seed_identity_warnings", []),
        }
        arrays[f"L{order}_samples"] = path.samples
        arrays[f"L{order}_proposed"] = (path.samples if path.provisional_samples is None
                                         else path.provisional_samples)
        arrays[f"L{order}_visible"] = path.visible
    np.savez_compressed(args.output / "tracking.npz", **arrays)
    report["comparisons"] = comparisons
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("gps_alignment", "layers", "tracking_seconds")},
                     indent=2))


if __name__ == "__main__":
    main()
