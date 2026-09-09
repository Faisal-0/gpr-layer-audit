"""RADAN analyst references. Coordinates and labels are never fitted to tracker output."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .io.dzt import DZTFile, fingerprint_file
from .io.dzx import read_dzx


def road_partition(path: str | Path) -> tuple[str, str]:
    name = str(path).upper()
    for token, group, split in (
        ("MANDIALI", "mandiali", "development"),
        ("GUJRAT", "gujrat", "development"),
        ("BAHAWALPUR", "bahawalpur", "calibration"),
        *(
            (x.upper(), x, "held_out")
            for x in ("jhang", "pattoki", "daska", "burewala", "jamshoro")
        ),
    ):
        if token in name:
            return group, split
    return "unresolved", "unassigned"


def audit_reference(dzx_path, dzt_path=None):
    metadata = read_dzx(dzx_path)
    source = Path(dzt_path) if dzt_path else Path(dzx_path).with_suffix(".DZT")
    report = {
        "schema": "radan-reference-v1",
        "metadata": asdict(metadata),
        "dzt_path": str(source.resolve()),
        "missing_picks": "unknown",
        "reference_kind": "analyst_layer_trace",
        "rfp_candidates_included": False,
        "physical_road_group": road_partition(source)[0],
        "split": road_partition(source)[1],
        "issues": [],
        "layers": [],
    }
    if not source.exists():
        report["issues"].append("paired_dzt_missing")
        return report
    radar = DZTFile(source)
    header = radar.header
    report.update(
        dzt_sha256=fingerprint_file(source),
        dimensions=[header.trace_count, header.samples_per_trace],
        dt_ns=header.sample_interval_ns,
        dx_m=header.distance_per_trace_m,
        header_time_origin_ns=header.position_ns,
        header_zero=header.zero,
    )
    if metadata.scan_range and metadata.scan_range != (0, header.trace_count - 1):
        report["issues"].append("declared_scan_range_differs_from_dzt")
    previous = {}
    for layer in metadata.layers:
        if not layer.picks:
            continue
        issues, amplitudes, times, ordering = [], [], [], []
        seen = set()
        for pick in layer.picks:
            key = (pick.trace, pick.channel)
            if key in seen:
                issues.append({"trace": pick.trace, "reason": "duplicate_observation"})
            seen.add(key)
            if not (
                0 <= pick.trace < header.trace_count
                and 0 <= pick.sample < header.samples_per_trace
                and 0 <= pick.channel < header.channels
            ):
                issues.append({"trace": pick.trace, "reason": "coordinate_out_of_bounds"})
                continue
            actual = float(radar.channel(pick.channel)[pick.trace, pick.sample])
            amplitudes.append(abs(actual - pick.recorded_amplitude))
            # Audit recorded time using header provenance, never use this residual to align radar.
            times.append(
                pick.time_ns - (pick.sample * header.sample_interval_ns + header.position_ns)
            )
            if key in previous and pick.sample <= previous[key]:
                ordering.append(pick.trace)
            previous[key] = pick.sample
        report["layers"].append(
            {
                "number": layer.number,
                "name": layer.name,
                "observations": len(layer.picks),
                "issues": issues,
                "amplitude_match_fraction": float(np.mean(np.array(amplitudes) <= 1))
                if amplitudes
                else None,
                "time_residual_ns_median": float(np.median(times)) if times else None,
                "time_residual_ns_range": [float(np.min(times)), float(np.max(times))]
                if times
                else None,
                "ordering_conflict_traces": ordering,
                "label_status": "analyst_reference; source layer number retained",
                "time_mapping_status": "verified_header"
                if times and np.max(np.abs(times)) < header.sample_interval_ns / 4
                else "requires_review",
            }
        )
    return report


def distributed_seeds(picks):
    """Three nearest observations to 10/50/90% of annotated trace extent."""
    ordered = sorted(picks, key=lambda p: p.trace)
    if not ordered:
        return []
    targets = [
        ordered[0].trace + f * (ordered[-1].trace - ordered[0].trace) for f in (0.1, 0.5, 0.9)
    ]
    chosen = {
        min(range(len(ordered)), key=lambda i: (abs(ordered[i].trace - t), ordered[i].trace))
        for t in targets
    }
    return [ordered[i] for i in sorted(chosen)]
