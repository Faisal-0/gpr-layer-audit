"""Benchmark deterministic seed-dropout parallelism on disclosed radar data.

This development benchmark never opens the case's interpretation/reference
file. It reuses the frozen radar-only clicks without changing their chainages
or samples, attaches event metadata from an independent zero-seed preview, and
then compares production serial and bounded-parallel leave-one-out runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import hilbert

from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import AcquisitionFileSet, SeedStation, VisibilityState
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/blinded-base-subbase-seeds-20260829.json"
DEFAULT_CASE = "burewala-vehari-002-base-validation"


def _normalise(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _normalise(value.tolist())
    if isinstance(value, np.generic):
        return _normalise(value.item())
    if isinstance(value, Enum):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        _normalise(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot(result) -> dict[str, Any]:
    parameters = dict(result.parameters)
    parameters.pop("seed_dropout_workers", None)
    return _normalise(
        {
            "parameters": parameters,
            "picks": [asdict(item) for item in result.picks],
            "thickness": [asdict(item) for item in result.thickness],
            "profile": [asdict(item) for item in result.profile],
            "review_issues": [asdict(item) for item in result.review_issues],
            "proposed_seed_requests": [
                asdict(item) for item in result.proposed_seed_requests
            ],
            "signal_only_paths": {
                str(order): values.tolist()
                for order, values in result.signal_only_paths.items()
            },
            "design_guided_paths": {
                str(order): values.tolist()
                for order, values in result.design_guided_paths.items()
            },
        }
    )


def _case(manifest: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    try:
        case = next(item for item in document["cases"] if item["case_id"] == case_id)
    except StopIteration as exc:
        raise ValueError(f"Unknown case {case_id!r} in {manifest}") from exc
    return document, case


def _stations(case: dict[str, Any]) -> list[SeedStation]:
    orders = sorted(int(order) for order in case["stations"][0]["samples"])
    return [
        SeedStation(
            station_id=f"frozen-{index}",
            chainage_m=float(item["chainage_m"]),
            samples={order: float(item["samples"][str(order)]) for order in orders},
            visibility={order: VisibilityState.VISIBLE for order in orders},
            user_confirmed={order: True for order in orders},
        )
        for index, item in enumerate(case["stations"], 1)
    ]


def _attach_preview_metadata(
    stations: list[SeedStation], preview
) -> list[dict[str, Any]]:
    """Mirror UI metadata capture, including explicitly confirmed free picks."""
    audit: list[dict[str, Any]] = []
    for station in stations:
        row = int(np.argmin(np.abs(preview.chainage_m - station.chainage_m)))
        preview_chainage = float(preview.chainage_m[row])
        if abs(preview_chainage - station.chainage_m) > 1e-6:
            raise ValueError(
                f"Frozen station {station.station_id} is not on a preview bin: "
                f"{station.chainage_m} versus {preview_chainage}"
            )
        for order, raw_sample in station.samples.items():
            events = [
                item
                for item in preview.candidate_events
                if item.layer_order == order
                and abs(item.chainage_m - station.chainage_m) < 1e-6
            ]
            nearest = (
                min(events, key=lambda item: abs(item.sample_index - raw_sample))
                if events
                else None
            )
            distance = (
                abs(float(nearest.sample_index) - raw_sample)
                if nearest is not None
                else None
            )
            if nearest is not None and distance is not None and distance <= 4.0:
                station.phase_class[order] = nearest.phase_class
                station.analytic_phase_rad[order] = nearest.analytic_phase_rad
                station.polarity[order] = nearest.polarity
                station.selected_lobe[order] = nearest.selected_lobe
                station.canonical_samples[order] = float(
                    nearest.canonical_sample_index
                    if nearest.canonical_sample_index is not None
                    else nearest.sample_index
                )
                station.pulse_width_samples[order] = max(
                    1.0, float(nearest.pulse_width_samples or 7.0)
                )
                station.event_ids[order] = nearest.event_id or (
                    f"L{order}:{station.chainage_m:.3f}:{nearest.sample_index}"
                )
                if nearest.event_family_id:
                    station.family_ids[order] = nearest.event_family_id
                if nearest.competing_family_id:
                    station.competing_family_ids[order] = nearest.competing_family_id
                competing_ids = set(nearest.competing_event_ids)
                station.competing_samples[order] = [
                    float(item.sample_index)
                    for item in events
                    if item.event_id in competing_ids
                ]
                mode = "candidate_event"
            else:
                sample = int(round(raw_sample))
                trace = preview.calibrated_radargram[row]
                phase = float(np.angle(hilbert(trace)[sample]))
                station.phase_class[order] = int(
                    np.floor(
                        ((phase + np.pi) % (2.0 * np.pi))
                        * 8.0
                        / (2.0 * np.pi)
                    )
                )
                station.analytic_phase_rad[order] = phase
                station.polarity[order] = int(np.sign(trace[sample]))
                station.selected_lobe[order] = (
                    "negative_trough" if trace[sample] < 0 else "positive_peak"
                )
                station.canonical_samples[order] = float(sample)
                station.pulse_width_samples[order] = 7.0
                station.event_ids[order] = (
                    f"free:L{order}:{station.chainage_m:.3f}:{sample}"
                )
                station.family_ids[order] = (
                    f"free:L{order}:{station.chainage_m:.3f}"
                )
                station.competing_samples[order] = []
                mode = "confirmed_free_pick"
            station.regime_ids[order] = "default"
            station.preview_status[order] = "confirmed"
            audit.append(
                {
                    "station_id": station.station_id,
                    "layer_order": order,
                    "frozen_sample": raw_sample,
                    "nearest_candidate_distance_samples": distance,
                    "metadata_mode": mode,
                }
            )
    return audit


def _run(source, plate, options, label: str):
    started = time.perf_counter()

    def progress(percent: int, message: str) -> None:
        print(f"{label:>8} {percent:3d}% {message}", flush=True)

    result = analyze_acquisition(source, plate, options, progress=progress)
    return result, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = args.manifest.resolve()
    _, case = _case(manifest, args.case)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    road = (ROOT / case["road"]).resolve()
    plate = (ROOT / case["plate"]).resolve()
    source = AcquisitionFileSet(road)
    plate_source = AcquisitionFileSet(plate)
    stations = _stations(case)
    enabled_orders = set(stations[0].samples)
    layer_specs = [
        replace(layer, analysis_enabled=layer.order in enabled_orders)
        for layer in AnalysisOptions().layer_specs
    ]

    preview_options = AnalysisOptions(
        survey_id=case["case_id"],
        layer_specs=layer_specs,
        validate_seed_dropout=False,
        auto_fine_retrack=False,
    )
    preview, preview_seconds = _run(
        source, plate_source, preview_options, "preview"
    )
    metadata_audit = _attach_preview_metadata(stations, preview)

    serial_options = AnalysisOptions(
        survey_id=case["case_id"],
        layer_specs=layer_specs,
        seed_stations=stations,
        seed_dropout_workers=1,
        auto_fine_retrack=False,
    )
    serial, serial_seconds = _run(source, plate_source, serial_options, "serial")
    parallel_options = AnalysisOptions(
        survey_id=case["case_id"],
        layer_specs=layer_specs,
        seed_stations=stations,
        seed_dropout_workers=3,
        auto_fine_retrack=False,
    )
    parallel, parallel_seconds = _run(
        source, plate_source, parallel_options, "parallel"
    )

    serial_snapshot = _snapshot(serial)
    parallel_snapshot = _snapshot(parallel)
    parity = serial_snapshot == parallel_snapshot
    summary = {
        "purpose": "development-only real-road seed-dropout parallelism benchmark",
        "field_accuracy_established": False,
        "reference_opened": False,
        "case_id": case["case_id"],
        "frozen_seed_chainages_and_samples_unchanged": True,
        "seed_metadata_audit": metadata_audit,
        "enabled_layer_orders": sorted(enabled_orders),
        "automatic_fine_retracking": False,
        "timing_scope": (
            "coarse production joint fit plus three leave-one-station-out fits; "
            "automatic fine-region retracking intentionally excluded"
        ),
        "preview_seconds": preview_seconds,
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "speedup": serial_seconds / parallel_seconds,
        "exact_output_parity": parity,
        "serial_snapshot_sha256": _fingerprint(serial_snapshot),
        "parallel_snapshot_sha256": _fingerprint(parallel_snapshot),
        "seed_dropout_audit_equal": (
            serial.parameters.get("seed_dropout_audit")
            == parallel.parameters.get("seed_dropout_audit")
        ),
        "input_fingerprints": {
            "road": fingerprint_file(road),
            "plate": fingerprint_file(plate),
            "seed_manifest": fingerprint_file(manifest),
            "benchmark_script": fingerprint_file(Path(__file__)),
            "pipeline": fingerprint_file(
                ROOT / "src/gpr_layer_audit/processing/pipeline.py"
            ),
            "tracker": fingerprint_file(
                ROOT / "src/gpr_layer_audit/processing/tracker.py"
            ),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if not parity:
        (output / "serial-snapshot.json").write_text(
            json.dumps(serial_snapshot, indent=2), encoding="utf-8"
        )
        (output / "parallel-snapshot.json").write_text(
            json.dumps(parallel_snapshot, indent=2), encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
