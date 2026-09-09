"""Conservative radar-only raw/processed registration, without layer observations."""

from dataclasses import asdict

import numpy as np
from scipy.signal import correlate

from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file

from .calibration import dewow
from .conventional_signal import (
    CoordinateTransform,
    numerical_extension,
    processed_boundary_mask,
    source_layout,
)


def register_sources(processed_path, raw_path, *, probes=21, max_trace_offset=4, cancel=None):
    processed, raw = DZTFile(processed_path), DZTFile(raw_path)
    p, r = processed.header, raw.header
    if p.distance_per_trace_m is None or r.distance_per_trace_m is None:
        raise ValueError("Physical trace spacing is required for registration")
    trace_scale = p.distance_per_trace_m / r.distance_per_trace_m
    sample_scale = p.sample_interval_ns / r.sample_interval_ns
    rows = np.unique(np.linspace(0, p.trace_count - 1, probes).astype(int))
    source = np.asarray(processed.channel()[rows], np.float32)
    source_valid = processed_boundary_mask(source)
    source = dewow(source, valid=source_valid)
    target_axis = np.arange(r.samples_per_trace)
    source_axis = np.arange(p.samples_per_trace) * sample_scale
    source = np.array([np.interp(target_axis, source_axis, t) for t in source])
    results = []
    for offset in range(-max_trace_offset, max_trace_offset + 1):
        shifts, qualities = [], []
        for row, trace in zip(rows, source, strict=True):
            if cancel and cancel():
                raise InterruptedError("Analysis cancelled")
            at = round(row * trace_scale + offset)
            if not 0 <= at < r.trace_count:
                continue
            target = np.asarray(raw.channel()[at : at + 1], np.float32)
            layout = source_layout(raw)
            target = dewow(numerical_extension(target, layout.mask(target)))[0]
            a, b = trace - np.mean(trace), target - np.mean(target)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            if norm <= 0:
                continue
            values = correlate(b, a, mode="full", method="fft") / norm
            lags = np.arange(-len(a) + 1, len(b))
            values[abs(lags) > r.samples_per_trace // 4] = -np.inf
            winner = int(np.argmax(values))
            shifts.append(float(lags[winner]))
            qualities.append(float(values[winner]))
        results.append(
            {
                "trace_offset": offset,
                "sample_shifts": shifts,
                "correlations": qualities,
                "quality": float(np.median(qualities)) if qualities else 0,
            }
        )
    results.sort(key=lambda v: -v["quality"])
    best = results[0]
    shifts = np.asarray(best["sample_shifts"])
    shift = float(np.median(shifts)) if len(shifts) else 0.0
    uncertainty = float(np.max(abs(shifts - shift))) if len(shifts) else float(r.samples_per_trace)
    trace_uncertainty = (
        0.0 if best["quality"] - results[1]["quality"] >= 0.02 else float(max_trace_offset + 1)
    )
    verified = bool(
        best["quality"] >= 0.8
        and uncertainty <= 1
        and trace_uncertainty <= 0.5
        and len(shifts) >= min(probes, p.trace_count) * 0.8
        and raw.path.stem.casefold() in processed.path.stem.casefold()
    )
    transform = CoordinateTransform(
        fingerprint_file(processed_path),
        fingerprint_file(raw_path),
        trace_scale,
        best["trace_offset"],
        sample_scale,
        shift,
        uncertainty,
        trace_uncertainty,
        verified,
        "radar_registration",
    )
    return {
        "schema": "radar-registration-v1",
        "transform": asdict(transform),
        "probes": results,
        "uses_layer_picks": False,
        "status": "verified" if verified else "rejected_for_accuracy_scoring",
        "limitations": (
            "Global affine mapping only; nonuniform warps require a separate verified transform"
        ),
    }
