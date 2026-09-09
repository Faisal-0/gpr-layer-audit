"""Reproducible conventional benchmark CLI. Never imports or enables ML."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .conventional_reference import audit_reference, distributed_seeds, road_partition
from .io.dzt import DZTFile, fingerprint_file
from .io.dzx import read_dzx
from .models import LayerSpec
from .processing.conventional_signal import (
    CoordinateTransform,
    numerical_extension,
    processed_boundary_mask,
)
from .processing.tracker import pick_interfaces


def peak_process_memory():
    """OS high-water resident memory, including NumPy, without tracing allocations."""
    import sys

    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        function = ctypes.windll.psapi.GetProcessMemoryInfo
        function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not function(handle, ctypes.byref(counter), counter.cb):
            raise OSError("Unable to query peak working set")
        return int(counter.PeakWorkingSetSize)
    import resource

    return int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024)
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def backend_fingerprint():
    return hashlib.sha256(
        b"".join(
            p.name.encode() + p.read_bytes()
            for p in sorted(Path(__file__).parent.joinpath("processing").glob("*.py"))
        )
    ).hexdigest()


def _same_lobe(trace, a, b):
    left, right = sorted((int(round(a)), int(round(b))))
    if left < 0 or right >= len(trace):
        return False
    sign = np.sign(trace[left])
    return bool(sign != 0 and np.all(np.sign(trace[left : right + 1]) == sign))


def layer_metrics(path, references, seeds, measurement, pulse_width, step):
    """No missing reference is scored as absence. Selected signed lobe is checked directly."""
    held = [p for p in references if p.trace not in seeds]
    tolerance = max(2, pulse_width / 4)
    proposed = path.provisional_samples if path.provisional_samples is not None else path.samples
    counts = dict(
        candidate_retained=0,
        correspondence_survived=0,
        proposed_agree=0,
        accepted=0,
        accepted_agree=0,
        reflector_switches=0,
    )
    errors, failed_rows, observations = [], [], []
    for point in held:
        row, sample = point.trace, point.sample
        candidates = path.candidate_components.get("audit_candidate_rank")
        if candidates is not None:
            cols = np.flatnonzero(np.isfinite(candidates[row]))
            counts["candidate_retained"] += any(
                abs(c - sample) <= tolerance and _same_lobe(measurement[row], c, sample)
                for c in cols
            )
        correspondence = path.candidate_components.get("hybrid_correspondence")
        if correspondence is not None:
            lo, hi = (
                max(0, int(sample - tolerance)),
                min(measurement.shape[1], int(sample + tolerance) + 1),
            )
            counts["correspondence_survived"] += any(
                correspondence[row, c] > 0 and _same_lobe(measurement[row], c, sample)
                for c in range(lo, hi)
            )
        proposed_ok = (
            proposed[row] >= 0
            and abs(proposed[row] - sample) <= tolerance
            and _same_lobe(measurement[row], proposed[row], sample)
        )
        counts["proposed_agree"] += int(proposed_ok)
        accepted = path.samples[row] >= 0 and bool(path.visible[row])
        counts["accepted"] += int(accepted)
        good = (
            accepted
            and abs(path.samples[row] - sample) <= tolerance
            and _same_lobe(measurement[row], path.samples[row], sample)
        )
        counts["accepted_agree"] += int(good)
        if accepted:
            errors.append(abs(float(path.samples[row]) - sample))
            counts["reflector_switches"] += int(
                not _same_lobe(measurement[row], path.samples[row], sample)
            )
        if not good:
            failed_rows.append(row)
        observations.append(
            {
                "row": int(row),
                "reference_sample": int(sample),
                "proposed_sample": int(proposed[row]),
                "proposed_correct": bool(proposed_ok),
                "correspondence": float(
                    path.evidence.get("hybrid_correspondence", np.zeros(len(measurement)))[row]
                ),
                "path_margin": float(
                    path.evidence.get("hybrid_path_margin", np.zeros(len(measurement)))[row]
                ),
                "measurement_support": float(
                    path.evidence.get("measurement_support", np.zeros(len(measurement)))[row]
                ),
                "accepted": bool(accepted),
                "accepted_correct": bool(good),
            }
        )
    n = len(held)
    return {
        "observations_excluding_seeds": n,
        "seed_rows": sorted(seeds),
        "tolerance_samples": tolerance,
        **counts,
        "candidate_retention": counts["candidate_retained"] / n if n else None,
        "correspondence_survival": counts["correspondence_survived"] / n if n else None,
        "proposed_path_agreement": counts["proposed_agree"] / n if n else None,
        "accepted_pick_agreement": counts["accepted_agree"] / counts["accepted"]
        if counts["accepted"]
        else None,
        "correct_coverage": counts["accepted_agree"] / n if n else None,
        "median_accepted_error_samples": float(np.median(errors)) if errors else None,
        "correction_observations": len(failed_rows),
        "correction_intervals": int(1 + np.count_nonzero(np.diff(failed_rows) * step > 2))
        if failed_rows
        else 0,
        "correction_burden_kind": "disagreement proxy; not measured analyst actions",
        "identity_rule": "same signed lobe without intervening zero/sign crossing",
        "evaluation_observations": observations,
    }


def evaluate_reference(
    dzx,
    *,
    output,
    methods=("joint_seed_adaptive", "seed_hybrid"),
    mode="processed",
    raw=None,
    mapping=None,
    stride=1,
    frozen=None,
    config=None,
    layer_map=None,
    seed_source=None,
    cancel=None,
):
    metadata = read_dzx(dzx)
    processed = Path(dzx).with_suffix(".DZT")
    source = DZTFile(processed)
    partition = road_partition(processed)
    if partition[1] == "held_out" and frozen is None:
        raise ValueError("Held-out roads require a frozen protocol file")
    if partition[1] == "held_out":
        from .processing.conventional_config import resolve_config

        protocol = json.loads(Path(frozen).read_text(encoding="utf-8"))
        digest = resolve_config(config).fingerprint
        if protocol.get("configuration_sha256") != digest or not protocol.get("settings_frozen"):
            raise ValueError("Held-out configuration differs from frozen protocol")
        if protocol.get("backend_source_sha256") != backend_fingerprint():
            raise ValueError("Held-out backend differs from frozen calibration")
    if stride < 1:
        raise ValueError("Stride must be positive")
    audit = audit_reference(dzx)
    if any(layer["issues"] for layer in audit["layers"]):
        raise ValueError("Reference coordinates or duplicates require review before evaluation")
    label_mapping = layer_map or {"0": 1, "1": 2, "2": 3}
    reused_seeds, seed_provenance = None, None
    if seed_source is not None:
        from .conventional_seeds import load_support

        reused_seeds, seed_provenance = load_support(
            seed_source, audit, label_mapping, stride, mode
        )
    transform = None
    if mode == "raw":
        if raw is None or mapping is None:
            raise ValueError("Raw benchmark requires --raw and --mapping")
        transform = CoordinateTransform(**json.loads(Path(mapping).read_text(encoding="utf-8")))
        transform.require_scoring(fingerprint_file(processed), fingerprint_file(raw), 2)
        source = DZTFile(raw)
        from .processing.calibration import calibrate

        calibrated = calibrate(source, None, stack_size=1, conventional=True)
        measurement = calibrated.measurement_radargram[::stride].copy()
    else:
        measurement = np.asarray(source.channel()[::stride], np.float32).copy()
    dt = source.header.sample_interval_ns
    reference_surface = (
        int(np.clip(round(-source.header.position_ns / dt), 0, measurement.shape[1] - 1))
        if mode == "processed"
        else calibrated.reference_surface_sample
    )
    step = source.header.distance_per_trace_m
    if step is None:
        raise ValueError("Benchmark requires physical horizontal sampling")
    step *= stride
    valid = (
        processed_boundary_mask(measurement)
        if mode == "processed"
        else calibrated.sample_validity[::stride]
    )
    # Baseline uses the identical processed radar; signal ablations explicitly opt into validity.
    use_validity = bool(config and config.get("signal_validity", True))
    working = numerical_extension(measurement, valid) if use_validity else measurement.copy()
    layers, anchors, observations, selected_native = [], {}, {}, {}
    from .io.dzx import DZXPick

    for group in metadata.layers:
        order = (layer_map or {}).get(str(group.number), group.number + 1)
        if order not in (1, 2, 3) or not group.picks:
            continue
        picks = []
        for p in group.picks:
            if p.channel != 0:
                continue
            trace, sample = (
                (p.trace, p.sample) if transform is None else transform.forward(p.trace, p.sample)
            )
            if transform is not None:
                raw_row = round(float(trace))
                sample += calibrated.reference_surface_sample - calibrated.surface_samples[raw_row]
            if abs(float(trace) / stride - round(float(trace) / stride)) > 1e-6:
                continue
            picks.append(
                DZXPick(
                    round(float(trace) / stride),
                    round(float(sample)),
                    0,
                    p.interpretation_property,
                    p.time_ns,
                    p.recorded_amplitude,
                    p.recorded_depth,
                    p.recorded_velocity,
                )
            )
        if not picks:
            continue
        selected = (
            distributed_seeds(picks)
            if reused_seeds is None
            else [p for p in picks if p.trace * stride in reused_seeds[order]]
        )
        if mode == "processed":
            native_rows = {p.trace * stride for p in selected}
            selected_native[order] = [
                p for p in group.picks if p.channel == 0 and p.trace in native_rows
            ]
        anchors[order] = {p.trace: p.sample for p in selected}
        observations[order] = picks
        # No evaluation depth enters search bounds or pulse estimation.
        layers.append(
            LayerSpec(
                order,
                ("Asphalt", "Base", "Subbase")[order - 1],
                1,
                working.shape[1] - 1 - reference_surface,
                1,
            )
        )
    result = {
        "schema": "conventional-evaluation-v2",
        "mode": mode,
        "audit": audit,
        "ml": "disabled",
        "reference_surface_sample": reference_surface,
        "surface_origin_provenance": (
            "processed DZT header time origin"
            if mode == "processed"
            else "raw radar surface alignment"
        ),
        "backend_source_sha256": backend_fingerprint(),
        "partition": partition,
        "stride": stride,
        "reference_label_mapping": label_mapping,
        "label_mapping_status": "explicit"
        if layer_map
        else "RADAN numbered layers; semantic labels require review",
        "config": config or {},
        "transform": asdict(transform) if transform else None,
        "methods": {},
        "promotion": {"promoted_layers": [], "subbase": "experimental"},
    }
    from .conventional_seeds import native_support

    result["seed_support"] = (
        native_support(audit, label_mapping, selected_native) if mode == "processed" else None
    )
    result["seed_selection"] = seed_provenance or {
        "basis": "10/50/90 percent of annotated extent on retained grid",
        "resolution_comparison_requires_reused_native_seeds": True,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay_paths = {}
    from .processing.conventional_config import ConventionalConfig, resolve_pulse

    scoring_pulses = {
        order: resolve_pulse(measurement, valid, seed, {}, dt, ConventionalConfig())
        for order, seed in anchors.items()
    }
    result["scoring_pulses"] = {
        str(order): pulse.metadata() for order, pulse in scoring_pulses.items()
    }
    result["scoring_tolerance_policy"] = (
        "Common seed-only pulse estimate for all comparators and ablations"
    )
    for method in methods:
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        start = time.perf_counter()
        kwargs = {}
        if config is not None:
            kwargs["conventional_config"] = config
        branches = {"hybrid_measurement": working}
        if use_validity:
            branches["sample_validity"] = valid
        try:
            paths = pick_interfaces(
                working,
                reference_surface,
                layers,
                anchor_samples=anchors,
                feature_branches=branches,
                method=method,
                horizontal_step_m=step,
                sample_interval_ns=dt,
                ml_policy="off",
                cancel=cancel,
                **kwargs,
            )
        except (ValueError, RuntimeError, IndexError) as exc:
            result["methods"][method] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_s": time.perf_counter() - start,
                "layers": {},
            }
            write_json(output, result)
            continue
        runtime = time.perf_counter() - start
        peak = peak_process_memory()
        metrics = {}
        for order, path in paths.items():
            pulse = scoring_pulses[order].lobe_samples
            metrics[str(order)] = layer_metrics(
                path, observations[order], anchors[order], measurement, pulse, step
            )
            metrics[str(order)]["diagnostics"] = path.provenance
        result["methods"][method] = {
            "runtime_s": runtime,
            "process_peak_memory_bytes": peak,
            "memory_scope": "process lifetime high-water resident set",
            "layers": metrics,
        }
        overlay_paths[method] = paths
        np.savez_compressed(
            output.with_name(output.stem + "-" + method + ".npz"),
            sample_validity=valid,
            **{
                f"layer{order}_{key}": value
                for order, p in paths.items()
                for key, value in (
                    ("accepted", p.samples),
                    (
                        "provisional",
                        p.provisional_samples if p.provisional_samples is not None else p.samples,
                    ),
                )
            },
        )
    if overlay_paths:
        _overlay(output.with_suffix(".png"), working, observations, anchors, overlay_paths, step)
    write_json(output, result)
    return result


def _overlay(path, radar, references, seeds, methods, step):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(methods), 1, figsize=(16, 5 * len(methods)), squeeze=False)
    scale = max(float(np.percentile(abs(radar), 98)), 1e-9)
    for ax, (method, paths) in zip(axes[:, 0], methods.items(), strict=True):
        ax.imshow(
            radar.T,
            cmap="gray",
            aspect="auto",
            vmin=-scale,
            vmax=scale,
            extent=(0, len(radar) * step, radar.shape[1], 0),
        )
        for order, picks in references.items():
            ax.scatter(
                [p.trace * step for p in picks],
                [p.sample for p in picks],
                s=5,
                color="cyan",
                label="RADAN reference" if order == min(references) else None,
            )
            p = paths[order]
            proposal = p.provisional_samples if p.provisional_samples is not None else p.samples
            ax.plot(
                np.arange(len(radar)) * step,
                np.where(proposal >= 0, proposal, np.nan),
                color="orange",
                lw=0.8,
                label="Provisional (gaps omitted)" if order == min(references) else None,
            )
            ax.plot(
                np.arange(len(radar)) * step,
                np.where(p.samples >= 0, p.samples, np.nan),
                color="lime",
                lw=1,
                label="Accepted" if order == min(references) else None,
            )
            ax.scatter(
                np.array(list(seeds[order])) * step,
                list(seeds[order].values()),
                c="magenta",
                s=35,
                marker="x",
                label="Seeds (excluded)" if order == min(references) else None,
            )
        ax.set(title=method, xlabel="Distance (m)", ylabel="Stored sample")
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def add_commands(subparsers):
    parser = subparsers.add_parser(
        "conventional", help="RADAN reference audit and conventional evaluation"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    register = commands.add_parser(
        "register", help="Audit a radar-only affine raw/processed mapping"
    )
    register.add_argument("source", type=Path)
    register.add_argument("raw", type=Path)
    register.add_argument("--output", type=Path, required=True)
    calibrate = commands.add_parser(
        "calibrate", help="Fit acceptance thresholds on development/calibration only"
    )
    calibrate.add_argument("sources", type=Path, nargs="+")
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze supported settings for subsequent held-out evaluation",
    )
    audit = commands.add_parser("audit")
    audit.add_argument("source", type=Path)
    audit.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("source", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--mode", choices=("processed", "raw"), default="processed")
    evaluate.add_argument("--raw", type=Path)
    evaluate.add_argument("--mapping", type=Path)
    evaluate.add_argument("--frozen", type=Path)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument(
        "--layer-map", type=Path, help="Explicit RADAN layer-number to interface-order JSON"
    )
    evaluate.add_argument("--stride", type=int, default=1)
    evaluate.add_argument(
        "--seed-source",
        type=Path,
        help="Reuse exact native seeds from a processed evaluation or seed manifest",
    )
    evaluate.add_argument("--methods", nargs="+", default=["joint_seed_adaptive", "seed_hybrid"])


def run_command(args):
    if args.operation == "calibrate":
        from .conventional_validation import calibrate_acceptance

        report = calibrate_acceptance(
            args.sources, json.loads(args.config.read_text()), freeze=args.freeze
        )
        write_json(args.output, report)
        write_json(args.output.with_name(args.output.stem + "-config.json"), report["config"])
    elif args.operation == "register":
        from .processing.radar_mapping import register_sources

        report = register_sources(args.source, args.raw)
        write_json(args.output, report)
        write_json(args.output.with_name(args.output.stem + "-transform.json"), report["transform"])
    elif args.operation == "audit":
        sources = sorted(args.source.rglob("*.DZX")) if args.source.is_dir() else [args.source]
        write_json(
            args.output,
            {"schema": "radan-audit-v1", "references": [audit_reference(p) for p in sources]},
        )
    else:
        evaluate_reference(
            args.source,
            output=args.output,
            methods=args.methods,
            mode=args.mode,
            raw=args.raw,
            mapping=args.mapping,
            stride=args.stride,
            frozen=args.frozen,
            config=json.loads(args.config.read_text()) if args.config else None,
            layer_map=json.loads(args.layer_map.read_text()) if args.layer_map else None,
            seed_source=args.seed_source,
        )
    return 0
