"""Replay actual seeded inference after radar-selected requests and one revealed answer.

Only this evaluator reads DZX reference picks. The request policy receives radar,
current routes and already supplied anchors. An unavailable exact answer is logged
and consumes a request; it is never snapped or interpolated. Scoring tolerance is
frozen from the initial observations before the first request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = (
    Path(sys.argv[sys.argv.index("--workspace") + 1])
    if "--workspace" in sys.argv
    else Path(__file__).resolve().parents[1]
)
SOURCE = Path(sys.argv[sys.argv.index("--source") + 1]) if "--source" in sys.argv else ROOT / "src"
sys.path.insert(0, str(SOURCE))

from gpr_layer_audit.conventional import (  # noqa: E402
    backend_fingerprint,
    layer_metrics,
    peak_process_memory,
)
from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file  # noqa: E402
from gpr_layer_audit.io.dzx import read_dzx  # noqa: E402
from gpr_layer_audit.models import LayerSpec  # noqa: E402
from gpr_layer_audit.processing.active_queries import request_observation  # noqa: E402
from gpr_layer_audit.processing.conventional_config import (  # noqa: E402
    ConventionalConfig,
    resolve_pulse,
)
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask  # noqa: E402
from gpr_layer_audit.processing.processed_tracking import (  # noqa: E402
    fit_processed,
    guard_local_order,
    merge_local_paths,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_fingerprint(source):
    digest = hashlib.sha256()
    for path in sorted(Path(source).rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_write(path, data):
    """Replace a complete file, leaving the previous version intact on interruption."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_path(output):
    return Path(output).with_suffix(".checkpoint")


def checkpoint_header(path):
    with Path(path).open("rb") as stream:
        header = json.loads(stream.readline())
    if header.get("schema") != "seeded-replay-checkpoint-v1":
        raise ValueError("Unsupported replay checkpoint")
    return header


def save_checkpoint(path, contract, state):
    """Store complete path state, including alternatives needed by the request policy.

    Checkpoints contain pickle and are strictly trusted local experiment artifacts.
    The header and payload hash detect accidental change, not malicious replacement.
    Never resume a checkpoint obtained from an untrusted source.
    """
    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
    header = {
        "schema": "seeded-replay-checkpoint-v1",
        "contract": contract,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    atomic_write(path, json.dumps(header, sort_keys=True).encode() + b"\n" + payload)


def load_checkpoint(path, contract):
    with Path(path).open("rb") as stream:
        header = json.loads(stream.readline())
        payload = stream.read()
    if header.get("schema") != "seeded-replay-checkpoint-v1":
        raise ValueError("Unsupported replay checkpoint")
    if header.get("contract") != contract:
        expected = header.get("contract", {})
        changed = sorted(
            k for k in set(expected) | set(contract) if expected.get(k) != contract.get(k)
        )
        raise ValueError("Replay checkpoint contract changed: " + ", ".join(changed))
    if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
        raise ValueError("Replay checkpoint payload hash mismatch")
    return pickle.loads(payload)  # noqa: S301 - trusted local artifact, explicitly documented above


def replay(args):
    if (args.output.exists() or checkpoint_path(args.output).exists()) and not args.resume:
        raise ValueError(
            "Replay output exists; choose a new output to preserve the previous experiment"
        )
    manifest = read(ROOT / "benchmarks/seeded-evaluation-inputs.json")
    case = manifest["cases"][args.case]
    for name, digest in (
        ("dzt", "dzt_sha256"),
        ("dzx", "dzx_sha256"),
        ("seed_source", "seed_sha256"),
    ):
        if fingerprint_file(ROOT / case[name]) != case[digest]:
            raise ValueError(f"Frozen input changed: {name}")
    config = read(args.config)
    source = DZTFile(ROOT / case["dzt"])
    stride = args.stride or case["stride"]
    measurement = np.asarray(source.channel()[::stride], np.float32).copy()
    valid = processed_boundary_mask(measurement)
    dt, step = source.header.sample_interval_ns, source.header.distance_per_trace_m * stride
    surface = int(np.clip(round(-source.header.position_ns / dt), 0, measurement.shape[1] - 1))
    native = read(ROOT / case["seed_source"])["observations"]
    anchors = {}
    for order, picks in native.items():
        if any(p["trace"] % stride for p in picks):
            raise ValueError("Seed grid mismatch; snapping prohibited")
        anchors[int(order)] = {p["trace"] // stride: p["sample"] for p in picks}
    # Separate endpoint-seeding experiment preserves two of the fixed observations.
    if args.seeding == "endpoints":
        anchors = {o: {r: a[r] for r in (min(a), max(a))} for o, a in anchors.items()}
    initial_anchors = {o: dict(a) for o, a in anchors.items()}
    contract = {
        "case": args.case,
        "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "source_snapshot": str(SOURCE.resolve()),
        "source_sha256": source_fingerprint(SOURCE),
        "script_sha256": fingerprint_file(__file__),
        "method": args.method,
        "policy": args.policy,
        "scope": args.scope,
        "seeding": args.seeding,
        "stride": stride,
        "layers": args.layers,
        "actions": args.actions,
        "radius_m": args.radius_m,
        "freeze_initial_pulse": args.freeze_pulse,
    }
    layers = [
        LayerSpec(
            o, ("Asphalt", "Base", "Subbase")[o - 1], 1, measurement.shape[1] - 1 - surface, 1
        )
        for o in anchors
    ]
    # Evaluation labels stay here, outside all production request/tracker calls.
    references = {}
    for group in read_dzx(ROOT / case["dzx"]).layers:
        order = case["reference_label_mapping"].get(str(group.number), group.number + 1)
        if order in anchors:
            references[order] = [
                replace(p, trace=p.trace // stride)
                for p in group.picks
                if p.channel == 0 and p.trace % stride == 0
            ]
    pulse = {
        o: resolve_pulse(measurement, valid, a, {}, dt, ConventionalConfig()).lobe_samples
        for o, a in initial_anchors.items()
    }
    observations = {o: {p.trace: p for p in points} for o, points in references.items()}
    log = {
        "schema": "seeded-interaction-replay-v1",
        "case": args.case,
        "physical_road_group": case["physical_road_group"],
        "claim": "processed interpretation development",
        "method": args.method,
        "source_snapshot": str(SOURCE.resolve()),
        "freeze_initial_pulse": args.freeze_pulse,
        "policy": args.policy,
        "scope": args.scope,
        "seeding": args.seeding,
        "stride": stride,
        "step_m": step,
        "dt_ns": dt,
        "road_length_m": (len(measurement) - 1) * step,
        "input_sha256": case["dzt_sha256"],
        "reference_sha256": case["dzx_sha256"],
        "initial_seeds_sha256": case["seed_sha256"],
        "backend_sha256": backend_fingerprint(),
        "script_sha256": fingerprint_file(__file__),
        "source_sha256": contract["source_sha256"],
        "replay_contract": contract,
        "configuration": config,
        "configuration_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "initial_anchors": initial_anchors,
        "initial_pulse_samples": pulse,
        "denominator": "reviewed retained-grid observations excluding initial and revealed support",
        "unknown_reference_policy": "unscored; no answer at requested row logged; no snapping",
        "query_selection_reads_references": False,
        "actions": [],
        "steps": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = None
    visited, correction_anchors = set(), {}
    resume_iteration = -1
    if args.resume:
        restored = load_checkpoint(checkpoint_path(args.output), contract)
        paths, anchors = restored["paths"], restored["anchors"]
        visited, correction_anchors = restored["visited"], restored["correction_anchors"]
        log, resume_iteration = restored["log"], restored["completed_iteration"]
        atomic_write(args.output, json.dumps(log, indent=2, allow_nan=False).encode())
        if log.get("stop_reason"):
            return log
    for iteration in range(max(0, resume_iteration), args.actions + 1):
        start = time.perf_counter()
        start_cpu = time.process_time()
        if paths is None:
            paths = fit_processed(
                measurement,
                valid,
                surface,
                dt,
                step,
                layers,
                anchors,
                method=args.method,
                config=config,
                pulse_anchors=initial_anchors if args.freeze_pulse else None,
            )
        elapsed = time.perf_counter() - start
        cpu_elapsed = time.process_time() - start_cpu
        metrics = {
            str(o): layer_metrics(paths[o], references[o], anchors[o], measurement, pulse[o], step)
            for o in anchors
        }
        if iteration and log["actions"][-1].get("retrack_runtime_s") is not None:
            elapsed = log["actions"][-1]["retrack_runtime_s"]
            cpu_elapsed = log["actions"][-1].get("retrack_process_cpu_s")
        for order, value in metrics.items():
            value["diagnostics"] = paths[int(order)].provenance
            value["global_graph_diagnostics_scope"] = (
                "initial fit; local regenerated fit diagnostics are recorded with each action"
                if paths[int(order)].provenance.get("local_updates")
                else "initial fit"
            )
            n = value["observations_excluding_seeds"]
            value["incorrect_coverage"] = (
                (value["accepted"] - value["accepted_agree"]) / n if n else None
            )
            value["unresolved_coverage"] = 1 - value["accepted"] / n if n else None
            rows = value["evaluation_observations"]
            wrong = np.array(
                [r["row"] for r in rows if r["accepted"] and not r["accepted_correct"]]
            )
            groups = np.split(wrong, np.flatnonzero(np.diff(wrong) > 1) + 1)
            value["longest_contiguous_wrong_span_m"] = max(
                (len(g) * step for g in groups), default=0
            )
            value["initial_seed_count"] = len(initial_anchors[int(order)])
            fixed_n = len(references[int(order)]) - len(initial_anchors[int(order)])
            value["fixed_initial_observations"] = fixed_n
            value["correct_automatic_coverage_initial_pool"] = value["accepted_agree"] / fixed_n
            value["incorrect_automatic_coverage_initial_pool"] = (
                value["accepted"] - value["accepted_agree"]
            ) / fixed_n
            value["additional_observations"] = len(anchors[int(order)]) - len(
                initial_anchors[int(order)]
            )
        if iteration != resume_iteration:
            log["steps"].append(
                {
                    "step": iteration,
                    "requests": len(log["actions"]),
                    "runtime_s": elapsed,
                    "process_cpu_runtime_s": cpu_elapsed,
                    "layers": metrics,
                    "actions_per_km": len(log["actions"]) / max(log["road_length_m"] / 1000, 1e-9),
                    "process_peak_memory_bytes": peak_process_memory(),
                }
            )
            np.savez_compressed(
                args.output.with_name(f"{args.output.stem}-step{iteration}.npz"),
                **{
                    f"layer{o}_{name}": a
                    for o, p in paths.items()
                    for name, a in (
                        ("accepted", p.samples),
                        ("provisional", p.provisional_samples),
                        ("visible", p.visible),
                    )
                    if a is not None
                },
            )
            save_checkpoint(
                checkpoint_path(args.output),
                contract,
                {
                    "paths": paths,
                    "anchors": anchors,
                    "visited": visited,
                    "correction_anchors": correction_anchors,
                    "log": log,
                    "completed_iteration": iteration,
                },
            )
            atomic_write(args.output, json.dumps(log, indent=2, allow_nan=False).encode())
            print(
                json.dumps(
                    {
                        "step": iteration,
                        "layers": {
                            o: (m["accepted_agree"], m["accepted"], m["correct_coverage"])
                            for o, m in metrics.items()
                        },
                    }
                ),
                flush=True,
            )
        if iteration == args.actions:
            break
        requested_paths = {o: p for o, p in paths.items() if o in args.layers}
        request = request_observation(
            requested_paths, measurement, valid, anchors, step, visited=visited, policy=args.policy
        )
        if request is None:
            log["stop_reason"] = "No measurable unvisited request"
            break
        order, row = request["layer_order"], request["row"]
        visited.add((order, row))
        action = {
            **request,
            "native_trace": row * stride,
            "chainage_m": row * step,
            "scope": args.scope,
        }
        # The only point where a previously hidden answer crosses into operation.
        answer = observations[order].get(row)
        if answer is None:
            action["action"] = "unavailable_reviewed_answer"
            log["actions"].append(action)
            continue
        proposed = paths[order].provisional_samples
        action["action"] = (
            "confirmation"
            if proposed is not None and proposed[row] == answer.sample
            else "correction"
        )
        action["answer_sample"] = int(answer.sample)
        anchors[order][row] = int(answer.sample)
        correction_anchors.setdefault(order, {})[row] = int(answer.sample)
        t = time.perf_counter()
        cpu_start = time.process_time()
        active = {o: dict(a) for o, a in initial_anchors.items()}
        lo = max(0, int(np.ceil(row - args.radius_m / step)))
        hi = min(len(measurement) - 1, int(np.floor(row + args.radius_m / step)))
        if args.scope == "interval":
            support_rows = sorted(initial_anchors[order])
            lo = max((r for r in support_rows if r <= row), default=0)
            hi = min((r for r in support_rows if r >= row), default=len(measurement) - 1)
        for o, values in correction_anchors.items():
            active[o].update({r: s for r, s in values.items() if lo <= r <= hi})
        regenerated = fit_processed(
            measurement,
            valid,
            surface,
            dt,
            step,
            layers,
            active,
            method=args.method,
            config=config,
            pulse_anchors=initial_anchors if args.freeze_pulse else None,
        )
        action["regenerated_fit_diagnostics"] = {
            name: regenerated[order].provenance.get(name)
            for name in ("resolved_pulse", "graph_diagnostics", "segments", "edges", "hypotheses")
        }
        action["regenerated_fit_diagnostics_scope"] = (
            "full-context attempted fit; only affected rows/layer enter the displayed result"
        )
        merged = merge_local_paths(paths, regenerated, orders={order}, start_row=lo, stop_row=hi)
        try:
            guard_local_order(merged, paths, layers, active, {order}, lo, hi)
        except ValueError as exc:
            action["retrack_status"] = "neighbor_observation_conflict"
            action["reason"] = str(exc)
        else:
            paths = merged
        action["affected_rows"] = [lo, hi]
        action["affected_layer_orders"] = [order]
        action["operation"] = (
            "explicit_interval_reseed" if args.scope == "interval" else "local_correction"
        )
        action["retrack_runtime_s"] = time.perf_counter() - t
        action["retrack_process_cpu_s"] = time.process_time() - cpu_start
        log["actions"].append(action)
    log.setdefault("stop_reason", "Requested action budget completed")
    save_checkpoint(
        checkpoint_path(args.output),
        contract,
        {
            "paths": paths,
            "anchors": anchors,
            "visited": visited,
            "correction_anchors": correction_anchors,
            "log": log,
            "completed_iteration": iteration,
        },
    )
    atomic_write(args.output, json.dumps(log, indent=2, allow_nan=False).encode())
    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="seed_hybrid")
    parser.add_argument("--policy", choices=("active", "midpoint"), default="active")
    parser.add_argument("--scope", choices=("local", "interval"), default="local")
    parser.add_argument("--seeding", choices=("three", "endpoints"), default="three")
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--radius-m", type=float, default=25.0)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume a trusted local checkpoint")
    pulse_group = parser.add_mutually_exclusive_group()
    pulse_group.add_argument(
        "--freeze-pulse",
        action="store_true",
        default=True,
        help="Preserve initial seed packet scale during corrections",
    )
    pulse_group.add_argument(
        "--refit-pulse",
        action="store_false",
        dest="freeze_pulse",
        help="Comparator only: recompute packet scale from all current observations",
    )
    args = parser.parse_args()
    if args.source is None:
        if args.resume:
            header = checkpoint_header(checkpoint_path(args.output))
            snapshot = Path(header["contract"]["source_snapshot"]).parent
            if not (snapshot / "replay.py").exists():
                raise ValueError("Original replay snapshot script is missing")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(snapshot / "replay.py"),
                    *sys.argv[1:],
                    "--workspace",
                    str(ROOT),
                    "--source",
                    str(snapshot / "src"),
                ],
                cwd=ROOT,
            )
            raise SystemExit(completed.returncode)
        source_hash = source_fingerprint(ROOT / "src")
        script_bytes = Path(__file__).read_bytes()
        digest = hashlib.sha256(source_hash.encode() + script_bytes)
        snapshot = ROOT / "exports/seeded-tracker" / ("replay-source-" + digest.hexdigest()[:16])
        if not snapshot.exists():
            shutil.copytree(
                ROOT / "src", snapshot / "src", ignore=shutil.ignore_patterns("__pycache__")
            )
            shutil.copyfile(__file__, snapshot / "replay.py")
        if (
            source_fingerprint(snapshot / "src") != source_hash
            or (snapshot / "replay.py").read_bytes() != script_bytes
        ):
            raise ValueError("Replay snapshot changed during creation or after it was frozen")
        completed = subprocess.run(
            [
                sys.executable,
                str(snapshot / "replay.py"),
                *sys.argv[1:],
                "--workspace",
                str(ROOT),
                "--source",
                str(snapshot / "src"),
            ],
            cwd=ROOT,
        )
        raise SystemExit(completed.returncode)
    replay(args)


if __name__ == "__main__":
    main()
