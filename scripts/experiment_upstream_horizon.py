"""Pinned upstream numerical differential and one frozen-graph-input experiment.

Use the isolated exports/seeded-tracker/upstream/venv interpreter. The default
differential chooses radar crops around operating seeds before reading labels.
``--run`` adds whole-profile warped cosine to correspondence only; all candidate,
phase, motion, packet-DTW, objective and final acceptance rules stay frozen.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "bf550594bd55d7741a0fcbfdef0e4ede2567d48b"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, document):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")


def load_upstream(checkout):
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN:
        raise ValueError("Unexpected upstream revision")
    subprocess.run(["git", "-C", str(checkout), "diff", "HEAD", "--exit-code"], check=True)
    spec = importlib.util.spec_from_file_location(
        "pinned_horizon_tracker", checkout / "HorizonTracker_functions.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    return upstream


def differential(case_name, case, upstream):
    import numpy as np

    from gpr_layer_audit.io.dzt import DZTFile
    from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
    from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask
    from gpr_layer_audit.processing.hybrid import constrained_dtw
    from gpr_layer_audit.processing.trace_registration import balance_waveforms, register_pairs
    from gpr_layer_audit.processing.upstream_horizon import profile_paths, valid_runs

    radar = DZTFile(ROOT / case["dzt"])
    stride = case["stride"]
    data = np.array(radar.channel()[::stride], dtype=np.float32)
    valid = processed_boundary_mask(data)
    document = json.loads((ROOT / case["seed_source"]).read_text())
    rows = []
    for layer in (2, 3):
        seeds = {p["trace"] // stride: p["sample"] for p in document["observations"][str(layer)]}
        pulse = resolve_pulse(data, valid, seeds, {}, case["dt_ns"], ConventionalConfig())
        for seed_row, sample in seeds.items():
            for gap in (1, max(1, round(.5 / (stride * case["native_dx_m"]))),
                        max(1, round(2 / (stride * case["native_dx_m"])))):
                target_row = seed_row + gap if seed_row + gap < len(data) else seed_row - gap
                if target_row < 0:
                    continue
                low = max(0, sample - 4 * pulse.context_radius)
                high = min(data.shape[1], sample + 4 * pulse.context_radius + 1)
                common = valid[seed_row] & valid[target_row]
                runs = [(a, b) for a, b in valid_runs(common) if a <= sample < b]
                if not runs:
                    continue
                low, high = max(low, runs[0][0]), min(high, runs[0][1])
                first, second = data[seed_row, low:high], data[target_row, low:high]
                if len(first) < 3 or first[-1] == 0 or second[-1] == 0:
                    continue
                cube = np.stack([first, second], axis=1)[:, :, None]
                with contextlib.redirect_stdout(io.StringIO()):
                    info = upstream.DynamicTimeWarping(cube, [[0, 0], [1, 0]], 1)
                reference = np.asarray(next(p[4] for p in info if p[:4] == [0, 0, 1, 0]))
                forward, reverse = profile_paths(first, second)
                if not np.array_equal(reference, forward):
                    raise AssertionError("Adapter differs from actual pinned upstream path")
                mapped = forward[forward[:, 0] == sample - low, 1] + low
                balanced = balance_waveforms(np.stack([first, second]), pulse.lobe_samples)
                f, _, _, _ = register_pairs(
                    balanced[:1], balanced[1:], max(1, int(pulse.lobe_samples)), pulse.lobe_samples
                )
                radius = pulse.context_radius
                a = data[seed_row, sample-radius:sample+radius+1].astype(float)
                b = data[target_row, sample-radius:sample+radius+1].astype(float)
                a, b = a-a.mean(), b-b.mean()
                a /= max(np.linalg.norm(a), 1e-9)
                b /= max(np.linalg.norm(b), 1e-9)
                agreement, shift = constrained_dtw(a, b, max(1, int(pulse.lobe_samples/4)))
                # Compare the repository's actual scalar DP to upstream on
                # identical packets with its band made nonrestrictive.
                packet_cube = np.stack([a, b], axis=1)[:, :, None]
                with contextlib.redirect_stdout(io.StringIO()):
                    packet_info = upstream.DynamicTimeWarping(packet_cube, [[0, 0], [1, 0]], 1)
                packet_path = np.asarray(
                    next(p[4] for p in packet_info if p[:4] == [0, 0, 1, 0])
                )
                packet_cost = np.sum((a[packet_path[:, 0]] - b[packet_path[:, 1]]) ** 2)
                packet_shift = np.mean(
                    packet_path[packet_path[:, 0] == len(a)//2, 1] - len(b)//2
                )
                full_agreement, full_shift = constrained_dtw(a, b, len(a))
                np.testing.assert_allclose(full_agreement, np.exp(-packet_cost), atol=1e-10)
                np.testing.assert_allclose(full_shift, packet_shift, atol=1e-10)
                rows.append({
                    "layer": layer, "seed_native_trace": seed_row * stride,
                    "target_native_trace": target_row * stride, "seed_sample": sample,
                    "crop_native_samples": [int(low), int(high)],
                    "dt_ns": case["dt_ns"], "distance_m": abs(target_row-seed_row)
                    * stride * case["native_dx_m"],
                    "timing_tolerance_samples": max(2, pulse.lobe_samples / 4),
                    "upstream_matches_adapter": True,
                    "upstream_mapped_samples": mapped.tolist(),
                    "upstream_path_pairs": len(forward), "reverse_path_pairs": len(reverse),
                    "repository_profile_map_sample": float(f[0, sample-low] + low),
                    "repository_packet_cosine_at_seed_sample": float(a @ b),
                    "repository_packet_dtw_agreement": agreement,
                    "repository_packet_centre_shift": shift,
                    "repository_unrestricted_packet_matches_upstream": True,
                    "repository_unrestricted_packet_agreement": full_agreement,
                    "repository_unrestricted_packet_centre_shift": full_shift,
                })
    # These reviewed observations are opened only after all matching is done.
    from gpr_layer_audit.conventional import _same_lobe
    from gpr_layer_audit.io.dzx import read_dzx

    references = {
        int(case["reference_label_mapping"][str(group.number)]):
        {p.trace: p.sample for p in group.picks if p.channel == 0}
        for group in read_dzx(ROOT / case["dzx"]).layers
        if str(group.number) in case["reference_label_mapping"]
    }
    for row in rows:
        sample = references[row["layer"]].get(row["target_native_trace"])
        row["reviewed_target_sample"] = sample
        if sample is not None:
            trace = data[row["target_native_trace"] // stride]
            row["upstream_contains_reviewed_signed_lobe"] = any(
                _same_lobe(trace, sample, value) for value in row["upstream_mapped_samples"]
            )
            row["upstream_contains_agreeing_sample"] = any(
                _same_lobe(trace, sample, value)
                and abs(sample - value) <= row["timing_tolerance_samples"]
                for value in row["upstream_mapped_samples"]
            )
    return {"case": case_name, "usage": "development diagnostic", "pairs": rows}


def run(args, case):
    from gpr_layer_audit.conventional import evaluate_reference, write_json
    from gpr_layer_audit.processing import hybrid
    from gpr_layer_audit.processing.upstream_horizon import ProfileCorrespondence

    original = hybrid.correspondence_graph
    source = inspect.getsource(original)
    needle = "similarity = matching_waveforms[left, ii] @ matching_waveforms[right, js].T"
    replacement = ("similarity = upstream_context.similarity("
                   "left, right, ii, js, table, matching_waveforms, pulse_width)")
    if source.count(needle) != 1:
        raise ValueError("Frozen correspondence numerical contract changed")
    source = source.replace(needle, replacement)
    patched = {}
    graphs = []

    def wrapped(*values, **kwargs):
        table = values[0]
        context = ProfileCorrespondence(
            kwargs["measurement"], table.component_maps.get("sample_validity")
        )
        environment = dict(original.__globals__, upstream_context=context)
        exec(compile(source, "<upstream-warped-context-experiment>", "exec"), environment, patched)
        result = patched["correspondence_graph"](*values, **kwargs)
        graphs.append(dict(context.counts))
        print(json.dumps(context.counts), flush=True)
        return result

    if not args.control:
        hybrid.correspondence_graph = wrapped
    config_path = ROOT / "benchmarks/conventional-motion-calibrated-development.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    provenance = {
        "source": str(args.source), "script_sha256": sha(__file__),
        "adapter_sha256": sha(args.source / "gpr_layer_audit/processing/upstream_horizon.py"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
    }
    start = time.perf_counter()
    try:
        result = evaluate_reference(
            ROOT / case["dzx"], output=args.output, stride=case["stride"],
            methods=["seed_hybrid"], seed_source=ROOT / case["seed_source"],
            config=config,
        )
    finally:
        hybrid.correspondence_graph = original
    result["upstream_experiment"] = {
        "upstream_pin": PIN, "mechanism": "reciprocal whole-profile warped-context cosine",
        "control": args.control,
        "graphs": graphs, "runtime_s": time.perf_counter()-start,
        **provenance,
    }
    write_json(args.output, result)
    print(json.dumps({order: {k: v[k] for k in (
        "accepted", "accepted_agree", "correct_coverage", "reflector_switches"
    )} for order, v in result["methods"]["seed_hybrid"]["layers"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--upstream", type=Path,
                        default=ROOT / "exports/seeded-tracker/upstream/HorizonTracker")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve previous experiments")
    if args.source is None:
        args.source = args.output.with_name(args.output.stem + "-source") / "src"
        args.source.parent.mkdir(parents=True, exist_ok=False)
        shutil.copytree(ROOT / "exports/seeded-tracker/baseline-source/src", args.source)
        shutil.copy2(ROOT / "src/gpr_layer_audit/processing/upstream_horizon.py",
                     args.source / "gpr_layer_audit/processing/upstream_horizon.py")
        shutil.copy2(__file__, args.source.parent / Path(__file__).name)
        command = [sys.executable, str(args.source.parent / Path(__file__).name),
                   "--output", str(args.output.resolve()), "--source", str(args.source.resolve()),
                   "--upstream", str(args.upstream.resolve())]
        if args.case:
            command += ["--case", args.case]
        if args.run:
            command += ["--run"]
        if args.control:
            command += ["--control"]
        # Snapshot script location is not the repository root.
        command += ["--workspace", str(ROOT)]
        return subprocess.run(command, check=True).returncode
    sys.path.insert(0, str(args.source))
    manifest_path = ROOT / "benchmarks/seeded-evaluation-inputs.json"
    manifest = json.loads(manifest_path.read_text())
    cases = {k: v for k, v in manifest["cases"].items() if args.case is None or k == args.case}
    for case in cases.values():
        for key, fingerprint in (("dzt", "dzt_sha256"), ("dzx", "dzx_sha256"),
                                 ("seed_source", "seed_sha256")):
            if sha(ROOT / case[key]) != case[fingerprint]:
                raise ValueError(f"Frozen input changed: {key}")
    upstream = load_upstream(args.upstream)
    if args.run:
        if len(cases) != 1:
            parser.error("Full run requires one explicit case")
        run(args, next(iter(cases.values())))
    else:
        write(args.output, {
            "upstream_pin": PIN, "upstream_license": "MIT, Aina Juell Bugge 2019",
            "upstream_function_sha256": sha(args.upstream / "HorizonTracker_functions.py"),
            "manifest_sha256": sha(manifest_path), "script_sha256": sha(__file__),
            "adapter_sha256": sha(args.source / "gpr_layer_audit/processing/upstream_horizon.py"),
            "source": str(args.source),
            "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
            "cases": [differential(name, case, upstream) for name, case in cases.items()],
        })
    return 0


if __name__ == "__main__":
    if "--workspace" in sys.argv:
        offset = sys.argv.index("--workspace")
        ROOT = Path(sys.argv[offset+1]).resolve()
        del sys.argv[offset:offset+2]
    raise SystemExit(main())
