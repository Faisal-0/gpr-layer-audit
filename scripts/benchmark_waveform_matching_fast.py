"""Reproducible exact-DTW microbenchmark with operating-seed radar packets only.

Run with the isolated upstream environment (Numba 0.67.0) or the application's
environment after optional integration. No DZX or withheld labels are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_reciprocal(left, right, band):
    # Retain the original NumPy function as an independent reference even after
    # production reciprocal_dtw dispatches to the compiled implementation.
    from gpr_layer_audit.processing.waveform_matching import batch_dtw

    agreement, shift = batch_dtw(left, right, band)
    reverse, reverse_shift = batch_dtw(right, left, band)
    return np.minimum(agreement, reverse), np.maximum(abs(shift), abs(reverse_shift))


def timed(function, left, right, band, repetitions, samples=5):
    wall, cpu = [], []
    for _ in range(samples):
        began, cpu_began = time.perf_counter(), time.process_time()
        for _ in range(repetitions):
            function(left, right, band)
        wall.append((time.perf_counter() - began) / repetitions)
        cpu.append((time.process_time() - cpu_began) / repetitions)
    return {"wall_seconds": wall, "wall_median_seconds": float(np.median(wall)),
            "process_cpu_seconds": cpu, "repetitions_per_sample": repetitions}


def real_packets(case_name, length):
    from gpr_layer_audit.io.dzt import DZTFile
    from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask
    from gpr_layer_audit.processing.seed_graph import _normalise_waveform

    manifest = json.loads((ROOT / "benchmarks/seeded-evaluation-inputs.json").read_text())
    case = manifest["cases"][case_name]
    radar_path, seeds_path = ROOT / case["dzt"], ROOT / case["seed_source"]
    assert sha(radar_path) == case["dzt_sha256"]
    assert sha(seeds_path) == case["seed_sha256"]
    source = DZTFile(radar_path)
    data = source.channel(0)
    seeds = json.loads(seeds_path.read_text())["observations"]
    left, right, coordinates = [], [], []
    offsets = np.arange(length) - length // 2
    # Deterministic alternatives around the six operating deep observations;
    # no reviewed scoring answer, candidate truth, or interpolation is used.
    for order in ("2", "3"):
        for seed in seeds.get(order, []):
            row, sample = seed["trace"], seed["sample"]
            for distance in (-40, -10, -1, 1, 10, 40):
                target = row + distance * case["stride"]
                for shift in (-8, -4, 0, 4, 8):
                    target_sample = sample + shift
                    if not 0 <= target < len(data):
                        continue
                    a_indices, b_indices = sample + offsets, target_sample + offsets
                    if min(a_indices[0], b_indices[0]) < 0:
                        continue
                    if max(a_indices[-1], b_indices[-1]) >= data.shape[1]:
                        continue
                    validity = processed_boundary_mask(np.asarray(data[[row, target]], float))
                    if not validity[0, a_indices].all() or not validity[1, b_indices].all():
                        continue
                    a = _normalise_waveform(data[row, a_indices])
                    b = _normalise_waveform(data[target, b_indices])
                    if a is None or b is None:
                        continue
                    left.append(a)
                    right.append(b)
                    coordinates.append([int(order), row, sample, target, target_sample])
    return np.asarray(left), np.asarray(right), {
        "case": case_name, "dzt_sha256": sha(radar_path), "seeds_sha256": sha(seeds_path),
        "coordinate_fields": ["layer", "source_native_trace", "source_native_sample",
                              "target_native_trace", "target_native_sample"],
        "coordinates": coordinates, "validity": "all packet samples must be measured",
        "selection": "fixed offsets around operating seeds; no reference labels loaded",
        "normalization": "production _normalise_waveform (float32 centre and unit norm)",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "exports/seeded-tracker/dtw-acceleration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # A fresh cache directory ensures that the measured first call includes
    # compilation, independently of earlier pytest or production cache state.
    cache = tempfile.TemporaryDirectory(prefix="gpr-dtw-cold-")
    os.environ["NUMBA_CACHE_DIR"] = cache.name
    began = time.perf_counter()
    from gpr_layer_audit.processing.waveform_matching import batch_dtw
    from gpr_layer_audit.processing.waveform_matching_fast import (
        batch_dtw_fast,
        reciprocal_dtw_fast,
    )
    import_seconds = time.perf_counter() - began
    rng = np.random.default_rng(20260909)
    left = rng.normal(size=(32, 89)).astype(np.float32)
    right = rng.normal(size=(32, 89)).astype(np.float32)
    began = time.perf_counter()
    batch_dtw_fast(left, right, 7)
    cold_seconds = time.perf_counter() - began
    cases, packets, radar_provenance = [], {}, []
    for n, band, pairs in ((31, 2, 128), (32, 2, 128), (89, 7, 32), (89, 7, 128)):
        a, b = rng.normal(size=(2, pairs, n))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        cases.append((f"random-n{n}-b{band}-p{pairs}", a, b, band))
    for road in ("mandiali-short", "gujrat-second"):
        for n, band, pairs in ((31, 2, 128), (89, 7, 32)):
            a, b, provenance = real_packets(road, n)
            key = f"{road}-n{n}"
            packets[f"{key}-left"], packets[f"{key}-right"] = a, b
            radar_provenance.append({"packet_key": key, **provenance})
            # Compare every selected pair; time the production maximum batch.
            for reference, fast in ((batch_dtw, batch_dtw_fast),
                                    (reference_reciprocal, reciprocal_dtw_fast)):
                for expected, observed in zip(reference(a, b, band), fast(a, b, band), strict=True):
                    np.testing.assert_array_equal(expected, observed)
            cases.append((key, a[:pairs], b[:pairs], band))
    packet_path = args.output / "measured-packets.npz"
    np.savez_compressed(packet_path, **packets)
    results = []
    for name, a, b, band in cases:
        for mode, reference, fast in (("one-way", batch_dtw, batch_dtw_fast),
                                      ("reciprocal", reference_reciprocal, reciprocal_dtw_fast)):
            expected, observed = reference(a, b, band), fast(a, b, band)
            errors = [float(np.max(abs(x - y), initial=0))
                      for x, y in zip(expected, observed, strict=True)]
            exact = all(np.array_equal(x, y)
                        for x, y in zip(expected, observed, strict=True))
            assert exact
            baseline = timed(reference, a, b, band, repetitions=5)
            compiled = timed(fast, a, b, band, repetitions=100)
            pairs, n = a.shape
            result = {
                "case": name, "mode": mode, "pairs": pairs, "length": n, "band": band,
                "input_dtype": str(a.dtype), "bitwise_equal": exact,
                "maximum_absolute_agreement_error": errors[0],
                "maximum_absolute_shift_error": errors[1], "reference": baseline,
                "compiled": compiled,
                "warm_speedup": baseline["wall_median_seconds"] / compiled["wall_median_seconds"],
                "reference_cost_parent_workspace_bytes": 8 * pairs * (n + 1) ** 2 + pairs * n ** 2,
                "compiled_cost_parent_workspace_bytes": 16 * (n + 1) + n ** 2,
                "compiled_result_workspace_bytes": 16 * pairs,
                "float32_input_conversion_bytes": 16 * pairs * n if a.dtype == np.float32 else 0,
            }
            results.append(result)
            print(f"{name} {mode}: {result['warm_speedup']:.1f}x; exact={exact}", flush=True)
    dist = importlib.metadata.distribution("numba")
    licenses = {str(p): sha(dist.locate_file(p)) for p in dist.files if "LICENSE" in str(p)}
    for p in dist.files:
        if str(p).endswith("licenses/LICENSE"):
            (args.output / "NUMBA-LICENSE.txt").write_bytes(dist.locate_file(p).read_bytes())
    report = {
        "schema": "exact-dtw-acceleration-v1", "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "numba": dist.version,
        "numba_license": "BSD-2-Clause, installed distribution license inspected",
        "license_sha256": licenses, "numba_import_seconds": import_seconds,
        "first_call_fresh_cache_compile_and_execution_seconds": cold_seconds,
        "kernel_contract": "float64; fastmath=False; diagonal/up/left ties; NumPy exp outside JIT",
        "parallelism": "single thread; nogil permits cancellation checks between batches",
        "complexity": {
            "reference": "O(pairs*n*n) cost/parent workspace, plus diagonal NumPy temporaries",
            "compiled": "O(n*n + n + pairs) workspace; sequential pair processing",
            "caveat": "reported workspace bytes are exact named arrays, not process peak RSS",
        },
        "sources": {p: sha(ROOT / p) for p in (
            "src/gpr_layer_audit/processing/waveform_matching.py",
            "src/gpr_layer_audit/processing/waveform_matching_fast.py",
            "scripts/benchmark_waveform_matching_fast.py")},
        "measured_packets_sha256": sha(packet_path), "radar_provenance": radar_provenance,
        "results": results,
        "claim_limit": "kernel timing only; graph/application latency must be tested separately",
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    cache.cleanup()


if __name__ == "__main__":
    main()
