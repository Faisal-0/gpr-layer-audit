"""Time one three-layer joint fit and compare it with a frozen pre-change path.

The disclosed reference file is never opened. Frozen manual clicks and the
pre-change path arrays are read only for exact radar-solver regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from evaluate_blinded_multiroad import (
    _chainage,
    _detect_anomalies,
    _effective_stack,
    _path_arrays,
    _run_tracker,
)

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.preprocessing import (
    measurement_packet_support,
    preprocess_for_interpretation,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmarks/blinded-base-subbase-seeds-20260829.json"
DEFAULT_CASE = "sohal-kalan-gujrat-second-portion-001-validation"
DEFAULT_FROZEN = ROOT / "exports/blinded-base-subbase-strict-scale-20260829"


def _sha256_arrays(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, values in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(values)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--frozen-directory", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = args.manifest.resolve()
    document = json.loads(manifest.read_text(encoding="utf-8"))
    case = next(item for item in document["cases"] if item["case_id"] == args.case)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    road_path = (ROOT / case["road"]).resolve()
    plate_path = (ROOT / case["plate"]).resolve()
    road, plate = DZTFile(road_path), DZTFile(plate_path)
    stack = _effective_stack(
        road.header.trace_count, 0, road.header.distance_per_trace_m
    )
    calibrated = calibrate(road, plate, stack_size=stack)
    dzx_path = road_path.with_suffix(".DZX")
    dzx = read_dzx(dzx_path) if dzx_path.exists() else None
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    station_rows = [
        int(np.argmin(np.abs(chainage - float(station["chainage_m"]))))
        for station in case["stations"]
    ]
    orders = sorted(int(value) for value in case["stations"][0]["samples"])
    anchors = {
        order: {
            row: int(station["samples"][str(order)])
            for row, station in zip(station_rows, case["stations"], strict=True)
        }
        for order in orders
    }
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
    )
    branches = {
        **interpreted.feature_branches,
        "measurement_support": measurement_packet_support(
            calibrated.measurement_radargram
        ),
    }
    anomaly_mask, _ = _detect_anomalies(
        chainage, branches.get("anomaly_score")
    )

    started = time.perf_counter()
    result = _run_tracker(
        interpreted.radargram,
        calibrated.reference_surface_sample,
        LayerSpec.defaults()[: max(orders)],
        branches,
        anomaly_mask,
        chainage,
        anchors,
    )
    elapsed = time.perf_counter() - started
    current = _path_arrays(result, orders)
    current_arrays = {
        f"layer_{order}_{name}": values
        for order, fields in current.items()
        for name, values in fields.items()
    }

    frozen_path = (
        args.frozen_directory.resolve() / f"{case['case_id']}-paths.npz"
    )
    with np.load(frozen_path) as stored:
        frozen_chainage = stored["chainage_m"]
        if not np.array_equal(chainage, frozen_chainage):
            raise ValueError("Current chainage does not match the frozen run")
        frozen_arrays = {
            f"layer_{order}_{name}": stored[f"layer_{order}_{name}"]
            for order in orders
            for name in ("samples", "alternate", "graph", "canonical")
        }

    comparisons = {}
    exact = True
    for name, values in current_arrays.items():
        reference = frozen_arrays[name]
        matches = np.array_equal(values, reference, equal_nan=True)
        comparisons[name] = {
            "exact": bool(matches),
            "exact_percent": 100.0 * float(np.mean(values == reference)),
        }
        exact &= bool(matches)

    summary = {
        "purpose": "development-only exact three-layer joint-runtime regression",
        "field_accuracy_established": False,
        "reference_opened": False,
        "case_id": case["case_id"],
        "history_budget": 128,
        "elapsed_seconds": elapsed,
        "exact_frozen_path_parity": exact,
        "current_path_sha256": _sha256_arrays(current_arrays),
        "frozen_path_sha256": _sha256_arrays(frozen_arrays),
        "comparisons": comparisons,
        "input_fingerprints": {
            "road": fingerprint_file(road_path),
            "plate": fingerprint_file(plate_path),
            "seed_manifest": fingerprint_file(manifest),
            "frozen_paths": fingerprint_file(frozen_path),
            "benchmark_script": fingerprint_file(Path(__file__)),
            "joint_graph": fingerprint_file(
                ROOT / "src/gpr_layer_audit/processing/joint_graph.py"
            ),
            "seed_graph": fingerprint_file(
                ROOT / "src/gpr_layer_audit/processing/seed_graph.py"
            ),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
