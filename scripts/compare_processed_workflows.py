"""Compare complete application replays without opening any scoring reference.

An explicitly declared time-origin shift can account for a separately verified
coordinate repair. Every other pick field, path sample and query must agree.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def compare(before, after, shift):
    old, new = read(before / "verification.json"), read(after / "verification.json")
    if not old.get("success") or not new.get("success"):
        raise ValueError("Both application workflows must have completed successfully")
    for name in (
        "case",
        "input_sha256",
        "reference_sha256",
        "initial_seed_sha256",
        "config_sha256",
        "initial_pulse_samples",
        "query_layer_orders",
    ):
        if old[name] != new[name]:
            raise ValueError(f"Controlled workflow input changed: {name}")
    if old["actions"] != new["actions"]:
        raise ValueError("Requested coordinates, answers or correction outcomes changed")
    comparisons = []
    for path in sorted(before.glob("*-paths.npz")):
        current = after / path.name
        with np.load(path) as a, np.load(current) as b:
            if set(a.files) != set(b.files) or any(
                not np.array_equal(a[key], b[key]) for key in a.files
            ):
                raise ValueError(f"Application path arrays changed: {path.name}")
        comparisons.append(
            {
                "file": path.name,
                "before_sha256": sha(path),
                "after_sha256": sha(current),
                "arrays_bitwise_equal": True,
            }
        )
    if not comparisons:
        raise ValueError("No path arrays available for comparison")
    signatures = []
    for path in sorted(before.glob("*-pick-signature.json")):
        current = after / path.name
        a, b = read(path), read(current)
        if len(a) != len(b):
            raise ValueError(f"Pick count changed: {path.name}")
        finite = 0
        for left, right in zip(a, b, strict=True):
            if left[:6] != right[:6] or left[7:] != right[7:]:
                raise ValueError(f"Pick coordinates, status or identity changed: {path.name}")
            if math.isfinite(left[6]) and math.isfinite(right[6]):
                finite += 1
                if not math.isclose(right[6] - left[6], shift, abs_tol=1e-12):
                    raise ValueError(f"Unexpected accepted TWTT change: {path.name}")
            elif not (math.isnan(left[6]) and math.isnan(right[6])):
                raise ValueError(f"Unresolved timing changed: {path.name}")
        signatures.append(
            {
                "file": path.name,
                "picks": len(a),
                "finite_twtt": finite,
                "before_sha256": sha(path),
                "after_sha256": sha(current),
            }
        )
    if len(signatures) != len(comparisons):
        raise ValueError("Every path stage needs its complete pick signature")
    return {
        "schema": "processed-workflow-parity-v1",
        "success": True,
        "before": str(before),
        "after": str(after),
        "before_verification_sha256": sha(before / "verification.json"),
        "after_verification_sha256": sha(after / "verification.json"),
        "comparison_script_sha256": sha(Path(__file__)),
        "declared_twtt_shift_ns": shift,
        "claim": "Exact application path, pick identity/status, query and answer parity; "
        "only the explicitly declared timing-origin change is permitted",
        "paths": comparisons,
        "pick_signatures": signatures,
        "actions": new["actions"],
        "before_runs": old["runs"],
        "after_runs": new["runs"],
        "runtime_limit": "Separate executions under concurrent workloads; historical "
        "wall times can include recorded suspension. These timings are "
        "not a controlled estimate of acceleration alone.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--twtt-shift-ns", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing evidence; choose a new output")
    report = compare(args.before, args.after, args.twtt_shift_ns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"success": True, "stages": len(report["paths"])}))


if __name__ == "__main__":
    main()
