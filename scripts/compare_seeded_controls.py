"""Verify identical saved numerical controls without rerunning inference or opening labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(before, after, method="seed_hybrid"):
    inputs = [Path(before), Path(after)]
    documents = [json.loads(p.read_text()) for p in inputs]
    a, b = documents
    fields = (
        "schema",
        "mode",
        "audit",
        "ml",
        "reference_surface_sample",
        "surface_origin_provenance",
        "partition",
        "stride",
        "reference_label_mapping",
        "label_mapping_status",
        "config",
        "transform",
        "seed_support",
        "seed_selection",
        "scoring_pulses",
        "scoring_tolerance_policy",
    )
    contracts = {key: key in a and key in b and a[key] == b[key] for key in fields}
    metrics_equal = a["methods"][method]["layers"] == b["methods"][method]["layers"]
    array_paths = [p.with_name(f"{p.stem}-{method}.npz") for p in inputs]
    arrays_equal = {}
    with np.load(array_paths[0]) as left, np.load(array_paths[1]) as right:
        names_equal = set(left.files) == set(right.files)
        for name in sorted(set(left.files) | set(right.files)):
            if name not in left or name not in right:
                arrays_equal[name] = False
                continue
            x, y = left[name], right[name]
            arrays_equal[name] = (
                x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
            )
    return {
        "schema": "seeded-control-parity-v1",
        "success": all(contracts.values())
        and metrics_equal
        and names_equal
        and all(arrays_equal.values()),
        "claim": "Exact saved control arrays and all per-layer numerical scoring fields",
        "method": method,
        "inputs": [
            {
                "evaluation": str(p),
                "evaluation_sha256": sha(p),
                "arrays": str(q),
                "arrays_sha256": sha(q),
                "backend_source_sha256": d["backend_source_sha256"],
                "runtime_s": d["methods"][method]["runtime_s"],
                "process_peak_memory_bytes": d["methods"][method]["process_peak_memory_bytes"],
            }
            for p, q, d in zip(inputs, array_paths, documents, strict=True)
        ],
        "input_and_scoring_contracts_equal": contracts,
        "all_layer_metrics_equal": metrics_equal,
        "array_names_equal": names_equal,
        "arrays_bitwise_equal": arrays_equal,
        "not_required_equal": [
            "Backend source hash: the compiled implementation is deliberately different",
            "Runtime and memory: instrumented and concurrent runs are not a latency benchmark",
            "Run-specific wrapper/source provenance outside numerical scoring fields",
        ],
        "limitations": "Numerical parity is not evidence of correct reflector identity",
        "comparison_script_sha256": sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--method", default="seed_hybrid")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve an existing comparison; choose a fresh output")
    result = compare(args.before, args.after, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"success": result["success"], "artifact": str(args.output)}))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
