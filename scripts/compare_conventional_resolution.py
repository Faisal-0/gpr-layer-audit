"""Score matched native observations after changing only the working trace stride.

Both runs must use the same source, exact native seeds, config and backend. This
is an accuracy comparison; process timings remain separately reported by each run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gpr_layer_audit.conventional import _same_lobe, write_json
from gpr_layer_audit.conventional_seeds import load_support
from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file


def compare(left_path, right_path, method="seed_hybrid"):
    paths = [Path(left_path), Path(right_path)]
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    left, right = runs
    if any(r["mode"] != "processed" or r.get("transform") is not None for r in runs):
        raise ValueError("Resolution comparison requires processed coordinates")
    for key in (
        "config",
        "backend_source_sha256",
        "reference_label_mapping",
        "reference_surface_sample",
    ):
        if left[key] != right[key]:
            raise ValueError(f"Resolution comparison differs in {key}")
    seeds = [
        load_support(p, left["audit"], left["reference_label_mapping"], r["stride"], "processed")[0]
        for p, r in zip(paths, runs, strict=True)
    ]
    if seeds[0] != seeds[1]:
        raise ValueError("Resolution comparison must use identical native seeds")
    source = Path(left["audit"]["dzt_path"])
    if fingerprint_file(source) != left["audit"]["dzt_sha256"]:
        raise ValueError("Processed radar fingerprint differs")
    radar = DZTFile(source).channel()
    arrays = [
        np.load(p.with_name(p.stem + "-" + method + ".npz"), allow_pickle=False) for p in paths
    ]
    try:
        result = {
            "schema": "conventional-resolution-comparison-v1",
            "runs": [
                {"artifact": str(p.resolve()), "sha256": fingerprint_file(p), "stride": r["stride"]}
                for p, r in zip(paths, runs, strict=True)
            ],
            "method": method,
            "ml": "disabled",
            "promotion": "none",
            "policy": "common annotated native rows, identical native support excluded",
            "backend_source_sha256": left["backend_source_sha256"],
            "layers": {},
        }
        for order in left["methods"][method]["layers"]:
            layers = [r["methods"][method]["layers"][order] for r in runs]
            if layers[0]["tolerance_samples"] != layers[1]["tolerance_samples"]:
                raise ValueError("Resolution comparison changed the scoring pulse")
            references = [
                {
                    p["row"] * r["stride"]: p["reference_sample"]
                    for p in layer["evaluation_observations"]
                }
                for r, layer in zip(runs, layers, strict=True)
            ]
            accepted_rows = [
                {p["row"] * r["stride"] for p in layer["evaluation_observations"] if p["accepted"]}
                for r, layer in zip(runs, layers, strict=True)
            ]
            common = sorted(set(references[0]) & set(references[1]) - set(seeds[0][int(order)]))
            if any(references[0][r] != references[1][r] for r in common):
                raise ValueError("Matched reference sample coordinates differ")
            values = []
            tolerance = layers[0]["tolerance_samples"]
            for run, data, visible in zip(runs, arrays, accepted_rows, strict=True):
                accepted, correct, proposed_correct, switches = 0, 0, 0, 0
                for native_row in common:
                    row, ref = native_row // run["stride"], references[0][native_row]
                    pick = int(data[f"layer{order}_accepted"][row])
                    proposal = int(data[f"layer{order}_provisional"][row])
                    if pick >= 0 and native_row in visible:
                        same_lobe = _same_lobe(radar[native_row], pick, ref)
                        accepted += 1
                        correct += int(abs(pick - ref) <= tolerance and same_lobe)
                        switches += int(not same_lobe)
                    proposed_correct += int(
                        proposal >= 0
                        and abs(proposal - ref) <= tolerance
                        and _same_lobe(radar[native_row], proposal, ref)
                    )
                values.append(
                    {
                        "accepted": accepted,
                        "accepted_agree": correct,
                        "accepted_pick_agreement": correct / accepted if accepted else None,
                        "correct_coverage": correct / len(common) if common else None,
                        "proposed_path_agreement": proposed_correct / len(common)
                        if common
                        else None,
                        "reflector_switches": switches,
                    }
                )
            result["layers"][order] = {
                "common_observations_excluding_seeds": len(common),
                "native_seed_rows": sorted(seeds[0][int(order)]),
                "tolerance_samples": tolerance,
                "runs": values,
            }
        return result
    finally:
        for data in arrays:
            data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="seed_hybrid")
    args = parser.parse_args()
    write_json(args.output, compare(args.left, args.right, args.method))
