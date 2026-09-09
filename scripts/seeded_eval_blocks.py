"""Paired spatial-block uncertainty for completed fixed-cohort replay results.

This postscorer never runs inference. Block sizes are fixed before examining
block outcomes; the 25 m primary scale is the existing correction radius.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from seeded_eval_endpoints import frozen_context, require, sha, validate_log

BLOCK_LENGTHS_M = (25.0, 10.0, 50.0)
REPLICATES = 4000
RNG_SEED = 20260909


def rates(counts):
    n, correct0, wrong0, correct1, wrong1, supplied = np.asarray(counts).T
    values = []
    for correct, wrong, manual in ((correct0, wrong0, n * 0), (correct1, wrong1, supplied)):
        with np.errstate(divide="ignore", invalid="ignore"):
            values.append(
                np.stack(
                    (
                        correct / (correct + wrong),
                        correct / n,
                        wrong / n,
                        manual / n,
                        (n - correct - wrong - manual) / n,
                    ),
                    axis=-1,
                )
            )
    return values


def score_states(path, context):
    log = json.loads(path.read_text())
    anchors, initial = validate_log(log, context)
    require(log.get("stop_reason") == "Requested action budget completed", "Replay incomplete")
    require(
        log["seeding"] == "three" and log["freeze_initial_pulse"],
        "Expected frozen three-seed replay",
    )
    final = log["steps"][-1]
    require(final["requests"] == log["replay_contract"]["actions"], "Budget mismatch")
    arrays = [path.with_name(f"{path.stem}-step{s['step']}.npz") for s in (initial, final)]
    rows_by_layer = {}
    with np.load(arrays[0]) as first, np.load(arrays[1]) as last:
        for order in ("2", "3"):
            observations = initial["layers"][order]["evaluation_observations"]
            rows = np.array([o["row"] for o in observations], dtype=int)
            require(len(set(rows)) == len(rows), "Duplicate scoring row")
            require(not set(rows) & set(anchors[order]), "Initial seed scored")
            supplied = {
                int(a["row"])
                for a in log["actions"]
                if str(a["layer_order"]) == order and "answer_sample" in a
            }
            counts = np.zeros((len(rows), 6), dtype=int)
            counts[:, 0] = 1
            counts[:, 5] = np.isin(rows, list(supplied))
            tolerance = context["tolerances"][order]
            for index, (step, data) in enumerate(((initial, first), (final, last))):
                predicted = data[f"layer{order}_accepted"]
                visible = data[f"layer{order}_visible"]
                require(
                    predicted.shape == visible.shape == (len(context["radar"]),),
                    "Array grid mismatch",
                )
                logs = {o["row"]: o for o in step["layers"][order]["evaluation_observations"]}
                for offset, original in enumerate(observations):
                    row, reference = original["row"], original["reference_sample"]
                    require(context["references"][order][row] == reference, "Reference changed")
                    accepted = bool(predicted[row] >= 0 and visible[row])
                    correct = (
                        accepted
                        and abs(predicted[row] - reference) <= tolerance
                        and context["same_lobe"](
                            context["radar"][row], int(predicted[row]), reference
                        )
                    )
                    if index == 1 and row in supplied:
                        continue
                    require(row in logs, "Automatic cohort row disappeared")
                    require(bool(logs[row]["accepted"]) == accepted, "Accepted array/log mismatch")
                    require(
                        bool(logs[row]["accepted_correct"]) == correct, "Scoring array/log mismatch"
                    )
                    counts[offset, 1 + 2 * index] = correct
                    counts[offset, 2 + 2 * index] = accepted and not correct
            rows_by_layer[order] = (rows, counts)
    return log, rows_by_layer, [{"path": str(p), "sha256": sha(p)} for p in arrays]


def block_result(rows, counts, length_m, dx, extent_m):
    block_ids = np.floor(rows * dx / length_m).astype(int)
    number = int(np.floor(extent_m / length_m)) + 1
    blocks = np.zeros((number, 6), dtype=int)
    np.add.at(blocks, block_ids, counts)
    occupied = int(np.sum(blocks[:, 0] > 0))
    initial, final = rates(blocks.sum(axis=0))
    names = (
        "accepted_agreement",
        "correct_automatic_coverage",
        "incorrect_automatic_coverage",
        "analyst_supplied_coverage",
        "unresolved_coverage",
    )
    result = {
        "block_length_m": length_m,
        "primary": length_m == 25,
        "physical_blocks": number,
        "blocks_with_reviewed_observations": occupied,
        "blocks_with_initial_acceptance": int(np.sum(blocks[:, 1:3].sum(axis=1) > 0)),
        "blocks_with_final_acceptance": int(np.sum(blocks[:, 3:5].sum(axis=1) > 0)),
        "block_counts_columns": [
            "reviewed",
            "initial_correct",
            "initial_wrong",
            "final_correct",
            "final_wrong",
            "analyst_supplied",
        ],
        "blocks": [
            {
                "id": i,
                "start_m": i * length_m,
                "end_m": min((i + 1) * length_m, extent_m),
                "counts": b.tolist(),
            }
            for i, b in enumerate(blocks)
        ],
        "initial": dict(zip(names, initial.tolist(), strict=True)),
        "final": dict(zip(names, final.tolist(), strict=True)),
        "confidence_intervals": None,
    }
    if occupied < 5:
        result["uncertainty_status"] = (
            "Unavailable: fewer than five physical blocks contain reviewed observations"
        )
        return result
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, number, size=(REPLICATES, number))
    before, after = rates(blocks[draws].sum(axis=1))
    intervals = {}
    for state, samples in (
        ("initial", before),
        ("final", after),
        ("paired_final_minus_initial", after - before),
    ):
        intervals[state] = {}
        for i, name in enumerate(names):
            eligible = samples[:, i][np.isfinite(samples[:, i])]
            intervals[state][name] = {
                "percentile_95_interval": np.quantile(eligible, [0.025, 0.975]).tolist()
                if len(eligible)
                else None,
                "finite_replicates": len(eligible),
            }
    result["confidence_intervals"] = intervals
    result["uncertainty_status"] = (
        "Conditional within-acquisition spatial-block bootstrap; "
        "not independent-road generalization"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing uncertainty output")
    results, sources, contexts = [], [], {}
    for path in args.paths:
        case = json.loads(path.read_text())["case"]
        if case not in contexts:
            contexts[case] = frozen_context(case)
        context = contexts[case]
        log, data, arrays = score_states(path, context)
        sources.append(
            {
                "path": str(path),
                "sha256": sha(path),
                "arrays": arrays,
                "source_sha256": log["source_sha256"],
                "configuration_sha256": log["configuration_sha256"],
            }
        )
        for layer, (rows, counts) in data.items():
            results.append(
                {
                    "case": case,
                    "physical_road_group": log["physical_road_group"],
                    "policy": log["policy"],
                    "layer": int(layer),
                    "fixed_initial_observations": len(rows),
                    "total_counts": counts.sum(axis=0).tolist(),
                    "blocks": [
                        block_result(rows, counts, length, context["step"], log["road_length_m"])
                        for length in BLOCK_LENGTHS_M
                    ],
                }
            )
    document = {
        "schema": "seeded-paired-spatial-block-uncertainty-v1",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha(__file__),
        "claim": "Development inference with conditional within-road spatial resampling",
        "design": (
            "Block lengths 25 m primary, 10/50 m sensitivity fixed before block-outcome analysis. "
            "Primary uses existing correction-radius scale; no empirical independence is claimed. "
            "Seed 20260909, 4000 paired resamples, percentile 95% intervals."
        ),
        "cohort": (
            "Initial reviewed nonseed rows remain fixed; additional answers stay in denominator "
            "and receive no automatic credit. No unknown-label interpolation "
            "or scoring-rule change."
        ),
        "resampling": (
            "Nonoverlapping physical blocks from stored origin zero, including empty and shortened "
            "terminal blocks; identical block draw for before/after; ratios recomputed from counts."
        ),
        "limitations": [
            "Only two previously used replay roads; "
            "no reliable independent-road interval is possible",
            "Dependence may extend beyond block width; intervals are not calibration "
            "or proof of 95% engineering acceptance",
            "Mandiali 11.45 m has too few blocks; suppress an interval instead of false certainty",
            "Subbase acceptance is sparse across blocks; finite-replicate counts are reported",
            "Physical reflector-switch events and thickness truth "
            "are not independently established",
        ],
        "sources": sources,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
