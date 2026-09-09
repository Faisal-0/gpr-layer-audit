"""Evaluation-only paired physical-block bootstrap of completed frozen replays.

Run ``python scripts/seeded_block_uncertainty.py --self-test`` for accounting tests,
then run without arguments to validate saved inputs and write the compact report.
No inference, threshold fitting, production mutation, or reference interpolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from seeded_eval import source_manifest, validate_frozen_helpers
from seeded_eval_endpoints import (
    BASELINE_SOURCE,
    MANIFEST,
    ROOT,
    frozen_context,
    read,
    require,
    sha,
    source_sha,
    validate_log,
)

INPUT_DIRECTORY = ROOT / "exports/seeded-tracker/interaction-prospective"
OUTPUT = ROOT / "exports/seeded-tracker/block-uncertainty/paired-block-report.json"
BLOCK_LENGTHS_M = (25.0, 10.0, 50.0)
REPLICATES = 10000
RNG_SEED = 20260909
MIN_REVIEWED_BLOCKS = 5
COUNT_COLUMNS = (
    "reviewed_initial_nonseed",
    "correct_automatic",
    "incorrect_automatic",
    "unresolved",
    "analyst_supplied",
)
METRIC_NAMES = (
    *COUNT_COLUMNS,
    *(name + "_footprint_m" for name in COUNT_COLUMNS),
    "accepted_precision",
)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def row_map(observations):
    result = {r["row"]: r for r in observations}
    require(len(result) == len(observations), "Duplicate reviewed observation")
    return result


def metric_array(counts, spacing_m):
    counts = np.asarray(counts)
    require(counts.shape[-1] == 5, "Invalid count columns")
    require(np.all(counts >= 0), "Negative category count")
    require(np.all(counts[..., 0] == counts[..., 1:].sum(axis=-1)), "Cohort lost or doubled")
    accepted = counts[..., 1] + counts[..., 2]
    precision = np.full(accepted.shape, np.nan, dtype=float)
    np.divide(counts[..., 1], accepted, out=precision, where=accepted > 0)
    return np.concatenate((counts, counts * spacing_m, precision[..., None]), axis=-1)


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def metric_dict(values):
    return dict(zip(METRIC_NAMES, (finite_number(v) for v in values), strict=True))


def physical_blocks(rows, paired_counts, spacing_m, extent_m, length_m):
    """Aggregate entire physical blocks BEFORE any random sampling.

    All original reviewed rows remain present. A shortened last block includes
    the final stored coordinate; an exact terminal boundary creates no empty
    zero-width block. Unknown coordinates contribute no counts or footprint.
    """
    rows, paired_counts = np.asarray(rows), np.asarray(paired_counts)
    require(spacing_m > 0 and extent_m > 0 and length_m > 0, "Invalid physical spacing")
    require(paired_counts.shape == (len(rows), 2, 5), "Invalid paired count shape")
    require(len(set(rows.tolist())) == len(rows), "Duplicate working row")
    require(np.all(rows == rows.astype(int)), "Noninteger working row")
    require(np.all((rows >= 0) & (rows * spacing_m <= extent_m + 1e-9)), "Row outside road")
    require(np.all(paired_counts[:, :, 0] == 1), "Each initial reviewed row must count once")
    metric_array(paired_counts, spacing_m)
    number = max(1, math.ceil(extent_m / length_m))
    block_ids = np.minimum(np.floor(rows * spacing_m / length_m).astype(int), number - 1)
    blocks = np.zeros((number, 2, 5), dtype=np.int64)
    np.add.at(blocks, block_ids, paired_counts)
    require(np.array_equal(blocks.sum(axis=0), paired_counts.sum(axis=0)), "Block count loss")
    return blocks


def paired_draws(blocks, draws):
    """One physical-block draw indexes BOTH states and every row within a block."""
    require(blocks.ndim == 3 and blocks.shape[1:] == (2, 5), "Invalid physical-block tensor")
    require(np.all(blocks[:, 0, 0] == blocks[:, 1, 0]), "Paired denominator changed")
    require(draws.ndim == 2 and draws.shape[1] == len(blocks), "Draw whole road block count")
    require(np.all((draws >= 0) & (draws < len(blocks))), "Invalid block draw")
    totals = blocks[draws].sum(axis=1)
    require(np.array_equal(totals[:, 0, 0], totals[:, 1, 0]), "Resampled denominator changed")
    return totals


def intervals(values):
    output = {}
    for index, name in enumerate(METRIC_NAMES):
        finite = values[:, index][np.isfinite(values[:, index])]
        output[name] = np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None
    return output


def bootstrap(blocks, spacing_m, draws):
    totals = paired_draws(blocks, draws)
    values = metric_array(totals, spacing_m)
    return {
        "percentile_95": {
            "before": intervals(values[:, 0]),
            "after_four_requests": intervals(values[:, 1]),
            "paired_after_minus_before": intervals(values[:, 1] - values[:, 0]),
        },
        "accepted_precision_replicates": {
            "requested": len(draws),
            "before_finite": int(np.isfinite(values[:, 0, -1]).sum()),
            "after_finite": int(np.isfinite(values[:, 1, -1]).sum()),
            "paired_delta_finite": int(np.isfinite(values[:, 1, -1] - values[:, 0, -1]).sum()),
            "before_zero_accepted": int((totals[:, 0, 1:3].sum(axis=1) == 0).sum()),
            "after_zero_accepted": int((totals[:, 1, 1:3].sum(axis=1) == 0).sum()),
            "rule": "Precision is undefined at zero acceptance; no zero/one imputation. Delta "
            "uses only draws with defined precision in BOTH states. Intervals with omitted "
            "draws are conditional on nonzero acceptance and require caution.",
        },
    }


def score_replay(path, comparison, context):
    """Recheck the existing frozen scoring rule against persisted step 0/4 arrays."""
    log = read(path)
    anchors, initial = validate_log(log, context)
    require(log["freeze_initial_pulse"] and log["seeding"] == "three", "Expected frozen replay")
    require(log["scope"] == "local", "Expected local replay")
    require(log["replay_contract"]["radius_m"] == 25.0, "Existing correction radius changed")
    require(log["replay_contract"]["actions"] == 4, "Expected four-request budget")
    require(log["stop_reason"] == "Requested action budget completed", "Incomplete replay")
    require(len(log["actions"]) == 4, "Observed request budget incomplete")
    require([s["step"] for s in log["steps"]] == list(range(5)), "Replay steps incomplete")
    require([s["requests"] for s in log["steps"]] == list(range(5)), "Replay requests incomplete")
    final = log["steps"][-1]
    require(source_sha(log["source_snapshot"]) == log["source_sha256"], "Replay source changed")
    provenance = [p for p in comparison["provenance"] if Path(p["artifact"]).name == path.name]
    require(len(provenance) == 1, "Comparison provenance missing or duplicated")
    require(provenance[0]["artifact_sha256"] == sha(path), "Comparison replay hash mismatch")
    require(comparison["manifest_sha256"] == sha(MANIFEST), "Comparison manifest changed")
    arrays_paths = [path.with_name(f"{path.stem}-step{s['step']}.npz") for s in (initial, final)]
    pairs = {}
    control_layers = read(context["control_path"])["methods"]["seed_hybrid"]["layers"]
    with (
        np.load(arrays_paths[0], allow_pickle=False) as before,
        np.load(arrays_paths[1], allow_pickle=False) as after,
    ):
        for order in ("2", "3"):
            original = row_map(initial["layers"][order]["evaluation_observations"])
            control_records = row_map(control_layers[order]["evaluation_observations"])
            require(
                {
                    row: (r["reference_sample"], r["accepted"], r["accepted_correct"])
                    for row, r in original.items()
                }
                == {
                    row: (r["reference_sample"], r["accepted"], r["accepted_correct"])
                    for row, r in control_records.items()
                },
                "Replay baseline row classifications differ from pinned frozen control",
            )
            cohort = {
                row: sample
                for row, sample in context["references"][order].items()
                if row not in anchors[order]
            }
            require(
                {row: r["reference_sample"] for row, r in original.items()} == cohort,
                "Initial cohort differs from frozen reviewed nonseed cohort",
            )
            rows = np.array(sorted(original), dtype=int)
            counts = np.zeros((len(rows), 2, 5), dtype=np.int64)
            counts[:, :, 0] = 1
            for state, (step, arrays) in enumerate(((initial, before), (final, after))):
                layer = step["layers"][order]
                current = row_map(layer["evaluation_observations"])
                require(set(current).issubset(original), "Unknown row added to scoring")
                answered = {
                    a["row"]
                    for a in log["actions"][: step["requests"]]
                    if str(a["layer_order"]) == order and a.get("answer_sample") is not None
                }
                supplied = set(original) - set(current)
                require(supplied == answered & set(original), "Revealed-answer cohort mismatch")
                tolerance = context["tolerances"][order]
                require(layer["tolerance_samples"] == tolerance, "Frozen tolerance changed")
                samples, visible = (
                    arrays[f"layer{order}_accepted"],
                    arrays[f"layer{order}_visible"],
                )
                require(
                    samples.shape == visible.shape == (len(context["radar"]),),
                    "Saved prediction grid mismatch",
                )
                for row, sample in anchors[order].items():
                    require(samples[row] == sample and visible[row], "Initial seed changed")
                for offset, row in enumerate(rows):
                    if row in supplied:
                        counts[offset, state, 4] = 1
                        continue
                    record = current[row]
                    require(record["reference_sample"] == cohort[row], "Reference sample changed")
                    accepted = bool(samples[row] >= 0 and visible[row])
                    correct = bool(
                        accepted
                        and abs(float(samples[row]) - cohort[row]) <= tolerance
                        and context["same_lobe"](context["radar"][row], samples[row], cohort[row])
                    )
                    require(record["accepted"] == accepted, "Saved accepted metric mismatch")
                    require(record["accepted_correct"] == correct, "Saved correct metric mismatch")
                    counts[offset, state, 1 if correct else 2 if accepted else 3] = 1
                totals = counts[:, state].sum(axis=0)
                summary = [
                    r
                    for r in comparison["results"]
                    if Path(r["artifact"]).name == path.name
                    and r["layer"] == int(order)
                    and r["segment"] == "whole"
                    and r["step"] == step["step"]
                ]
                require(len(summary) == 1, "Missing four-action whole-cohort summary")
                summary = summary[0]
                expected = (
                    summary["fixed_initial_observation_denominator"],
                    summary["accepted_agree"],
                    summary["accepted_wrong"],
                    summary["fixed_initial_observation_denominator"]
                    - summary["accepted"]
                    - summary["revealed_observations_excluded_from_automatic_scoring"],
                    summary["revealed_observations_excluded_from_automatic_scoring"],
                )
                require(np.array_equal(totals, expected), "Comparison count accounting differs")
                require(
                    np.isclose(
                        totals[1] * context["step"], summary["correct_automatic_footprint_m"]
                    )
                    and np.isclose(
                        totals[2] * context["step"], summary["incorrect_automatic_footprint_m"]
                    ),
                    "Comparison footprint differs",
                )
            metric_array(counts, context["step"])
            pairs[order] = (rows, counts)
    return log, pairs, [artifact(p) for p in arrays_paths]


def run_self_tests():
    """Small adversarial tests, independent of any road outcome."""
    names = []
    rows = np.array([0, 1, 2, 30, 31])
    pair = np.zeros((5, 2, 5), dtype=int)
    pair[:, :, 0] = 1
    pair[:3, 0, 1] = 1
    pair[3:, 0, 2] = 1
    pair[:3, 1, 3] = 1
    pair[3:, 1, 1] = 1
    blocks = physical_blocks(rows, pair, 1, 49, 25)
    np.testing.assert_array_equal(blocks[:, 0, 0], [3, 2])
    draws = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    totals = paired_draws(blocks, draws)
    np.testing.assert_array_equal(totals[:, 0, 0], [6, 4, 5, 5])
    np.testing.assert_array_equal(totals[:, 0, 0], totals[:, 1, 0])
    np.testing.assert_array_equal(totals[:, 0, 1], totals[:, 1, 3])
    np.testing.assert_array_equal(totals[:, 0, 2], totals[:, 1, 1])
    names.append("paired draw preserves entire correlated blocks and common denominator")
    result = bootstrap(blocks, 1, draws)
    require(result["accepted_precision_replicates"]["after_zero_accepted"] == 1, "Undefined lost")
    require(
        result["accepted_precision_replicates"]["paired_delta_finite"] == 3, "Pairwise mask lost"
    )
    names.append("zero acceptance remains undefined and precision delta uses joint finite mask")
    empty_acceptance = blocks.copy()
    empty_acceptance[:, :, 1:] = 0
    empty_acceptance[:, :, 3] = empty_acceptance[:, :, 0]
    undefined = bootstrap(empty_acceptance, 1, draws)
    require(
        undefined["percentile_95"]["before"]["accepted_precision"] is None, "Fabricated precision"
    )
    require(
        undefined["accepted_precision_replicates"]["paired_delta_finite"] == 0, "Fabricated delta"
    )
    json.dumps(undefined, allow_nan=False)
    names.append("all-zero accepted replicates serialize as null precision intervals")
    # Revealing a correct row gives analyst credit only; unknown coordinates stay absent.
    revealed = pair.copy()
    revealed[0, 1, :] = [1, 0, 0, 0, 1]
    revealed_blocks = physical_blocks(rows, revealed, 1, 75, 25)
    require(int(revealed_blocks[:, 1, 4].sum()) == 1, "Analyst supply lost")
    require(int(revealed_blocks[:, 1, 1].sum()) == 2, "Revealed row got automatic credit")
    require(int(revealed_blocks[:, 1, 0].sum()) == 5, "Unknown coordinates filled")
    require(len(revealed_blocks) == 3 and revealed_blocks[2].sum() == 0, "Empty block lost")
    names.append("revealed row retains denominator without automatic credit or unknown filling")
    boundary = physical_blocks(np.array([0, 25]), pair[:2], 1, 25, 25)
    require(len(boundary) == 1 and boundary[0, 0, 0] == 2, "Terminal boundary mishandled")
    short = physical_blocks(np.array([0, 1]), pair[:2], 1, 11.45, 25)
    require(len(short) == 1, "Short road invented multiple blocks")
    names.append("exact terminal boundary and short road do not invent extra blocks")
    malformed = blocks.copy()
    malformed[0, 1, 0] -= 1
    try:
        paired_draws(malformed, draws)
    except ValueError:
        pass
    else:
        raise AssertionError("Unpaired denominator accepted")
    names.append("changed paired denominator fails closed")
    return {"passed": len(names), "tests": names}


def build_report():
    results, cases, sources = [], {}, []
    helper_hashes = validate_frozen_helpers(BASELINE_SOURCE)
    for short, case_name in (("mandiali", "mandiali-short"), ("gujrat", "gujrat-second")):
        context = frozen_context(case_name)
        comparison_path = INPUT_DIRECTORY / f"{short}-four-action-comparison.json"
        comparison = read(comparison_path)
        sources.extend([artifact(comparison_path), artifact(context["control_path"])])
        sources.extend(artifact(ROOT / context["case"][k]) for k in ("dzt", "dzx", "seed_source"))
        extent = (len(context["radar"]) - 1) * context["step"]
        common = None
        draws_by_length = {}
        cases[case_name] = {
            "physical_road_group": context["case"]["physical_road_group"],
            "stored_road_span_m": extent,
            "working_spacing_m": context["step"],
            "block_designs": [],
        }
        for length in BLOCK_LENGTHS_M:
            number = max(1, math.ceil(extent / length))
            draws = np.random.default_rng(RNG_SEED).integers(0, number, (REPLICATES, number))
            draws_by_length[length] = draws
            cases[case_name]["block_designs"].append(
                {
                    "block_length_m": length,
                    "primary": length == 25.0,
                    "physical_blocks": number,
                    "block_bounds_m": [
                        [i * length, min((i + 1) * length, extent)] for i in range(number)
                    ],
                    "block_draws_sha256_int64_little_endian": hashlib.sha256(
                        draws.astype("<i8").tobytes(order="C")
                    ).hexdigest(),
                }
            )
        for policy in ("active", "midpoint"):
            path = INPUT_DIRECTORY / f"{short}-fixed-{policy}.json"
            log, pairs, arrays = score_replay(path, comparison, context)
            require(log["policy"] == policy, "Replay policy mismatch")
            if common is None:
                common = (log, pairs)
            else:
                for key in (
                    "initial_anchors",
                    "initial_pulse_samples",
                    "source_sha256",
                    "configuration_sha256",
                ):
                    require(log[key] == common[0][key], f"Policy comparator differs: {key}")
                for order, (rows, counts) in pairs.items():
                    require(
                        np.array_equal(rows, common[1][order][0]), "Policy initial cohort differs"
                    )
                    require(
                        np.array_equal(counts[:, 0], common[1][order][1][:, 0]),
                        "Policy baseline differs",
                    )
            sources.append(
                {
                    **artifact(path),
                    "arrays": arrays,
                    "replay_source_snapshot": log["source_snapshot"],
                    "replay_source_sha256": log["source_sha256"],
                    "replay_backend_sha256": log["backend_sha256"],
                    "replay_script_sha256": log["script_sha256"],
                    "configuration_sha256": log["configuration_sha256"],
                }
            )
            for order, (rows, counts) in pairs.items():
                point = metric_array(counts.sum(axis=0), context["step"])
                item = {
                    "case": case_name,
                    "policy": policy,
                    "layer": int(order),
                    "artifact": path.relative_to(ROOT).as_posix(),
                    "common_initial_cohort_sha256": canonical_sha(
                        [[int(row), context["references"][order][row]] for row in rows]
                    ),
                    "before": metric_dict(point[0]),
                    "after_four_requests": metric_dict(point[1]),
                    "paired_after_minus_before": metric_dict(point[1] - point[0]),
                    "blocks": [],
                }
                for length in BLOCK_LENGTHS_M:
                    blocks = physical_blocks(rows, counts, context["step"], extent, length)
                    occupied = int(np.count_nonzero(blocks[:, 0, 0]))
                    block_result = {
                        "block_length_m": length,
                        "physical_blocks": len(blocks),
                        "blocks_with_reviewed_observations": occupied,
                        "before_blocks_with_acceptance": int(
                            np.count_nonzero(blocks[:, 0, 1:3].sum(axis=1))
                        ),
                        "after_blocks_with_acceptance": int(
                            np.count_nonzero(blocks[:, 1, 1:3].sum(axis=1))
                        ),
                        "counts_by_block_before_after": blocks.tolist(),
                        "uncertainty_status": "insufficient_blocks"
                        if occupied < MIN_REVIEWED_BLOCKS
                        else "descriptive_within_road_only",
                        "bootstrap": None,
                    }
                    if occupied >= MIN_REVIEWED_BLOCKS:
                        block_result["bootstrap"] = bootstrap(
                            blocks, context["step"], draws_by_length[length]
                        )
                    item["blocks"].append(block_result)
                results.append(item)
    return {
        "schema": "seeded-paired-physical-block-uncertainty-v2",
        "claim": "Descriptive within-road uncertainty on historically used development roads",
        "baseline": "Saved fixed-pulse replay step 0; same initial native support and common "
        "reviewed nonseed cohort as the pinned original frozen control. Before/after compare "
        "the completed active or midpoint policy with its own identical step-zero state.",
        "design": {
            "primary_block_length_m": 25.0,
            "sensitivity_block_lengths_m": [10.0, 50.0],
            "choice": "25 m fixed from the existing local correction radius. 10/50 m sensitivity "
            "scales inherit the existing seeded_eval_blocks.py design. No scale selected "
            "from interval width or outcomes; all scales retained. This is a post hoc descriptive "
            "analysis of completed historically used-road replays, not preregistered validation.",
            "replicates": REPLICATES,
            "seed": RNG_SEED,
            "generator": "NumPy PCG64",
            "interval": "2.5th and 97.5th percentile, NumPy linear quantile",
            "minimum_blocks_with_reviewed_observations_for_interval": MIN_REVIEWED_BLOCKS,
            "pairing": "Aggregate every reviewed row in each nonoverlapping physical block; "
            "draw K whole blocks with replacement from K physical blocks. Identical block draws "
            "apply to before/after, policies, and layers on a road. No IID row resampling.",
            "denominator": "Exact original reviewed nonseed cohort retained in both states. "
            "Bootstrap duplicates complete blocks so replicate cohort size can vary, but is "
            "identical before/after within each draw. Unknown labels never filled or interpolated.",
            "categories": "Correct automatic + incorrect automatic + unresolved + analyst "
            "supplied = reviewed initial nonseed denominator. Revealed answers have no automatic "
            "credit; unavailable answers never enter the scored cohort.",
            "footprint": "Category counts times working row spacing; labelled observation "
            "footprint only, not continuous classified road length. Ratios recomputed from totals.",
            "count_columns": COUNT_COLUMNS,
            "evaluated_layers": [2, 3],
        },
        "limitations": [
            "All available roads were historically used; the user has no unused annotations. "
            "These intervals provide no independent-road validation or new-road guarantee.",
            "Mandiali spans 11.45 m: one 25 m block, so no interval is reported. Even the "
            "10 m sensitivity has too few blocks for the fixed five-reviewed-block guard.",
            "Gujrat spans 287.4 m: only 12 primary physical blocks. Sparse layer acceptance "
            "and missing reviews further reduce effective information.",
            "The 25 m scale matches correction radius, not a demonstrated independence length; "
            "the full correction window can span 50 m. Dependence can cross block boundaries.",
            "10/50 m sensitivity reflects dependence-scale uncertainty, not tuning. No block "
            "length, threshold, scorer, input, seed, model, or operating gate was optimized here.",
            "The last block is shorter, and review coverage is uneven; counts and footprint "
            "intervals describe resampled blocks, not fixed-length new-road totals.",
            "Precision intervals dropping zero-acceptance draws are conditional; a degenerate "
            "interval or zero paired difference can reflect unchanged predictions, not certainty.",
            "Frozen signed-lobe interpretation agreement is not independent physical thickness "
            "truth or independently established semantic reflector-switch validation.",
        ],
        "verification": {
            "self_tests": run_self_tests(),
            "saved_arrays_match_frozen_rule_and_summary": True,
            "baseline_row_classifications_match_pinned_frozen_control": True,
            "initial_cohorts_and_baselines_identical_across_policies": True,
            "frozen_helpers": helper_hashes,
        },
        "provenance": {
            "script": artifact(__file__),
            "manifest": artifact(MANIFEST),
            "helper_sources": [
                artifact(Path(__file__).with_name(name))
                for name in ("seeded_eval.py", "seeded_eval_endpoints.py", "seeded_eval_blocks.py")
            ],
            "frozen_scorer_source": source_manifest(BASELINE_SOURCE),
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "inputs": sources,
        },
        "cases": cases,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_self_tests(), indent=2))
        return
    require(not args.output.exists(), "Preserve completed output; choose a new output path")
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
