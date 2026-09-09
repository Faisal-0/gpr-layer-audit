"""Postscore step-zero endpoint/three-seed replay arrays on the same frozen cohort.

No inference runs here. Operating anchors and prediction arrays remain untouched.
The original three support rows are excluded for BOTH comparators, using the
original three-seed tolerance. Each replay's own pulse/cohort score is also
recomputed and checked against its existing log, then reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SOURCE = ROOT / "exports/seeded-tracker/baseline-source/src"
MANIFEST = ROOT / "benchmarks/seeded-evaluation-inputs.json"
MANIFEST_SHA = "a8e23ba1e6df9fd34f3e8245e34237a8530dacf8c3ba0dbeda123bf1003ea7a8"
SCORER_SHA = "3d17e790891dad14a0a76794eca81cb783ef0ffb6639d7d2d683f6b276a1a894"
CONTROLS = {
    "mandiali-short": (
        "mandiali-control.json",
        "16ec8f1cad7c7f99749e5386a8ffe13dd473edf2bb97f44b4c158bd39a8aa8f7",
    ),
    "gujrat-second": (
        "gujrat-control.json",
        "0bb6a6f388de1c447bade8c0f2b17fa2449725664f25f8f2b39a4188b552c737",
    ),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_sha(source):
    digest = hashlib.sha256()
    for path in sorted(Path(source).rglob("*.py")):
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array_hash(array):
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256_c_order_bytes": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def frozen_context(case_name):
    require(sha(MANIFEST) == MANIFEST_SHA, "Frozen manifest fingerprint mismatch")
    scorer = BASELINE_SOURCE / "gpr_layer_audit/conventional.py"
    require(sha(scorer) == SCORER_SHA, "Frozen scoring source fingerprint mismatch")
    sys.path.insert(0, str(BASELINE_SOURCE))
    from gpr_layer_audit.conventional import _same_lobe
    from gpr_layer_audit.io.dzt import DZTFile
    from gpr_layer_audit.io.dzx import read_dzx

    require(case_name in CONTROLS, "Unsupported frozen case")
    case = read(MANIFEST)["cases"][case_name]
    for field, digest in (
        ("dzt", "dzt_sha256"),
        ("dzx", "dzx_sha256"),
        ("seed_source", "seed_sha256"),
    ):
        require(sha(ROOT / case[field]) == case[digest], f"Frozen {field} fingerprint mismatch")
    filename, expected = CONTROLS[case_name]
    control_path = ROOT / "exports/seeded-tracker/baseline" / filename
    require(sha(control_path) == expected, "Original control fingerprint mismatch")
    control = read(control_path)
    require(control["stride"] == case["stride"], "Original control grid mismatch")
    require(control["audit"]["dzt_sha256"] == case["dzt_sha256"], "Original control data mismatch")
    require(
        control["audit"]["metadata"]["source_sha256"] == case["dzx_sha256"],
        "Original control reference mismatch",
    )
    native = read(ROOT / case["seed_source"])["observations"]
    require(
        control["seed_support"]["observations"] == native, "Original control seed support mismatch"
    )
    anchors = {
        o: {p["trace"] // case["stride"]: p["sample"] for p in values}
        for o, values in native.items()
    }
    require(
        all(len(a) == 3 for a in anchors.values()), "Frozen support budget is not three per layer"
    )
    require(
        all(p["trace"] % case["stride"] == 0 for values in native.values() for p in values),
        "Frozen native seed grid mismatch",
    )
    source = DZTFile(ROOT / case["dzt"])
    radar = np.asarray(source.channel()[:: case["stride"]], np.float32)
    refs = {}
    for group in read_dzx(ROOT / case["dzx"]).layers:
        order = str(case["reference_label_mapping"].get(str(group.number), group.number + 1))
        if order in anchors:
            eligible = [p for p in group.picks if p.channel == 0 and p.trace % case["stride"] == 0]
            refs[order] = {p.trace // case["stride"]: p.sample for p in eligible}
            require(len(refs[order]) == len(eligible), "Duplicate reviewed rows")
            require(
                all(refs[order].get(r) == s for r, s in anchors[order].items()),
                "Frozen seed differs from reviewed observation",
            )
    tolerances = {
        o: control["methods"]["seed_hybrid"]["layers"][o]["tolerance_samples"] for o in anchors
    }
    for order, original in control["methods"]["seed_hybrid"]["layers"].items():
        cohort = {r: s for r, s in refs[order].items() if r not in anchors[order]}
        recorded = {v["row"]: v["reference_sample"] for v in original["evaluation_observations"]}
        require(cohort == recorded, "Original frozen scoring cohort mismatch")
        require(
            original["seed_rows"] == sorted(anchors[order]), "Original frozen seed rows mismatch"
        )
    return {
        "case_name": case_name,
        "case": case,
        "anchors": anchors,
        "references": refs,
        "radar": radar,
        "tolerances": tolerances,
        "same_lobe": _same_lobe,
        "control_path": control_path,
        "control_sha256": expected,
        "step": case["native_dx_m"] * case["stride"],
        "dt": case["dt_ns"],
    }


def validate_log(log, context):
    case = context["case"]
    require(log.get("schema") == "seeded-interaction-replay-v1", "Unsupported replay schema")
    require(log.get("case") == context["case_name"], "Replay case mismatch")
    require(log.get("input_sha256") == case["dzt_sha256"], "Replay data fingerprint mismatch")
    require(
        log.get("reference_sha256") == case["dzx_sha256"], "Replay reference fingerprint mismatch"
    )
    require(
        log.get("initial_seeds_sha256") == case["seed_sha256"], "Replay seed fingerprint mismatch"
    )
    require(log.get("stride") == case["stride"], "Replay grid mismatch")
    require(
        log.get("step_m") == context["step"] and log.get("dt_ns") == context["dt"],
        "Replay physical coordinates mismatch",
    )
    require(
        abs(log.get("road_length_m", -1) - (len(context["radar"]) - 1) * context["step"]) < 1e-9,
        "Replay road span mismatch",
    )
    require(
        log.get("configuration_sha256") == json_sha(log["configuration"]),
        "Replay configuration fingerprint mismatch",
    )
    contract = log["replay_contract"]
    require(
        contract.get("case_sha256") == json_sha(case), "Replay case contract fingerprint mismatch"
    )
    for field in (
        "case",
        "source_snapshot",
        "source_sha256",
        "script_sha256",
        "method",
        "policy",
        "scope",
        "seeding",
        "stride",
        "configuration_sha256",
        "freeze_initial_pulse",
    ):
        require(contract.get(field) == log.get(field), f"Replay contract mismatch: {field}")
    require(log["seeding"] in ("three", "endpoints"), "Unsupported initial seed arrangement")
    expected = deepcopy(context["anchors"])
    if log["seeding"] == "endpoints":
        expected = {o: {r: a[r] for r in (min(a), max(a))} for o, a in expected.items()}
    actual = {o: {int(r): s for r, s in a.items()} for o, a in log["initial_anchors"].items()}
    require(actual == expected, "Replay initial seeds changed")
    steps = [s for s in log.get("steps", []) if s.get("step") == 0]
    require(len(steps) == 1 and steps[0].get("requests") == 0, "Missing or invalid step-zero state")
    step = steps[0]
    require(set(step["layers"]) == set(expected), "Replay tracked layer set mismatch")
    for order, values in step["layers"].items():
        require(
            values["seed_rows"] == sorted(expected[order]),
            "Step-zero operational seed rows mismatch",
        )
        require(
            values["initial_seed_count"] == len(expected[order]), "Step-zero seed budget mismatch"
        )
        require(
            values["tolerance_samples"] == max(2, log["initial_pulse_samples"][order] / 4),
            "Operational pulse/tolerance mismatch",
        )
    return expected, step


def summarize_rows(rows, step, dt):
    accepted = [v for v in rows if v["accepted"]]
    correct = sum(v["accepted_correct"] for v in rows)
    n, a = len(rows), len(accepted)
    wrong = a - correct
    wrong_rows = np.array(
        [v["row"] for v in rows if v["accepted"] and not v["accepted_correct"]], dtype=int
    )
    groups = np.split(wrong_rows, np.flatnonzero(np.diff(wrong_rows) != 1) + 1)
    lobe_rows = np.array(
        [v["row"] for v in rows if v["accepted"] and not v["same_signed_lobe"]], dtype=int
    )
    lobe_episodes = int(len(lobe_rows) > 0) + int(np.count_nonzero(np.diff(lobe_rows) != 1))
    errors = [v["absolute_error_samples"] for v in accepted]
    return {
        "scored_observations": n,
        "accepted": a,
        "accepted_agree": correct,
        "accepted_wrong": wrong,
        "unresolved": n - a,
        "accepted_agreement": correct / a if a else None,
        "correct_coverage": correct / n if n else None,
        "incorrect_coverage": wrong / n if n else None,
        "unresolved_coverage": (n - a) / n if n else None,
        "scored_observation_footprint_m": n * step,
        "correct_accepted_footprint_m": correct * step,
        "incorrect_accepted_footprint_m": wrong * step,
        "unresolved_footprint_m": (n - a) * step,
        "accepted_wrong_signed_lobe_observations": len(lobe_rows),
        "accepted_wrong_signed_lobe_episodes": lobe_episodes,
        "longest_contiguous_incorrectly_accepted_footprint_m": max(
            (len(g) * step for g in groups), default=0
        ),
        "median_accepted_error_samples": float(np.median(errors)) if errors else None,
        "median_accepted_error_ns": float(np.median(errors)) * dt if errors else None,
        "max_accepted_error_samples": float(max(errors)) if errors else None,
        "row_sample_sha256": json_sha([(v["row"], v["reference_sample"]) for v in rows]),
    }


def score_layer(samples, visible, references, excluded, tolerance, bracket, context):
    rows = []
    for row, reference in sorted(references.items()):
        if row in excluded:
            continue
        sample = int(samples[row])
        accepted = bool(visible[row]) and sample >= 0
        same = context["same_lobe"](context["radar"][row], sample, reference) if accepted else False
        error = abs(sample - reference) if accepted else None
        rows.append(
            {
                "row": row,
                "native_trace": row * context["case"]["stride"],
                "reference_sample": reference,
                "accepted_sample": sample,
                "accepted": accepted,
                "accepted_correct": accepted and same and error <= tolerance,
                "same_signed_lobe": same,
                "absolute_error_samples": error,
            }
        )
    first, last = bracket
    partitions = {
        "whole": rows,
        "bracketed": [r for r in rows if first < r["row"] < last],
        "left_tail": [r for r in rows if r["row"] < first],
        "right_tail": [r for r in rows if r["row"] > last],
        "tails": [r for r in rows if r["row"] < first or r["row"] > last],
    }
    return {
        "tolerance_samples": tolerance,
        "excluded_native_traces": [r * context["case"]["stride"] for r in sorted(excluded)],
        "bracket_native_trace_bounds": [
            first * context["case"]["stride"],
            last * context["case"]["stride"],
        ],
        "segments": {
            name: summarize_rows(selected, context["step"], context["dt"])
            for name, selected in partitions.items()
        },
        "evaluation_observations": rows,
    }


def score_log(path, context):
    path = Path(path).resolve()
    log_bytes = path.read_bytes()
    log = json.loads(log_bytes)
    anchors, step_zero = validate_log(log, context)
    snapshot = Path(log["source_snapshot"])
    require(
        snapshot.is_dir() and source_sha(snapshot) == log["source_sha256"],
        "Replay source snapshot fingerprint mismatch",
    )
    require(
        sha(snapshot.parent / "replay.py") == log["script_sha256"],
        "Replay script snapshot fingerprint mismatch",
    )
    archive_path = path.with_name(path.stem + "-step0.npz")
    archive_before = sha(archive_path)
    arrays = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for name in archive.files:
            arrays[name] = archive[name].copy()
    for order in anchors:
        samples, visible = arrays[f"layer{order}_accepted"], arrays[f"layer{order}_visible"]
        require(
            samples.shape == visible.shape == (len(context["radar"]),),
            "Replay array grid shape mismatch",
        )
        require(
            samples.dtype.kind in "iu" and visible.dtype == np.bool_,
            "Replay array numerical contract mismatch",
        )
        require(
            np.all((samples >= -1) & (samples < context["radar"].shape[1])),
            "Replay sample coordinates out of bounds",
        )
        require(
            all(samples[r] == s for r, s in anchors[order].items()),
            "Step-zero arrays do not preserve operating seed samples",
        )
    common, operational = {}, {}
    for order, expected in context["anchors"].items():
        samples, visible = arrays[f"layer{order}_accepted"], arrays[f"layer{order}_visible"]
        bracket = min(expected), max(expected)
        common[order] = score_layer(
            samples,
            visible,
            context["references"][order],
            expected,
            context["tolerances"][order],
            bracket,
            context,
        )
        operational[order] = score_layer(
            samples,
            visible,
            context["references"][order],
            anchors[order],
            step_zero["layers"][order]["tolerance_samples"],
            bracket,
            context,
        )
        fresh = operational[order]["segments"]["whole"]
        recorded = step_zero["layers"][order]
        for fresh_key, old_key in (
            ("scored_observations", "observations_excluding_seeds"),
            ("accepted", "accepted"),
            ("accepted_agree", "accepted_agree"),
        ):
            require(
                fresh[fresh_key] == recorded[old_key],
                f"Operational score differs from replay: layer {order} {fresh_key}",
            )
        existing = {r["row"]: r for r in recorded["evaluation_observations"]}
        require(
            len(existing) == len(operational[order]["evaluation_observations"]),
            "Operational reference cohort mismatch",
        )
        for row in operational[order]["evaluation_observations"]:
            require(row["row"] in existing, "Operational reference row missing")
            prior = existing[row["row"]]
            require(
                all(
                    row[key] == prior[key]
                    for key in ("reference_sample", "accepted", "accepted_correct")
                ),
                "Operational per-observation score differs from replay",
            )
    require(sha(archive_path) == archive_before, "Replay arrays changed during scoring")
    return {
        "path": str(path),
        "log_sha256_at_read": json_sha(log),
        "log_file_sha256_at_read": hashlib.sha256(log_bytes).hexdigest(),
        "step_zero_npz": str(archive_path),
        "step_zero_npz_sha256": archive_before,
        "array_hashes": {name: array_hash(a) for name, a in sorted(arrays.items())},
        "case": log["case"],
        "seeding": log["seeding"],
        "method": log["method"],
        "freeze_initial_pulse": log["freeze_initial_pulse"],
        "source_snapshot": log["source_snapshot"],
        "source_sha256": log["source_sha256"],
        "backend_sha256": log["backend_sha256"],
        "replay_script_sha256": log["script_sha256"],
        "configuration": log["configuration"],
        "configuration_sha256": log["configuration_sha256"],
        "input_sha256": log["input_sha256"],
        "reference_sha256": log["reference_sha256"],
        "original_native_support_file_sha256": log["initial_seeds_sha256"],
        "actual_operating_native_anchors": {
            o: [[r * context["case"]["stride"], s] for r, s in sorted(a.items())]
            for o, a in anchors.items()
        },
        "actual_operating_native_anchors_sha256": json_sha(
            {
                o: [[r * context["case"]["stride"], s] for r, s in sorted(a.items())]
                for o, a in anchors.items()
            }
        ),
        "actual_operating_anchors_sha256": json_sha(anchors),
        "actual_initial_seeds_per_tracked_interface": {o: len(a) for o, a in anchors.items()},
        "actual_initial_seed_total": sum(map(len, anchors.values())),
        "actual_initial_deep_seed_total": sum(
            len(a) for o, a in anchors.items() if o in ("2", "3")
        ),
        "requested_correction_interfaces": log["replay_contract"]["layers"],
        "step_zero_runtime_s": step_zero["runtime_s"],
        "common_three_seed_scoring": common,
        "operational_pulse_scoring": operational,
        "operational_score_matches_logged_counts_and_observations": True,
    }


def compare(endpoint_path, three_path):
    endpoint_log = read(endpoint_path)
    context = frozen_context(endpoint_log["case"])
    endpoint = score_log(endpoint_path, context)
    three = score_log(three_path, context)
    require(
        endpoint["seeding"] == "endpoints" and three["seeding"] == "three",
        "Expected endpoint2 versus distributed3 comparators",
    )
    for field in (
        "case",
        "method",
        "source_sha256",
        "replay_script_sha256",
        "configuration_sha256",
        "input_sha256",
        "reference_sha256",
        "freeze_initial_pulse",
    ):
        require(endpoint[field] == three[field], f"Comparator contract mismatch: {field}")
    deltas = []
    for order in context["anchors"]:
        for segment in ("whole", "bracketed", "left_tail", "right_tail", "tails"):
            a = endpoint["common_three_seed_scoring"][order]["segments"][segment]
            b = three["common_three_seed_scoring"][order]["segments"][segment]
            require(
                a["row_sample_sha256"] == b["row_sample_sha256"], "Common scoring cohort differs"
            )
            deltas.append(
                {
                    "layer": int(order),
                    "segment": segment,
                    "common_observations": a["scored_observations"],
                    "endpoint2": a,
                    "distributed3": b,
                    "distributed3_minus_endpoint2": {
                        k: b[k] - a[k] if a[k] is not None and b[k] is not None else None
                        for k in (
                            "accepted",
                            "accepted_agree",
                            "accepted_wrong",
                            "unresolved",
                            "accepted_agreement",
                            "correct_coverage",
                            "incorrect_coverage",
                            "unresolved_coverage",
                        )
                    },
                }
            )
    return {
        "schema": "endpoint-common-three-seed-cohort-v1",
        "case": context["case_name"],
        "claim": (
            "Processed interpretation development comparison; no physical thickness accuracy claim"
        ),
        "scoring_script": str(Path(__file__).resolve()),
        "scoring_script_sha256": sha(__file__),
        "frozen_manifest": str(MANIFEST),
        "frozen_manifest_sha256": MANIFEST_SHA,
        "frozen_control": str(context["control_path"]),
        "frozen_control_sha256": context["control_sha256"],
        "frozen_scoring_source_sha256": SCORER_SHA,
        "stride": context["case"]["stride"],
        "step_m": context["step"],
        "dt_ns": context["dt"],
        "header_time_origin_ns": context["case"]["header_time_origin_ns"],
        "array_grid_shape": list(context["radar"].shape),
        "coordinate_contract": (
            "Exact stored native trace/sample; working row=native trace/stride; "
            "no registration, snapping or resampling"
        ),
        "common_cohort_contract": (
            "Both arms exclude all original three native support rows per interface, "
            "even though endpoint inference consumes only two. Tolerance is taken unchanged "
            "from the original frozen three-seed control. Operating anchors and all inference "
            "arrays are untouched."
        ),
        "operational_score_contract": (
            "Each arm's original initial-pulse tolerance and actual initial-anchor exclusion "
            "remain available separately; recomputed from arrays and checked against every "
            "logged step-zero observation"
        ),
        "coverage_denominator": (
            "Reviewed retained-grid observations excluding the common original3 support rows; "
            "fractions are not percentages of total road length. Footprints=count*retained "
            "spacing; no interpolation over unknown labels."
        ),
        "lobe_error_contract": (
            "Wrong signed-lobe observations/contiguous episodes are interpretation-error "
            "diagnostics, not independently proven physical reflector switches; a missing "
            "reviewed row breaks a contiguous episode."
        ),
        "unknown_reference_policy": (
            "Unknown labels are unscored, never background or inferred absence"
        ),
        "endpoint2": endpoint,
        "distributed3": three,
        "comparison": deltas,
    }


def self_test(three_path):
    log = read(three_path)
    context = frozen_context(log["case"])
    run = score_log(three_path, context)
    require(log["seeding"] == "three", "Self-test requires a real distributed-three replay")
    for order in context["anchors"]:
        require(
            run["common_three_seed_scoring"][order] == run["operational_pulse_scoring"][order],
            "Three-seed common/operational scores should match",
        )
    rejections = []
    for label, change in (
        ("wrong_input_fingerprint", lambda x: x.update(input_sha256="0" * 64)),
        ("wrong_grid", lambda x: x.update(stride=x["stride"] * 2)),
        (
            "changed_initial_seed",
            lambda x: x["initial_anchors"]["2"].update({next(iter(x["initial_anchors"]["2"])): -1}),
        ),
    ):
        corrupted = deepcopy(log)
        change(corrupted)
        try:
            validate_log(corrupted, context)
        except ValueError as exc:
            rejections.append({"test": label, "rejected": True, "reason": str(exc)})
        else:
            raise AssertionError(f"Failed to reject {label}")
    return {
        "schema": "endpoint-common-score-selftest-v1",
        "case": log["case"],
        "real_array_path": run["step_zero_npz"],
        "real_array_sha256": run["step_zero_npz_sha256"],
        "real_three_seed_common_and_operational_parity": True,
        "rejections": rejections,
        "script_sha256": sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument("--three", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing output; choose a new path")
    require(args.self_test or args.endpoint is not None, "Comparison requires --endpoint")
    result = self_test(args.three) if args.self_test else compare(args.endpoint, args.three)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
