"""Freeze and evaluate one prospective correction-scope query-ranking experiment.

The tracker, measured inputs, fixed native seeds, correction operation and scorer
are unchanged. Only the frozen active-query module is patched. No evaluation
references are available to the ranking function. Run prepare, then run, then report.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports/seeded-tracker/scoped-queries"
QUERY = Path("gpr_layer_audit/processing/active_queries.py")
CONFIG = ROOT / "benchmarks/conventional-motion-calibrated-development.json"
OLD = ROOT / "exports/seeded-tracker/interaction-prospective"
RADIUS_M = 25.0

HELPERS = '''
def _window_sum(values, lower, upper):
    """Inclusive correction-window sums, exactly matching local replay geometry."""
    prefix = np.concatenate(([0], np.cumsum(values, dtype=np.int64)))
    return prefix[upper + 1] - prefix[lower]


def _scope_geometry(hypotheses, rows, step, radius_m):
    if not np.isfinite(step) or step <= 0 or not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("Correction scope requires positive step and nonnegative radius")
    indices = np.arange(rows)
    lower = np.maximum(0, np.ceil(indices - radius_m / step).astype(int))
    upper = np.minimum(rows - 1, np.floor(indices + radius_m / step).astype(int))
    # Routes differing only outside a query's scope represent one local route.
    # Otherwise remote changes could multiply identical local pair contributions.
    multiplicities = np.ones((len(hypotheses), rows), dtype=np.int32)
    for i, left in enumerate(hypotheses):
        for j in range(i + 1, len(hypotheses)):
            equal_here = _window_sum(left != hypotheses[j], lower, upper) == 0
            multiplicities[i] += equal_here
            multiplicities[j] += equal_here
    return lower, upper, multiplicities


def _observable_disputes(left, right, measurement, valid, tolerance):
    """A hypothetical benefit needs two measured, valid, distinct radar samples."""
    use = (left >= 0) & (right >= 0)
    use &= (left < measurement.shape[1]) & (right < measurement.shape[1])
    rows = np.flatnonzero(use)
    use[rows] &= valid[rows, left[rows]] & valid[rows, right[rows]]
    use[rows] &= measurement[rows, left[rows]] != 0
    use[rows] &= measurement[rows, right[rows]] != 0
    use &= abs(left.astype(np.int64) - right.astype(np.int64)) > tolerance
    return use

'''


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files(source):
    return {
        p.relative_to(source).as_posix(): sha(p)
        for p in sorted(source.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one frozen patch target: {old[:80]}")
    return text.replace(old, new, 1)


def variant(text):
    text = replace_once(text, "def request_observation(", HELPERS + "\ndef request_observation(")
    text = replace_once(
        text,
        "def request_observation(paths, measurement, valid, anchors, step, *, visited=(), "
        'policy="active"):',
        "def request_observation(paths, measurement, valid, anchors, step, *, visited=(), "
        'policy="active", correction_radius_m=25.0):',
    )
    text = replace_once(
        text,
        "        scores = np.zeros(rows)",
        "        lower, upper, multiplicities = _scope_geometry(\n"
        "            hypotheses, rows, step, correction_radius_m\n"
        "        )\n"
        "        scores = np.zeros(rows)",
    )
    text = replace_once(
        text,
        "            for right in hypotheses[i + 1 :]",
        "            for j, right in enumerate(hypotheses[i + 1 :], start=i + 1)",
    )
    text = replace_once(
        text,
        "                disagreement_m = float(\n"
        "                    np.count_nonzero((left >= 0) & (right >= 0) & "
        "(abs(left - right) > tolerance))\n"
        "                    * step\n"
        "                )",
        "                disputes = _observable_disputes(\n"
        "                    left, right, measurement, valid, tolerance\n"
        "                )\n"
        "                disagreement_m = (\n"
        "                    _window_sum(disputes, lower[indices], upper[indices]) * step\n"
        "                    / (multiplicities[i, indices] * multiplicities[j, indices])\n"
        "                )",
    )
    return replace_once(
        text,
        "                # Each observable answer separates this pair over its disputed span.",
        "                # Credit each distinct local pair only over valid disputes in scope.",
    )


def module_at(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def regression():
    scoped = module_at(OUT / "scoped-source/src" / QUERY, "scoped_queries_regression")
    original = module_at(OUT / "original-source/src" / QUERY, "original_queries_regression")
    rows, columns, query, radius = 81, 24, 20, 5.0
    measurement = np.ones((rows, columns))
    valid = np.ones_like(measurement, dtype=bool)
    a = np.full(rows, 5, dtype=np.int32)
    b = a.copy()
    b[15:26] = 12

    def request(routes, fn, radar=measurement, mask=valid, **kwargs):
        path = SimpleNamespace(
            provisional_samples=routes[0],
            samples=routes[0],
            alternate_samples=routes[1],
            visible=np.ones(rows, bool),
            candidate_components={},
            evidence={},
            provenance={"hypothesis_samples": routes, "resolved_pulse": {"lobe_samples": 8.0}},
        )
        # Fix the candidate to one row without changing any measured dispute.
        anchors = {2: {r: int(a[r]) for r in range(rows) if r != query}}
        return fn({2: path}, radar, mask, anchors, 1.0, **kwargs)

    local = request([a, b], scoped.request_observation, correction_radius_m=radius)
    remote = b.copy()
    remote[50:80] = 12
    remote_score = request([a, remote], scoped.request_observation, correction_radius_m=radius)
    assert local == remote_score, (local, remote_score)
    # A route split visible only far away must not manufacture duplicate local pairs.
    split = a.copy()
    split[55:70] = 18
    duplicates = request([a, remote, split], scoped.request_observation, correction_radius_m=radius)
    assert local == duplicates, (local, duplicates)
    assert local["priority"] == 11.0
    old_local = request([a, b], original.request_observation)
    old_remote = request([a, remote], original.request_observation)
    assert old_remote["priority"] > old_local["priority"]
    bad = valid.copy()
    bad[15, 12] = False
    radar = measurement.copy()
    radar[16, 12] = 0
    damaged = b.copy()
    damaged[17], damaged[18] = -1, columns + 10
    masked = request(
        [a, damaged], scoped.request_observation, radar, bad, correction_radius_m=radius
    )
    assert masked["priority"] == 7.0, masked
    whole = request([a, remote], scoped.request_observation, correction_radius_m=100.0)
    assert whole == old_remote
    result = {
        "remote_disputes_do_not_change_local_request": local == remote_score,
        "remote_route_splitting_does_not_multiply_local_benefit": local == duplicates,
        "inclusive_scope_boundary_disputes": 11,
        "invalid_zero_negative_and_out_of_bounds_disputes_excluded": masked["priority"] == 7,
        "whole_road_valid_disputes_equal_original": whole == old_remote,
        "scoped_priority_before_and_after_remote_disputes": [
            local["priority"],
            remote_score["priority"],
        ],
        "original_priority_before_and_after_remote_disputes": [
            old_local["priority"],
            old_remote["priority"],
        ],
    }
    write(OUT / "regression.json", result)
    return result


def prepare():
    if (OUT / "experiment.json").exists():
        raise ValueError("Experiment already frozen; preserve its artifacts")
    OUT.mkdir(parents=True, exist_ok=True)
    original, scoped = OUT / "original-source", OUT / "scoped-source"
    before = files(ROOT / "src")
    if not original.exists():
        shutil.copytree(
            ROOT / "src", original / "src", ignore=shutil.ignore_patterns("__pycache__")
        )
    if not scoped.exists():
        shutil.copytree(original / "src", scoped / "src")
    for destination in (original, scoped):
        shutil.copyfile(ROOT / "scripts/replay_seeded_interaction.py", destination / "replay.py")
    target = scoped / "src" / QUERY
    old_text = (original / "src" / QUERY).read_text(encoding="utf-8")
    new_text = variant(old_text)
    target.write_text(new_text, encoding="utf-8", newline="\n")
    frozen_original, frozen_scoped = files(original / "src"), files(scoped / "src")
    assert files(ROOT / "src") == before == frozen_original
    changed = [p for p in before if before[p] != frozen_scoped[p]]
    assert changed == [QUERY.as_posix()], changed
    assert "gpr_layer_audit/processing/waveform_matching_fast.py" in before
    (OUT / "ranking-only.patch").write_text(
        "".join(
            difflib.unified_diff(
                old_text.splitlines(True),
                new_text.splitlines(True),
                fromfile="a/src/" + QUERY.as_posix(),
                tofile="b/src/" + QUERY.as_posix(),
            )
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": "scoped-query-experiment-v1",
        "variant_count": 1,
        "status": "frozen before replay outcomes",
        "claim": "Processed interpretation development; all roads historically used in development",
        "declared_radius_m": RADIUS_M,
        "policy": (
            "Original pairwise heuristic, with valid in-scope dispute length "
            "and local route deduplication"
        ),
        "scope_geometry": (
            "Inclusive ceil(row - radius/step) through floor(row + radius/step), clipped to road"
        ),
        "no_reference_query_selection": True,
        "no_threshold_or_weight_sweep": True,
        "local_duplicate_reason": (
            "Remote-only route splitting must not multiply local pair contributions"
        ),
        "unchanged_graph_frontier_scores": True,
        "configuration": str(CONFIG.relative_to(ROOT)),
        "configuration_file_sha256": sha(CONFIG),
        "replay_script_sha256": sha(original / "replay.py"),
        "experiment_script_sha256_at_freeze": sha(Path(__file__)),
        "source_files_original": frozen_original,
        "source_files_scoped": frozen_scoped,
        "changed_source_files": changed,
        "compiled_dtw_files_preserved": {
            name: digest for name, digest in before.items() if "waveform_matching" in name.lower()
        },
        "comparators": {
            p.name: {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
            for p in [
                OLD / "gujrat-fixed-active.json",
                OLD / "gujrat-fixed-midpoint.json",
                OLD / "gujrat-four-action-comparison.json",
            ]
        },
    }
    write(OUT / "experiment.json", manifest)
    print(json.dumps(regression(), indent=2))


def verify_frozen():
    frozen = read(OUT / "experiment.json")
    for name in ("original", "scoped"):
        assert files(OUT / f"{name}-source/src") == frozen[f"source_files_{name}"]
        assert sha(OUT / f"{name}-source/replay.py") == frozen["replay_script_sha256"]
    assert sha(CONFIG) == frozen["configuration_file_sha256"]
    for item in frozen["comparators"].values():
        assert sha(ROOT / item["path"]) == item["sha256"]
    assert variant((OUT / "original-source/src" / QUERY).read_text(encoding="utf-8")) == (
        OUT / "scoped-source/src" / QUERY
    ).read_text(encoding="utf-8")


def run():
    verify_frozen()
    regression()
    command = [
        sys.executable,
        str(OUT / "scoped-source/replay.py"),
        "--workspace",
        str(ROOT),
        "--source",
        str(OUT / "scoped-source/src"),
        "--case",
        "gujrat-second",
        "--config",
        str(CONFIG),
        "--output",
        str(OUT / "gujrat-scoped-active.json"),
        "--actions",
        "4",
        "--scope",
        "local",
        "--seeding",
        "three",
        "--policy",
        "active",
        "--radius-m",
        str(RADIUS_M),
        "--freeze-pulse",
    ]
    write(OUT / "run-command.json", {"command": command, "started_unix_s": time.time()})
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, check=False)
    write(
        OUT / "run-completion.json",
        {
            "returncode": result.returncode,
            "wall_runtime_s": time.perf_counter() - started,
            "completed_unix_s": time.time(),
        },
    )
    raise SystemExit(result.returncode)


def mandiali():
    """Compare both decisions at saved complete local checkpoints, without references."""
    verify_frozen()
    sys.path.insert(0, str(OUT / "original-source/src"))
    original = module_at(OUT / "original-source/src" / QUERY, "mandiali_original")
    scoped = module_at(OUT / "scoped-source/src" / QUERY, "mandiali_scoped")
    from gpr_layer_audit.io.dzt import DZTFile
    from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

    case = read(ROOT / "benchmarks/seeded-evaluation-inputs.json")["cases"]["mandiali-short"]
    source = DZTFile(ROOT / case["dzt"])
    radar = np.asarray(source.channel()[:: case["stride"]], np.float32).copy()
    valid = processed_boundary_mask(radar)
    step = source.header.distance_per_trace_m * case["stride"]
    length = (len(radar) - 1) * step
    assert length < RADIUS_M
    checks = []
    for name in ("mandiali-fixed-active", "mandiali-fixed-midpoint"):
        checkpoint = OLD / f"{name}.checkpoint"
        with checkpoint.open("rb") as stream:
            header = json.loads(stream.readline())
            payload = stream.read()
        assert hashlib.sha256(payload).hexdigest() == header["payload_sha256"]
        state = pickle.loads(payload)  # Trusted local repository replay artifact only.
        paths = {o: p for o, p in state["paths"].items() if o in (2, 3)}
        args = (paths, radar, valid, state["anchors"], step)
        keywords = {"visited": state["visited"], "policy": "active"}
        left = original.request_observation(*args, **keywords)
        right = scoped.request_observation(*args, **keywords)
        checks.append(
            {
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "sha256": sha(checkpoint),
                "completed_iteration": state["completed_iteration"],
                "original_request": left,
                "scoped_request": right,
                "exact_request_equal": left == right,
                "decision_equal": left is None
                and right is None
                or left is not None
                and right is not None
                and (left["layer_order"], left["row"]) == (right["layer_order"], right["row"]),
            }
        )
        del state, payload
    write(
        OUT / "mandiali-checkpoint-equivalence.json",
        {
            "road_length_m": length,
            "correction_radius_m": RADIUS_M,
            "entire_road_in_every_query_scope": True,
            "limit": (
                "Saved final checkpoints establish next-decision equivalence, "
                "not unrecorded earlier states"
            ),
            "checks": checks,
        },
    )
    print(json.dumps(checks, indent=2))


def report():
    verify_frozen()
    shutil.copyfile(__file__, OUT / "experiment-driver.py")
    paths = [
        OLD / "gujrat-fixed-active.json",
        OLD / "gujrat-fixed-midpoint.json",
        OUT / "gujrat-scoped-active.json",
    ]
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/seeded_eval.py"),
            "interaction",
            *map(str, paths),
            "--output",
            str(OUT / "gujrat-comparison.json"),
        ],
        cwd=ROOT,
        check=True,
    )
    metrics = read(OUT / "gujrat-comparison.json")
    result = [r for r in metrics["results"] if r["segment"] == "whole" and r["layer"] in (2, 3)]
    for row in result:
        assert row["fixed_initial_observation_denominator"] == {2: 1586, 3: 1067}[row["layer"]]
        if row["step"] == 0:
            assert (row["accepted_agree"], row["accepted"]) == {2: (369, 547), 3: (4, 63)}[
                row["layer"]
            ]
    frozen_counts = {
        ("gujrat-fixed-active", 2): (369, 547),
        ("gujrat-fixed-active", 3): (25, 60),
        ("gujrat-fixed-midpoint", 2): (457, 641),
        ("gujrat-fixed-midpoint", 3): (4, 63),
    }
    for row in result:
        key = (Path(row["artifact"]).stem, row["layer"])
        if row["step"] == 4 and key in frozen_counts:
            assert (row["accepted_agree"], row["accepted"]) == frozen_counts[key]
    fields = [
        "layer",
        "step",
        "requests_all_layers",
        "requests_this_layer",
        "accepted_agree",
        "accepted",
        "accepted_wrong",
        "accepted_agreement",
        "correct_automatic_coverage_initial_pool",
        "incorrect_automatic_coverage_initial_pool",
        "fixed_initial_observation_denominator",
        "revealed_observations_excluded_from_automatic_scoring",
        "new_correct_automatic_observations",
        "lost_correct_automatic_observations_not_revealed",
        "new_wrong_accepted_observations",
        "previous_wrong_now_correct_automatic",
        "previous_wrong_now_unresolved",
        "runtime_s",
    ]
    compact = [{"run": Path(r["artifact"]).stem, **{f: r[f] for f in fields}} for r in result]
    write(OUT / "gujrat-compact-metrics.json", compact)
    runs = [read(path) for path in paths]
    for path in paths[1:]:
        with (
            np.load(paths[0].with_name(f"{paths[0].stem}-step0.npz")) as expected,
            np.load(path.with_name(f"{path.stem}-step0.npz")) as actual,
        ):
            assert expected.files == actual.files
            assert all(np.array_equal(expected[k], actual[k]) for k in expected.files)
    confinement = []
    for path, run in zip(paths, runs, strict=True):
        for index, action in enumerate(run["actions"], start=1):
            with (
                np.load(path.with_name(f"{path.stem}-step{index - 1}.npz")) as before,
                np.load(path.with_name(f"{path.stem}-step{index}.npz")) as after,
            ):
                for key in before.files:
                    order = int(key.split("_")[0].removeprefix("layer"))
                    mutable = np.zeros(len(before[key]), bool)
                    if action.get("affected_rows") and action["layer_order"] == order:
                        lo, hi = action["affected_rows"]
                        expected_lo = max(0, int(np.ceil(action["row"] - RADIUS_M / run["step_m"])))
                        expected_hi = min(
                            len(mutable) - 1,
                            int(np.floor(action["row"] + RADIUS_M / run["step_m"])),
                        )
                        assert (lo, hi) == (expected_lo, expected_hi)
                        mutable[lo : hi + 1] = True
                    assert np.array_equal(before[key][~mutable], after[key][~mutable]), (
                        path,
                        index,
                        key,
                    )
            confinement.append({"run": path.stem, "step": index, "scope_preserved": True})
    write(
        OUT / "array-contract-verification.json",
        {
            "initial_arrays_exactly_equal_across_all_three_runs": True,
            "all_actions_preserve_every_other_layer_and_rows_outside_declared_scope": True,
            "checks": confinement,
        },
    )
    contracts = [run["replay_contract"] for run in runs]
    unchanged = [
        "case_sha256",
        "configuration_sha256",
        "method",
        "scope",
        "seeding",
        "stride",
        "layers",
        "actions",
        "radius_m",
        "freeze_initial_pulse",
    ]
    for field in unchanged:
        assert len({json.dumps(c[field], sort_keys=True) for c in contracts}) == 1, field
    for field in ["initial_anchors", "initial_pulse_samples", "input_sha256", "reference_sha256"]:
        assert len({json.dumps(r[field], sort_keys=True) for r in runs}) == 1, field
    for order in ("1", "2", "3"):
        initial_scoring = [run["steps"][0]["layers"][order] for run in runs]
        assert len({layer["tolerance_samples"] for layer in initial_scoring}) == 1
        cohorts = [
            [(r["row"], r["reference_sample"]) for r in layer["evaluation_observations"]]
            for layer in initial_scoring
        ]
        assert all(cohort == cohorts[0] for cohort in cohorts)
    actions = {
        path.stem: [
            {k: v for k, v in action.items() if k != "regenerated_fit_diagnostics"}
            for action in run["actions"]
        ]
        for path, run in zip(paths, runs, strict=True)
    }
    write(OUT / "actions.json", actions)
    final = {
        r["layer"]: r for r in compact if r["run"] == "gujrat-scoped-active" and r["step"] == 4
    }
    active = {
        r["layer"]: r for r in compact if r["run"] == "gujrat-fixed-active" and r["step"] == 4
    }
    keep = all(
        final[o]["accepted_agree"] >= active[o]["accepted_agree"]
        and final[o]["accepted_wrong"] <= active[o]["accepted_wrong"]
        for o in (2, 3)
    )
    strictly = any(
        final[o]["accepted_agree"] > active[o]["accepted_agree"]
        or final[o]["accepted_wrong"] < active[o]["accepted_wrong"]
        for o in (2, 3)
    )
    decision = (
        "retain as development candidate only"
        if keep and strictly
        else "reject performance promotion"
    )
    write(
        OUT / "verdict.json",
        {
            "decision": decision,
            "criterion": (
                "Pareto improvement versus original active: neither deep layer loses "
                "correct automatic rows or gains wrong rows"
            ),
            "criterion_scope": (
                "Descriptive development readout, not a held-out claim or parameter selection"
            ),
            "gujrat_final_counts": {
                str(o): {k: final[o][k] for k in ("accepted_agree", "accepted", "accepted_wrong")}
                for o in (2, 3)
            },
            "reliable_95_percent_deep_coverage_achieved": False,
            "replay_and_metric_hashes": {
                p.name: sha(p) for p in paths + [OUT / "gujrat-comparison.json"]
            },
            "identical_replay_contract_fields": unchanged,
            "final_driver_sha256": sha(OUT / "experiment-driver.py"),
            "final_driver_reproduces_frozen_query_source_exactly": True,
            "runtime_comparison_limit": (
                "Historical comparators used the older backend and include recorded suspension; "
                "these concurrent instrumented runs do not establish a speedup"
            ),
        },
    )
    lines = [
        "# Scoped query ranking: one prospective development experiment",
        "",
        f"Decision: **{decision}**. No reliable 95% deep coverage claim is supported.",
        "",
        "One frozen variant counts observable radar/model pair disagreements only inside "
        "the declared ±25 m local correction window. It collapses routes identical inside "
        "that window so remote-only route splits cannot multiply local benefit. "
        "Original graph frontier priorities, distance weighting, tracker acceptance "
        "and correction behavior are unchanged.",
        "",
        "All compared runs use three fixed native seeds per layer, four requests, the same "
        "configuration, frozen initial pulse, signed-lobe and timing scoring, and common "
        "initial denominators (base 1586, subbase 1067). Revealed answers receive no automatic "
        "accuracy credit and remain in these denominators. All roads are development data.",
        "",
        "| Run | Step | Layer | Correct / accepted | Wrong | Correct coverage | New correct | "
        "Lost correct (excluding revealed) | New wrong |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in compact:
        lines.append(
            f"| {r['run']} | {r['step']} | {['', '', 'Base', 'Subbase'][r['layer']]} | "
            f"{r['accepted_agree']}/{r['accepted']} | {r['accepted_wrong']} | "
            f"{100 * r['correct_automatic_coverage_initial_pool']:.3f}% | "
            f"{r['new_correct_automatic_observations']} | "
            f"{r['lost_correct_automatic_observations_not_revealed']} | "
            f"{r['new_wrong_accepted_observations']} |"
        )
    lines += [
        "",
        "| Run | Request | Layer | Native trace | Chainage (m) | Answer outcome |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, requests in actions.items():
        for index, action in enumerate(requests, start=1):
            lines.append(
                f"| {name} | {index} | {action['layer_order']} | {action['native_trace']} | "
                f"{action['chainage_m']:.2f} | {action['action']} |"
            )
    mandiali_file = OUT / "mandiali-checkpoint-equivalence.json"
    if mandiali_file.exists():
        equivalence = read(mandiali_file)
        lines += [
            "",
            f"Mandiali spans {equivalence['road_length_m']:.2f} m, so every 25 m radius "
            "query window covers the entire road. Both saved final checkpoints produced "
            "exactly identical next requests under the original and scoped policies "
            "(including priority and candidate samples). This verifies those saved "
            "states; earlier complete states were not retained by the original replay.",
        ]
        assert all(check["exact_request_equal"] for check in equivalence["checks"])
    lines += [
        "",
        "Reproduce with `.venv/Scripts/python.exe scripts/experiment_scoped_queries.py prepare`, "
        "then `run`, `mandiali`, and `report`. Preparation deliberately refuses to overwrite "
        "the frozen experiment. `run-command.json` records the exact replay command; "
        "`ranking-only.patch` contains the source change. "
        "`experiment.json`, `actions.json`, `gujrat-comparison.json`, `regression.json`, and "
        "`mandiali-checkpoint-equivalence.json` retain source hashes, decisions, metrics "
        "and exact checks.",
    ]
    (OUT / "readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(read(OUT / "verdict.json"), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run", "mandiali", "report"))
    args = parser.parse_args()
    {"prepare": prepare, "check": regression, "run": run, "mandiali": mandiali, "report": report}[
        args.command
    ]()


if __name__ == "__main__":
    main()
