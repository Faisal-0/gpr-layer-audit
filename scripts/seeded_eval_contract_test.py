"""Executable integration assertions for the frozen real-road evaluation contract."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from seeded_eval import (
    BASELINE,
    ROOT,
    read,
    sha,
    summarize,
    summarize_interaction,
    validate_frozen_helpers,
    write,
)


def main():
    root = ROOT / "exports/seeded-tracker/baseline"
    expected = {
        "mandiali-control": {"2": (443, 4, 4), "3": (404, 44, 42)},
        "gujrat-control": {"2": (1586, 547, 369), "3": (1067, 63, 4)},
        "gujrat-control-coarse": {"2": (395, 29, 29), "3": (261, 0, 0)},
    }
    for name, layers in expected.items():
        path = root / f"{name}.json"
        run = read(path)
        values = run["methods"]["seed_hybrid"]["layers"]
        for order, counts in layers.items():
            actual = tuple(
                values[order][key]
                for key in ("observations_excluding_seeds", "accepted", "accepted_agree")
            )
            assert actual == counts, (name, order, actual, counts)
    checked = []
    for path in root.glob("*.json"):
        run = read(path)
        if run.get("schema") != "conventional-evaluation-v2":
            continue
        assert sha(run["audit"]["dzt_path"]) == run["audit"]["dzt_sha256"]
        for method, result in run["methods"].items():
            with np.load(path.with_name(f"{path.stem}-{method}.npz")) as arrays:
                for order, values in result["layers"].items():
                    seeds = run["seed_support"]["observations"][order]
                    seed_rows = {p["trace"] // run["stride"] for p in seeds}
                    assert not seed_rows.intersection(
                        p["row"] for p in values["evaluation_observations"]
                    )
                    assert all(
                        arrays[f"layer{order}_accepted"][p["trace"] // run["stride"]] == p["sample"]
                        for p in seeds
                    ), (path, order, "Exact seed changed")
        checked.append(path.name)
    original = read(ROOT / "benchmarks/conventional-frozen-comparators.json")
    assert all(sha(Path(original["location"]) / k) == v for k, v in original["files"].items())
    # A tracker exception must leave the same scored denominator, all unresolved.
    with tempfile.TemporaryDirectory(dir=ROOT / "exports/seeded-tracker") as temporary:
        package = Path(temporary) / "source" / "gpr_layer_audit"
        package.mkdir(parents=True)
        for name in ("conventional.py", "conventional_reference.py", "conventional_seeds.py"):
            frozen = (BASELINE / "gpr_layer_audit" / name).read_bytes()
            (package / name).write_bytes(frozen.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        validate_frozen_helpers(package.parent)
        with (package / "conventional.py").open("ab") as handle:
            handle.write(b"\r\n# Not part of the frozen scorer\r\n")
        try:
            validate_frozen_helpers(package.parent)
        except ValueError:
            pass
        else:
            raise AssertionError("Edited scorer passed the line-ending-only contract")
        case = read(root / "mandiali-control.json")
        case["methods"] = {
            "seed_hybrid": {"status": "failed", "error": "test failure", "layers": {}}
        }
        path, output = Path(temporary) / "failure.json", Path(temporary) / "summary.json"
        write(path, case)
        summarize([path], output)
        rows = {r["layer"]: r for r in read(output)["results"]}
        assert rows[2]["scored_observations"] == 443
        assert rows[3]["scored_observations"] == 404
        assert all(r["unresolved_coverage"] == 1 for r in rows.values())
        case["audit"]["dzt_sha256"] = "0" * 64
        write(path, case)
        try:
            summarize([path], output)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed input fingerprint was accepted")
        # An actual four-correction replay must preserve its original cohort.
        # The old replay intentionally remains useful as a failed comparator.
        replay = ROOT / "exports/seeded-tracker/interaction/mandiali-control-active.json"
        replay_output = Path(temporary) / "replay-summary.json"
        summarize_interaction([replay], replay_output)
        rows = [r for r in read(replay_output)["results"] if r["segment"] == "whole"]
        for row in rows:
            if row["layer"] in (2, 3):
                assert (
                    row["fixed_initial_observation_denominator"] == {2: 443, 3: 404}[row["layer"]]
                )
            fractions = [
                row[k]
                for k in (
                    "correct_automatic_coverage_initial_pool",
                    "incorrect_automatic_coverage_initial_pool",
                    "analyst_supplied_coverage_initial_pool",
                    "unresolved_coverage_initial_pool",
                )
            ]
            assert abs(sum(fractions) - 1) < 1e-12
        final_base = next(r for r in rows if r["layer"] == 2 and r["step"] == 4)
        assert final_base["correct_automatic_coverage_initial_pool"] == 14 / 443
        assert final_base["revealed_observations_excluded_from_automatic_scoring"] == 3
        assert final_base["remaining_automatic_observations"] == 440
        first_subbase = next(r for r in rows if r["layer"] == 3 and r["step"] == 1)
        assert first_subbase["lost_correct_automatic_observations_not_revealed"] > 0
    print(
        json.dumps(
            {
                "baseline_counts_reproduced": True,
                "old_comparators_unchanged": True,
                "exact_seed_and_exclusion_checks": checked,
                "failure_denominator_and_hash_rejection": "passed",
                "real_replay_fixed_cohort_and_measured_regression": "passed",
                "line_ending_only_scorer_equivalence_and_tamper_rejection": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
