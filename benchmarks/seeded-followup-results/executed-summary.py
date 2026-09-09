"""Compact completed experiments without rerunning inference or altering evidence."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from seeded_eval import sha

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports/seeded-tracker"


def read(path):
    return json.loads(path.read_text())


def compact(value):
    if isinstance(value, dict):
        return {
            k: compact(v)
            for k, v in value.items()
            if k not in ("evaluation_observations", "diagnostics")
        }
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def summarize(output):
    if output.exists():
        raise ValueError("Preserve previous summary")
    output.mkdir(parents=True)
    report = {
        "claim": "All four followups rejected for application promotion",
        "source_commit": "c442b60ad5f519564893a8da4f824b9327a7f195",
        "script_sha256": sha(__file__),
        "claim_scope": "Historically used development roads; no physical thickness accuracy",
        "limitations": [
            "Legacy correspondence zeros are unavailable instrumentation, not graph-loss counts",
            "Signed-lobe disagreement transitions are proxies, not expert-verified semantic switches",
            "Dense margins are uncalibrated and depend on the full conditioned interval length",
            "Short Mandiali is 11.45m; four requests mean 349.34 requests/km",
            "The local +/-25m correction covers this whole short road",
            "Pointwise alternative events in dense replay are not coherent competing routes",
            "Rows are correlated; no independent-road uncertainty claim from these experiments",
        ],
        "inputs": {},
        "formatting_parity": {},
        "replay": {},
        "peak_states": {},
        "relative_gap": {},
    }

    def record(relative):
        path = EXPORTS / relative
        report["inputs"][relative] = sha(path)
        return read(path)

    for experiment, folder, arms in (
        ("peak_states", "dense-peak-modes-v1/mandiali", ("sample", "peak")),
        ("relative_gap", "dense-relative-gap-v1/mandiali", ("absolute", "relative")),
    ):
        report[experiment]["contract"] = record(f"{folder}/contract.json")
        for arm in arms:
            report[experiment][arm] = compact(record(f"{folder}/{arm}.json"))

    # Exact old arrays establish that controls did not drift across mechanisms.
    before = np.load(EXPORTS / "dense-peak-modes-v1/mandiali/sample.npz")
    after = np.load(EXPORTS / "dense-relative-gap-v1/mandiali/absolute.npz")
    for key in before.files:
        np.testing.assert_array_equal(before[key], after[key])
    report["mandiali_control_arrays_exact"] = True

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for name in ("legacy-active", "legacy-midpoint", "timing-active", "timing-midpoint"):
        data = record(f"dense-replay-v1/mandiali/{name}.json")
        report["replay"][name] = compact(data)
        for column, order in enumerate(("2", "3")):
            step_values = [step["layers"][order] for step in data["steps"]]
            actions = [step["requests"] for step in data["steps"]]
            for row, key in enumerate(
                ("correct_automatic_coverage_initial_pool", "incorrect_automatic_coverage_initial_pool")
            ):
                axes[row, column].plot(
                    actions, [100 * step[key] for step in step_values], "o-", label=name
                )
    for column, layer in enumerate(("Base", "Subbase")):
        axes[0, column].set_title(layer)
        axes[0, column].set_ylabel("Correct automatic coverage (%)")
        axes[1, column].set_ylabel("Wrong automatic coverage (%)")
        axes[1, column].set_xlabel("Additional requests (0-4); initial 3 seeds per layer")
        for row in range(2):
            axes[row, column].set_xticks(range(5))
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Mandiali 11.45m: fixed initial reviewed pools; revealed answers earn no automatic credit"
    )
    fig.tight_layout()
    fig.savefig(output / "intervention-curves.png", dpi=160)
    plt.close(fig)

    folder = "patchnet-v3-far-negatives"
    report["far_negatives"] = {
        key: record(f"{folder}/{filename}")
        for key, filename in (
            ("contract", "contract.json"),
            ("training", "training-result.json"),
            ("training_data", "training-data.json"),
            ("parity", "preprediction-parity.json"),
            ("independent_review", "independent-review/review.json"),
            ("diagnosis", "diagnosis/diagnosis.json"),
        )
    }
    # Formatting wraps preserve code and dynamically generated function text.
    for folder, names in (
        (
            "dense-replay-v1/mandiali",
            ("dense_replay_adapter.py", "experiment_dense_replay.py"),
        ),
        (
            "dense-peak-modes-v1/mandiali",
            ("dense_peak_modes.py", "experiment_dense_peak_modes.py"),
        ),
    ):
        for name in names:
            archived = EXPORTS / folder / "source" / name
            current = ROOT / "scripts" / name
            equal = ast.dump(ast.parse(archived.read_text())) == ast.dump(
                ast.parse(current.read_text())
            )
            assert equal, name
            report["formatting_parity"][name] = {
                "ast_exact": equal,
                "executed_sha256": sha(archived),
                "current_sha256": sha(current),
            }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "executed-summary.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"summary_sha256": sha(output / "summary.json"), "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.output)
