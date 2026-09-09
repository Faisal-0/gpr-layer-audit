"""Verify real correction scope and preserve full-road basin experiment readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from seeded_eval import sha

from gpr_layer_audit.io.dzt import DZTFile

ROOT = Path(__file__).resolve().parents[1]


def compact(value):
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        output = {
            k: compact(v)
            for k, v in value.items()
            if k
            not in (
                "evaluation_observations",
                "diagnostics",
                "correspondence_survived",
                "correspondence_survival",
            )
        }
        if "reflector_switches" in output:
            output["accepted_wrong_signed_lobe_observations"] = output.pop("reflector_switches")
        return output
    return value


def run(folder):
    output = folder / "readout"
    if output.exists():
        raise ValueError("Preserve previous readout")
    output.mkdir()
    report = {
        "decision": "Reject application promotion: longer-road precision and useful coverage fail",
        "claim": "Previously used development roads; no physical thickness accuracy",
        "script_sha256": sha(__file__),
        "limits": [
            "Exact graph margins retain uncalibrated full-interval normalization",
            "A measured waveform basin is not proof of persistent physical identity",
            "Legacy reflector_switches counts wrong-signed-lobe observations, not switch events",
            "Missing correspondence instrumentation is unavailable, not measured zero",
            "Mandiali is one 11.45m block; four requests mean 349.34 requests/km",
        ],
        "sources": {},
        "cases": {},
    }
    manifest = json.loads((ROOT / "benchmarks/seeded-evaluation-inputs.json").read_text())
    for case, replay_folder in (("mandiali-short", "replay"), ("gujrat-second", "gujrat-replay")):
        info = manifest["cases"][case]
        assert sha(ROOT / info["dzt"]) == info["dzt_sha256"]
        datafile = DZTFile(ROOT / info["dzt"])
        data = np.asarray(datafile.channel())[:: info["stride"]]
        dt, dx = (
            datafile.header.sample_interval_ns,
            datafile.header.distance_per_trace_m * info["stride"],
        )
        origin = datafile.header.position_ns
        figure, axes = plt.subplots(2, 2, figsize=(13, 8))
        case_result = {}
        for policy in ("active", "midpoint"):
            path = folder / replay_folder / f"{policy}.json"
            log = json.loads(path.read_text())
            report["sources"][str(path.relative_to(ROOT))] = sha(path)
            arrays = [np.load(path.with_name(f"{policy}-step{i}.npz")) for i in range(5)]
            for i, action in enumerate(log["actions"]):
                for key in arrays[i].files:
                    affected = np.zeros(len(data), bool)
                    if "affected_rows" in action and key.startswith(
                        f"layer{action['layer_order']}_"
                    ):
                        lo, hi = action["affected_rows"]
                        affected[lo : hi + 1] = True
                    np.testing.assert_array_equal(
                        arrays[i][key][~affected], arrays[i + 1][key][~affected]
                    )
            for column, order in enumerate(("2", "3")):
                for row, key in enumerate(
                    (
                        "correct_automatic_coverage_initial_pool",
                        "incorrect_automatic_coverage_initial_pool",
                    )
                ):
                    axes[row, column].plot(
                        range(5),
                        [100 * s["layers"][order][key] for s in log["steps"]],
                        "o-",
                        label=policy,
                    )
            case_result[policy] = compact(log)
            case_result[policy]["outside_scope_arrays_verified_every_action"] = True
            if policy != "active":
                continue
            for order in (2, 3):
                radar_fig, radar_axes = plt.subplots(
                    2, 1, figsize=(14, 7), sharex=True, sharey=True
                )
                for axis, step in zip(radar_axes, (0, 4), strict=True):
                    layer = log["steps"][step]["layers"][str(order)]
                    points = layer["evaluation_observations"]
                    rr = np.array([p["row"] for p in points])
                    proposal = arrays[step][f"layer{order}_provisional"][rr]
                    accepted = np.array([p["accepted"] for p in points])
                    good = np.array([p["accepted_correct"] for p in points])
                    vmax = np.percentile(abs(data), 99)
                    axis.imshow(
                        data.T,
                        aspect="auto",
                        cmap="gray",
                        vmin=-vmax,
                        vmax=vmax,
                        origin="upper",
                        extent=(
                            -dx / 2,
                            (len(data) - 0.5) * dx,
                            (data.shape[1] - 0.5) * dt + origin,
                            origin - dt / 2,
                        ),
                    )
                    axis.scatter(
                        rr * dx,
                        np.array([p["reference_sample"] for p in points]) * dt + origin,
                        s=4,
                        c="#3bb3e3",
                        label="Reviewed reference",
                    )
                    for mask, color, label in (
                        ((proposal >= 0) & ~accepted, "#e4a025", "Unresolved proposal"),
                        (accepted & good, "#20c55e", "Accepted agreeing"),
                        (accepted & ~good, "#ef3b45", "Accepted wrong"),
                    ):
                        axis.scatter(
                            rr[mask] * dx, proposal[mask] * dt + origin, s=6, c=color, label=label
                        )
                    seeds = log["initial_anchors"][str(order)]
                    axis.scatter(
                        np.array(list(seeds), int) * dx,
                        np.array(list(seeds.values())) * dt + origin,
                        s=80,
                        marker="*",
                        c="white",
                        edgecolors="black",
                        label="Initial seed",
                    )
                    for action in log["actions"][:step]:
                        if action["layer_order"] == order and "answer_sample" in action:
                            axis.scatter(
                                action["row"] * dx,
                                action["answer_sample"] * dt + origin,
                                marker="D",
                                s=35,
                                c="#bb80ff",
                                edgecolors="black",
                            )
                    axis.set_title(
                        f"{case}, layer {order}, requests {step}: "
                        f"{layer['accepted_agree']}/{layer['accepted']} agreeing accepted"
                    )
                    axis.set_ylabel("Header time (ns)")
                    times = np.array(list(seeds.values())) * dt + origin
                    axis.set_ylim(max(times) + 2, min(times) - 2)
                radar_axes[0].legend(loc="upper right", ncol=3, fontsize=8)
                radar_axes[-1].set_xlabel(
                    "Native distance (m); supplied answers have no automatic credit"
                )
                radar_fig.tight_layout()
                radar_fig.savefig(output / f"{case}-layer{order}-replay.png", dpi=150)
                plt.close(radar_fig)
        for column, layer in enumerate(("Base", "Subbase")):
            axes[0, column].set_title(layer)
            axes[0, column].set_ylabel("Correct coverage (%)")
            axes[1, column].set_ylabel("Wrong coverage (%)")
            axes[1, column].set_xlabel("Additional requests; initially 3 seeds per layer")
            for row in range(2):
                axes[row, column].set_xticks(range(5))
                axes[row, column].grid(alpha=0.2)
                axes[row, column].legend()
        figure.suptitle(f"{case}; fixed initial reviewed pools, supplied answers excluded")
        figure.tight_layout()
        figure.savefig(output / f"{case}-curves.png", dpi=150)
        plt.close(figure)
        report["cases"][case] = case_result
    (output / "readout.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "executed-readout.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"decision": report["decision"], "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder.resolve())
