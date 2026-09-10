"""Verify scope and plot actual local-edge correction replay; no inference changes."""

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


def run(folder):
    output = folder / "readout-v2"
    if output.exists():
        raise ValueError("Preserve existing readout")
    output.mkdir()
    config = json.loads((folder / "config.json").read_text())
    frozen = Path(config["edge_experiment"])
    contract = json.loads((frozen / "contract.json").read_text())
    entry = contract["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    data = np.asarray(DZTFile(entry["dzt"]).channel())[:: contract["stride"]]
    dt, dx, origin = entry["dt_ns"], entry["dx_m"] * contract["stride"], entry["origin_ns"]
    report = {
        "decision": "Reject: active corrections add wrong accepted length and fail 95% precision",
        "script_sha256": sha(__file__),
        "claim": "Development interpretation, Gujrat only",
        "initial_seeds_per_layer": 3,
        "policies": {},
        "switch_limit": "Legacy reflector_switches counts wrong-lobe observations, not events",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for policy in ("active", "midpoint"):
        log = json.loads((folder / f"{policy}.json").read_text())
        snapshots = [np.load(folder / f"{policy}-step{i}.npz") for i in range(len(log["steps"]))]
        for order in (2, 3):
            previous = np.load(frozen / f"{config['arm']}-layer{order}.npz")
            for key in ("samples", "provisional", "visible"):
                mapped = "accepted" if key == "samples" else key
                np.testing.assert_array_equal(snapshots[0][f"layer{order}_{mapped}"], previous[key])
        for i, action in enumerate(log["actions"]):
            for key in snapshots[i].files:
                affected = np.zeros(len(data), bool)
                if "affected_rows" in action and key.startswith(f"layer{action['layer_order']}_"):
                    lo, hi = action["affected_rows"]
                    affected[lo : hi + 1] = True
                np.testing.assert_array_equal(
                    snapshots[i][key][~affected], snapshots[i + 1][key][~affected]
                )
            if (
                "answer_sample" in action
                and action.get("retrack_status") != "neighbor_observation_conflict"
            ):
                assert (
                    snapshots[i + 1][f"layer{action['layer_order']}_accepted"][action["row"]]
                    == action["answer_sample"]
                )
        steps = []
        for step in log["steps"]:
            layers = {}
            for order, layer in step["layers"].items():
                layers[order] = {
                    k: layer[k]
                    for k in (
                        "observations_excluding_seeds",
                        "fixed_initial_observations",
                        "accepted",
                        "accepted_agree",
                        "proposed_agree",
                        "accepted_pick_agreement",
                        "longest_contiguous_wrong_span_m",
                        "correct_automatic_coverage_initial_pool",
                        "incorrect_automatic_coverage_initial_pool",
                    )
                }
                layers[order]["accepted_wrong_signed_lobe_observations"] = layer[
                    "reflector_switches"
                ]
            steps.append({"step": step["step"], "layers": layers})
        report["policies"][policy] = {
            "log_sha256": sha(folder / f"{policy}.json"),
            "initial_saved_display_arrays_exact": True,
            "outside_scope_saved_display_arrays_exact_every_action": True,
            "checkpoint_fields": ["accepted samples", "provisional samples", "visible"],
            "exact_answer_preservation": True,
            "road_length_m": log["road_length_m"],
            "requests_per_km": len(log["actions"]) * 1000 / log["road_length_m"],
            "actions": log["actions"],
            "steps": steps,
        }
        for column, order in enumerate(("2", "3")):
            for row, metric in enumerate(
                (
                    "correct_automatic_coverage_initial_pool",
                    "incorrect_automatic_coverage_initial_pool",
                )
            ):
                axes[row, column].plot(
                    range(len(steps)),
                    [100 * s["layers"][order][metric] for s in steps],
                    "o-",
                    label=policy,
                )
        if policy != "active":
            continue
        for order in (2, 3):
            radar_fig, radar_axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True)
            seeds = log["initial_anchors"][str(order)]
            # Display contrast uses operating-seed depth range only; same for both panels.
            lower = max(0, int(min(seeds.values()) - 2 / dt))
            upper = min(data.shape[1], int(max(seeds.values()) + 2 / dt) + 1)
            vmax = np.percentile(abs(data[:, lower:upper]), 99)
            for axis, step in zip(radar_axes, (0, len(steps) - 1), strict=True):
                metrics = log["steps"][step]["layers"][str(order)]
                points = metrics["evaluation_observations"]
                rr = np.array([p["row"] for p in points])
                pp = snapshots[step][f"layer{order}_provisional"][rr]
                good = np.array([p["accepted_correct"] for p in points])
                accepted = np.array([p["accepted"] for p in points])
                axis.imshow(
                    data.T,
                    cmap="gray",
                    aspect="auto",
                    vmin=-vmax,
                    vmax=vmax,
                    extent=(
                        -dx / 2,
                        (len(data) - 0.5) * dx,
                        origin + (data.shape[1] - 0.5) * dt,
                        origin - dt / 2,
                    ),
                )
                axis.scatter(
                    rr * dx,
                    np.array([p["reference_sample"] for p in points]) * dt + origin,
                    c="#3bb3e3",
                    s=4,
                    label="Reviewed reference",
                )
                for mask, color, label in (
                    ((pp >= 0) & ~accepted, "#e4a025", "Unresolved proposal"),
                    (accepted & good, "#20c55e", "Accepted agreeing"),
                    (accepted & ~good, "#ef3b45", "Accepted wrong"),
                ):
                    axis.scatter(rr[mask] * dx, pp[mask] * dt + origin, s=6, c=color, label=label)
                axis.scatter(
                    np.array(list(seeds), int) * dx,
                    np.array(list(seeds.values())) * dt + origin,
                    marker="*",
                    s=70,
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
                axis.set_ylim(upper * dt + origin, lower * dt + origin)
                axis.set_title(
                    f"Gujrat layer {order}; requests {step}; "
                    f"{metrics['accepted_agree']}/{metrics['accepted']} agreeing accepted"
                )
                axis.set_ylabel("Header time (ns)")
            radar_axes[0].legend(ncol=3, fontsize=8)
            radar_axes[-1].set_xlabel(
                "Native distance (m); revealed answers excluded from automatic credit"
            )
            radar_fig.tight_layout()
            radar_fig.savefig(output / f"layer{order}-active-replay.png", dpi=150)
            plt.close(radar_fig)
    for column, label in enumerate(("Base", "Subbase")):
        axes[0, column].set_title(label)
        for row in range(2):
            axes[row, column].set_ylabel(
                ("Correct" if row == 0 else "Wrong") + " accepted coverage (%)"
            )
            axes[row, column].set_xticks(range(5))
            axes[row, column].grid(alpha=0.2)
            axes[row, column].legend()
        axes[1, column].set_xlabel("Additional requests; initial 3 seeds per layer")
    fig.suptitle("Gujrat: fixed reviewed pools 1586/1067; supplied answers excluded")
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=150)
    plt.close(fig)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "executed-readout.py").write_bytes(Path(__file__).read_bytes())
    print(report["decision"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder.resolve())
