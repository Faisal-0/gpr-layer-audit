"""Publish unchanged scored dense comparisons, with incorrect acceptances in red."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from diagnose_patchnet_dense import compact_metrics
from seeded_eval import sha

from gpr_layer_audit.conventional import _same_lobe
from gpr_layer_audit.io.dzt import DZTFile


def summarize(folder):
    if (folder / "summary.json").exists():
        raise ValueError("Preserve the published summary")
    runs = {
        arm: json.loads((folder / f"{arm}.json").read_text()) for arm in ("classical", "pairwise")
    }
    first = runs["classical"]
    radar_path = Path(first["source"]["dzt"])
    assert sha(radar_path) == first["source"]["dzt_sha256"]
    radar = DZTFile(radar_path)
    data = np.asarray(radar.channel())[:: first["stride"]]
    dx = radar.header.distance_per_trace_m * first["stride"]
    dt, origin = radar.header.sample_interval_ns, radar.header.position_ns
    result = {
        "decision": "Reject: more wrong accepted observations and less correct coverage in both layers",
        "claim": "Development interpretation only, no physical thickness claim",
        "script_sha256": sha(__file__),
        "contract_sha256": sha(folder / "contract.json"),
        "input_hashes": {
            k: first["source"][k] for k in ("dzt_sha256", "dzx_sha256", "seed_sha256")
        },
        "array_hashes": {arm: sha(folder / f"{arm}.npz") for arm in runs},
        "score_hashes": {arm: sha(folder / f"{arm}.json") for arm in runs},
        "notes": [
            "Metres are reviewed observation footprints, not interpolated road length",
            "Legacy correspondence-survival zeros mean unavailable diagnostics",
            "Wrong signed-lobe observations are not independent semantic switch events",
        ],
        "layers": {},
    }
    for order in (2, 3):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, sharey=True)
        stats = {}
        vmax = np.percentile(abs(data[:, 110:]), 99)
        for axis, (arm, run) in zip(axes, runs.items(), strict=True):
            layer = run["layers"][str(order)]
            points = layer["evaluation_observations"]
            seeds = layer["seed_rows"]
            arrays = np.load(folder / f"{arm}.npz")
            accepted = arrays[f"layer{order}_accepted"]
            proposal = arrays[f"layer{order}_provisional"]
            for p in points:
                p["accepted_wrong_signed_lobe"] = bool(
                    p["accepted"]
                    and not _same_lobe(data[p["row"]], accepted[p["row"]], p["reference_sample"])
                )
            metrics = compact_metrics(points, dx, dt)
            metrics["bracketed"] = compact_metrics(
                [p for p in points if min(seeds) < p["row"] < max(seeds)], dx, dt
            )
            metrics["tails"] = compact_metrics(
                [p for p in points if not min(seeds) <= p["row"] <= max(seeds)], dx, dt
            )
            metrics["runtime_s"] = layer["runtime_s"]
            stats[arm] = metrics
            axis.imshow(
                data.T,
                aspect="auto",
                cmap="gray",
                vmin=-vmax,
                vmax=vmax,
                extent=(0, len(data) * dx, data.shape[1] * dt + origin, origin),
            )
            rr = np.array([p["row"] for p in points])
            samples = np.array([p["reference_sample"] for p in points])
            axis.scatter(
                rr * dx, samples * dt + origin, c="#35b7e3", s=8, label="Reviewed reference"
            )
            good = np.array([p["accepted_correct"] for p in points])
            visible = np.array([p["accepted"] for p in points])
            for mask, color, label in (
                ((proposal[rr] >= 0) & ~visible, "#e49c1b", "Unresolved proposal"),
                (visible & good, "#16c35b", "Accepted agreeing"),
                (visible & ~good, "#f33040", "Accepted wrong"),
            ):
                axis.scatter(
                    rr[mask] * dx, proposal[rr[mask]] * dt + origin, c=color, s=9, label=label
                )
            axis.scatter(
                np.array(seeds) * dx,
                accepted[seeds] * dt + origin,
                marker="*",
                s=100,
                c="white",
                edgecolors="black",
                label="Operating seed",
            )
            axis.set_title(
                f"Mandiali {'base' if order == 2 else 'subbase'} / {arm}: "
                f"{metrics['accepted_correct']} correct, {metrics['accepted_wrong']} "
                f"wrong, {metrics['unresolved']} unresolved / {len(points)}"
            )
            axis.set_ylabel("Header time (ns)")
            # Display range follows initial operating observations only.
            seed_times = accepted[seeds] * dt + origin
            axis.set_ylim(max(seed_times) + 2, min(seed_times) - 2)
        axes[0].legend(loc="upper right", ncol=3, fontsize=8)
        axes[-1].set_xlabel(
            "Native trace distance (m); supplied observations excluded from accuracy"
        )
        fig.tight_layout()
        fig.savefig(folder / f"layer{order}-accepted-comparison.png", dpi=160)
        plt.close(fig)
        result["layers"][str(order)] = stats
    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    summarize(parser.parse_args().folder)
