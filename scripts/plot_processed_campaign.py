"""Render frozen native predictions on preregistered failure contexts, CPU only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def draw_context(axis, data, traces, prediction, observations, first, last, layer, *, full=False):
    record = data["record"]
    dx, dt, origin = (
        record[k] for k in ("horizontal_step_m", "sample_interval_ns", "time_origin_ns")
    )
    take = (traces >= first) & (traces <= last)
    rows = np.flatnonzero(take)
    if not len(rows):
        return
    native = traces[rows]
    refs = data["labels"][layer - 1, native]
    if full or not np.isfinite(refs).any():
        lo, hi = 0, record["shape"][1] - 1
    else:
        lo = max(0, int(np.nanmin(refs)) - 36)
        hi = min(record["shape"][1] - 1, int(np.nanmax(refs)) + 36)
    image_rows = native[:: max(1, len(native) // 2400)]
    image_step = float(image_rows[1] - image_rows[0]) if len(image_rows) > 1 else 1.0
    radar = np.asarray(data["amplitudes"][image_rows, lo : hi + 1])
    scale = max(float(np.percentile(abs(radar), 98)), 1)
    axis.imshow(
        radar.T,
        cmap="gray",
        aspect="auto",
        vmin=-scale,
        vmax=scale,
        extent=(
            (image_rows[0] - image_step / 2) * dx,
            (image_rows[-1] + image_step / 2) * dx,
            origin + (hi + 0.5) * dt,
            origin + (lo - 0.5) * dt,
        ),
    )
    known = np.isfinite(refs)
    axis.scatter(
        native[known] * dx,
        origin + refs[known] * dt,
        s=4 if full else 12,
        c="#f5c542",
        label="Stored reference",
        zorder=3,
    )
    x = native * dx
    y = origin + prediction[rows] * dt
    correct = np.array([observations.get(int(r), {}).get("proposed_correct", False) for r in rows])
    scored = np.array([int(r) in observations for r in rows])
    for mask, color, label in (
        (scored & correct, "#1caf70", "Correct proposal"),
        (scored & ~correct, "#e44c55", "Wrong proposal"),
    ):
        axis.scatter(x[mask], y[mask], s=3 if full else 10, c=color, label=label, zorder=4)
    axis.set(xlabel="Stored distance (m)", ylabel="Header time (ns)")
    axis.set_ylim(origin + (hi + 0.5) * dt, origin + (lo - 0.5) * dt)


def plot_run(evaluation, dataset, atlas, output):
    from gpr_layer_audit.ml.processed_data import open_processed_record

    evaluation, output = Path(evaluation), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new figure directory")
    output.mkdir(parents=True)
    atlas_data = read(atlas)
    figure_paths = []
    for report_path in sorted(evaluation.glob("*/baseline/layer*/report.json")):
        report = read(report_path)
        layer = report["layer"]
        if layer == 1:
            continue
        data = open_processed_record(dataset, report["record_id"])
        arrays = np.load(report_path.parent / "stage-arrays.npz", allow_pickle=False)
        traces = arrays["native_trace_indices"]
        proposal = arrays["dense_proposal"]
        stage = report["stages"]["dense_ungated"]
        observed = {int(o["row"]): o for o in stage["evaluation_observations"]}
        windows = [
            entry
            for entry in atlas_data["entries"]
            if entry["record_id"] == report["record_id"] and entry["layer"] == layer
        ]
        # Short Mandiali is fully displayed; no new favorable failure windows are selected.
        fig = plt.figure(figsize=(15, 4 + 2.1 * len(windows)), layout="constrained")
        grid = fig.add_gridspec(1 + len(windows), 1, height_ratios=[1.6] + [1] * len(windows))
        axis = fig.add_subplot(grid[0])
        draw_context(
            axis, data, traces, proposal, observed, traces[0], traces[-1], layer, full=True
        )
        axis.set_title("Complete acquisition; unchanged signed radar; proposals before acceptance")
        axis.legend(loc="upper right", ncols=3, fontsize=8)
        for i, entry in enumerate(windows, 1):
            axis = fig.add_subplot(grid[i])
            draw_context(
                axis,
                data,
                traces,
                proposal,
                observed,
                entry["start_native_trace"],
                entry["stop_native_trace"],
                layer,
            )
            axis.set_title(entry["category"].replace("_", " "), loc="left", fontsize=10)
        gated = report["stages"]["dense_gated"]
        n = stage["fixed_initial_nonseed_N"]
        fig.suptitle(
            f"{report['case_id']} / layer {layer}: "
            f"{stage['correct_proposals_before_gating']:,}/{n:,} correct proposals; "
            f"{gated['correct_accepted']:,} correct, "
            f"{gated['wrong_accepted']:,} wrong accepted\n"
            "Fixed atlas windows; analyst-reference agreement, not physical thickness accuracy",
            fontsize=13,
        )
        path = output / f"{report['case_id']}-layer{layer}-fixed-atlas.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figure_paths.append(str(path.resolve()))
    provenance = {
        "evaluation": str(evaluation.resolve()),
        "figures": figure_paths,
        "atlas_sha256": hashlib.sha256(Path(atlas).read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Plotting only; no labels or predictions altered; unknown rows not scored",
    }
    (output / "figures.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot_run(args.evaluation, args.dataset, args.atlas, args.output)
