"""Plot saved tracker lobes and workbook depths without rerunning tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpr_layer_audit.io.dzt import read_dzt_header
from gpr_layer_audit.reference import normalize_reference_workbook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--epsilon", type=float, default=7.0)
    parser.add_argument("--center", type=float)
    parser.add_argument("--half-width", type=float, default=50.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error("epsilon must be positive")
    manifest = json.loads((args.snapshot_dir / "manifest.json").read_text())
    root = Path(__file__).resolve().parents[1]
    header = read_dzt_header(root / manifest["source"]["dzt"])
    surface = manifest["diagnostics"]["reference_surface_sample"]
    mm_per_sample = header.sample_interval_ns * 299.792458 / (2 * np.sqrt(args.epsilon))
    points = normalize_reference_workbook(args.workbook)
    if len({p.road_id for p in points}) != 1:
        raise ValueError("Choose a workbook containing exactly one survey line")
    colors = {1: "#00bfff", 2: "#ff9900"}
    workbook_colors = {1: "#e600a9", 2: "#2ee65c"}
    with np.load(args.snapshot_dir / "tracking.npz") as data:
        x = data["chainage_m"]
        select = np.ones(len(x), dtype=bool) if args.center is None else abs(x - args.center) <= args.half_width
        if not np.any(select):
            raise ValueError("Requested window is outside the survey")
        lo, hi = max(0, surface - 5), min(data["radargram"].shape[1], surface + 190)
        radar = data["radargram"][select, lo:hi]
        limit = max(float(np.percentile(abs(radar), 97)), 1e-9)
        fig, ax = plt.subplots(figsize=(17, 7))
        fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.20)
        dx = float(np.median(np.diff(x)))
        ax.imshow(radar.T, cmap="gray", aspect="auto", interpolation="nearest",
                  vmin=-limit, vmax=limit,
                  extent=(x[select][0] - dx / 2, x[select][-1] + dx / 2,
                          (hi - .5 - surface) * mm_per_sample,
                          (lo - .5 - surface) * mm_per_sample))
        handles = []
        for layer, name in ((1, "Asphalt bottom"), (2, "Base bottom")):
            display = data[f"layer_{layer}_display"].astype(float)
            auto = data[f"layer_{layer}_automatic"].astype(bool)
            depth = (display - surface) * mm_per_sample
            visible = (display >= 0) & np.isfinite(display)
            # Keep every road bin: excluded samples become NaN, never bridged.
            ax.plot(x, np.where(visible & auto, depth, np.nan), color=colors[layer], lw=1.5)
            ax.plot(x, np.where(visible & ~auto, depth, np.nan), color=colors[layer], lw=.9, ls=":")
            ax.scatter(x[visible & ~auto & select], depth[visible & ~auto & select],
                       color=colors[layer], s=2, alpha=.65)
            for seed in manifest.get("seeds", []):
                sample = seed.get("samples", {}).get(str(layer))
                if sample is not None and sample >= 0 and seed.get("visibility", {}).get(str(layer)) == "visible":
                    ax.plot(seed["chainage_m"], (sample - surface) * mm_per_sample,
                            "o", ms=7, mec="black", mfc=colors[layer], zorder=6)
            rows = sorted((p for p in points if p.layer_order == layer), key=lambda p: p.chainage_m)
            # Workbook points only: do not invent a reflector between reported stations.
            for derived in (False, True):
                subset = [p for p in rows if (p.label_origin.value != "manual") == derived]
                ax.scatter([p.chainage_m for p in subset], [p.cumulative_depth_mm for p in subset],
                           marker="x" if derived else "_", s=24 if derived else 48,
                           linewidths=1.2, color=workbook_colors[layer], zorder=5)
            handles.extend([
                Line2D([], [], color=colors[layer], lw=2, label=f"Tracker: {name}"),
                Line2D([], [], color=workbook_colors[layer], marker="_", ls="none", ms=10, label=f"Workbook: {name}"),
            ])
        handles.extend([
            Line2D([], [], color="0.3", ls=":", label="Dotted / dots: needs review"),
            Line2D([], [], color="0.3", marker="o", ls="none", label="Circle: supplied seed"),
            Line2D([], [], color="0.3", marker="x", ls="none", label="Cross: derived workbook value"),
        ])
        ax.set_xlim(x[select][0] - dx / 2, x[select][-1] + dx / 2)
        ax.set_ylim((hi - .5 - surface) * mm_per_sample, (lo - .5 - surface) * mm_per_sample)
        ax.set_xlabel("Chainage (m)")
        ax.set_ylabel(f"Depth equivalent (mm), assumed relative permittivity = {args.epsilon:g}")
        window = "full survey" if args.center is None else f"{x[select][0]:.0f}–{x[select][-1]:.0f} m"
        fig.suptitle(f"{args.title} | tracked interfaces vs workbook | {window}", y=.97, fontsize=17)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925), ncol=4, frameon=False, fontsize=10)
        fig.text(.065, .105, "Solid tracker lines = automatically accepted; missing picks remain gaps. Workbook marks are reported stations, not ground truth.", fontsize=10)
        fig.text(.065, .071, "Background: saved processed, surface-flattened radar. Tracker uses displayed reflection lobes; workbook uses cumulative depth (inches × 25.4).", fontsize=10)
        fig.text(.065, .037, f"Depth conversion assumes uniform εr={args.epsilon:g}; lobe/canonical timing conventions may differ. No depth offset fitted to workbook; field accuracy remains unconfirmed.", fontsize=10)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
