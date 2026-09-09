"""Plot frozen workbook-development runs without altering paths or rerunning tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-m", type=float, default=0)
    parser.add_argument("--stop-m", type=float, default=150)
    args = parser.parse_args()
    height = 5 * len(args.runs)
    fig, axes = plt.subplots(len(args.runs), 1, figsize=(16, height), squeeze=False)
    fig.subplots_adjust(left=.06, right=.985, bottom=.8 / height,
                        top=1 - 1.3 / height, hspace=.35)
    colors = {1: "#00c5ff", 2: "#ff961f", 3: "#d690ff"}
    for ax, directory in zip(axes[:, 0], args.runs, strict=True):
        report = json.loads((directory / "summary.json").read_text())
        conversion = report["conversion"]
        scale = conversion["mm_per_sample"]
        surface = conversion["reference_surface_sample"]
        with np.load(directory / "tracking.npz") as data:
            x = data["chainage_m"]
            select = (x >= args.start_m) & (x <= args.stop_m)
            if not np.any(select):
                raise ValueError(f"No observations in the requested window: {directory}")
            lo, hi = surface + 10, min(data["measurement"].shape[1], surface + 170)
            radar = data["measurement"][select, lo:hi].copy()
            # Display-only depth gain. Zero rows remain zero; it cannot alter
            # path acceptance or provide additional measurement support.
            radar /= np.maximum(np.sqrt(np.mean(radar * radar, axis=0)), 1e-9)[None, :]
            limit = max(float(np.percentile(abs(radar), 98)), 1e-9)
            ax.imshow(radar.T, aspect="auto", cmap="gray", interpolation="nearest",
                      vmin=-limit, vmax=limit,
                      extent=(x[select][0], x[select][-1],
                              (hi - .5 - surface) * scale, (lo - .5 - surface) * scale))
            for order in (1, 2, 3):
                if f"L{order}_samples" not in data:
                    continue
                proposed = data[f"L{order}_proposed"]
                accepted = data[f"L{order}_visible"].astype(bool)
                depth = (proposed - surface) * scale
                ax.plot(x, np.where((proposed >= 0) & accepted, depth, np.nan),
                        lw=1.8, color=colors[order])
                # Never join across missing states, even for provisional paths.
                ax.plot(x, np.where((proposed >= 0) & ~accepted, depth, np.nan),
                        lw=1, ls=":", color=colors[order])
                isolated = (proposed >= 0) & ~accepted & select
                ax.scatter(x[isolated], depth[isolated], s=3, color=colors[order], alpha=.65)
                points = [c for c in report["comparisons"] if c["layer"] == order
                          and c["origin"] == "manual" and not c["is_guide"]]
                ax.scatter([p["chainage_m"] for p in points],
                           [(p["projected_sample"] - surface) * scale for p in points],
                           marker="_", color="#f42cec" if order == 1 else "#65ff53",
                           s=70, linewidths=1.5, zorder=5)
            for guide in report["guides"]:
                ax.scatter(guide["chainage_m"], (guide["sample"] - surface) * scale,
                           s=55, facecolors=colors[guide["layer"]], edgecolors="black", zorder=6)
            ax.set_xlim(max(args.start_m, x[0]), min(args.stop_m, x[-1]))
            ax.set_ylim((hi - surface) * scale, (lo - surface) * scale)
            ax.set_xlabel("Road distance (m)")
            ax.set_ylabel("Depth equivalent (mm)")
            ax.set_title(f'{directory.name} | {report["method"]}', fontsize=12)
    fig.suptitle("Talagang: frozen conventional tracking results", fontsize=17,
                 y=1 - .13 / height)
    handles = [Line2D([], [], color=colors[1], lw=2, label="Asphalt trace"),
               Line2D([], [], color=colors[2], lw=2, label="Base trace"),
               Line2D([], [], color="gray", ls=":", label="Dotted / dots: provisional"),
               Line2D([], [], color="#65ff53", marker="_", ls="none", markersize=12,
                      label="Original base workbook checkpoints"),
               Line2D([], [], color="#f42cec", marker="_", ls="none", markersize=12,
                      label="Original asphalt workbook checkpoints"),
               Line2D([], [], color="gray", marker="o", ls="none", label="Development guide")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1 - .45 / height),
               ncol=3, frameon=False)
    fig.text(.06, .15 / height, "Solid = accepted by that backend, not verified correct. "
             "Gaps remain gaps. "
             "Acquisition dielectric 7; workbook depth origin unverified; no fitted offset.",
             fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
