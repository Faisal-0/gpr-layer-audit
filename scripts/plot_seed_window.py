"""Render a radar-only seed decision window, without workbook or predicted depth."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--chainage", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.snapshot) as snapshot:
        x = snapshot["chainage_m"]
        data = snapshot["radargram"]
        row = int(np.argmin(abs(x - args.chainage)))
        selected = abs(x - x[row]) <= 25
        low, high = 160, min(340, data.shape[1])
        limit = float(np.percentile(abs(data[selected, low:high]), 97))
        figure, (radar, scan) = plt.subplots(
            1,
            2,
            figsize=(15, 7),
            sharey=True,
            gridspec_kw={"width_ratios": [4, 1]},
            constrained_layout=True,
        )
        radar.imshow(
            data[selected, low:high].T,
            cmap="gray",
            aspect="auto",
            vmin=-limit,
            vmax=limit,
            extent=(x[selected][0], x[selected][-1], high, low),
        )
        radar.axvline(x[row], color="#0088cc", linewidth=1)
        radar.set(
            xlabel="Chainage (m)",
            ylabel="Sample index (surface reference: 156)",
            title="Clean radar — no tracker or workbook overlay",
        )
        trace = data[row, low:high]
        scale = max(float(np.max(abs(trace))), 1e-9)
        scan.plot(trace / scale, np.arange(low, high), color="black", linewidth=1)
        scan.axvline(0, color="0.7", linewidth=0.6)
        peaks, _ = find_peaks(abs(trace), distance=7, prominence=0.03 * scale)
        for peak in peaks:
            value = trace[peak] / scale
            scan.plot(value, low + peak, "o", color="#0088cc", markersize=3)
            scan.annotate(
                str(low + peak),
                (value, low + peak),
                xytext=(5, -3),
                textcoords="offset points",
                fontsize=8,
            )
        scan.set(xlabel="Normalized amplitude", title=f"A-scan at {x[row]:.1f} m")
        scan.set_ylim(high, low)
        figure.suptitle("Talagang base-interface seed decision · ±25 m continuation", fontsize=14)
        figure.savefig(args.output, dpi=150)
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
