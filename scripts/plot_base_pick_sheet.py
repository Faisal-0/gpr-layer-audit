"""Prepare unoverlaid Talagang radar windows for user-supplied base seeds."""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "exports/seed-assisted-v17-20260827/talagang"
    with np.load(folder / "tracking.npz") as saved:
        x, radar = saved["chainage_m"], saved["radargram"]
        fig, axes = plt.subplots(3, 2, figsize=(16, 12),
                                 gridspec_kw={"width_ratios": [5, 1]})
        fig.subplots_adjust(left=.065, right=.955, top=.91, bottom=.085,
                            hspace=.36, wspace=.12)
        low, high = 215, 315
        for (ax, scan), target in zip(axes, (1000, 1100, 1200)):
            row = int(np.argmin(abs(x - target)))
            selected = abs(x - x[row]) <= 50
            window = radar[selected, low:high]
            limit = max(float(np.percentile(abs(window), 97)), 1e-9)
            spacing = float(np.median(np.diff(x)))
            ax.imshow(window.T, cmap="gray", aspect="auto", interpolation="nearest",
                      vmin=-limit, vmax=limit,
                      extent=(x[selected][0]-spacing/2, x[selected][-1]+spacing/2,
                              high-.5, low-.5))
            ax.axvline(x[row], color="#0088cc", lw=.9, ls="--")
            ax.set(title=f"Pick base at dashed line: {x[row]:.1f} m",
                   xlabel="Chainage (m)", ylabel="Flattened sample index")
            trace = radar[row, low:high]
            scale = max(float(np.max(abs(trace))), 1e-9)
            scan.plot(trace/scale, np.arange(low, high), color="black", lw=.9)
            scan.axvline(0, color=".7", lw=.6)
            peaks, _ = find_peaks(abs(trace), distance=6, prominence=.05*scale)
            for p in peaks:
                scan.plot(trace[p]/scale, low+p, ".", color="#0088cc")
                scan.annotate(str(low+p), (trace[p]/scale, low+p),
                              xytext=(5, 0), textcoords="offset points", fontsize=9)
            scan.set(ylim=(high-.5, low-.5), xlim=(-1.1, 1.5),
                     title="A-scan / sample labels", xlabel="Relative amplitude")
            scan.tick_params(labelleft=False)
        fig.suptitle("Talagang | choose three base-interface seeds | 950–1250 m", fontsize=18)
        fig.text(.065, .945, "Processed, surface-flattened radar only — no tracker, workbook or design overlay.", fontsize=11)
        fig.text(.065, .035, "Reply with one sample index per dashed chainage (any sample, not just a labelled peak), or 'unclear'. Labels are amplitude extrema, not proposed interfaces.", fontsize=10)
        output = folder / "base-manual-pick-sheet.png"
        fig.savefig(output, dpi=160)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
