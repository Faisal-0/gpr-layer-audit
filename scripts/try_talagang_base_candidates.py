"""Bounded candidate-seed experiment; not field validation or a full pipeline rerun."""

from pathlib import Path
import argparse
import json
import sys
import time

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.seed_graph import pick_seed_conditioned_interfaces
from gpr_layer_audit.reference import normalize_reference_workbook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "exports/talagang-candidate-seeds-20260827")
    parser.add_argument("--reuse", action="store_true", help="Explicitly reuse saved runs for plotting only")
    parser.add_argument("--before", type=Path, help="Previous run folder for before/after radargrams")
    args = parser.parse_args()
    source = ROOT / "exports/seed-assisted-v17-20260827/talagang"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source / "tracking.npz") as saved:
        chosen = (saved["chainage_m"] >= 950) & (saved["chainage_m"] <= 1250)
        x = saved["chainage_m"][chosen]
        radar = saved["radargram"][chosen].copy()
        baseline = saved["layer_2_display"][chosen].copy()
        asphalt = saved["layer_1_display"][chosen].copy()
    manifest = json.loads((source / "manifest.json").read_text())
    surface = manifest["diagnostics"]["reference_surface_sample"]
    rows = [int(np.argmin(abs(x - c))) for c in (1000, 1100, 1200)]
    hypotheses = [
        ("A — flat negative trough", [249, 249, 251]),
        ("B — deeper negative troughs", [290, 273, 293]),
        ("C — deeper positive peaks", [276, 286, 264]),
    ]
    arrays = {"chainage_m": x, "radargram": radar, "baseline": baseline}
    summary = {"purpose": "assistant-selected candidate hypotheses, NOT confirmed manual picks",
               "method": "existing joint graph on cached processed crop; no design corridors; fallback feature branches; no pipeline acceptance gates",
               "window_m": [float(x[0]), float(x[-1])], "runs": []}
    results = []
    for index, (name, seeds) in enumerate(hypotheses):
        cache = output / f"candidate-{index}.npz"
        start = time.perf_counter()
        print(f"Running {name}: {seeds}", flush=True)
        if args.reuse and cache.exists():
            with np.load(cache) as stored:
                samples, raw, canonical = (stored[k].copy() for k in ("samples", "raw", "canonical"))
        else:
            paths = pick_seed_conditioned_interfaces(
                radar, surface, LayerSpec.defaults()[:2],
                anchor_samples={1: {r: int(asphalt[r]) for r in rows if asphalt[r] >= 0},
                                2: dict(zip(rows, seeds))},
                feature_branches={}, search_corridors={}, design_weight=0.0,
                pulse_width_samples=14.0, break_rows=set(),
                anomaly_mask=np.zeros(len(x), dtype=bool), max_interpolation_rows=0,
                horizontal_step_m=float(np.median(np.diff(x))),
            )
            path = paths[2]
            samples = path.samples
            raw = path.evidence["graph_selected_sample"]
            canonical = path.evidence["canonical_event_sample"]
            np.savez_compressed(cache, samples=samples, raw=raw, canonical=canonical,
                                confidence=path.confidence)
        visible = samples >= 0
        record = {"name": name, "seeds": [{"chainage_m": float(x[r]), "sample": s} for r, s in zip(rows, seeds)],
                  "retained_percent": 100 * float(np.mean(visible)),
            "retained_within_5_samples_of_flat_249_percent": 100 * float(np.mean(visible & (abs(samples-249) <= 5))),
                  "raw_graph_within_5_samples_of_flat_249_percent": 100 * float(np.mean((raw >= 0) & (abs(raw-249) <= 5))),
                  "retained_sample_range": [int(samples[visible].min()), int(samples[visible].max())] if visible.any() else None,
                  "elapsed_seconds_this_invocation": time.perf_counter()-start}
        summary["runs"].append(record)
        results.append((name, seeds, samples, raw, canonical))
        print(json.dumps(record), flush=True)
    # Workbook is loaded only AFTER every solver run has completed.
    workbook = ROOT / "GPR Data/talagang/Talagang_layer_thickness_.xlsx"
    points = [p for p in normalize_reference_workbook(workbook)
              if p.layer_order == 2 and x[0] <= p.chainage_m <= x[-1]]
    mm_per_sample = (15.0 / 512) * 299.792458 / (2*np.sqrt(7))
    fig, axes = plt.subplots(3, 1, figsize=(17, 12), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.965, top=.885, bottom=.115, hspace=.25)
    lo, hi = 215, 320
    limit = max(float(np.percentile(abs(radar[:, lo:hi]), 97)), 1e-9)
    for index, (ax, (name, seeds, samples, raw, canonical)) in enumerate(zip(axes, results)):
        ax.imshow(radar[:, lo:hi].T, cmap="gray", aspect="auto", interpolation="nearest",
                  vmin=-limit, vmax=limit, extent=(x[0]-.2, x[-1]+.2, hi-.5, lo-.5))
        ax.plot(x, np.where(baseline >= 0, baseline, np.nan), color="#ff9900", lw=.8, ls="--", alpha=.7)
        ax.plot(x, np.where(samples >= 0, samples, np.nan), color="#00cfff", lw=1.5)
        rejected = (raw >= 0) & (samples < 0)
        ax.scatter(x[rejected], raw[rejected], s=3, color="#ff55cc")
        for derived in (False, True):
            selected = [p for p in points if (p.label_origin.value != "manual") == derived]
            ax.scatter([p.chainage_m for p in selected],
                       [surface + p.cumulative_depth_mm/mm_per_sample for p in selected],
                       marker="x" if derived else "_", s=28 if derived else 65,
                       color="#38ee65", linewidths=1.3, zorder=5)
        ax.scatter(x[rows], seeds, marker="o", s=60, facecolor="white", edgecolor="#006b9c", zorder=6)
        for r, s in zip(rows, seeds):
            ax.annotate(str(s), (x[r], s), xytext=(5, -8), textcoords="offset points", color="#005b86", fontsize=10,
                        bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none", "pad": 1})
        ax.set_title(f"{name} | seeds {seeds} | retained {summary['runs'][index]['retained_percent']:.1f}%", loc="left", fontsize=12)
        ax.set_ylabel("Flattened sample")
        ax.set_xlim(950, 1250)
        ax.set_ylim(hi-.5, lo-.5)
        # Disagreement only at actual manual workbook stations with a nearby retained pick.
        diffs = []
        for p in points:
            row = int(np.argmin(abs(x - p.chainage_m)))
            if p.label_origin.value == "manual" and samples[row] >= 0 and abs(x[row]-p.chainage_m) <= .21:
                diffs.append(abs((canonical[row]-surface)*mm_per_sample-p.cumulative_depth_mm))
        summary["runs"][index]["manual_workbook_comparison_stations"] = len(diffs)
        summary["runs"][index]["canonical_workbook_median_absolute_disagreement_mm"] = float(np.median(diffs)) if diffs else None
    axes[-1].set_xlabel("Chainage (m)")
    fig.suptitle("Talagang | three alternative base seed sets through the existing graph", fontsize=17, y=.975)
    fig.legend(handles=[Line2D([], [], color="#00cfff", lw=2, label="Retained candidate path"),
                        Line2D([], [], color="#ff55cc", marker=".", ls="none", label="Graph proposal rejected by gates"),
                        Line2D([], [], color="#ff9900", ls="--", label="Previous v17 base"),
                        Line2D([], [], color="#38ee65", marker="_", ls="none", label="Workbook (× = interpolated)"),
                        Line2D([], [], color="#006b9c", marker="o", mfc="white", ls="none", label="Assistant candidate seed")],
               loc="upper center", bbox_to_anchor=(.5, .95), ncol=3, frameon=False)
    fig.text(.065, .065, "Seed hypotheses are NOT confirmed interfaces. Retained ≠ physically correct or full-pipeline automatic acceptance. Missing picks remain gaps.", fontsize=10)
    fig.text(.065, .04, "Same cached radar / crop / graph settings for all runs; no design or workbook fitting. Workbook projected with εr=7; lobe vs canonical timing may differ.", fontsize=10)
    fig.savefig(output / "candidate-comparison.png", dpi=170)
    plt.close(fig)
    if args.before:
        fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
        fig.subplots_adjust(left=.055, right=.985, bottom=.12, top=.88, hspace=.23, wspace=.06)
        comparison = []
        for plot_row, index in enumerate((1, 2)):
            name, seeds, samples, raw, _ = results[index]
            with np.load(args.before / f"candidate-{index}.npz") as stored:
                before_samples, before_raw = stored["samples"].copy(), stored["raw"].copy()
            comparison.append({"hypothesis": name,
                "retained_before_percent": 100*float(np.mean(before_samples >= 0)),
                "retained_after_percent": 100*float(np.mean(samples >= 0)),
                "raw_graph_near_sample_249_before_percent": 100*float(np.mean(abs(before_raw-249) <= 5)),
                "raw_graph_near_sample_249_after_percent": 100*float(np.mean(abs(raw-249) <= 5))})
            for plot_col, (label, displayed, proposal) in enumerate((
                ("Before", before_samples, before_raw), ("After: seed-identity guard", samples, raw)
            )):
                ax = axes[plot_row, plot_col]
                ax.imshow(radar[:, lo:hi].T, cmap="gray", aspect="auto", interpolation="nearest",
                          vmin=-limit, vmax=limit, extent=(x[0]-.2, x[-1]+.2, hi-.5, lo-.5))
                ax.plot(x, np.where(baseline >= 0, baseline, np.nan), color="#ff9900", lw=.8, ls="--")
                ax.plot(x, np.where(displayed >= 0, displayed, np.nan), color="#00cfff", lw=1.4)
                hidden = (proposal >= 0) & (displayed < 0)
                ax.scatter(x[hidden], proposal[hidden], color="#ff55cc", s=3)
                ax.scatter([p.chainage_m for p in points],
                           [surface + p.cumulative_depth_mm/mm_per_sample for p in points],
                           color="#38ee65", marker="_", s=36)
                ax.scatter(x[rows], seeds, s=45, facecolor="white", edgecolor="#006b9c", zorder=6)
                ax.set_title(f"{label} | {name[0]} | retained {np.mean(displayed >= 0):.1%}", loc="left", fontsize=12)
                ax.set(xlim=(950,1250), ylim=(hi-.5,lo-.5))
                if plot_col == 0:
                    ax.set_ylabel(f"{name}\nFlattened sample")
                if plot_row == 1:
                    ax.set_xlabel("Chainage (m)")
        fig.suptitle("Talagang | prevent deeper seeds reverting to the horizontal packet", fontsize=18, y=.97)
        fig.text(.055, .92, "Cyan: retained candidate   ·   Pink: rejected graph proposal   ·   Orange dashed: old v17 base   ·   Green: workbook (including interpolated)   ·   Circles: candidate seeds", fontsize=10)
        fig.text(.055, .06, "Same 300 m radar crop and seeds. Workbook is comparison only (εr=7); no design fitting. Retained coverage is NOT accuracy or automatic acceptance.", fontsize=10)
        fig.text(.055, .033, "This guard excludes a separate persistent packet, not all horizontal reflectors. Remaining paths still need physical identification; gaps are deliberate.", fontsize=10)
        fig.savefig(output / "before-after.png", dpi=170)
        plt.close(fig)
        summary["before_after"] = comparison
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
