"""Scientific comparison figures from completed scored artifacts, CPU only.

No model, source data, label parser, or scorer is imported. Outer threshold curves
are diagnostic; only the already selected TRAIN thresholds are operating points.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from summarize_processed_campaign import preserve_write, read, sha  # noqa: E402

LAYER_NAMES = {1: "Asphalt", 2: "Base", 3: "Subbase"}
ROADS = ("gujrat", "mandiali", "jamshoro")
BLUE, ORANGE, GREEN, GRAY, RED = "#2166ac", "#e08214", "#16806a", "#737373", "#b2182b"


def setup():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "svg.fonttype": "none",
        }
    )


def save(fig, output, stem, records):
    paths = []
    for suffix in ("png", "svg"):
        target = output / f"{stem}.{suffix}"
        if target.exists():
            backup = (
                output / "history" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") / target.name
            )
            backup.parent.mkdir(parents=True)
            shutil.copy2(target, backup)
        fig.savefig(target, dpi=160, bbox_inches="tight", facecolor="white")
        paths.append({"path": str(target.resolve()), "sha256": sha(target)})
    records.append({"name": stem, "outputs": paths})
    plt.close(fig)


def bars(axis, values, labels, *, colors=None):
    y = np.arange(len(values))
    axis.barh(y, values, color=colors or BLUE, height=0.62)
    for row, value in enumerate(values):
        axis.text(min(value + 1.1, 100.5), row, f"{value:.1f}", va="center", fontsize=9)
    axis.set(yticks=y, yticklabels=labels, xlim=(0, 110), xticks=[0, 25, 50, 75, 100])
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.18)


def primary_comparison(document, output, figures):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    chart = []
    for col, layer in enumerate((1, 2, 3)):
        members = [r for r in document["primary_layer_only_comparisons"] if r["layer"] == layer]
        members.sort(key=lambda r: ROADS.index(r["road_group"]))
        y = np.arange(len(members))
        for row, stage in enumerate(("direct_ungated", "dense_gated")):
            axis = axes[row, col]
            for family, offset, color, label in (
                ("primary", -0.18, BLUE, "Seed-conditioned primary"),
                ("layer_only", 0.18, ORANGE, "Layer-only model"),
            ):
                values = [r[family][stage]["correct_coverage"] * 100 for r in members]
                axis.barh(y + offset, values, height=0.32, color=color, label=label)
                for yi, value in zip(y + offset, values, strict=True):
                    axis.text(min(value + 1.0, 100.2), yi, f"{value:.1f}", va="center", fontsize=9)
                chart.extend(
                    {
                        "layer": layer,
                        "stage": stage,
                        "family": family,
                        "road_group": r["road_group"],
                        **r[family][stage],
                    }
                    for r in members
                )
            axis.set(
                yticks=y,
                yticklabels=[r["road_group"].title() for r in members],
                xlim=(0, 112),
                xticks=[0, 25, 50, 75, 100],
            )
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.18)
            axis.set_title(
                f"{LAYER_NAMES[layer]}: "
                + ("direct proposals" if row == 0 else "automatic accepts")
            )
            axis.set_xlabel("Correct observations / initial nonseed N (%)")
            if row == 1 and layer in (2, 3):
                axis.text(
                    35,
                    0.5,
                    "Both models abstain\nNo TRAIN gate met 95% objective",
                    color=RED,
                    fontsize=9,
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    fig.suptitle(
        "Seed conditioning improves deep proposals; reliable automatic coverage remains unproven\n"
        "Frozen initial observations; all three deep-layer outer folds; seed 42",
        fontsize=14,
    )
    save(fig, output, "primary-vs-layer-only", figures)
    return chart


def stage_comparison(document, output, figures):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    stages = ("seed_interpolation_no_radar", "direct_ungated", "dense_ungated", "dense_gated")
    labels = ("Seed interpolation", "Direct evidence", "Dense path", "Automatic gate")
    for row, road in enumerate(ROADS):
        for col, layer in enumerate((1, 2, 3)):
            axis = axes[row, col]
            values = document["primary_aggregates"][str(layer)]["stages"]
            if road not in values["direct_ungated"]["roads"]:
                axis.text(
                    0.5,
                    0.5,
                    "Not enabled in frozen case registry",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_axis_off()
                continue
            coverage = [values[s]["roads"][road]["correct_coverage"] * 100 for s in stages]
            bars(axis, coverage, labels, colors=[GRAY, BLUE, GREEN, RED])
            axis.set_title(f"{road.title()} · {LAYER_NAMES[layer]}")
            axis.set_xlabel("Correct observations / initial nonseed N (%)")
    fig.suptitle(
        "Where predictions gain and where acceptance fails\n"
        "First three rows of each panel are ungated diagnostics; last row uses TRAIN gate",
        fontsize=14,
    )
    save(fig, output, "primary-decoder-interpolation-gate", figures)


def aggregate_curves(curves):
    if not curves:
        return []
    thresholds = [p["threshold"] for p in curves[0]]
    if any([p["threshold"] for p in c] != thresholds for c in curves):
        raise ValueError("Saved outer curves use different thresholds; do not interpolate silently")
    result = []
    for index, threshold in enumerate(thresholds):
        n = sum(c[index]["N"] for c in curves)
        accepted = sum(c[index]["accepted"] for c in curves)
        correct = sum(c[index]["correct_accepted"] for c in curves)
        result.append(
            {
                "threshold": threshold,
                "N": n,
                "accepted": accepted,
                "correct_accepted": correct,
                "wrong_accepted": accepted - correct,
                "correct_coverage": correct / n if n else None,
                "accepted_agreement": correct / accepted if accepted else None,
            }
        )
    return result


def precision_coverage(document, root, output, figures, sources):
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    chart = []
    for row, road in enumerate(ROADS):
        name = f"primary-{road}-bf16-s42"
        calibration_path = root / "models" / name / "calibration-fp32-stride4.json"
        calibration = read(calibration_path)
        sources[str(calibration_path)] = sha(calibration_path)
        if "float32" not in calibration.get("inference_precision", ""):
            raise ValueError(
                "Precision figure requires calibration arithmetic matching road inference"
            )
        reports = {}
        for detail in document["per_case_details"]:
            if detail["run"] == name:
                path = Path(detail["path"])
                reports.setdefault(detail["layer"], []).append(read(path))
        for col, layer in enumerate((1, 2, 3)):
            axis = axes[row, col]
            for stage, linestyle in (("direct", "-"), ("dense", "--")):
                gate = calibration["calibrations"][str(layer)][stage]
                train = [p for p in gate["curve"] if p["accepted_agreement"] is not None]
                axis.plot(
                    [p["macro_road_correct_coverage"] * 100 for p in train],
                    [p["accepted_agreement"] * 100 for p in train],
                    color=BLUE,
                    linestyle=linestyle,
                    linewidth=1.4,
                )
                if gate["selected_operating_point"] is not None:
                    point = gate["selected_operating_point"]
                    axis.scatter(
                        point["macro_road_correct_coverage"] * 100,
                        point["accepted_agreement"] * 100,
                        marker="*",
                        s=70,
                        color=BLUE,
                        zorder=4,
                    )
                outer = aggregate_curves(
                    [
                        r["stages"][stage + "_ungated"]["precision_coverage_curve"]
                        for r in reports.get(layer, [])
                    ]
                )
                defined = [p for p in outer if p["accepted_agreement"] is not None]
                axis.plot(
                    [p["correct_coverage"] * 100 for p in defined],
                    [p["accepted_agreement"] * 100 for p in defined],
                    color=ORANGE,
                    linestyle=linestyle,
                    linewidth=1.4,
                )
                actual = document["primary_aggregates"][str(layer)]["stages"][stage + "_gated"][
                    "roads"
                ].get(road)
                if actual and actual["accepted_agreement"] is not None:
                    axis.scatter(
                        actual["correct_coverage"] * 100,
                        actual["accepted_agreement"] * 100,
                        marker="D",
                        s=35,
                        facecolor=ORANGE,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=5,
                    )
                chart.append(
                    {
                        "run": name,
                        "layer": layer,
                        "stage": stage,
                        "train_calibration_path": str(calibration_path),
                        "train_selected_threshold": gate["selected_threshold"],
                        "outer_curve_is_diagnostic": True,
                        "outer_curve": outer,
                        "actual_registered_operating_point": actual,
                    }
                )
            axis.axhline(95, color=GRAY, linewidth=0.8, linestyle=":")
            axis.set(
                title=f"{road.title()} · {LAYER_NAMES[layer]}",
                xlim=(0, 100),
                ylim=(0, 102),
                xlabel="Correct coverage (%)",
                ylabel="Accepted agreement (%)",
            )
            axis.grid(alpha=0.12)
            if layer in (2, 3):
                axis.text(
                    0.97,
                    0.06,
                    "No qualified TRAIN gate\nAutomatic coverage = 0",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    color=RED,
                    fontsize=8,
                )
            elif layer not in reports:
                axis.text(
                    0.97,
                    0.06,
                    "Outer asphalt not enabled",
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    color=GRAY,
                    fontsize=8,
                )
    handles = [
        Line2D([], [], color=BLUE, label="TRAIN inner validation"),
        Line2D([], [], color=ORANGE, label="Outer diagnostic sweep"),
        Line2D([], [], color="black", linestyle="-", label="Direct"),
        Line2D([], [], color="black", linestyle="--", label="Dense"),
        Line2D([], [], color=BLUE, marker="*", linestyle="none", label="TRAIN selected"),
        Line2D([], [], color=ORANGE, marker="D", linestyle="none", label="Registered gate result"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    fig.suptitle(
        "Precision versus correct coverage: qualified gates come only from TRAIN\n"
        "FP32 arithmetic. TRAIN x: macro roads; outer x: pooled acquisitions within road.\n"
        "A pooled TRAIN curve above 95% can still fail the required per-road objective.",
        fontsize=12,
    )
    save(fig, output, "primary-train-vs-outer-precision-coverage", figures)
    return chart


def ablations(document, output, figures):
    names = [
        "primary-gujrat-bf16-s42",
        "primary-gujrat-bf16-s84",
        "layer-only-gujrat-bf16-s42",
        "depth-rms-gujrat-bf16-s42",
        "correction-gujrat-bf16-s42",
        "no-context-gujrat-bf16-s42",
    ]
    labels = [
        "Primary · seed 42",
        "Primary · seed 84",
        "Layer-only",
        "Depth RMS",
        "Correction-trained",
        "No wide context",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    y = np.arange(len(names))
    for axis, layer in zip(axes, (1, 2, 3), strict=True):
        for stage, offset, color in (
            ("direct_ungated", -0.17, BLUE),
            ("dense_ungated", 0.17, GREEN),
        ):
            values = [
                next(
                    r["correct_coverage"] * 100
                    for r in document["run_aggregates"]
                    if r["run"] == name and r["layer"] == layer and r["stage"] == stage
                )
                for name in names
            ]
            axis.barh(y + offset, values, height=0.3, color=color)
            for yi, value in zip(y + offset, values, strict=True):
                axis.text(value + 0.8, yi, f"{value:.1f}", va="center", fontsize=8)
        axis.set(
            yticks=y,
            yticklabels=labels,
            xlim=(0, 109),
            xticks=[0, 25, 50, 75, 100],
            title=LAYER_NAMES[layer],
            xlabel="Correct proposals / initial nonseed N (%)",
        )
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.18)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, linewidth=5, label="Direct evidence"),
            Line2D([], [], color=GREEN, linewidth=5, label="Dense path"),
        ],
        loc="outside lower center",
        ncol=2,
    )
    fig.suptitle(
        "Gujrat ablations and second initialization · identical two-acquisition cohort\n"
        "All deep-layer TRAIN gates are undefined; these bars show proposal diagnostics",
        fontsize=13,
    )
    save(fig, output, "gujrat-ablations-and-initialization", figures)


def seed_controls(document, output, figures):
    order = (
        "baseline",
        "removed-seeds",
        "changed-target",
        "seed-shuffle",
        "context-off",
        "conditioning-off",
    )
    labels = (
        "Original seeds",
        "Remove seeds",
        "Alternate target seeds",
        "Permute seed order",
        "Disable wide context",
        "Disable conditioning",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for col, layer in enumerate((2, 3)):
        controls = {c["control"]: c for c in document["seed_controls"] if c["layer"] == layer}
        values = [controls[c]["stages"]["direct_ungated"]["correct_coverage"] * 100 for c in order]
        bars(axes[0, col], values, labels, colors=[BLUE, RED, ORANGE, BLUE, GRAY, RED])
        axes[0, col].set(
            title=f"{LAYER_NAMES[layer]}: original-target agreement",
            xlabel="Correct proposals / original nonseed N (%)",
        )
        shifts = [
            controls[c]["conditioning_change"]["mean_absolute_argmax_shift_samples"]
            if controls[c]["conditioning_change"]
            else 0
            for c in order
        ]
        axis = axes[1, col]
        axis.barh(
            np.arange(len(order)), shifts, color=[BLUE, RED, ORANGE, BLUE, GRAY, RED], height=0.62
        )
        axis.set(
            yticks=np.arange(len(order)),
            yticklabels=labels,
            xlim=(0, max(shifts) * 1.22),
            title="Response to the intervention",
            xlabel="Mean absolute change in native sample argmax",
        )
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.18)
        for y, shift in enumerate(shifts):
            axis.text(shift + max(shifts) * 0.02, y, f"{shift:.2f}", va="center", fontsize=9)
    fig.suptitle(
        "Same Gujrat radar, changed conditioning · frozen original scoring cohort\n"
        "Alternate seeds change predictions; correct tracking of that target is not verified.\n"
        "All control gates withheld; permutation leaves predictions exactly unchanged.",
        fontsize=12,
    )
    save(fig, output, "gujrat-seed-context-controls", figures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    document = read(args.report)
    root = args.report.resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures, sources = [], {str(args.report.resolve()): sha(args.report)}
    chart_data = {"primary_comparison": primary_comparison(document, output, figures)}
    stage_comparison(document, output, figures)
    chart_data["precision_coverage"] = precision_coverage(document, root, output, figures, sources)
    ablations(document, output, figures)
    seed_controls(document, output, figures)
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError(f"Input changed during plotting: {path}")
    manifest = {
        "schema": "processed-campaign-scientific-comparisons-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "figures": figures,
        "input_sha256": sources,
        "plot_script_sha256": sha(__file__),
        "all_outer_threshold_curves_are_diagnostic": True,
        "precision_calibration_arithmetic": "FP32 matches ordinary road inference",
        "chart_data": chart_data,
    }
    preserve_write(output / "manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
    lines = [
        "# Campaign comparison figures",
        "",
        "Static scientific figures from completed artifacts. "
        "PNG and editable SVG exports preserve the same plotted data. Outer threshold curves "
        "are diagnostic; the operating markers use TRAIN-selected gates.",
        "",
    ]
    for figure in figures:
        lines += [
            f"## {figure['name']}",
            "",
            " | ".join(
                f"[{Path(p['path']).suffix[1:].upper()}]({p['path']})" for p in figure["outputs"]
            ),
            "",
        ]
    preserve_write(output / "index.md", "\n".join(lines))
    print(json.dumps({"figures": len(figures), "output": str(output)}))


if __name__ == "__main__":
    main()
