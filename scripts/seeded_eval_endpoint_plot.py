"""Render the frozen endpoint/common-cohort summaries without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

DIRECTORY = Path(__file__).resolve().parents[1] / "exports/seeded-tracker/endpoint-common"
INPUTS = {
    "mandiali": (
        "mandiali-common-score-v1.json",
        "41c84cc4b18a04922560624022e6b61d620b319a7f091d3deb044d10980a6f20",
    ),
    "gujrat": (
        "gujrat-common-score-v1.json",
        "25d6b47aa762bb2f9622b6dbaeb5a13ed76732eb76fd7ff77062a12ee45f39cd",
    ),
}
COLORS = ("#087E8B", "#C4413D", "#D8DDE3")
FIELDS = ("correct_coverage", "incorrect_coverage", "unresolved_coverage")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DIRECTORY / "endpoint2-versus-distributed3-v1.png"
    )
    args = parser.parse_args()
    sidecar = args.output.with_suffix(".json")
    if args.output.exists() or sidecar.exists():
        raise ValueError("Preserve existing artifacts; choose a new output filename")
    inputs, plot_rows = {}, []
    for group, (filename, expected) in INPUTS.items():
        path = DIRECTORY / filename
        if sha(path) != expected:
            raise ValueError(f"Frozen common-score input changed: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        inputs[group] = {"path": str(path), "sha256": expected, "case": document["case"]}
        for layer in (2, 3):
            selected = [
                v for v in document["comparison"] if v["segment"] == "whole" and v["layer"] == layer
            ]
            if len(selected) != 1:
                raise ValueError("Expected exactly one whole-cohort result per deep interface")
            row = selected[0]
            for side, budget in (("endpoint2", 2), ("distributed3", 3)):
                if (
                    document[side]["actual_initial_seeds_per_tracked_interface"][str(layer)]
                    != budget
                ):
                    raise ValueError("Initial seed budget differs from plotted label")
                metrics = row[side]
                if metrics["scored_observations"] != row["common_observations"]:
                    raise ValueError("Common observation count mismatch")
                fractions = [metrics[field] for field in FIELDS]
                if not np.isclose(sum(fractions), 1.0, atol=1e-12):
                    raise ValueError("Coverage fractions must sum to one")
                plot_rows.append(
                    {
                        "road_group": group,
                        "case": document["case"],
                        "layer": layer,
                        "seeding": side,
                        "initial_seeds_per_interface": budget,
                        "common_observations": row["common_observations"],
                        **{field: metrics[field] for field in FIELDS},
                        **{
                            field: metrics[field]
                            for field in ("accepted_agree", "accepted_wrong", "unresolved")
                        },
                    }
                )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "text.color": "#172331",
            "axes.labelcolor": "#394756",
            "xtick.color": "#52606D",
            "ytick.color": "#273541",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.25), dpi=180)
    figure.subplots_adjust(
        left=0.145, right=0.978, bottom=0.245, top=0.765, wspace=0.49, hspace=0.70
    )
    figure.text(
        0.5,
        0.947,
        "Two endpoint seeds versus three distributed seeds",
        ha="center",
        va="top",
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.5,
        0.898,
        "Processed radar interpretation · development roads · "
        "identical scoring observations and tolerances",
        ha="center",
        va="top",
        fontsize=11,
        color="#52606D",
    )
    figure.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for color, label in zip(
                COLORS, ("Correct accepted", "Wrong accepted", "Unresolved"), strict=True
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.866),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        handleheight=0.9,
        columnspacing=2.3,
    )
    for row_index, group in enumerate(("mandiali", "gujrat")):
        for column_index, layer in enumerate((2, 3)):
            axis = axes[row_index, column_index]
            values = [v for v in plot_rows if v["road_group"] == group and v["layer"] == layer]
            road = "Mandiali short" if group == "mandiali" else "Gujrat second portion"
            interface = "Base" if layer == 2 else "Subbase"
            axis.set_title(
                f"{road} · {interface}   (n = {values[0]['common_observations']:,})",
                loc="left",
                pad=13,
                weight="bold",
            )
            for y, value in zip((1, 0), values, strict=True):
                percentages = [100 * value[field] for field in FIELDS]
                offset = 0.0
                for width, color in zip(percentages, COLORS, strict=True):
                    axis.barh(
                        y, width, left=offset, height=0.29, color=color, linewidth=0, zorder=3
                    )
                    offset += width
                axis.text(
                    0,
                    y + 0.22,
                    f"Correct {percentages[0]:.1f}%   ·   Wrong {percentages[1]:.1f}%   ·   "
                    f"Unresolved {percentages[2]:.1f}%",
                    fontsize=9.0,
                    va="bottom",
                )
            axis.set_yticks(
                [1, 0], ["Endpoints\n2 seeds/interface", "Distributed\n3 seeds/interface"]
            )
            axis.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
            axis.set_xlim(0, 100)
            axis.set_ylim(-0.32, 1.53)
            axis.grid(axis="x", color="#E7EAEE", linewidth=0.8, zorder=0)
            axis.tick_params(axis="y", length=0, pad=12, labelsize=9)
            axis.tick_params(axis="x", length=0, pad=6, labelsize=9)
            for spine in axis.spines.values():
                spine.set_visible(False)
    figure.text(
        0.145,
        0.158,
        "Denominator: reviewed retained-grid observations, "
        "excluding the same original three support traces in both arms.",
        fontsize=9.7,
        va="top",
    )
    figure.text(
        0.145,
        0.119,
        "Colors describe agreement with reviewed signed-lobe/time references. "
        "Unknown locations remain unscored; fractions are not total-road coverage.",
        fontsize=9.3,
        va="top",
        color="#52606D",
    )
    figure.text(
        0.145,
        0.080,
        "Initial budget: 2 vs 3 seeds per tracked interface "
        "(6 vs 9 total, including asphalt; 4 vs 6 deep-interface observations).",
        fontsize=9.3,
        va="top",
        color="#52606D",
    )
    figure.text(
        0.145,
        0.041,
        "Step zero only. No additional corrections shown. These results establish neither "
        "untouched-road generalization nor physical-thickness accuracy.",
        fontsize=9.1,
        va="top",
        color="#52606D",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, facecolor="white")
    plt.close(figure)
    report = {
        "schema": "endpoint-common-figure-v1",
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha(__file__),
        "input_files": inputs,
        "output": str(args.output.resolve()),
        "output_sha256": sha(args.output),
        "output_dimensions_pixels": [2160, 1305],
        "matplotlib_version": matplotlib.__version__,
        "plot_source": "Existing common-score-v1 JSON fractions only; no inference or rescoring",
        "denominator": (
            "Common reviewed retained-grid observations excluding all original three "
            "support rows in both arms"
        ),
        "displayed_rows": plot_rows,
        "claims": (
            "Processed interpretation development only; no untouched-road generalization "
            "or physical thickness accuracy claim"
        ),
        "limitations": [
            "Step-zero comparison; does not show correction benefit",
            "Unknown reviewed locations are unscored, "
            "so these fractions are not total-road coverage",
            "The Mandiali acquisition is only 11.45 m; "
            "road/block correlation is not quantified in this figure",
            "Bars show whole-cohort results; "
            "separate bracket/tail metrics remain in the input summaries",
        ],
    }
    sidecar.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve())
    print(sidecar.resolve())


if __name__ == "__main__":
    main()
