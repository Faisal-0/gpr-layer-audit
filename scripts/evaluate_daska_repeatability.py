"""Evaluate forward/back radar tracking repeatability on the Daska-Pasrur road.

The two acquisitions traverse the same road in opposite directions. They are
processed and tracked independently, aligned by reversed chainage, and compared
without using the legacy workbook in either solver. Workbook values are loaded
only after tracking as a secondary disagreement diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.pipeline import _chainage, _effective_stack
from gpr_layer_audit.processing.preprocessing import (
    measurement_packet_support,
    preprocess_for_interpretation,
)
from gpr_layer_audit.processing.tracker import pick_interfaces
from gpr_layer_audit.reference import normalize_reference_workbook

ROOT = Path(__file__).resolve().parents[1]
ZONE = ROOT / "GPR Data/NORTH ZONE"
FORWARD = ZONE / (
    "DASKA PASRUR ROAD, SIALKOT.PRJ/"
    "DASKA PASRUR ROAD, SIALKOT_001.DZT"
)
BACKWARD = ZONE / (
    "DASKA PASRUR ROAD,SIALKOT BACK.PRJ/"
    "DASKA PASRUR ROAD,SIALKOT BACK_001.DZT"
)
PLATE = ZONE / (
    "METAL PLATE CONFIGRATION SIALKOT.PRJ/"
    "METAL PLATE CONFIGRATION SIALKOT_001.DZT"
)
WORKBOOK = ZONE / "Sialkot Thicknesses_graph.xlsx"
ALL_PHYSICAL_SEED_CHAINAGES = (100.0, 300.0, 600.0, 900.0, 1100.0)
# Radar-only development clicks on the negative lobe of the repeated packet.
# Each acquisition is snapped independently because the passes follow nearby,
# not identical, wheel paths. These are reproducibility constraints, not truth.
ALL_FORWARD_SEED_SAMPLES = {
    1: (183, 184, 177, 182, 173),
    2: (256, 245, 256, 257, 257),
}
ALL_BACKWARD_SEED_SAMPLES = {
    1: (178, 180, 180, 185, 178),
    2: (256, 245, 243, 256, 256),
}


def _prepare(
    path: Path,
    plate: DZTFile,
    seed_samples: dict[int, tuple[int, ...]],
    physical_seed_chainages: tuple[float, ...],
    *,
    reverse: bool,
):
    road = DZTFile(path)
    stack = _effective_stack(
        road.header.trace_count, 0, road.header.distance_per_trace_m
    )
    calibrated = calibrate(road, plate, stack_size=stack)
    metadata = read_dzx(path.with_suffix(".DZX"))
    chainage = _chainage(road.header, calibrated.trace_centres, metadata)
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
    )
    interpreted.feature_branches["measurement_support"] = measurement_packet_support(
        calibrated.measurement_radargram
    )
    physical = float(chainage[-1]) - chainage if reverse else chainage
    seed_rows = [
        int(np.argmin(np.abs(physical - value)))
        for value in physical_seed_chainages
    ]
    anchors = {
        order: dict(zip(seed_rows, samples, strict=True))
        for order, samples in seed_samples.items()
    }
    paths = pick_interfaces(
        interpreted.radargram,
        calibrated.reference_surface_sample,
        LayerSpec.defaults()[:2],
        anchor_samples=anchors,
        feature_branches=interpreted.feature_branches,
        search_corridors={},
        design_weight=0.0,
        anomaly_mask=np.zeros(len(chainage), dtype=bool),
        break_rows=set(),
        max_interpolation_rows=0,
        horizontal_step_m=float(np.median(np.diff(chainage))),
    )
    return road, calibrated, interpreted, chainage, paths, anchors


def _aligned_backward(forward_x: np.ndarray, backward_x: np.ndarray) -> np.ndarray:
    physical = float(backward_x[-1]) - backward_x
    order = np.argsort(physical)
    return np.interp(forward_x, physical[order], np.arange(len(physical))[order])


def _sample(values: np.ndarray, fractional_rows: np.ndarray) -> np.ndarray:
    rows = np.arange(len(values), dtype=float)
    return np.interp(fractional_rows, rows, np.asarray(values, dtype=float))


def _workbook_samples(surface: int, sample_interval_ns: float):
    mm_per_sample = sample_interval_ns * 299.792458 / (2 * np.sqrt(7.0))
    return [
        {
            "chainage_m": float(point.chainage_m),
            "layer_order": int(point.layer_order),
            "sample": float(surface + point.cumulative_depth_mm / mm_per_sample),
            "manual": point.label_origin.value == "manual",
        }
        for point in normalize_reference_workbook(WORKBOOK)
    ]


def _path_metrics(forward, backward, rows, pulse_width: float) -> dict[str, float | int]:
    forward_graph = np.asarray(forward.evidence["graph_selected_sample"], dtype=float)
    backward_graph = _sample(backward.evidence["graph_selected_sample"], rows)
    forward_visible = forward.samples >= 0
    backward_visible = _sample((backward.samples >= 0).astype(float), rows) >= 0.5
    both_graph = (forward_graph >= 0) & (backward_graph >= 0)
    both_visible = forward_visible & backward_visible
    graph_error = np.abs(forward_graph[both_graph] - backward_graph[both_graph])
    visible_error = np.abs(
        forward.samples[both_visible] - _sample(backward.samples, rows)[both_visible]
    )
    forward_canonical = np.asarray(
        forward.evidence["canonical_event_sample"], dtype=float
    )
    backward_canonical = _sample(
        backward.evidence["canonical_event_sample"], rows
    )
    both_canonical = (forward_canonical >= 0) & (backward_canonical >= 0)
    canonical_error = np.abs(
        forward_canonical[both_canonical] - backward_canonical[both_canonical]
    )
    return {
        "graph_compared_rows": int(np.sum(both_graph)),
        "graph_within_one_pulse_width_percent": (
            100.0 * float(np.mean(graph_error <= pulse_width))
            if len(graph_error)
            else 0.0
        ),
        "graph_median_absolute_sample_difference": (
            float(np.median(graph_error)) if len(graph_error) else float("nan")
        ),
        "both_visible_rows": int(np.sum(both_visible)),
        "both_visible_percent": 100.0 * float(np.mean(both_visible)),
        "visible_within_one_pulse_width_percent": (
            100.0 * float(np.mean(visible_error <= pulse_width))
            if len(visible_error)
            else 0.0
        ),
        "visible_median_absolute_sample_difference": (
            float(np.median(visible_error)) if len(visible_error) else float("nan")
        ),
        "canonical_compared_rows": int(np.sum(both_canonical)),
        "canonical_within_one_pulse_width_percent": (
            100.0 * float(np.mean(canonical_error <= pulse_width))
            if len(canonical_error)
            else 0.0
        ),
        "canonical_median_absolute_sample_difference": (
            float(np.median(canonical_error))
            if len(canonical_error)
            else float("nan")
        ),
    }


def _render(output: Path, forward, backward, rows, workbook):
    figure, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True, sharey=True)
    low, high = 145, 345
    extent = (forward[3][0], forward[3][-1], high, low)
    for ax, title, case, reverse_rows in (
        (axes[0], "Forward acquisition", forward, None),
        (axes[1], "Backward acquisition aligned to forward chainage", backward, rows),
    ):
        radar = case[2].radargram
        paths = case[4]
        if reverse_rows is not None:
            radar = np.stack(
                [
                    _sample(radar[:, sample], reverse_rows)
                    for sample in range(radar.shape[1])
                ],
                axis=1,
            )
        window = radar[:, low:high]
        limit = max(float(np.percentile(np.abs(window), 98.0)), 1e-7)
        ax.imshow(
            window.T,
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            vmin=-limit,
            vmax=limit,
            extent=extent,
        )
        colours = {1: "#00bde3", 2: "#ff5f8f"}
        for order, path in paths.items():
            samples = np.where(path.samples >= 0, path.samples, np.nan)
            if reverse_rows is not None:
                samples = _sample(samples, reverse_rows)
            ax.plot(forward[3], samples, color=colours[order], lw=1.1)
        for order, anchors in case[5].items():
            seed_x = [
                (
                    float(case[3][-1] - case[3][row])
                    if reverse_rows is not None
                    else float(case[3][row])
                )
                for row in anchors
            ]
            ax.scatter(
                seed_x,
                list(anchors.values()),
                s=34,
                facecolor="white",
                edgecolor=colours[order],
                linewidths=1.1,
                zorder=5,
            )
        for order, marker in ((1, "_"), (2, "x")):
            points = [
                item
                for item in workbook
                if item["layer_order"] == order and item["manual"]
            ]
            ax.scatter(
                [item["chainage_m"] for item in points],
                [item["sample"] for item in points],
                marker=marker,
                s=18,
                color="#59d86c",
                linewidths=0.8,
            )
        ax.set_title(title, loc="left")
        ax.set_ylabel("Flattened sample")
    axes[-1].set_xlabel("Forward chainage (m)")
    figure.suptitle(
        "Daska-Pasrur forward/back repeatability\n"
        "cyan asphalt, pink base, green legacy workbook (comparison only)",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exports/daska-repeatability-20260829",
    )
    parser.add_argument("--seed-count", type=int, choices=(3, 5), default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected_indices = (0, 2, 4) if args.seed_count == 3 else tuple(range(5))
    physical_seed_chainages = tuple(
        ALL_PHYSICAL_SEED_CHAINAGES[index] for index in selected_indices
    )
    forward_seed_samples = {
        order: tuple(values[index] for index in selected_indices)
        for order, values in ALL_FORWARD_SEED_SAMPLES.items()
    }
    backward_seed_samples = {
        order: tuple(values[index] for index in selected_indices)
        for order, values in ALL_BACKWARD_SEED_SAMPLES.items()
    }
    plate = DZTFile(PLATE)
    forward = _prepare(
        FORWARD,
        plate,
        forward_seed_samples,
        physical_seed_chainages,
        reverse=False,
    )
    backward = _prepare(
        BACKWARD,
        plate,
        backward_seed_samples,
        physical_seed_chainages,
        reverse=True,
    )
    rows = _aligned_backward(forward[3], backward[3])
    workbook = _workbook_samples(
        forward[1].reference_surface_sample,
        forward[0].header.sample_interval_ns,
    )
    metrics = {
        str(order): _path_metrics(forward[4][order], backward[4][order], rows, 7.0)
        for order in (1, 2)
    }
    summary = {
        "purpose": "independent-pass repeatability; not physical layer accuracy",
        "forward": str(FORWARD),
        "backward": str(BACKWARD),
        "forward_length_m": float(forward[3][-1]),
        "backward_length_m": float(backward[3][-1]),
        "metrics": metrics,
        "solver_used_workbook": False,
        "solver_used_design": False,
        "solver_used_seeds": True,
        "seed_source": "radar-only development clicks; not validation truth",
        "physical_seed_chainages_m": list(physical_seed_chainages),
        "forward_seed_samples": {
            str(order): list(values) for order, values in forward_seed_samples.items()
        },
        "backward_seed_samples": {
            str(order): list(values) for order, values in backward_seed_samples.items()
        },
        "field_accuracy_established": False,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _render(args.output / "paired-radargrams.png", forward, backward, rows, workbook)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
