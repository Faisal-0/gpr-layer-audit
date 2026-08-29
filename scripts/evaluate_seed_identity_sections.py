"""Bounded field-data check for seed identity; not physical validation.

Runs the same cached-resolution tracker twice on one Talagang and one Pattoki
section: once without the persistent-packet identity guard and once with it.
Workbooks are loaded only after both solver variants finish and are plotted as
comparison-only marks. The script also renders the retained paths on the
surface-flattened, dewow-only road signal before plate subtraction and on the
processed interpretation branch.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import hilbert

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.pipeline import _chainage, _effective_stack
from gpr_layer_audit.processing.preprocessing import (
    measurement_packet_support,
    preprocess_for_interpretation,
)
from gpr_layer_audit.processing.seed_graph import pick_seed_conditioned_interfaces
from gpr_layer_audit.reference import normalize_reference_workbook

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    name: str
    road: Path
    plate: Path
    workbook: Path
    window_m: tuple[float, float]
    seed_chainages_m: tuple[float, ...]
    asphalt_samples: tuple[int, ...] | None
    base_samples: tuple[int, ...]
    persistent_sample: int
    pulse_width_samples: float
    plot_samples: tuple[int, int]


CASES = (
    Case(
        "Talagang — deeper candidate interpretation",
        ROOT / "GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT",
        ROOT / "GPR Data/talagang/TALAGANG METAL PLATE.PRJ/TALAGANG METAL PLATE_001.DZT",
        ROOT / "GPR Data/talagang/Talagang_layer_thickness_.xlsx",
        (950.0, 1250.0),
        (1000.0, 1100.0, 1200.0),
        None,
        (290, 273, 293),
        249,
        14.0,
        (235, 320),
    ),
    Case(
        "Pattoki-Jhoru — seeded persistent interpretation",
        ROOT
        / (
            "GPR Data/CENTRAL ZONE/PATTOKI MEGA ROAD TO JHORU VILLAGE.PRJ/"
            "PATTOKI MEGA ROAD TO JHORU VILLAGE_001.DZT"
        ),
        ROOT
        / "GPR Data/CENTRAL ZONE/METAL PLATE 11 AUGUST 2026.PRJ/METAL PLATE 11 AUGUST 2026_001.DZT",
        ROOT / "GPR Data/CENTRAL ZONE/kasur_pattoki_jhoru_road_thickness with_graphs.xlsx",
        (300.0, 750.0),
        (325.675, 728.65),
        (186, 190),
        (278, 282),
        280,
        7.0,
        (235, 325),
    ),
)


def _no_exclusions(radargram, candidates, *_args, **_kwargs):
    return np.zeros(candidates.shape, dtype=bool), np.zeros(radargram.shape, dtype=np.float32)


def _run(case: Case):
    road = DZTFile(case.road)
    plate = DZTFile(case.plate)
    stack = _effective_stack(road.header.trace_count, 0, road.header.distance_per_trace_m)
    calibrated = calibrate(road, plate, stack_size=stack)
    dzx_path = case.road.with_suffix(".DZX")
    dzx = read_dzx(dzx_path) if dzx_path.exists() else None
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    processed = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
    ).radargram
    measurement = calibrated.measurement_radargram
    chosen = (chainage >= case.window_m[0]) & (chainage <= case.window_m[1])
    x = chainage[chosen]
    radar = processed[chosen].copy()
    raw = measurement[chosen].copy()
    rows = [int(np.argmin(np.abs(x - value))) for value in case.seed_chainages_m]
    base = dict(zip(rows, case.base_samples, strict=True))
    if case.asphalt_samples is None:
        # Match the frozen bounded Talagang experiment: use the saved v17
        # asphalt observation only to enforce layer ordering at the three
        # candidate-base stations. These are not new manual asphalt labels.
        with np.load(ROOT / "exports/seed-assisted-v17-20260827/talagang/tracking.npz") as saved:
            saved_x = saved["chainage_m"]
            saved_asphalt = saved["layer_1_display"]
            asphalt = {
                row: int(saved_asphalt[int(np.argmin(np.abs(saved_x - x[row])))]) for row in rows
            }
    else:
        asphalt = dict(zip(rows, case.asphalt_samples, strict=True))
    measurement_support = measurement_packet_support(
        raw, pulse_width_samples=case.pulse_width_samples
    )
    arguments = dict(
        radargram=radar,
        reference_surface_sample=calibrated.reference_surface_sample,
        layers=LayerSpec.defaults()[:2],
        anchor_samples={1: asphalt, 2: base},
        search_corridors={},
        design_weight=0.0,
        pulse_width_samples=case.pulse_width_samples,
        break_rows=set(),
        anomaly_mask=np.zeros(len(x), dtype=bool),
        max_interpolation_rows=0,
        horizontal_step_m=float(np.median(np.diff(x))),
    )
    with patch(
        "gpr_layer_audit.processing.seed_identity.stationary_competitor_mask",
        _no_exclusions,
    ):
        before = pick_seed_conditioned_interfaces(
            **arguments,
            feature_branches={"measurement_support": measurement_support.copy()},
        )[2]
    after = pick_seed_conditioned_interfaces(
        **arguments,
        feature_branches={"measurement_support": measurement_support.copy()},
    )[2]
    return road.header, calibrated.reference_surface_sample, x, raw, radar, rows, before, after


def _path_metrics(path, raw, plot_samples, persistent_sample):
    samples = path.samples
    graph = path.evidence["graph_selected_sample"]
    visible = samples >= 0
    envelope = maximum_filter1d(np.abs(hilbert(raw, axis=1)), 7, axis=1)
    low, high = plot_samples
    local = envelope[:, low:high]
    columns = np.clip(samples, 0, raw.shape[1] - 1).astype(int)
    strength = envelope[np.arange(len(raw)), columns]
    raw_rank = np.mean(local <= strength[:, None], axis=1)
    measurement_support = np.asarray(path.evidence["measurement_support"], dtype=float)
    return {
        "retained_percent": 100.0 * float(np.mean(visible)),
        "explicit_gap_percent": 100.0 * float(np.mean(~visible)),
        "raw_graph_near_persistent_packet_percent": 100.0
        * float(np.mean((graph >= 0) & (np.abs(graph - persistent_sample) <= 5))),
        "retained_near_persistent_packet_percent": 100.0
        * float(np.mean(visible & (np.abs(samples - persistent_sample) <= 5))),
        "retained_raw_packet_rank_median": (
            float(np.median(raw_rank[visible])) if np.any(visible) else None
        ),
        "retained_raw_packet_rank_below_0_5_percent": (
            100.0 * float(np.mean(raw_rank[visible] < 0.5)) if np.any(visible) else None
        ),
        "retained_measurement_support_median": (
            float(np.median(measurement_support[visible])) if np.any(visible) else None
        ),
        "retained_measurement_support_p10": (
            float(np.percentile(measurement_support[visible], 10)) if np.any(visible) else None
        ),
    }


def _workbook_samples(case, x, surface, sample_interval_ns):
    mm_per_sample = sample_interval_ns * 299.792458 / (2 * np.sqrt(7.0))
    return [
        (
            point.chainage_m,
            surface + point.cumulative_depth_mm / mm_per_sample,
            point.label_origin.value != "manual",
        )
        for point in normalize_reference_workbook(case.workbook)
        if point.layer_order == 2 and x[0] <= point.chainage_m <= x[-1]
    ]


def _workbook_disagreement(path, x, workbook, sample_interval_ns):
    mm_per_sample = sample_interval_ns * 299.792458 / (2 * np.sqrt(7.0))
    canonical = path.evidence["canonical_event_sample"]
    differences = []
    half_bin = float(np.median(np.diff(x))) / 2
    for chainage_m, workbook_sample, derived in workbook:
        if derived:
            continue
        row = int(np.argmin(np.abs(x - chainage_m)))
        if abs(float(x[row]) - chainage_m) <= half_bin and path.samples[row] >= 0:
            differences.append(abs(float(canonical[row]) - workbook_sample) * mm_per_sample)
    return {
        "manual_workbook_points_compared": len(differences),
        "manual_workbook_median_absolute_disagreement_mm": (
            float(np.median(differences)) if differences else None
        ),
    }


def _radar(ax, x, values, limits, sample_window):
    low, high = sample_window
    window = values[:, low:high]
    limit = max(float(np.percentile(np.abs(window), limits)), 1e-9)
    dx = float(np.median(np.diff(x)))
    ax.imshow(
        window.T,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        vmin=-limit,
        vmax=limit,
        extent=(x[0] - dx / 2, x[-1] + dx / 2, high - 0.5, low - 0.5),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "exports/independent-takeover-20260827/seed-identity-sections",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    completed = []
    for case in CASES:
        values = _run(case)
        header, surface, x, raw, processed, rows, before, after = values
        workbook = _workbook_samples(case, x, surface, header.sample_interval_ns)
        before_metrics = _path_metrics(before, raw, case.plot_samples, case.persistent_sample)
        after_metrics = _path_metrics(after, raw, case.plot_samples, case.persistent_sample)
        before_metrics.update(
            _workbook_disagreement(before, x, workbook, header.sample_interval_ns)
        )
        after_metrics.update(
            _workbook_disagreement(after, x, workbook, header.sample_interval_ns)
        )
        record = {
            "case": case.name,
            "purpose": "seed-adherence mechanism check, not physical validation",
            "window_m": [float(x[0]), float(x[-1])],
            "seeds": [
                {"chainage_m": float(x[row]), "sample": int(sample)}
                for row, sample in zip(rows, case.base_samples, strict=True)
            ],
            "before": before_metrics,
            "after": after_metrics,
            "changed_retained_rows": int(np.sum(before.samples != after.samples)),
            "seed_samples_preserved": bool(
                all(
                    after.samples[row] == sample
                    for row, sample in zip(rows, case.base_samples, strict=True)
                )
            ),
            "field_accuracy_established": False,
        }
        records.append(record)
        completed.append((case, *values, workbook))
        print(json.dumps(record, indent=2), flush=True)

    figure, axes = plt.subplots(2, 2, figsize=(18, 11), sharex="row", sharey="row")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.12, hspace=0.25, wspace=0.06)
    for row_index, item in enumerate(completed):
        case, _header, _surface, x, _raw, processed, seed_rows, before, after, workbook = item
        exclusion = after.candidate_components.get("seed_mismatched_persistent_packet")
        for column, (label, path) in enumerate((("Before", before), ("After", after))):
            ax = axes[row_index, column]
            _radar(ax, x, processed, 97, case.plot_samples)
            display = np.where(path.samples >= 0, path.samples, np.nan)
            ax.plot(x, display, color="#00bde3", lw=1.4)
            rejected = (path.evidence["graph_selected_sample"] >= 0) & (path.samples < 0)
            ax.scatter(
                x[rejected], path.evidence["graph_selected_sample"][rejected], s=3, color="#e500a4"
            )
            gaps = path.samples < 0
            ax.scatter(
                x[gaps],
                np.full(np.sum(gaps), case.plot_samples[1] - 2),
                marker="|",
                s=11,
                color="#d23434",
                alpha=0.45,
            )
            if column == 1 and exclusion is not None:
                baseline = before.evidence["graph_selected_sample"].astype(int)
                valid = baseline >= 0
                excluded = valid & (
                    exclusion[np.arange(len(x)), np.clip(baseline, 0, exclusion.shape[1] - 1)] > 0
                )
                ax.scatter(
                    x[excluded], baseline[excluded], marker="x", s=8, color="#e500a4", alpha=0.7
                )
            manual = [(a, b) for a, b, derived in workbook if not derived]
            derived = [(a, b) for a, b, is_derived in workbook if is_derived]
            if manual:
                ax.scatter(
                    *zip(*manual, strict=True), marker="_", s=42, color="#25d05f", linewidths=1.2
                )
            if derived:
                ax.scatter(
                    *zip(*derived, strict=True), marker="x", s=18, color="#25d05f", linewidths=1.0
                )
            ax.scatter(
                x[seed_rows],
                case.base_samples,
                s=55,
                facecolor="white",
                edgecolor="#006c91",
                zorder=6,
            )
            metrics = records[row_index][label.lower()]
            title = (
                f"{label} | retained {metrics['retained_percent']:.1f}% | "
                f"gaps {metrics['explicit_gap_percent']:.1f}%"
            )
            ax.set_title(
                title,
                loc="left",
            )
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(case.plot_samples[1] - 0.5, case.plot_samples[0] - 0.5)
            ax.set_xlabel("Chainage (m)")
            if column == 0:
                ax.set_ylabel(f"{case.name}\nFlattened sample")
    figure.suptitle("Seed identity on bounded field sections", fontsize=18, y=0.97)
    figure.legend(
        handles=[
            Line2D([], [], color="#00bde3", lw=2, label="Retained tracked event"),
            Line2D(
                [],
                [],
                color="#e500a4",
                marker=".",
                ls="none",
                label="Rejected/competing graph event",
            ),
            Line2D([], [], color="#d23434", marker="|", ls="none", label="Explicit gap row"),
            Line2D([], [], color="#006c91", marker="o", mfc="white", ls="none", label="Seed"),
            Line2D(
                [],
                [],
                color="#25d05f",
                marker="_",
                ls="none",
                label="Workbook comparison only (× derived)",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.06,
        0.065,
        "Talagang seeds are assistant-selected hypotheses, not confirmed interfaces. "
        "Pattoki uses supplied user-confirmed development seeds.",
    )
    figure.text(
        0.06,
        0.037,
        "Background is processed radar. Lines break at no-picks; workbooks are "
        "plotted only after tracking and never enter scoring.",
    )
    figure.savefig(args.output / "before-after-overlays.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(18, 11), sharex="row", sharey="row")
    figure.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.11, hspace=0.24, wspace=0.06)
    for row_index, item in enumerate(completed):
        case, _header, _surface, x, raw, processed, seed_rows, _before, after, _workbook = item
        for column, (label, radar, percentile) in enumerate(
            (
                ("Dewow-only road signal before plate subtraction", raw, 99),
                ("Processed interpretation branch", processed, 97),
            )
        ):
            ax = axes[row_index, column]
            _radar(ax, x, radar, percentile, case.plot_samples)
            ax.plot(x, np.where(after.samples >= 0, after.samples, np.nan), color="#00bde3", lw=1.4)
            ax.scatter(
                x[seed_rows],
                case.base_samples,
                s=55,
                facecolor="white",
                edgecolor="#006c91",
                zorder=6,
            )
            ax.set_title(label, loc="left")
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(case.plot_samples[1] - 0.5, case.plot_samples[0] - 0.5)
            ax.set_xlabel("Chainage (m)")
            if column == 0:
                ax.set_ylabel(f"{case.name}\nFlattened sample")
    figure.suptitle("Retained events in measurement and processed branches", fontsize=18, y=0.97)
    figure.text(
        0.06,
        0.052,
        "Identical retained path over both branches. Visibility in the processed "
        "panel alone is not proof of an event in the road measurement.",
    )
    figure.text(
        0.06,
        0.025,
        "The left branch is surface-flattened and dewow-only, before the "
        "gain-incompatible plate waveform is subtracted; no workbook is shown.",
    )
    figure.savefig(args.output / "raw-processed-evidence.png", dpi=170)
    plt.close(figure)
    (args.output / "summary.json").write_text(json.dumps(records, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
