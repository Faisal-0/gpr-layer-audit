"""Render radar-only seed sheets for roads reserved for blinded validation.

This script deliberately never imports or opens a reference workbook. Station
locations are fixed survey fractions, not selected from labels or tracker
output. Extrema labels describe the centre trace only; they are not proposed
interfaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import find_peaks

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.pipeline import _chainage, _effective_stack
from gpr_layer_audit.processing.preprocessing import preprocess_for_interpretation

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Case:
    case_id: str
    road: Path
    plate: Path


CASES = (
    Case(
        "jhang",
        ROOT / "GPR Data/CENTRAL ZONE/JHANG LOCAL ROAD TPV.PRJ/JHANG LOCAL ROAD TPV_001.DZT",
        ROOT
        / (
            "GPR Data/CENTRAL ZONE/JHANG METAL PLATE CALIBRATION TPV.PRJ/"
            "JHANG METAL PLATE CALIBRATION TPV_001.DZT"
        ),
    ),
    Case(
        "rawalpindi",
        ROOT / "GPR Data/NORTH ZONE/RAWALPINDI ROAD.PRJ/RAWALPINDI ROAD_001.DZT",
        ROOT
        / "GPR Data/NORTH ZONE/RAWALPINDI METAL PLATE.PRJ/RAWALPINDI METAL PLATE_001.DZT",
    ),
    Case(
        "burewala-vehari-001",
        ROOT / "GPR Data/SOUTH ZONE/BUREWALA VEHARI.PRJ/BUREWALA VEHARI_001.DZT",
        ROOT
        / (
            "GPR Data/SOUTH ZONE/BUREWALA VEHARI METAL PLATE.PRJ/"
            "BUREWALA VEHARI METAL PLATE_001.DZT"
        ),
    ),
)


def _render_case(case: Case, output: Path) -> dict[str, object]:
    road = DZTFile(case.road)
    plate = DZTFile(case.plate)
    stack = _effective_stack(
        road.header.trace_count, 0, road.header.distance_per_trace_m
    )
    calibrated = calibrate(road, plate, stack_size=stack)
    dzx_path = case.road.with_suffix(".DZX")
    metadata = read_dzx(dzx_path) if dzx_path.exists() else None
    chainage = _chainage(road.header, calibrated.trace_centres, metadata)
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
    )
    radar = interpreted.radargram
    stations = [
        float(chainage[0] + fraction * (chainage[-1] - chainage[0]))
        for fraction in (0.15, 0.50, 0.85)
    ]
    low = max(0, calibrated.reference_surface_sample + 5)
    high = min(radar.shape[1], calibrated.reference_surface_sample + 195)
    figure, axes = plt.subplots(3, 2, figsize=(16, 13), gridspec_kw={"width_ratios": [4, 1]})
    records = []
    for row_index, station in enumerate(stations):
        row = int(np.argmin(np.abs(chainage - station)))
        selected = np.abs(chainage - chainage[row]) <= 30.0
        window = radar[selected, low:high]
        limit = max(float(np.percentile(np.abs(window), 98.5)), 1e-8)
        image_axis, trace_axis = axes[row_index]
        image_axis.imshow(
            window.T,
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            vmin=-limit,
            vmax=limit,
            extent=(chainage[selected][0], chainage[selected][-1], high, low),
        )
        image_axis.axvline(chainage[row], color="#00d4ff", lw=1.1, ls="--")
        image_axis.set_title(
            f"Station {row_index + 1} · {chainage[row]:.1f} m · radar only",
            loc="left",
        )
        image_axis.set_ylabel("Surface-flattened sample")
        trace = radar[row, low:high]
        sample_axis = np.arange(low, high)
        trace_axis.plot(trace, sample_axis, color="#17243a", lw=1.0)
        trace_axis.axvline(0.0, color="0.7", lw=0.7)
        extrema, _ = find_peaks(np.abs(trace), distance=4)
        strongest = extrema[np.argsort(np.abs(trace[extrema]))[-14:]]
        strongest = strongest[np.argsort(strongest)]
        trace_axis.scatter(trace[strongest], sample_axis[strongest], s=14, color="#e45826")
        for index in strongest:
            trace_axis.annotate(
                str(int(sample_axis[index])),
                (trace[index], sample_axis[index]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=7,
                va="center",
            )
        trace_axis.set_ylim(high, low)
        trace_axis.set_title("Centre trace extrema", loc="left")
        trace_axis.set_xlabel("Signed amplitude")
        records.append(
            {
                "station_number": row_index + 1,
                "chainage_m": float(chainage[row]),
                "stacked_row": row,
                "labeled_extrema_samples": [
                    int(value) for value in sample_axis[strongest]
                ],
                "labeled_extrema": [
                    {
                        "sample": int(sample_axis[index]),
                        "signed_amplitude": float(trace[index]),
                        "polarity": int(np.sign(trace[index])),
                    }
                    for index in strongest
                ],
            }
        )
    axes[-1, 0].set_xlabel("Chainage (m)")
    figure.suptitle(
        f"{case.case_id} · blinded radar-only seed sheet\n"
        "Choose the same physical interface/lobe at each station; use 'not visible' when uncertain",
        fontsize=16,
    )
    figure.text(
        0.06,
        0.015,
        "No workbook, design thickness, tracker path, or reference overlay was loaded. "
        "Orange labels are waveform extrema, not interface recommendations.",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.94))
    image_path = output / f"{case.case_id}-seed-sheet.png"
    figure.savefig(image_path, dpi=170)
    plt.close(figure)
    return {
        "case_id": case.case_id,
        "road": str(case.road),
        "plate": str(case.plate),
        "stack_size": stack,
        "reference_surface_sample": calibrated.reference_surface_sample,
        "station_rule": "15%, 50%, and 85% of recorded chainage",
        "reference_workbook_opened": False,
        "design_used": False,
        "tracker_path_used": False,
        "image": str(image_path),
        "stations": records,
    }


def main() -> int:
    output = ROOT / "exports/blinded-seed-sheets-20260829"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "radar-only seed selection before opening validation references",
        "cases": [_render_case(case, output) for case in CASES],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
