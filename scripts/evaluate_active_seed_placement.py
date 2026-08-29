"""Audit radar-only follow-up seed placement on disclosed development roads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.pipeline import _chainage, _effective_stack
from gpr_layer_audit.processing.preprocessing import preprocess_for_interpretation
from gpr_layer_audit.processing.tracker import propose_seed_rows
from gpr_layer_audit.seeds import load_seed_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "exports/active-seed-placement-development-20260829"

CASES = (
    {
        "case_id": "talagang-two-base-observations",
        "road": ROOT / "GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT",
        "plate": ROOT
        / "GPR Data/talagang/TALAGANG METAL PLATE.PRJ/TALAGANG METAL PLATE_001.DZT",
        "seeds": ROOT / "benchmarks/talagang-development-seeds.json",
        "retained_base_station_indices": (0, 1),
        "withheld_base_station_index": None,
    },
    {
        "case_id": "pattoki-two-of-three-base-observations",
        "road": ROOT
        / "GPR Data/CENTRAL ZONE/PATTOKI MEGA ROAD TO JHORU VILLAGE.PRJ"
        / "PATTOKI MEGA ROAD TO JHORU VILLAGE_001.DZT",
        "plate": ROOT
        / "GPR Data/CENTRAL ZONE/METAL PLATE 11 AUGUST 2026.PRJ"
        / "METAL PLATE 11 AUGUST 2026_001.DZT",
        "seeds": ROOT / "benchmarks/pattoki-development-seeds.json",
        "retained_base_station_indices": (0, 2),
        "withheld_base_station_index": 1,
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_case(case: dict[str, object], output: Path) -> dict[str, object]:
    road_path, plate_path = case["road"], case["plate"]
    road, plate = DZTFile(road_path), DZTFile(plate_path)
    stack = _effective_stack(
        road.header.trace_count, 0, road.header.distance_per_trace_m
    )
    calibrated = calibrate(road, plate, stack_size=stack)
    dzx_path = road_path.with_suffix(".DZX")
    dzx = read_dzx(dzx_path) if dzx_path.exists() else None
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
    )
    candidate = interpreted.feature_branches["candidate"]

    _survey_id, stations = load_seed_file(case["seeds"])
    base_stations = [item for item in stations if item.visible_sample(2) is not None]
    retained = [base_stations[index] for index in case["retained_base_station_indices"]]
    occupied_rows = np.asarray(
        [
            int(np.argmin(np.abs(chainage - float(station.chainage_m))))
            for station in retained
        ],
        dtype=np.int32,
    )
    proposed_row = int(
        propose_seed_rows(candidate, count=1, occupied_rows=occupied_rows)[0]
    )
    proposed_chainage = float(chainage[proposed_row])
    withheld_index = case["withheld_base_station_index"]
    withheld_chainage = (
        float(base_stations[withheld_index].chainage_m)
        if withheld_index is not None
        else None
    )
    low = max(0, calibrated.reference_surface_sample + 20)
    high = min(interpreted.radargram.shape[1], calibrated.reference_surface_sample + 190)
    view = interpreted.radargram[:, low:high]
    limit = max(float(np.percentile(np.abs(view), 98.0)), 1e-9)
    figure, axis = plt.subplots(figsize=(17, 6), constrained_layout=True)
    spacing = float(np.median(np.diff(chainage))) if len(chainage) > 1 else 1.0
    axis.imshow(
        view.T,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        vmin=-limit,
        vmax=limit,
        extent=(
            float(chainage[0] - spacing / 2.0),
            float(chainage[-1] + spacing / 2.0),
            high - 0.5,
            low - 0.5,
        ),
    )
    for station in retained:
        axis.plot(
            station.chainage_m,
            station.samples[2],
            "o",
            color="#ffc857",
            markeredgecolor="black",
            markersize=7,
            zorder=4,
        )
    axis.axvline(proposed_chainage, color="#00b7c7", linewidth=1.6, linestyle="--")
    if withheld_chainage is not None:
        axis.axvline(withheld_chainage, color="#d946ef", linewidth=1.2, linestyle=":")
    axis.set(
        title=f"{case['case_id']} · radar-only follow-up placement",
        xlabel="Chainage (m)",
        ylabel="Surface-flattened sample index",
    )
    handles = [
        Line2D(
            [],
            [],
            color="#ffc857",
            marker="o",
            markeredgecolor="black",
            linestyle="none",
            label="Retained manual base observation",
        ),
        Line2D(
            [],
            [],
            color="#00b7c7",
            linestyle="--",
            label="Radar-only proposed follow-up",
        ),
    ]
    if withheld_chainage is not None:
        handles.append(
            Line2D(
                [],
                [],
                color="#d946ef",
                linestyle=":",
                label="Withheld development station (comparison only)",
            )
        )
    axis.legend(handles=handles, loc="upper right")
    image_name = f"{case['case_id']}-placement.png"
    figure.savefig(output / image_name, dpi=170)
    plt.close(figure)
    return {
        "case_id": case["case_id"],
        "counts_as_blind_release_evidence": False,
        "reference_workbook_opened": False,
        "retained_base_chainages_m": [float(item.chainage_m) for item in retained],
        "proposed_followup_chainage_m": proposed_chainage,
        "distance_to_nearest_retained_station_m": min(
            abs(proposed_chainage - float(item.chainage_m)) for item in retained
        ),
        "withheld_development_station_chainage_m": withheld_chainage,
        "distance_to_withheld_development_station_m": (
            abs(proposed_chainage - withheld_chainage)
            if withheld_chainage is not None
            else None
        ),
        "visualization": image_name,
        "road_sha256": _sha256(road_path),
        "plate_sha256": _sha256(plate_path),
        "seed_file_sha256": _sha256(case["seeds"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "purpose": "radar-only active follow-up seed placement development audit",
        "selection_uses_design": False,
        "selection_uses_reference_workbook": False,
        "implementation_sha256": {
            "tracker": _sha256(
                ROOT / "src/gpr_layer_audit/processing/tracker.py"
            ),
            "pipeline": _sha256(
                ROOT / "src/gpr_layer_audit/processing/pipeline.py"
            ),
            "audit_script": _sha256(Path(__file__)),
        },
        "cases": [_audit_case(case, output) for case in CASES],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
