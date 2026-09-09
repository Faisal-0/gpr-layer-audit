"""Render representative tracked radargram sections around frozen manual seeds."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from gpr_layer_audit.io import DZTFile, read_dzx
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.pipeline import _chainage, _effective_stack
from gpr_layer_audit.processing.preprocessing import preprocess_for_interpretation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports/tracked-road-sections-20260901"


def _manifest_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def _load_radar(case: dict) -> tuple[np.ndarray, np.ndarray]:
    road_path = ROOT / case["road"]
    plate_path = ROOT / case["plate"]
    road = DZTFile(road_path)
    plate = DZTFile(plate_path)
    stack = _effective_stack(road.header.trace_count, 0, road.header.distance_per_trace_m)
    calibrated = calibrate(road, plate, stack_size=stack)
    dzx = read_dzx(road_path.with_suffix(".DZX"))
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
    )
    return chainage, interpreted.radargram


def _path_arrays(path: Path) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    data = np.load(path)
    chainage = data["chainage_m"]
    accepted: dict[int, np.ndarray] = {}
    proposed: dict[int, np.ndarray] = {}
    for order in (1, 2, 3):
        for key in (f"layer_{order}_samples", f"layer_{order}_sample"):
            if key in data:
                accepted[order] = np.asarray(data[key], dtype=float)
                break
        for key in (f"layer_{order}_graph", f"layer_{order}_raw_graph"):
            if key in data:
                proposed[order] = np.asarray(data[key], dtype=float)
                break
    return chainage, accepted, proposed


def _render(case: dict, path_file: Path) -> Path:
    path_x, accepted, proposed = _path_arrays(path_file)
    if "radargram" in np.load(path_file).files:
        stored = np.load(path_file)
        radar_x, radar = stored["chainage_m"], stored["radargram"]
    else:
        radar_x, radar = _load_radar(case)
    stations = case["stations"]
    count = len(stations)
    fig, axes = plt.subplots(count, 1, figsize=(15, 3.25 * count), squeeze=False)
    colors = {1: "#00e5ff", 2: "#ffb000", 3: "#e95cff"}
    labels = {1: "asphalt bottom", 2: "base bottom", 3: "subbase bottom"}
    for panel, station in enumerate(stations):
        ax = axes[panel, 0]
        centre = float(station["chainage_m"])
        half = min(55.0, max(8.0, 0.12 * (float(radar_x[-1]) - float(radar_x[0]))))
        mask = (radar_x >= centre - half) & (radar_x <= centre + half)
        low, high = 155, min(340, radar.shape[1])
        window = radar[mask, low:high]
        limit = max(float(np.percentile(np.abs(window), 98.5)), 1e-8)
        ax.imshow(
            window.T,
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            vmin=-limit,
            vmax=limit,
            extent=(radar_x[mask][0], radar_x[mask][-1], high, low),
        )
        path_mask = (path_x >= radar_x[mask][0]) & (path_x <= radar_x[mask][-1])
        for order in sorted(set(accepted) | set(proposed)):
            if order in proposed:
                y = proposed[order][path_mask].copy()
                y[y < 0] = np.nan
                ax.plot(path_x[path_mask], y, color=colors[order], lw=1.0, ls=(0, (2, 2)), alpha=0.72)
            if order in accepted:
                y = accepted[order][path_mask].copy()
                y[y < 0] = np.nan
                ax.plot(path_x[path_mask], y, color=colors[order], lw=2.0, label=labels[order] if panel == 0 else None)
            if str(order) in station["samples"]:
                ax.scatter(
                    [centre],
                    [float(station["samples"][str(order)])],
                    s=72,
                    facecolor=colors[order],
                    edgecolor="white",
                    linewidth=1.3,
                    zorder=8,
                )
        ax.axvline(centre, color="white", lw=0.8, ls="--", alpha=0.65)
        ax.set_ylabel("Sample")
        ax.set_title(f"Manual seed station {panel + 1} · {centre:.1f} m", loc="left", fontsize=11)
        ax.grid(False)
    axes[-1, 0].set_xlabel("Chainage (m)")
    if accepted:
        axes[0, 0].legend(loc="upper right", ncol=min(3, len(accepted)), framealpha=0.85)
    fig.suptitle(
        f"{case['case_id']} · {count} manual stations · solid accepted / dotted proposal",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{case['case_id']}-tracked-sections.jpg"
    fig.savefig(target, dpi=135, pil_kwargs={"quality": 91})
    plt.close(fig)
    return target


def main() -> int:
    primary = _manifest_cases(ROOT / "benchmarks/blinded-radar-only-seeds-20260829.json")
    deep = _manifest_cases(ROOT / "benchmarks/blinded-base-subbase-seeds-20260829.json")
    cases = {case["case_id"]: case for case in primary + deep}
    for case_id, filename in (
        ("talagang", "talagang-development-seeds.json"),
        ("pattoki-jhoru", "pattoki-development-seeds.json"),
    ):
        raw = json.loads((ROOT / "benchmarks" / filename).read_text(encoding="utf-8"))
        cases[case_id] = {
            "case_id": case_id,
            "stations": [
                {
                    "chainage_m": station["chainage_m"],
                    "samples": {
                        order: pick["sample_index"]
                        for order, pick in station["picks"].items()
                        if pick.get("visibility") == "visible"
                    },
                }
                for station in raw["stations"]
            ],
        }
    paths = {
        "jhang": "exports/blinded-multiroad-strict-scale-20260829/jhang-paths.npz",
        "rawalpindi": "exports/blinded-multiroad-strict-scale-20260829/rawalpindi-paths.npz",
        "burewala-vehari-001": "exports/blinded-multiroad-strict-scale-20260829/burewala-vehari-001-paths.npz",
        "bahawalpur-local-road-sub-engr-001": "exports/bahawalpur-direct-lineage-transfer-20260901/bahawalpur-local-road-sub-engr-001-paths.npz",
        "jamshoro-survey-files-001": "exports/blinded-multiroad-strict-scale-20260829/jamshoro-survey-files-001-paths.npz",
        "mandiali-to-puranalongwala-001": "exports/blinded-multiroad-strict-scale-20260829/mandiali-to-puranalongwala-001-paths.npz",
        "sohl-kalan-gujrat-001": "exports/blinded-multiroad-strict-scale-20260829/sohl-kalan-gujrat-001-paths.npz",
        "bahawalpur-local-road-sub-engr-002": "exports/blinded-multiroad-strict-scale-20260829/bahawalpur-local-road-sub-engr-002-paths.npz",
        "daska-pasrur-forward-development": "exports/blinded-multiroad-strict-scale-20260829/daska-pasrur-forward-development-paths.npz",
        "burewala-vehari-002-base-validation": "exports/blinded-base-subbase-strict-scale-20260829/burewala-vehari-002-base-validation-paths.npz",
        "sohal-kalan-gujrat-second-portion-001-validation": "exports/blinded-base-subbase-strict-scale-20260829/sohal-kalan-gujrat-second-portion-001-validation-paths.npz",
        "talagang": "exports/seed-assisted-v17-20260827/talagang/tracking.npz",
        "pattoki-jhoru": "exports/seed-assisted-v17-20260827/pattoki/tracking.npz",
    }
    rendered = []
    for case_id, relative in paths.items():
        rendered.append(str(_render(cases[case_id], ROOT / relative)))
    print(json.dumps(rendered, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
