"""Deterministic end-to-end warm correction runtime and isolation check (synthetic)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

# Imports below follow the explicit local source/test paths for standalone use.
# ruff: noqa: E402
from conftest import pulse, write_dzt

from gpr_layer_audit.conventional import backend_fingerprint, peak_process_memory, write_json
from gpr_layer_audit.models import AcquisitionFileSet, LayerSpec
from gpr_layer_audit.processing.conventional_config import resolve_config
from gpr_layer_audit.processing.pipeline import (
    AnalysisOptions,
    analyze_acquisition,
    retrack_segment,
)
from gpr_layer_audit.project import ProjectStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "exports/conventional/local-runtime-continued"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--include-subbase", action="store_true")
    args = parser.parse_args()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    data = np.tile(-4e6 * pulse(256, 70) + 8e5 * pulse(256, 97) - 5e5 * pulse(256, 134), (600, 1))
    if args.include_subbase:
        data += 4e5 * pulse(256, 175)
    data[:, :2] = 2e9
    road = directory / "SYNTHETIC_001.DZT"
    write_dzt(road, data, scans_per_meter=5)
    road.with_suffix(".DZX").write_text(
        "<DZX><DataCollection><system>SIR-30</system></DataCollection></DZX>"
    )
    options = AnalysisOptions(
        tracker_method="seed_hybrid",
        ml_policy="off",
        stack_size=2,
        validate_seed_dropout=False,
        auto_fine_retrack=False,
        layer_specs=LayerSpec.defaults()[: 3 if args.include_subbase else 2],
        conventional_config=json.loads(args.config.read_text()) if args.config else {},
        anchors={
            1: [(10.0, 97.0), (60.0, 97.0), (110.0, 97.0)],
            2: [(10.0, 134.0), (60.0, 134.0), (110.0, 134.0)],
        },
    )
    if args.include_subbase:
        options.anchors[3] = [(10.0, 175.0), (60.0, 175.0), (110.0, 175.0)]
    start = time.perf_counter()
    result = analyze_acquisition(AcquisitionFileSet(road), None, options)
    full = time.perf_counter() - start

    def outside():
        return json.dumps(
            [asdict(p) for p in result.picks if p.chainage_m < 35 or p.chainage_m > 85],
            sort_keys=True,
            default=str,
        )

    original = outside()
    timings = []
    for iteration in range(2):
        if iteration == 1:
            options.anchors[2][1] = (60.0, 135.0)
        start = time.perf_counter()
        retrack_segment(result, options, 35.0, 85.0)
        timings.append(time.perf_counter() - start)
        if outside() != original:
            raise AssertionError("Correction modified an outside observation")
    # Reopening preserves the versioned settings used by the UI's existing replay path.
    store_path = directory / f"runtime-{time.time_ns()}.gprproj"
    store = ProjectStore.create(store_path, "Conventional runtime")
    store.save_analysis(result)
    reopened = ProjectStore(store_path)
    if (
        resolve_config(reopened.latest_parameters()["conventional_config"]).fingerprint
        != resolve_config(result.parameters["conventional_config"]).fingerprint
    ):
        raise AssertionError("Reopened conventional settings changed")
    write_json(
        directory / "metrics.json",
        {
            "schema": "conventional-runtime-v1",
            "backend_source_sha256": backend_fingerprint(),
            "config": asdict(resolve_config(options.conventional_config)),
            "layers": [layer.order for layer in options.layer_specs],
            "data": "synthetic; 120 m",
            "full_road_s": full,
            "local_update_s": timings,
            "warm_local_update_s": timings[-1],
            "warm_update_includes_seed_edit": True,
            "target_s": 10,
            "target_met": timings[-1] <= 10,
            "correction_scope_m": [35, 85],
            "outside_observations_unchanged": True,
            "project_settings_reopened": True,
            "peak_process_memory_bytes": peak_process_memory(),
            "ml": "disabled",
        },
    )
    print(json.dumps({"full_road_s": full, "local_update_s": timings}), flush=True)


if __name__ == "__main__":
    main()
