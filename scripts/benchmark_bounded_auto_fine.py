"""Benchmark the bounded automatic fine-refinement policy on disclosed radar.

The reference workbook is never opened. Frozen radar-only clicks are unchanged;
an independent preview supplies only the event metadata captured by the UI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from benchmark_seed_dropout_parallelism import (
    DEFAULT_MANIFEST,
    ROOT,
    _attach_preview_metadata,
    _case,
    _run,
    _stations,
)

from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import AcquisitionFileSet
from gpr_layer_audit.processing import AnalysisOptions

DEFAULT_CASE = "sohal-kalan-gujrat-second-portion-001-validation"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = args.manifest.resolve()
    _, case = _case(manifest, args.case)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    road = (ROOT / case["road"]).resolve()
    plate = (ROOT / case["plate"]).resolve()
    source = AcquisitionFileSet(road)
    plate_source = AcquisitionFileSet(plate)
    stations = _stations(case)
    enabled_orders = set(stations[0].samples)
    layer_specs = [
        replace(layer, analysis_enabled=layer.order in enabled_orders)
        for layer in AnalysisOptions().layer_specs
    ]

    preview, preview_seconds = _run(
        source,
        plate_source,
        AnalysisOptions(
            survey_id=case["case_id"],
            layer_specs=layer_specs,
            auto_fine_retrack=False,
            validate_seed_dropout=False,
        ),
        "preview",
    )
    metadata_audit = _attach_preview_metadata(stations, preview)
    result, elapsed_seconds = _run(
        source,
        plate_source,
        AnalysisOptions(
            survey_id=case["case_id"],
            layer_specs=layer_specs,
            seed_stations=stations,
            auto_fine_retrack=True,
            max_auto_fine_regions=1,
            validate_seed_dropout=False,
        ),
        "bounded",
    )

    plan = result.parameters["automatic_fine_retrack_plan"]
    segments = result.parameters["fine_retracked_segments"]
    windows = plan["selected_windows_m"]
    policy_valid = bool(
        plan["policy"] == "bounded_high_information"
        and len(windows) <= 1
        and all(end - start <= 10.0 + 1e-9 for start, end in windows)
        and len(segments) == len(windows)
    )
    summary = {
        "purpose": "development-only bounded automatic fine-refinement benchmark",
        "field_accuracy_established": False,
        "reference_opened": False,
        "case_id": case["case_id"],
        "frozen_seed_chainages_and_samples_unchanged": True,
        "seed_metadata_audit": metadata_audit,
        "preview_seconds": preview_seconds,
        "bounded_seeded_run_seconds": elapsed_seconds,
        "policy_valid": policy_valid,
        "fine_plan": plan,
        "fine_segments": segments,
        "remaining_review_regions": len(result.review_issues),
        "remaining_seed_requests": [
            asdict(item) for item in result.proposed_seed_requests
        ],
        "required_seed_orders": result.parameters["required_seed_orders"],
        "input_fingerprints": {
            "road": fingerprint_file(road),
            "plate": fingerprint_file(plate),
            "seed_manifest": fingerprint_file(manifest),
            "benchmark_script": fingerprint_file(Path(__file__)),
            "pipeline": fingerprint_file(
                ROOT / "src/gpr_layer_audit/processing/pipeline.py"
            ),
            "joint_graph": fingerprint_file(
                ROOT / "src/gpr_layer_audit/processing/joint_graph.py"
            ),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if policy_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
