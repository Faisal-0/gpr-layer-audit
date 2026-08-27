"""Radar-only development diagnostics. Never loads an interpretation workbook.

Examples:
  uv run python scripts/diagnose_tracking.py --output exports/joint-diagnostic
  uv run python scripts/diagnose_tracking.py --output exports/dropout --drop-one-seed

Existing development seeds are not relabeled as blinded ground truth. This
script measures solver behavior and stability, not field accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from gpr_layer_audit.design import quick_layer_designs
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import AcquisitionFileSet, PickStatus
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition
from gpr_layer_audit.seeds import load_seed_file


def _snapshot(result, output, elapsed):
    arrays = {"chainage_m": result.chainage_m, "radargram": result.calibrated_radargram}
    summary = {
        "elapsed_seconds": elapsed,
        "layers": {},
        "proposed_seed_chainages_m": list(result.proposed_seed_chainages),
        "field_accuracy_established": False,
    }
    for order in sorted({item.layer_order for item in result.picks}):
        picks = sorted(
            (item for item in result.picks if item.layer_order == order),
            key=lambda item: item.chainage_m,
        )
        prefix = f"layer_{order}_"
        for name, values in {
            "sample": [item.sample_index for item in picks],
            "display": [
                item.selected_lobe_sample if item.selected_lobe_sample is not None else -1
                for item in picks
            ],
            "raw_graph": [item.evidence.graph_selected_sample for item in picks],
            "confidence": [item.confidence for item in picks],
            "pre_gate_confidence": [item.evidence.pre_gate_confidence for item in picks],
            "automatic": [item.status == PickStatus.HIGH_CONFIDENCE for item in picks],
            "family": [item.evidence.event_family_index for item in picks],
        }.items():
            arrays[prefix + name] = np.asarray(values)
        for field in (
            "signal_score",
            "absolute_strength",
            "seed_correlation",
            "phase_score",
            "coherence_score",
            "candidate_margin",
            "tracklet_support",
            "preprocessing_agreement",
            "forward_backward_agreement",
            "joint_hypothesis_support",
            "neighborhood_support",
            "waveform_similarity",
            "drop_seed_stability",
            "cycle_slip_risk",
            "spatial_lineage_index",
            "seed_reachable",
            "lineage_break",
            "seed_position_conflict",
        ):
            arrays[prefix + field] = np.asarray([getattr(item.evidence, field) for item in picks])
        events = [item for item in result.candidate_events if item.layer_order == order]
        for field in (
            "chainage_m",
            "sample_index",
            "canonical_sample_index",
            "rank",
            "radar_score",
            "graph_selected",
            "joint_hypothesis_count",
        ):
            arrays[prefix + "candidate_" + field] = np.asarray(
                [getattr(item, field) for item in events]
            )
        for field in ("spatial_lineage_id", "seed_reachable"):
            arrays[prefix + "candidate_" + field] = np.asarray([
                getattr(item, field) if getattr(item, field) is not None else -1 for item in events
            ], dtype=np.int32)
        summary["layers"][order] = {
            "status_counts": dict(Counter(str(item.status) for item in picks)),
            "automatic_fraction": float(np.mean(arrays[prefix + "automatic"])),
            "graph_coverage": float(np.mean(arrays[prefix + "raw_graph"] >= 0)),
            "visible_fraction": float(np.mean(arrays[prefix + "display"] >= 0)),
            "candidate_count": len(events),
        }
    np.savez_compressed(output / "tracking.npz", **arrays)
    (output / "manifest.json").write_text(json.dumps(result.manifest(), indent=2, default=str))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return arrays, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--road", type=Path, default=Path("GPR Data/talagang/TALAGANG.PRJ/TALAGANG_001.DZT")
    )
    parser.add_argument(
        "--plate",
        type=Path,
        default=Path("GPR Data/talagang/TALAGANG METAL PLATE.PRJ/TALAGANG METAL PLATE_001.DZT"),
    )
    parser.add_argument(
        "--seeds", type=Path, default=Path("benchmarks/talagang-development-seeds.json")
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asphalt", default="2in")
    parser.add_argument("--base", default="4in")
    parser.add_argument("--dielectric", type=float, default=7.0)
    parser.add_argument("--stack", type=int, default=0)
    parser.add_argument("--fine", action="store_true")
    parser.add_argument(
        "--validate-seeds",
        action="store_true",
        help="Run optional seed-withholding diagnostics (not required for seeded operation)",
    )
    parser.add_argument("--drop-one-seed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    survey_id, stations = load_seed_file(args.seeds)
    options = AnalysisOptions(
        survey_id=survey_id,
        seed_stations=stations,
        stack_size=args.stack,
        layer_designs=quick_layer_designs(
            args.asphalt, args.base, None, dielectric=args.dielectric
        ),
        auto_fine_retrack=args.fine,
        validate_seed_dropout=args.validate_seeds,
    )
    # No field evidence exists for subbase in this diagnostic. Do not let an
    # unconstrained third-interface hypothesis alter the asphalt/base experiment.
    options.layer_specs = [
        replace(layer, analysis_enabled=layer.order <= 2) for layer in options.layer_specs
    ]
    code_diff = subprocess.check_output(["git", "diff", "HEAD", "--", "src", "scripts"])
    source_hashes = {
        "road": fingerprint_file(args.road),
        "plate": fingerprint_file(args.plate),
        "seeds": fingerprint_file(args.seeds),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "tracked_diff_sha256": hashlib.sha256(code_diff).hexdigest(),
        # Include new/untracked modules too; git diff alone is not a code freeze.
        "python_sources": {
            str(path): fingerprint_file(path) for path in sorted(Path("src").rglob("*.py"))
        },
        "options": asdict(options),
        "purpose": "development_diagnostic_not_blinded_validation",
    }
    (args.output / "inputs.json").write_text(json.dumps(source_hashes, indent=2, default=str))

    def run(run_options, destination):
        started = time.perf_counter()
        try:
            result = analyze_acquisition(
                AcquisitionFileSet(args.road),
                AcquisitionFileSet(args.plate),
                run_options,
                progress=lambda percent, message: print(f"{percent:3d}% {message}", flush=True),
            )
        except Exception:
            (destination / "failure.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "elapsed_seconds": time.perf_counter() - started,
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                )
            )
            raise
        arrays, summary = _snapshot(result, destination, time.perf_counter() - started)
        print(json.dumps(summary, indent=2), flush=True)
        return arrays

    baseline = run(options, args.output)
    if args.drop_one_seed:
        comparisons = []
        for index, station in enumerate(stations):
            destination = args.output / f"drop-{index}"
            destination.mkdir()
            dropped = run(
                replace(options, seed_stations=[s for s in stations if s is not station]),
                destination,
            )
            for order in (1, 2):
                prefix = f"layer_{order}_"
                original, alternative = baseline[prefix + "display"], dropped[prefix + "display"]
                # Genuine no-picks and loss of the original family are failures
                # of autonomous support, not zero-error observations.
                stable = (
                    (original >= 0) & (alternative >= 0) & (np.abs(original - alternative) <= 7)
                )
                accepted = baseline[prefix + "automatic"]
                comparisons.append(
                    {
                        "dropped_station": station.station_id,
                        "layer_order": order,
                        "automatic_points": int(np.sum(accepted)),
                        "unstable_automatic_points": int(np.sum(accepted & ~stable)),
                        "requires_review_chainages_m": baseline["chainage_m"][
                            accepted & ~stable
                        ].tolist(),
                    }
                )
        (args.output / "dropout.json").write_text(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
