from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gpr_layer_audit.design import compare_with_design, read_design_schedule
from gpr_layer_audit.export import export_audit_package
from gpr_layer_audit.io import DZTFile, read_dzg, read_dzx
from gpr_layer_audit.models import AcquisitionFileSet
from gpr_layer_audit.processing import (
    AnalysisOptions,
    PreprocessingOptions,
    analyze_acquisition,
)
from gpr_layer_audit.reference import evaluate_manual_reference, read_manual_reference


def _info(path: Path) -> int:
    source = AcquisitionFileSet(path)
    dzt = DZTFile(path)
    header = dzt.header
    output = {
        "path": str(path),
        "antenna": header.antenna,
        "system_code": header.system_code,
        "version_code": header.version_code,
        "traces": header.trace_count,
        "samples_per_trace": header.samples_per_trace,
        "bits_per_sample": header.bits_per_sample,
        "channels": header.channels,
        "range_ns": header.range_ns,
        "scans_per_meter": header.scans_per_meter,
        "dielectric_metadata": header.dielectric,
        "data_offset": header.data_offset,
        "trailing_bytes": header.trailing_bytes,
        "gps_observations": len(read_dzg(source.dzg_path)) if source.dzg_path else 0,
        "dzx": asdict(read_dzx(source.dzx_path)) if source.dzx_path else None,
    }
    print(json.dumps(output, indent=2, default=str))
    return 0


def _analyze(args) -> int:
    road = AcquisitionFileSet(args.road)
    plate = AcquisitionFileSet(args.plate) if args.plate else None
    options = AnalysisOptions(
        stack_size=args.stack,
        report_interval_m=args.interval,
        accept_scan_dielectric=args.accept_scan_dielectric,
        preprocessing=PreprocessingOptions(enabled=not args.no_preprocessing),
    )
    result = analyze_acquisition(
        road,
        plate,
        options,
        progress=lambda value, message: print(f"[{value:3d}%] {message}"),
    )
    if args.design:
        result.thickness = compare_with_design(result.thickness, read_design_schedule(args.design))
    if args.reference:
        diagnostics = evaluate_manual_reference(result, read_manual_reference(args.reference))
        reviewed = [
            item
            for item in diagnostics
            if not item.interpolated_reference
            and item.pick_status.value == "high_confidence"
            and item.within_release_target is not None
        ]
        passing = sum(item.within_release_target is True for item in reviewed)
        print(
            "Manual-reference diagnostic: "
            f"{passing}/{len(reviewed)} non-interpolated high-confidence points "
            "meet the layer-specific target."
        )
    package = export_audit_package(result, args.output)
    print(f"Audit package: {package}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpr-layer-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    info = subparsers.add_parser("info", help="Inspect GSSI acquisition metadata")
    info.add_argument("dzt", type=Path)
    analyze = subparsers.add_parser("analyze", help="Analyze and export a road acquisition")
    analyze.add_argument("road", type=Path)
    analyze.add_argument("--plate", type=Path)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--stack", type=int, default=10)
    analyze.add_argument("--interval", type=float, default=5.0)
    analyze.add_argument("--accept-scan-dielectric", action="store_true")
    analyze.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable interpretation-only visibility enhancement for controlled comparisons",
    )
    analyze.add_argument("--design", type=Path)
    analyze.add_argument("--reference", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        return _info(args.dzt)
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
