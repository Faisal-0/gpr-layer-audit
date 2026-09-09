from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from gpr_layer_audit.benchmark import run_benchmark_manifest, write_benchmark_result
from gpr_layer_audit.catalog import (
    calibration_candidates_for,
    calibration_pairing_is_ambiguous,
    catalog_as_dict,
    discover_survey_catalog,
)
from gpr_layer_audit.design import (
    compare_with_design,
    quick_layer_designs,
    read_design_schedule,
)
from gpr_layer_audit.export import export_audit_package
from gpr_layer_audit.io import DZTFile, read_dzg, read_dzx
from gpr_layer_audit.models import AcquisitionFileSet, LayerSpec
from gpr_layer_audit.processing import (
    TRACKER_METHODS,
    AnalysisOptions,
    PreprocessingOptions,
    analyze_acquisition,
)
from gpr_layer_audit.reference import evaluate_manual_reference, read_manual_reference
from gpr_layer_audit.seeds import load_seed_file


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


def _analysis_options(args, *, survey_id: str | None = None) -> AnalysisOptions:
    options = AnalysisOptions(
        input_mode=getattr(args, "input_mode", "raw"),
        processed_stride=getattr(args, "processed_stride", 1),
        query_layer_orders=getattr(args, "query_layers", [2, 3]),
        survey_id=survey_id,
        tracker_method=args.method,
        ml_model=str(args.model) if args.model else None,
        ml_policy=args.ml,
        hybrid_calibration=str(args.calibration) if args.calibration else None,
        conventional_config=json.loads(args.conventional_config.read_text())
        if args.conventional_config
        else {},
        stack_size=args.stack,
        report_interval_m=args.interval,
        accept_scan_dielectric=args.accept_scan_dielectric,
        preprocessing=PreprocessingOptions(enabled=not args.no_preprocessing),
    )
    if getattr(args, "seeds", None):
        seed_survey_id, options.seed_stations = load_seed_file(args.seeds)
        if survey_id and seed_survey_id != survey_id:
            raise ValueError(
                f"Seed file targets {seed_survey_id!r}, not selected survey {survey_id!r}."
            )
        options.survey_id = seed_survey_id
    if getattr(args, "design", None):
        options.design_segments = read_design_schedule(args.design)
    layer_dielectric = {
        order: value
        for order, value in (
            (1, getattr(args, "asphalt_dielectric", None)),
            (2, getattr(args, "base_dielectric", None)),
            (3, getattr(args, "subbase_dielectric", None)),
        )
        if value is not None
    }
    if any(
        getattr(args, name, None) is not None
        for name in (
            "asphalt_thickness",
            "base_thickness",
            "subbase_thickness",
            "dielectric",
            "asphalt_dielectric",
            "base_dielectric",
            "subbase_dielectric",
        )
    ):
        options.layer_designs = quick_layer_designs(
            getattr(args, "asphalt_thickness", None),
            getattr(args, "base_thickness", None),
            getattr(args, "subbase_thickness", None),
            dielectric=getattr(args, "dielectric", None),
            dielectric_by_layer=layer_dielectric,
        )
    subbase_requested = bool(
        getattr(args, "track_subbase", False)
        or getattr(args, "subbase_thickness", None) is not None
        or getattr(args, "subbase_dielectric", None) is not None
        or any(3 in station.samples or 3 in station.visibility for station in options.seed_stations)
        or any(
            "subbase" in segment.layer_name.casefold().replace("-", "")
            for segment in options.design_segments
        )
    )
    # Deep third-interface fitting is expensive and the supplied road set does
    # not support a general subbase reliability claim. Keep it opt-in at the
    # command line, while automatically honoring an explicit thickness,
    # dielectric, design segment, or saved layer-3 seed.
    options.layer_specs = LayerSpec.defaults()
    options.layer_specs[2].analysis_enabled = subbase_requested
    options.layer_specs[2].audit_enabled = subbase_requested
    return options


def _run_analysis(args, road, plate=None, *, survey_id: str | None = None) -> int:
    road = AcquisitionFileSet(road)
    plate = AcquisitionFileSet(plate) if plate else None
    options = _analysis_options(args, survey_id=survey_id)
    if getattr(args, "native_seeds", None):
        from .native_seed_io import load_native_observations

        if options.input_mode != "processed" or getattr(args, "seeds", None):
            raise ValueError("--native-seeds requires processed mode and replaces --seeds")
        options.seed_stations = load_native_observations(args.native_seeds, road.dzt_path)
        seeded_orders = {o for s in options.seed_stations for o in s.samples | s.visibility}
        for layer in options.layer_specs:
            if layer.order in seeded_orders:
                layer.analysis_enabled = layer.audit_enabled = True
    result = analyze_acquisition(
        road,
        plate,
        options,
        progress=lambda value, message: print(f"[{value:3d}%] {message}"),
    )
    if getattr(args, "design", None):
        result.thickness = compare_with_design(result.thickness, read_design_schedule(args.design))
    if getattr(args, "reference", None):
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


def _analyze(args) -> int:
    return _run_analysis(args, args.road, args.plate)


def _catalog(args) -> int:
    catalog = discover_survey_catalog(args.directory)
    print(json.dumps(catalog_as_dict(catalog), indent=2))
    return 0


def _analyze_folder(args) -> int:
    catalog = discover_survey_catalog(args.directory)
    roads = catalog.roads
    if args.survey_id:
        matches = [item for item in roads if item.survey_id == args.survey_id]
        if not matches:
            available = "\n  ".join(item.survey_id for item in roads)
            raise ValueError(f"Unknown survey id {args.survey_id!r}. Available:\n  {available}")
        road = matches[0]
    elif len(roads) == 1:
        road = roads[0]
    else:
        available = "\n  ".join(item.survey_id for item in roads)
        raise ValueError(
            "Directory contains multiple road surveys; select one with --survey-id:\n  " + available
        )
    plate = None
    if args.plate_id:
        matches = [item for item in catalog.calibrations if item.survey_id == args.plate_id]
        if not matches:
            raise ValueError(f"Unknown calibration id {args.plate_id!r}.")
        plate = matches[0]
    else:
        candidates = calibration_candidates_for(catalog, road.survey_id)
        if candidates:
            top = candidates[0]
            tied = calibration_pairing_is_ambiguous(candidates)
            if tied:
                raise ValueError(
                    "Calibration pairing is ambiguous; review `catalog` output and pass --plate-id."
                )
            if not top.waveform_compatible:
                print(
                    "Skipping proposed calibration because its waveform dimensions are "
                    "incompatible with the road acquisition."
                )
            else:
                plate = next(
                    item
                    for item in catalog.calibrations
                    if item.survey_id == top.calibration_survey_id
                )
                print(
                    "Proposed calibration: "
                    f"{plate.survey_id} (score {top.compatibility_score:.2f}; "
                    f"problems: {', '.join(top.problems) or 'none'}). "
                    + (
                        "Waveform timing/ringdown will be used, but amplitude dielectric "
                        "inversion remains disabled."
                        if not top.gain_compatible
                        else "Amplitude calibration is compatible."
                    )
                )
    return _run_analysis(
        args,
        road.dzt_path,
        plate.dzt_path if plate else None,
        survey_id=road.survey_id,
    )


def _benchmark(args) -> int:
    result = run_benchmark_manifest(args.manifest)
    if args.output:
        write_benchmark_result(result, args.output)
        print(f"Benchmark result: {args.output}")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


def _analysis_arguments(parser) -> None:
    parser.add_argument(
        "--conventional-config", type=Path, help="Versioned physical conventional settings"
    )
    parser.add_argument("--model", type=Path, help="Optional validated hybrid model bundle")
    parser.add_argument("--ml", choices=("auto", "off", "require"), default="off")
    parser.add_argument("--calibration", type=Path, help="Frozen hybrid acceptance calibration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stack", type=int, default=0, help="0 selects adaptive stacking")
    parser.add_argument(
        "--input-mode",
        choices=("raw", "processed"),
        default="raw",
        help="Processed retains the supplied sample coordinates and time origin",
    )
    parser.add_argument(
        "--processed-stride",
        type=int,
        default=1,
        help="Explicit processed trace subsampling; incompatible seed grids fail",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--query-layers",
        type=int,
        nargs="+",
        choices=(1, 2, 3),
        default=[2, 3],
        help="Interfaces eligible for processed observation requests (default: base/subbase)",
    )
    parser.add_argument(
        "--method",
        choices=TRACKER_METHODS,
        default="joint_seed_adaptive",
        help="Tracking method: joint_seed_adaptive is the default; seed_hybrid is experimental",
    )
    parser.add_argument("--accept-scan-dielectric", action="store_true")
    parser.add_argument("--seeds", type=Path, help="Versioned JSON seed stations")
    parser.add_argument(
        "--native-seeds",
        type=Path,
        help="Explicit native processed operating observations, checked by DZT hash",
    )
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable interpretation-only enhancement for controlled comparisons",
    )
    parser.add_argument("--design", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument(
        "--asphalt",
        "--asphalt-thickness",
        dest="asphalt_thickness",
        help="Individual asphalt thickness, e.g. 2in",
    )
    parser.add_argument(
        "--base",
        "--base-thickness",
        dest="base_thickness",
        help="Individual base thickness, e.g. 4in",
    )
    parser.add_argument(
        "--subbase",
        "--subbase-thickness",
        dest="subbase_thickness",
        help="Individual subbase thickness; supplying it enables subbase tracking",
    )
    parser.add_argument(
        "--track-subbase",
        action="store_true",
        help=(
            "Explicitly enable the provisional subbase tracker without a design value; "
            "use only when a distinct reflector can be manually seeded"
        ),
    )
    parser.add_argument(
        "--dielectric",
        type=float,
        help=(
            "Optional layer dielectric; otherwise physical thickness requires valid "
            "calibration or an explicitly accepted scan value"
        ),
    )
    parser.add_argument(
        "--asphalt-dielectric",
        type=float,
        help="Optional asphalt relative permittivity; overrides --dielectric",
    )
    parser.add_argument(
        "--base-dielectric",
        type=float,
        help="Optional base relative permittivity; overrides --dielectric",
    )
    parser.add_argument(
        "--subbase-dielectric",
        type=float,
        help="Optional subbase relative permittivity; overrides --dielectric",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpr-layer-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    info = subparsers.add_parser("info", help="Inspect GSSI acquisition metadata")
    info.add_argument("dzt", type=Path)
    catalog = subparsers.add_parser("catalog", help="Discover surveys and proposed pairings")
    catalog.add_argument("directory", type=Path)
    analyze = subparsers.add_parser("analyze", help="Analyze and export a road acquisition")
    analyze.add_argument("road", type=Path)
    analyze.add_argument("--plate", type=Path)
    _analysis_arguments(analyze)
    folder = subparsers.add_parser(
        "analyze-folder", help="Analyze one cataloged survey from a directory"
    )
    folder.add_argument("directory", type=Path)
    folder.add_argument("--survey-id")
    folder.add_argument("--plate-id")
    _analysis_arguments(folder)
    benchmark = subparsers.add_parser(
        "benchmark", help="Run deterministic blocked validation from a dataset manifest"
    )
    benchmark.add_argument("manifest", type=Path)
    benchmark.add_argument("--output", type=Path)
    from gpr_layer_audit.ml.cli import add_commands

    add_commands(subparsers)
    from gpr_layer_audit.conventional import add_commands as add_conventional

    add_conventional(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "conventional":
        from gpr_layer_audit.conventional import run_command

        return run_command(args)
    if args.command in ("dataset", "ml", "hybrid"):
        from gpr_layer_audit.ml.cli import run_command

        return run_command(args)
    if args.command == "info":
        return _info(args.dzt)
    if args.command == "catalog":
        return _catalog(args)
    if args.command == "analyze-folder":
        return _analyze_folder(args)
    if args.command == "benchmark":
        return _benchmark(args)
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
