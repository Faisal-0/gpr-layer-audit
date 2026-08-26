from __future__ import annotations

import ctypes
import json
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gpr_layer_audit.design import quick_layer_designs, read_design_schedule
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    PickStatus,
)
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition
from gpr_layer_audit.reference import evaluate_manual_reference, read_manual_reference
from gpr_layer_audit.seeds import load_seed_file

MANIFEST_SCHEMA_VERSION = 1


@dataclass(slots=True)
class BenchmarkMetric:
    case_id: str
    configuration_id: str
    split_type: str
    observation_kind: str
    layer_order: int
    trustworthy_points: int
    holdout_points: int
    measured_points: int
    within_target: int
    pass_rate: float | None
    mae_mm: float | None
    p90_error_mm: float | None
    target_mm: float
    review_fraction: float
    seed_stations: int
    elapsed_seconds: float
    seconds_per_km: float | None
    peak_memory_mib: float | None
    passed: bool
    failure_reasons: list[str]


def _split_mask(chainages: np.ndarray, split_type: str) -> np.ndarray:
    """Frozen 100 m Talagang split: 0-2 development, 3 calibration, 4 test."""
    values = np.asarray(chainages, dtype=float)
    if not len(values):
        return np.zeros(0, dtype=bool)
    blocks = np.floor(np.maximum(values, 0.0) / 100.0).astype(int) % 5
    if split_type == "development":
        return blocks <= 2
    if split_type == "confidence_calibration":
        return blocks == 3
    if split_type in {"final_test", "blocked_span"}:
        return blocks == 4
    if split_type == "leave_one_road_out":
        return np.ones(len(values), dtype=bool)
    raise ValueError(f"Unknown benchmark split: {split_type}")


def _holdout_block(case_id: str, chainages: np.ndarray, block_count: int = 5) -> np.ndarray:
    """Compatibility wrapper for the frozen final-test split.

    ``case_id`` and ``block_count`` are intentionally ignored: changing a case
    label must not change which reference values are final holdout evidence.
    """
    del case_id, block_count
    return _split_mask(chainages, "final_test")


def _peak_working_set_mib() -> float | None:
    if not hasattr(ctypes, "WinDLL"):
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 1024**2 if succeeded else None


def _review_fraction(result, layer_order: int) -> float:
    picks = [item for item in result.picks if item.layer_order == layer_order]
    if not picks:
        return 1.0
    return sum(
        item.status not in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED} for item in picks
    ) / len(picks)


def _case_metric(
    case_id: str,
    configuration_id: str,
    layer_order: int,
    diagnostics,
    result,
    elapsed: float,
    split_type: str,
    peak_memory_mib: float | None,
    observation_kind: str,
) -> BenchmarkMetric:
    trustworthy = [
        item
        for item in diagnostics
        if item.layer_order == layer_order and not item.interpolated_reference
    ]
    chainages = np.asarray([item.chainage_m for item in trustworthy], dtype=float)
    holdout_mask = _split_mask(chainages, split_type)
    seed_chainages = np.asarray(
        [station.chainage_m for station in result.seed_stations], dtype=float
    )
    autonomous = np.ones(len(trustworthy), dtype=bool)
    if len(seed_chainages):
        autonomous = np.min(
            np.abs(chainages[:, None] - seed_chainages[None, :]), axis=1
        ) > 10.0
    holdout = [
        item
        for item, keep, is_autonomous in zip(
            trustworthy, holdout_mask, autonomous, strict=False
        )
        if keep and is_autonomous
    ]
    automatically_accepted = [
        item
        for item in holdout
        if item.pick_status == PickStatus.HIGH_CONFIDENCE and item.absolute_error_mm is not None
    ]
    if observation_kind == "cumulative_interface":
        raw_errors = [item.absolute_error_mm for item in automatically_accepted]
    elif observation_kind == "individual_thickness":
        raw_errors = [item.individual_absolute_error_mm for item in automatically_accepted]
    else:
        raise ValueError(f"Unknown observation kind: {observation_kind}")
    errors = np.asarray([item for item in raw_errors if item is not None], dtype=float)
    target = 12.7 if layer_order == 1 else 25.4
    within = int(np.count_nonzero(errors <= target))
    pass_rate = float(within / len(errors)) if len(errors) else None
    review_fraction = _review_fraction(result, layer_order)
    length_km = float(result.chainage_m[-1] - result.chainage_m[0]) / 1000.0
    seconds_per_km = elapsed / length_km if length_km > 0 else None
    reasons: list[str] = []
    if len(trustworthy) >= 30:
        if pass_rate is None or pass_rate < 0.85:
            reasons.append("held-out accuracy below 85%")
        if review_fraction > 0.10:
            reasons.append("review/unresolved chainage above 10%")
    if len(result.seed_stations) > 5:
        reasons.append("more than five seed stations")
    if seconds_per_km is not None and seconds_per_km > 30.0:
        reasons.append("runtime above 30 seconds per kilometre")
    if peak_memory_mib is not None and peak_memory_mib > 1024.0:
        reasons.append("peak working set above 1 GiB")
    return BenchmarkMetric(
        case_id=case_id,
        configuration_id=configuration_id,
        split_type=split_type,
        observation_kind=observation_kind,
        layer_order=layer_order,
        trustworthy_points=len(trustworthy),
        holdout_points=len(holdout),
        measured_points=len(errors),
        within_target=within,
        pass_rate=pass_rate,
        mae_mm=float(np.mean(errors)) if len(errors) else None,
        p90_error_mm=float(np.percentile(errors, 90)) if len(errors) else None,
        target_mm=target,
        review_fraction=review_fraction,
        seed_stations=len(result.seed_stations),
        elapsed_seconds=elapsed,
        seconds_per_km=seconds_per_km,
        peak_memory_mib=peak_memory_mib,
        passed=not reasons,
        failure_reasons=reasons,
    )


def run_benchmark_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported benchmark manifest schema {document.get('schema_version')}; "
            f"expected {MANIFEST_SCHEMA_VERSION}."
        )
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Benchmark manifest must contain at least one case.")
    configurations = document.get("configurations") or [{"id": "default"}]
    if not isinstance(configurations, list) or not configurations:
        raise ValueError("Benchmark configurations must be a non-empty list.")
    metrics: list[BenchmarkMetric] = []
    case_summaries: list[dict] = []
    seen_configuration_ids: set[str] = set()
    configuration_order: list[str] = []
    for configuration_index, configuration in enumerate(configurations, 1):
        if not isinstance(configuration, dict):
            raise ValueError("Each benchmark configuration must be an object.")
        configuration_id = str(configuration.get("id") or f"configuration-{configuration_index}")
        if configuration_id in seen_configuration_ids:
            raise ValueError(f"Duplicate benchmark configuration id: {configuration_id}")
        seen_configuration_ids.add(configuration_id)
        configuration_order.append(configuration_id)
        for index, case in enumerate(cases, 1):
            case_id = str(case.get("id") or f"case-{index}")
            road = (manifest_path.parent / case["road"]).resolve()
            plate = (manifest_path.parent / case["plate"]).resolve() if case.get("plate") else None
            reference = (manifest_path.parent / case["reference"]).resolve()
            expected_hashes = case.get("source_hashes") or {}
            for role, source_path in (
                ("road", road),
                ("plate", plate),
                ("reference", reference),
            ):
                expected = expected_hashes.get(role)
                if expected is None or source_path is None:
                    continue
                actual = fingerprint_file(source_path)
                if actual != expected:
                    raise ValueError(
                        f"Benchmark case {case_id!r} {role} hash changed: "
                        f"expected {expected}, found {actual}."
                    )

            def setting(
                name: str,
                default,
                config_values=configuration,
                case_values=case,
            ):
                return config_values.get(name, case_values.get(name, default))

            options = AnalysisOptions(
                survey_id=case_id,
                tracker_method=str(setting("tracker_method", "joint_seed_adaptive")),
                stack_size=int(setting("stack_size", 0)),
                report_interval_m=float(setting("report_interval_m", 1.0)),
                confidence_threshold=float(setting("confidence_threshold", 0.45)),
                accept_scan_dielectric=bool(setting("accept_scan_dielectric", True)),
            )
            seed_file = setting("seeds", None)
            seed_selection_source = setting("seed_selection_source", None)
            if seed_file:
                if seed_selection_source != "radar_evidence":
                    raise ValueError(
                        f"Benchmark case {case_id!r} must declare "
                        "seed_selection_source='radar_evidence'; reference-selected seeds "
                        "are not valid holdout inputs."
                    )
                options.survey_id, options.seed_stations = load_seed_file(
                    (manifest_path.parent / seed_file).resolve()
                )
            if case.get("design"):
                options.design_segments = read_design_schedule(
                    (manifest_path.parent / case["design"]).resolve()
                )
            quick_design = setting("layer_designs", None)
            if quick_design:
                options.layer_designs = quick_layer_designs(
                    quick_design.get("asphalt"),
                    quick_design.get("base"),
                    quick_design.get("subbase"),
                    dielectric=quick_design.get("dielectric"),
                )
            started = time.perf_counter()
            result = analyze_acquisition(
                AcquisitionFileSet(road),
                AcquisitionFileSet(plate) if plate else None,
                options,
            )
            if bool(setting("simulate_radar_seeds", False)):
                raise ValueError(
                    f"Benchmark case {case_id!r} requests simulated seeds. The tracker may "
                    "not promote its own path into manual evidence; capture and supply a "
                    "schema-2 user-confirmed seed file instead."
                )
            elapsed = time.perf_counter() - started
            peak_memory_mib = _peak_working_set_mib()
            diagnostics = evaluate_manual_reference(result, read_manual_reference(reference))
            layers = sorted({item.layer_order for item in diagnostics})
            case_metrics = [
                _case_metric(
                    case_id,
                    configuration_id,
                    layer_order,
                    diagnostics,
                    result,
                    elapsed,
                    split_type,
                    peak_memory_mib,
                    observation_kind,
                )
                for split_type in (
                    "development",
                    "confidence_calibration",
                    "final_test",
                    "leave_one_road_out",
                )
                for layer_order in layers
                for observation_kind in ("cumulative_interface", "individual_thickness")
            ]
            metrics.extend(case_metrics)
            case_summaries.append(
                {
                    "case_id": case_id,
                    "configuration_id": configuration_id,
                    "road": str(road),
                    "reference": str(reference),
                    "seed_stations": len(options.seed_stations),
                    "seed_selection_source": seed_selection_source,
                    "elapsed_seconds": elapsed,
                    "peak_memory_mib": peak_memory_mib,
                    "review_groups": len(result.review_issues),
                }
            )
    configuration_summaries: list[dict] = []
    for configuration_id in configuration_order:
        required = [
            item
            for item in metrics
            if item.configuration_id == configuration_id and item.trustworthy_points >= 30
        ]
        mae_values = [item.mae_mm for item in required if item.mae_mm is not None]
        run_summaries = [
            item for item in case_summaries if item["configuration_id"] == configuration_id
        ]
        configuration_summaries.append(
            {
                "configuration_id": configuration_id,
                "passed": bool(required) and all(item.passed for item in required),
                "mean_review_fraction": (
                    float(np.mean([item.review_fraction for item in required]))
                    if required
                    else None
                ),
                "mean_mae_mm": float(np.mean(mae_values)) if mae_values else None,
                "elapsed_seconds": float(sum(item["elapsed_seconds"] for item in run_summaries)),
            }
        )
    eligible = [item for item in configuration_summaries if item["passed"]]
    selected = min(
        eligible,
        key=lambda item: (
            item["mean_review_fraction"],
            item["mean_mae_mm"] if item["mean_mae_mm"] is not None else float("inf"),
            item["elapsed_seconds"],
        ),
        default=None,
    )
    failed = [item for item in metrics if not item.passed and item.trustworthy_points >= 30]
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "passed": selected is not None,
        "selected_configuration": selected["configuration_id"] if selected else None,
        "configuration_summaries": configuration_summaries,
        "cases": case_summaries,
        "metrics": [asdict(item) for item in metrics],
        "failures": [asdict(item) for item in failed] if selected is None else [],
        "failed_candidate_metrics": [asdict(item) for item in failed],
        "selection_rule": (
            "Require accuracy/review gates; among passing configurations choose lowest review, "
            "then lowest MAE, then runtime."
        ),
    }


def write_benchmark_result(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
