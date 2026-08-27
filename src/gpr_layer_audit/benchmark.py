from __future__ import annotations

import ctypes
import json
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gpr_layer_audit.checkpoints import evaluate_retention_audit, load_checkpoint_file
from gpr_layer_audit.design import quick_layer_designs, read_design_schedule
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    PickStatus,
)
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition
from gpr_layer_audit.processing.dielectric import thickness_from_twtt_mm
from gpr_layer_audit.reference import evaluate_manual_reference, read_manual_reference
from gpr_layer_audit.seeds import load_seed_file

MANIFEST_SCHEMA_VERSION = 1


def _missing_checkpoint_pairs(cases: list[dict], checkpoints: list[dict]) -> list[dict]:
    """An empty checkpoint list must never satisfy a release gate via all([])."""
    present = {
        (item["case_id"], item["layer_order"])
        for item in checkpoints if item["checkpoints"] >= 30
    }
    return [
        {"case_id": case["case_id"], "layer_order": order}
        for case in cases
        for order in case.get("required_layer_orders", [1, 2])
        if (case["case_id"], order) not in present
    ]


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
    # Runtime and memory remain reproducibility diagnostics.  During the
    # accuracy-first prototype phase they do not disqualify an otherwise
    # reliable reflector-family configuration.
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
    checkpoint_summaries: list[dict] = []
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
            checkpoint_file = setting("checkpoints", None)
            if checkpoint_file:
                checkpoint_survey_id, checkpoints = load_checkpoint_file(
                    (manifest_path.parent / checkpoint_file).resolve()
                )
                if checkpoint_survey_id != options.survey_id:
                    raise ValueError(
                        f"Checkpoint survey {checkpoint_survey_id!r} does not match "
                        f"analysis survey {options.survey_id!r}."
                    )
                retention = evaluate_retention_audit(result, checkpoints)
                checkpoint_by_layer = {
                    layer_order: [
                        item for item in checkpoints if item.layer_order == layer_order
                    ]
                    for layer_order in {item.layer_order for item in checkpoints}
                }
                audit_by_id = {item.checkpoint_id: item for item in retention}
                for layer_order in sorted({item.layer_order for item in checkpoints}):
                    layer_records = [
                        item for item in retention if item.layer_order == layer_order
                    ]
                    visible_records = [
                        item
                        for item in layer_records
                        if item.expected_visibility.value == "visible"
                    ]
                    candidate_retention = (
                        sum(item.candidate_generated for item in visible_records)
                        / len(visible_records)
                        if visible_records
                        else None
                    )
                    selected_identity = (
                        sum(item.graph_selected for item in visible_records)
                        / len(visible_records)
                        if visible_records
                        else None
                    )
                    dielectric_item = result.parameters.get("dielectric_by_layer", {}).get(
                        str(layer_order), {}
                    )
                    dielectric = dielectric_item.get("value")
                    millimetres_per_sample = (
                        thickness_from_twtt_mm(
                            result.header.sample_interval_ns, float(dielectric)
                        )
                        if dielectric is not None
                        else None
                    )
                    accepted_records = [item for item in visible_records if item.accepted]
                    cumulative_errors = [
                        float(item.canonical_sample_error) * millimetres_per_sample
                        for item in accepted_records
                        if item.canonical_sample_error is not None
                        and millimetres_per_sample is not None
                    ]
                    individual_errors: list[float] = []
                    if layer_order == 1:
                        individual_errors = list(cumulative_errors)
                    elif millimetres_per_sample is not None:
                        prior_checkpoints = checkpoint_by_layer.get(layer_order - 1, [])
                        for checkpoint in checkpoint_by_layer[layer_order]:
                            current_audit = audit_by_id[checkpoint.checkpoint_id]
                            if (
                                not current_audit.accepted
                                or current_audit.selected_canonical_sample is None
                                or checkpoint.canonical_sample_index is None
                                or not prior_checkpoints
                            ):
                                continue
                            prior = min(
                                prior_checkpoints,
                                key=lambda item: abs(item.chainage_m - checkpoint.chainage_m),
                            )
                            if abs(prior.chainage_m - checkpoint.chainage_m) > 0.75:
                                continue
                            prior_audit = audit_by_id[prior.checkpoint_id]
                            if (
                                not prior_audit.accepted
                                or prior_audit.selected_canonical_sample is None
                                or prior.canonical_sample_index is None
                            ):
                                continue
                            expected_gap = (
                                checkpoint.canonical_sample_index
                                - prior.canonical_sample_index
                            )
                            selected_gap = (
                                current_audit.selected_canonical_sample
                                - prior_audit.selected_canonical_sample
                            )
                            individual_errors.append(
                                abs(selected_gap - expected_gap) * millimetres_per_sample
                            )
                    target_mm = 12.7 if layer_order == 1 else 25.4
                    cumulative_pass_rate = (
                        sum(value <= target_mm for value in cumulative_errors)
                        / len(cumulative_errors)
                        if cumulative_errors
                        else None
                    )
                    individual_pass_rate = (
                        sum(value <= target_mm for value in individual_errors)
                        / len(individual_errors)
                        if individual_errors
                        else None
                    )
                    checkpoint_summaries.append(
                        {
                            "case_id": case_id,
                            "configuration_id": configuration_id,
                            "layer_order": layer_order,
                            "checkpoints": len(layer_records),
                            "visible_checkpoints": len(visible_records),
                            "candidate_retention": candidate_retention,
                            "selected_identity": selected_identity,
                            "accepted_checkpoints": len(accepted_records),
                            "cumulative_pass_rate": cumulative_pass_rate,
                            "individual_pass_rate": individual_pass_rate,
                            "target_mm": target_mm,
                            "passed": len(layer_records) >= 30
                            and candidate_retention is not None
                            and candidate_retention >= 0.95
                            and selected_identity is not None
                            and selected_identity >= 0.85
                            and cumulative_pass_rate is not None
                            and cumulative_pass_rate >= 0.85
                            and individual_pass_rate is not None
                            and individual_pass_rate >= 0.85,
                        }
                    )
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
                    "required_layer_orders": [
                        layer.order for layer in options.layer_specs
                        if layer.analysis_enabled and layer.order <= 2
                    ],
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
        checkpoint_required = [
            item
            for item in checkpoint_summaries
            if item["configuration_id"] == configuration_id
        ]
        missing_checkpoints = _missing_checkpoint_pairs(run_summaries, checkpoint_required)
        configuration_summaries.append(
            {
                "configuration_id": configuration_id,
                "passed": bool(required)
                and all(item.passed for item in required)
                and bool(checkpoint_required)
                and not missing_checkpoints
                and all(item["passed"] for item in checkpoint_required),
                "missing_checkpoint_pairs": missing_checkpoints,
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
            item["configuration_id"],
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
        "checkpoint_metrics": checkpoint_summaries,
        "metrics": [asdict(item) for item in metrics],
        "failures": [asdict(item) for item in failed] if selected is None else [],
        "failed_candidate_metrics": [asdict(item) for item in failed],
        "selection_rule": (
            "Require accuracy/review gates; among passing configurations choose lowest review, "
            "then lowest MAE, then deterministic configuration ID. "
            "Runtime/memory are diagnostic only."
        ),
    }


def write_benchmark_result(result: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result, indent=2), encoding="utf-8")
