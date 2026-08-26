from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.ndimage import gaussian_filter1d

from gpr_layer_audit.io import DZTFile, interpolate_gps, read_dzg, read_dzx
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    AnalysisResult,
    DielectricSource,
    InterfacePick,
    LayerSpec,
    PickSource,
    PickStatus,
    ReviewIssue,
    ThicknessResult,
)

from .calibration import calibrate
from .dielectric import resolve_dielectric, surface_reflection_dielectric, thickness_from_twtt_mm
from .picker import pick_interfaces
from .preprocessing import PreprocessingOptions, preprocess_for_interpretation


class AnalysisCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class AnalysisOptions:
    stack_size: int = 10
    report_interval_m: float = 5.0
    confidence_threshold: float = 0.72
    accept_scan_dielectric: bool = False
    layer_specs: list[LayerSpec] = field(default_factory=LayerSpec.defaults)
    analyst_dielectric: dict[int, float] = field(default_factory=dict)
    anchors: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    preprocessing: PreprocessingOptions = field(default_factory=PreprocessingOptions)


def _chainage(header, trace_centres: np.ndarray, dzx_metadata) -> np.ndarray:
    distance = header.distance_per_trace_m
    if dzx_metadata and dzx_metadata.units_per_scan and dzx_metadata.units_per_scan > 0:
        distance = dzx_metadata.units_per_scan
    if distance is None:
        return trace_centres.copy()
    return trace_centres * distance


def _anchor_rows(
    anchors: dict[int, list[tuple[float, float]]],
    chainage: np.ndarray,
) -> dict[int, dict[int, int]]:
    output: dict[int, dict[int, int]] = {}
    for layer, values in anchors.items():
        layer_anchors: dict[int, int] = {}
        for distance, sample in values:
            row = int(np.argmin(np.abs(chainage - distance)))
            layer_anchors[row] = int(round(sample))
        output[layer] = layer_anchors
    return output


def _aggregate_results(
    picks: list[InterfacePick],
    layers: list[LayerSpec],
    header,
    report_interval_m: float,
    dielectric_by_layer: dict[int, tuple[float | None, DielectricSource]],
    reference_surface_sample: int,
) -> list[ThicknessResult]:
    if not picks:
        return []
    grouped: dict[int, list[InterfacePick]] = {}
    for item in picks:
        grouped.setdefault(item.layer_order, []).append(item)
    max_chainage = max(item.chainage_m for item in picks)
    edges = np.arange(0.0, max_chainage + report_interval_m * 1.01, report_interval_m)
    if len(edges) < 2:
        edges = np.asarray([0.0, report_interval_m])
    output: list[ThicknessResult] = []
    layer_lookup = {item.order: item for item in layers}
    for bin_index in range(len(edges) - 1):
        start, end = float(edges[bin_index]), float(edges[bin_index + 1])
        centre = (start + end) / 2.0
        previous_bottom = float(reference_surface_sample)
        previous_interface_resolved = True
        for order in sorted(grouped):
            candidates = [
                item
                for item in grouped[order]
                if start <= item.chainage_m < end
                or (bin_index == len(edges) - 2 and item.chainage_m == end)
            ]
            if not candidates:
                continue
            bottom = float(np.median([item.sample_index for item in candidates]))
            confidence = float(np.median([item.confidence for item in candidates]))
            if any(item.status == PickStatus.UNRESOLVED for item in candidates):
                status = PickStatus.UNRESOLVED
            elif all(item.status == PickStatus.HIGH_CONFIDENCE for item in candidates):
                status = PickStatus.HIGH_CONFIDENCE
            else:
                status = PickStatus.REVIEW
            latitude_values = [item.latitude for item in candidates if item.latitude is not None]
            longitude_values = [item.longitude for item in candidates if item.longitude is not None]
            dielectric, dielectric_source = dielectric_by_layer.get(
                order, (None, DielectricSource.UNRESOLVED)
            )
            top = previous_bottom
            delta_samples = max(0.0, bottom - top)
            twtt_ns = delta_samples * header.sample_interval_ns
            thickness = None
            low = None
            high = None
            interface_resolved = status != PickStatus.UNRESOLVED
            if dielectric is not None and interface_resolved and previous_interface_resolved:
                thickness = thickness_from_twtt_mm(twtt_ns, dielectric)
                sample_uncertainty = 1.0 + (1.0 - confidence) * 7.0
                timing_uncertainty = thickness_from_twtt_mm(
                    sample_uncertainty * header.sample_interval_ns, dielectric
                )
                dielectric_fraction = (
                    0.12 if dielectric_source == DielectricSource.ASSUMED_SCAN else 0.06
                )
                uncertainty = timing_uncertainty + thickness * dielectric_fraction / 2.0
                low = max(0.0, thickness - uncertainty)
                high = thickness + uncertainty
            elif dielectric is None:
                status = PickStatus.UNRESOLVED
            layer_name = layer_lookup.get(order, LayerSpec(order, f"Layer {order}", 0, 0)).name
            output.append(
                ThicknessResult(
                    layer_order=order,
                    layer_name=layer_name,
                    chainage_m=centre,
                    start_chainage_m=start,
                    end_chainage_m=end,
                    top_sample=top,
                    bottom_sample=bottom,
                    twtt_ns=twtt_ns,
                    dielectric=dielectric,
                    dielectric_source=dielectric_source,
                    thickness_mm=thickness,
                    uncertainty_low_mm=low,
                    uncertainty_high_mm=high,
                    confidence=confidence,
                    status=status,
                    latitude=float(np.median(latitude_values)) if latitude_values else None,
                    longitude=float(np.median(longitude_values)) if longitude_values else None,
                )
            )
            previous_bottom = bottom
            previous_interface_resolved = interface_resolved
    return output


def _review_issues(picks: list[InterfacePick], bin_width_m: float = 10.0) -> list[ReviewIssue]:
    output: list[ReviewIssue] = []
    for layer_order in sorted({item.layer_order for item in picks}):
        layer = [item for item in picks if item.layer_order == layer_order]
        open_group: list[InterfacePick] = []
        for item in layer:
            uncertain = item.status != PickStatus.HIGH_CONFIDENCE
            contiguous = (
                not open_group or item.chainage_m - open_group[-1].chainage_m <= bin_width_m
            )
            if uncertain and contiguous:
                open_group.append(item)
                continue
            if open_group:
                output.append(_issue_from_group(open_group))
                open_group = []
            if uncertain:
                open_group = [item]
        if open_group:
            output.append(_issue_from_group(open_group))
    return output


def _issue_from_group(group: list[InterfacePick]) -> ReviewIssue:
    confidence = min(item.confidence for item in group)
    reasons = ["Weak or conflicting reflector evidence"]
    if confidence < 0.5:
        reasons.append("Very low path confidence")
    return ReviewIssue(
        issue_id=str(uuid4()),
        layer_order=group[0].layer_order,
        layer_name=group[0].layer_name,
        start_chainage_m=group[0].chainage_m,
        end_chainage_m=group[-1].chainage_m,
        reasons=reasons,
        suggested_action="Inspect the A-scan and add one or more interface anchors.",
    )


def analyze_acquisition(
    source: AcquisitionFileSet,
    plate_source: AcquisitionFileSet | None = None,
    options: AnalysisOptions | None = None,
    *,
    progress: Callable[[int, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> AnalysisResult:
    options = options or AnalysisOptions()

    def update(value: int, message: str) -> None:
        if cancel and cancel():
            raise AnalysisCancelled("Analysis cancelled")
        if progress:
            progress(value, message)

    update(2, "Reading GSSI metadata")
    road = DZTFile(source.dzt_path)
    plate = DZTFile(plate_source.dzt_path) if plate_source else None
    dzx = read_dzx(source.dzx_path) if source.dzx_path and Path(source.dzx_path).exists() else None
    source.antenna = road.header.antenna
    source.antenna_serial = dzx.antenna_serial if dzx else None
    update(8, "Fingerprinting source data")
    source.fingerprint = fingerprint_file(source.dzt_path)
    update(15, "Calibrating surface and metal-plate waveform")
    try:
        calibrated = calibrate(
            road,
            plate,
            stack_size=options.stack_size,
            cancel=cancel,
        )
    except InterruptedError as exc:
        raise AnalysisCancelled(str(exc)) from exc
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    update(35, "Enhancing reflector visibility")
    interpretation_input = calibrated.radargram.copy()
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
        options.preprocessing,
    )
    calibrated.radargram = interpreted.radargram
    calibrated.diagnostics.preprocessing_steps = interpreted.steps
    calibrated.diagnostics.preprocessing_metrics = interpreted.metrics
    calibrated.diagnostics.messages.append(
        "Automatic interpretation preprocessing applied on a branch isolated from "
        "dielectric amplitude calibration."
    )

    update(40, "Attaching GPS observations")
    latitude = longitude = None
    if source.dzg_path and Path(source.dzg_path).exists():
        observations = read_dzg(source.dzg_path)
        latitude, longitude = interpolate_gps(observations, calibrated.trace_centres)

    reflection_dielectric = np.full(len(chainage), np.nan)
    if calibrated.diagnostics.valid_for_dielectric:
        reflection_dielectric = surface_reflection_dielectric(
            calibrated.surface_amplitudes,
            calibrated.diagnostics.plate_peak_amplitude,
        )

    dielectric_by_layer: dict[int, tuple[float | None, DielectricSource]] = {}
    for layer in options.layer_specs:
        reflected = None
        if layer.order == 1 and np.count_nonzero(np.isfinite(reflection_dielectric)):
            reflected = float(np.nanmedian(reflection_dielectric))
        dielectric_by_layer[layer.order] = resolve_dielectric(
            reflected,
            options.analyst_dielectric.get(layer.order, layer.dielectric),
            dzx.dielectric if dzx and dzx.dielectric else road.header.dielectric,
            options.accept_scan_dielectric,
        )

    update(50, "Tracking pavement interfaces")
    anchors = _anchor_rows(options.anchors, chainage)
    try:
        paths = pick_interfaces(
            calibrated.radargram,
            calibrated.reference_surface_sample,
            options.layer_specs,
            anchor_samples=anchors,
            matched_template=interpreted.matched_template,
            cancel=cancel,
        )
    except InterruptedError as exc:
        raise AnalysisCancelled(str(exc)) from exc

    update(78, "Calculating confidence and travel time")
    layer_lookup = {item.order: item for item in options.layer_specs}
    picks: list[InterfacePick] = []
    for order, path in paths.items():
        layer = layer_lookup[order]
        for index, sample in enumerate(path.samples):
            confidence = float(path.confidence[index])
            if confidence < 0.25:
                status = PickStatus.UNRESOLVED
            elif confidence >= options.confidence_threshold:
                status = PickStatus.HIGH_CONFIDENCE
            else:
                status = PickStatus.REVIEW
            source_type = PickSource.ANCHOR if index in anchors.get(order, {}) else PickSource.AUTO
            picks.append(
                InterfacePick(
                    layer_order=order,
                    layer_name=layer.name,
                    trace_index=int(round(calibrated.trace_centres[index])),
                    chainage_m=float(chainage[index]),
                    sample_index=float(sample),
                    twtt_ns=(float(sample) - calibrated.reference_surface_sample)
                    * road.header.sample_interval_ns,
                    amplitude=float(calibrated.radargram[index, sample]),
                    confidence=confidence,
                    status=status,
                    source=source_type,
                    latitude=(
                        float(latitude[index])
                        if latitude is not None and np.isfinite(latitude[index])
                        else None
                    ),
                    longitude=(
                        float(longitude[index])
                        if longitude is not None and np.isfinite(longitude[index])
                        else None
                    ),
                    polarity=int(np.sign(calibrated.radargram[index, sample])),
                )
            )

    picks.sort(key=lambda item: (item.layer_order, item.chainage_m))
    thickness = _aggregate_results(
        picks,
        options.layer_specs,
        road.header,
        options.report_interval_m,
        dielectric_by_layer,
        calibrated.reference_surface_sample,
    )
    issues = _review_issues(picks)
    update(95, "Preparing auditable results")
    calibrated.radargram[:] = gaussian_filter1d(calibrated.radargram, sigma=0.45, axis=1)
    result = AnalysisResult(
        source=source,
        header=road.header,
        stack_size=options.stack_size,
        chainage_m=chainage,
        calibrated_radargram=calibrated.radargram,
        surface_samples_raw=calibrated.surface_samples,
        reference_surface_sample=calibrated.reference_surface_sample,
        picks=picks,
        thickness=thickness,
        review_issues=issues,
        diagnostics=calibrated.diagnostics,
        gps_latitude=latitude,
        gps_longitude=longitude,
        parameters={
            "stack_size": options.stack_size,
            "report_interval_m": options.report_interval_m,
            "confidence_threshold": options.confidence_threshold,
            "preprocessing": asdict(options.preprocessing),
            "accept_scan_dielectric": options.accept_scan_dielectric,
            "dielectric_by_layer": {
                str(order): {"value": value, "source": str(source_type)}
                for order, (value, source_type) in dielectric_by_layer.items()
            },
        },
        interpretation_input_radargram=interpretation_input,
        matched_template=interpreted.matched_template,
    )
    update(100, "Analysis complete")
    return result


def retrack_segment(
    result: AnalysisResult,
    options: AnalysisOptions,
    start_chainage_m: float,
    end_chainage_m: float,
    *,
    context_m: float = 10.0,
) -> AnalysisResult:
    """Jointly re-track a bounded segment while preserving every outside pick."""
    core = (result.chainage_m >= start_chainage_m) & (result.chainage_m <= end_chainage_m)
    context = (result.chainage_m >= start_chainage_m - context_m) & (
        result.chainage_m <= end_chainage_m + context_m
    )
    context_indices = np.flatnonzero(context)
    if not len(context_indices) or not np.any(core):
        return result
    subset_chainage = result.chainage_m[context_indices]
    anchor_rows = _anchor_rows(options.anchors, subset_chainage)
    paths = pick_interfaces(
        result.calibrated_radargram[context_indices],
        result.reference_surface_sample,
        options.layer_specs,
        anchor_samples=anchor_rows,
        matched_template=result.matched_template,
    )
    core_chainages = set(float(value) for value in result.chainage_m[core])
    existing = {
        (item.layer_order, item.chainage_m): item
        for item in result.picks
        if item.chainage_m in core_chainages
    }
    replacements: dict[tuple[int, float], InterfacePick] = {}
    layer_lookup = {item.order: item for item in options.layer_specs}
    for order, path in paths.items():
        layer = layer_lookup[order]
        anchor_indices = anchor_rows.get(order, {})
        for local_index, global_index in enumerate(context_indices):
            chainage = float(result.chainage_m[global_index])
            if chainage not in core_chainages:
                continue
            sample = int(path.samples[local_index])
            confidence = float(path.confidence[local_index])
            if confidence < 0.25:
                status = PickStatus.UNRESOLVED
            elif confidence >= options.confidence_threshold:
                status = PickStatus.HIGH_CONFIDENCE
            else:
                status = PickStatus.REVIEW
            prior = existing[(order, chainage)]
            replacements[(order, chainage)] = InterfacePick(
                layer_order=order,
                layer_name=layer.name,
                trace_index=prior.trace_index,
                chainage_m=chainage,
                sample_index=float(sample),
                twtt_ns=(sample - result.reference_surface_sample)
                * result.header.sample_interval_ns,
                amplitude=float(result.calibrated_radargram[global_index, sample]),
                confidence=confidence,
                status=status,
                source=(PickSource.ANCHOR if local_index in anchor_indices else PickSource.AUTO),
                latitude=prior.latitude,
                longitude=prior.longitude,
                polarity=int(np.sign(result.calibrated_radargram[global_index, sample])),
            )
    result.picks = [
        replacements.get((item.layer_order, item.chainage_m), item) for item in result.picks
    ]
    dielectric_by_layer = {
        int(order): (
            item.get("value"),
            DielectricSource(item.get("source", DielectricSource.UNRESOLVED)),
        )
        for order, item in result.parameters.get("dielectric_by_layer", {}).items()
    }
    result.thickness = _aggregate_results(
        result.picks,
        options.layer_specs,
        result.header,
        options.report_interval_m,
        dielectric_by_layer,
        result.reference_surface_sample,
    )
    result.review_issues = _review_issues(result.picks)
    return result
