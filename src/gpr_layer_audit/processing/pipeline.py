from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter1d, label, maximum_filter1d

from gpr_layer_audit.io import DZTFile, interpolate_gps, read_dzg, read_dzx
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    AnalysisResult,
    AnomalyRegion,
    CandidateEvent,
    DesignSegment,
    DielectricSource,
    InterfacePick,
    LayerDesign,
    LayerProfilePoint,
    LayerSpec,
    PickSource,
    PickStatus,
    ReviewIssue,
    SearchCorridor,
    SeedStation,
    ThicknessResult,
    TrackingEvidence,
    TrackingProvenance,
    VisibilityState,
)
from gpr_layer_audit.seeds import stations_as_anchors

from .calibration import calibrate
from .corridor import build_search_corridors
from .dielectric import (
    LIGHT_SPEED_M_PER_S,
    resolve_dielectric,
    surface_reflection_dielectric,
    thickness_from_twtt_mm,
)
from .picker import TRACKER_METHODS, PickPath, pick_interfaces, propose_seed_rows
from .preprocessing import PreprocessingOptions, preprocess_for_interpretation


class AnalysisCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class AnalysisOptions:
    # Zero selects an adaptive stack targeting roughly 5,000 coarse traces.
    stack_size: int = 0
    survey_id: str | None = None
    tracker_method: str = "joint_seed_adaptive"
    report_interval_m: float = 1.0
    confidence_threshold: float = 0.45
    layer_confidence_thresholds: dict[int, float] = field(
        default_factory=lambda: {2: 0.45, 3: 0.48}
    )
    accept_scan_dielectric: bool = False
    layer_specs: list[LayerSpec] = field(default_factory=LayerSpec.defaults)
    analyst_dielectric: dict[int, float] = field(default_factory=dict)
    seed_stations: list[SeedStation] = field(default_factory=list)
    layer_designs: list[LayerDesign] = field(default_factory=list)
    design_segments: list[DesignSegment] = field(default_factory=list)
    design_weight: float = 0.10
    automation_mode: str = "aggressive"
    structural_breaks_m: list[float] = field(default_factory=list)
    auto_fine_retrack: bool = True
    max_auto_fine_regions: int = 3
    anchors: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    preprocessing: PreprocessingOptions = field(default_factory=PreprocessingOptions)


def _effective_stack(trace_count: int, requested: int) -> int:
    if requested > 0:
        return requested
    return int(np.clip(math.ceil(trace_count / 5_000), 4, 30))


def _chainage(header, trace_centres: np.ndarray, dzx_metadata) -> np.ndarray:
    distance = header.distance_per_trace_m
    if dzx_metadata and dzx_metadata.units_per_scan and dzx_metadata.units_per_scan > 0:
        distance = dzx_metadata.units_per_scan
    if distance is None:
        return trace_centres.copy()
    return trace_centres * distance


def _shift_sample_axis(data: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return np.asarray(data, dtype=np.float32)
    output = np.zeros_like(data, dtype=np.float32)
    if shift > 0:
        output[:, shift:] = data[:, : data.shape[1] - shift]
    else:
        output[:, :shift] = data[:, -shift:]
    return output


def _complete_seed_count(stations: list[SeedStation], layers: list[LayerSpec]) -> int:
    enabled_orders = [item.order for item in layers if item.analysis_enabled]
    return sum(
        all(
            station.visible_sample(order) is not None
            or station.visibility.get(order)
            in {VisibilityState.NOT_VISIBLE, VisibilityState.ABSENT}
            for order in enabled_orders
        )
        for station in stations
    )


def _dielectric_from_result(
    result: AnalysisResult,
) -> dict[int, tuple[float | None, DielectricSource]]:
    return {
        int(order): (
            item.get("value"),
            DielectricSource(item.get("source", DielectricSource.UNRESOLVED)),
        )
        for order, item in result.parameters.get("dielectric_by_layer", {}).items()
    }


def _anchor_rows(
    anchors: dict[int, list[tuple[float, float]]], chainage: np.ndarray
) -> dict[int, dict[int, int]]:
    output: dict[int, dict[int, int]] = {}
    for layer, values in anchors.items():
        for distance, sample in values:
            row = int(np.argmin(np.abs(chainage - distance)))
            output.setdefault(layer, {})[row] = int(round(sample))
    return output


def _all_anchor_values(options: AnalysisOptions) -> dict[int, list[tuple[float, float]]]:
    output = {order: list(values) for order, values in options.anchors.items()}
    for order, values in stations_as_anchors(options.seed_stations).items():
        output.setdefault(order, []).extend(values)
    return output


def _samples_per_mm(sample_interval_ns: float, dielectric: float | None) -> float | None:
    if dielectric is None or dielectric <= 1:
        return None
    light_speed_mm_ns = LIGHT_SPEED_M_PER_S * 1e-6
    return 2.0 * math.sqrt(dielectric) / (light_speed_mm_ns * sample_interval_ns)


def _segment_at(
    segments: list[DesignSegment], layer_name: str, chainage_m: float
) -> DesignSegment | None:
    return next(
        (
            item
            for item in segments
            if item.layer_name.casefold() == layer_name.casefold()
            and item.start_chainage_m <= chainage_m <= item.end_chainage_m
        ),
        None,
    )


def _design_priors(
    chainage: np.ndarray,
    reference_surface_sample: int,
    header,
    layers: list[LayerSpec],
    segments: list[DesignSegment],
    dielectric_by_layer: dict[int, tuple[float | None, DielectricSource]],
    stations: list[SeedStation],
    pulse_width_samples: float = 7.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    if not segments:
        return {}, {}
    priors = {layer.order: np.full(len(chainage), np.nan) for layer in layers}
    widths = {layer.order: np.full(len(chainage), pulse_width_samples) for layer in layers}
    layer_lookup = {layer.order: layer for layer in layers}
    seed_ratios: dict[int, list[float]] = {}
    for station in stations:
        previous = float(reference_surface_sample)
        for order in sorted(layer_lookup):
            sample = station.visible_sample(order)
            segment = _segment_at(segments, layer_lookup[order].name, station.chainage_m)
            if sample is not None and segment and segment.design_thickness_mm > 0:
                gap = sample - previous
                if gap > 0:
                    seed_ratios.setdefault(order, []).append(gap / segment.design_thickness_mm)
                previous = sample
            elif sample is not None:
                previous = sample
    ratios: dict[int, float | None] = {}
    for order in layer_lookup:
        ratios[order] = (
            float(np.median(seed_ratios[order]))
            if seed_ratios.get(order)
            else _samples_per_mm(
                header.sample_interval_ns, dielectric_by_layer.get(order, (None, None))[0]
            )
        )
    for row, distance in enumerate(chainage):
        previous = float(reference_surface_sample)
        cumulative_valid = True
        for order in sorted(layer_lookup):
            segment = _segment_at(segments, layer_lookup[order].name, float(distance))
            ratio = ratios.get(order)
            if not cumulative_valid or segment is None or ratio is None:
                cumulative_valid = False
                continue
            previous += segment.design_thickness_mm * ratio
            priors[order][row] = previous
            tolerance_mm = max(
                abs(segment.tolerance_low_mm or 0.0),
                abs(segment.tolerance_high_mm or 0.0),
                segment.design_thickness_mm * 0.30,
            )
            widths[order][row] = max(pulse_width_samples, tolerance_mm * ratio)
    return priors, widths


def _evidence_at(path: PickPath, index: int) -> TrackingEvidence:
    values = {
        key: float(items[index])
        for key, items in path.evidence.items()
        if index < len(items) and np.isfinite(items[index])
    }
    return TrackingEvidence(
        signal_score=values.get("signal_score", 0.0),
        absolute_strength=values.get("absolute_strength", 0.0),
        seed_correlation=values.get("seed_correlation", 0.0),
        phase_score=values.get("phase_score", 0.0),
        coherence_score=values.get("coherence_score", 0.0),
        candidate_margin=values.get("candidate_margin", 0.0),
        forward_backward_agreement=values.get("forward_backward_agreement", 0.0),
        perturbation_stability=values.get("perturbation_stability", 0.0),
        design_score=values.get("design_score", 0.0),
        local_snr=values.get("local_snr", 0.0),
        ensemble_agreement=values.get("ensemble_agreement", 0.0),
        design_tiebreak=values.get("design_tiebreak", 0.0),
        hypothesis_agreement=values.get("hypothesis_agreement", 0.0),
        neighborhood_support=values.get("neighborhood_support", 0.0),
        residual_improvement=values.get("residual_improvement", 0.0),
        waveform_similarity=values.get("waveform_similarity", 0.0),
        reflectivity_strength=values.get("reflectivity_strength", 0.0),
        phase_cycle_agreement=values.get("phase_cycle_agreement", 0.0),
        path_margin=values.get("path_margin", 0.0),
        edge_condition=values.get("edge_condition", 0.0),
    )


def _interface_picks(
    paths: dict[int, PickPath],
    layers: list[LayerSpec],
    radargram: np.ndarray,
    reference_surface_sample: int,
    sample_interval_ns: float,
    trace_centres: np.ndarray,
    chainage: np.ndarray,
    anchors: dict[int, dict[int, int]],
    confidence_threshold: float,
    layer_confidence_thresholds: dict[int, float] | None = None,
    latitude: np.ndarray | None = None,
    longitude: np.ndarray | None = None,
    search_corridors: dict[int, SearchCorridor] | None = None,
    anomaly_mask: np.ndarray | None = None,
) -> list[InterfacePick]:
    layer_lookup = {item.order: item for item in layers}
    output: list[InterfacePick] = []
    layer_confidence_thresholds = layer_confidence_thresholds or {}
    search_corridors = search_corridors or {}
    anomaly_mask = (
        np.asarray(anomaly_mask, dtype=bool)
        if anomaly_mask is not None
        else np.zeros(len(chainage), dtype=bool)
    )
    for order, path in paths.items():
        layer = layer_lookup[order]
        layer_threshold = layer_confidence_thresholds.get(order, confidence_threshold)
        seeded_layer = bool(anchors.get(order))
        for index, sample0 in enumerate(path.samples):
            sample = int(sample0)
            confidence = float(path.confidence[index])
            is_seed = index in anchors.get(order, {})
            interpolated = bool(path.interpolated[index])
            anomaly = bool(anomaly_mask[index])
            evidence = _evidence_at(path, index)
            conflict = (
                bool(path.design_conflict[index]) if path.design_conflict is not None else False
            )
            graph_supported = (
                evidence.hypothesis_agreement > 0.0
                and evidence.hypothesis_agreement >= 0.72
                and evidence.forward_backward_agreement >= math.exp(-1.0)
                and evidence.neighborhood_support >= 0.48
                and evidence.signal_score >= 0.18
                and evidence.edge_condition < 0.5
            )
            if anomaly or sample < 0:
                status = PickStatus.UNRESOLVED
                visibility = VisibilityState.NOT_VISIBLE
            elif is_seed and not conflict:
                status = PickStatus.ACCEPTED
                visibility = VisibilityState.VISIBLE
                confidence = max(confidence, 0.95)
            elif interpolated:
                status = PickStatus.REVIEW
                visibility = VisibilityState.UNCERTAIN
            elif (
                (seeded_layer or path.design_constrained)
                and confidence >= layer_threshold
                and (
                    graph_supported
                    or (
                        evidence.ensemble_agreement >= 0.75
                        and evidence.forward_backward_agreement >= math.exp(-1.0)
                        and (
                            evidence.local_snr >= 2.0
                            or evidence.seed_correlation >= 0.60
                        )
                        and (
                            evidence.candidate_margin >= 0.60
                            or evidence.ensemble_agreement >= 0.99
                        )
                    )
                )
                and not conflict
            ):
                status = PickStatus.HIGH_CONFIDENCE
                visibility = VisibilityState.VISIBLE
            elif confidence >= 0.20:
                status = PickStatus.REVIEW
                visibility = VisibilityState.UNCERTAIN
            else:
                status = PickStatus.UNRESOLVED
                visibility = VisibilityState.NOT_VISIBLE
            if interpolated:
                provenance = TrackingProvenance.INTERPOLATED
            elif anomaly:
                provenance = TrackingProvenance.SIGNAL_ONLY
            elif conflict:
                provenance = TrackingProvenance.DESIGN_CONFLICT
            elif path.design_constrained:
                provenance = TrackingProvenance.DESIGN_CONSTRAINED
            elif (
                path.design_guided_samples is not None
                and path.signal_only_samples is not None
                and path.design_guided_samples[index] >= 0
                and path.signal_only_samples[index] >= 0
            ):
                provenance = TrackingProvenance.DESIGN_AGREEMENT
            else:
                provenance = TrackingProvenance.SIGNAL_ONLY
            valid_sample = 0 <= sample < radargram.shape[1]
            amplitude = float(radargram[index, sample]) if valid_sample else float("nan")
            output.append(
                InterfacePick(
                    layer_order=order,
                    layer_name=layer.name,
                    trace_index=int(round(trace_centres[index])),
                    chainage_m=float(chainage[index]),
                    sample_index=float(sample),
                    twtt_ns=(
                        (sample - reference_surface_sample) * sample_interval_ns
                        if valid_sample
                        else float("nan")
                    ),
                    amplitude=amplitude,
                    confidence=confidence,
                    status=status,
                    source=(
                        PickSource.SEED
                        if is_seed
                        else PickSource.INTERPOLATED
                        if interpolated
                        else PickSource.AUTO
                    ),
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
                    polarity=int(np.sign(amplitude)) if np.isfinite(amplitude) else 0,
                    visibility=visibility,
                    provenance=provenance,
                    signal_only_sample=(
                        float(path.signal_only_samples[index])
                        if path.signal_only_samples is not None
                        and path.signal_only_samples[index] >= 0
                        else None
                    ),
                    design_guided_sample=(
                        float(path.design_guided_samples[index])
                        if path.design_guided_samples is not None
                        and path.design_guided_samples[index] >= 0
                        else None
                    ),
                    interpolated=interpolated,
                    evidence=evidence,
                    corridor_lower_sample=(
                        float(search_corridors[order].lower_sample[index])
                        if order in search_corridors
                        and np.isfinite(search_corridors[order].lower_sample[index])
                        else None
                    ),
                    corridor_centre_sample=(
                        float(search_corridors[order].centre_sample[index])
                        if order in search_corridors
                        and np.isfinite(search_corridors[order].centre_sample[index])
                        else None
                    ),
                    corridor_upper_sample=(
                        float(search_corridors[order].upper_sample[index])
                        if order in search_corridors
                        and np.isfinite(search_corridors[order].upper_sample[index])
                        else None
                    ),
                    anomaly=anomaly,
                )
            )
    output.sort(key=lambda item: (item.layer_order, item.chainage_m))
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
            resolved = [item for item in candidates if item.sample_index >= 0]
            bottom = (
                float(np.median([item.sample_index for item in resolved])) if resolved else -1.0
            )
            confidence = float(np.median([item.confidence for item in candidates]))
            if not resolved or any(item.status == PickStatus.UNRESOLVED for item in candidates):
                status = PickStatus.UNRESOLVED
            elif all(
                item.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}
                for item in candidates
            ):
                status = PickStatus.HIGH_CONFIDENCE
            else:
                status = PickStatus.REVIEW
            latitude_values = [item.latitude for item in resolved if item.latitude is not None]
            longitude_values = [item.longitude for item in resolved if item.longitude is not None]
            dielectric, dielectric_source = dielectric_by_layer.get(
                order, (None, DielectricSource.UNRESOLVED)
            )
            top = previous_bottom if previous_interface_resolved else -1.0
            interface_resolved = status != PickStatus.UNRESOLVED and bottom >= 0
            timing_resolved = interface_resolved and previous_interface_resolved and top >= 0
            twtt_ns = (
                max(0.0, bottom - top) * header.sample_interval_ns
                if timing_resolved
                else float("nan")
            )
            thickness = low = high = None
            if dielectric is not None and timing_resolved:
                thickness = thickness_from_twtt_mm(twtt_ns, dielectric)
                sample_uncertainty = 1.0 + (1.0 - confidence) * 7.0
                timing_uncertainty = thickness_from_twtt_mm(
                    sample_uncertainty * header.sample_interval_ns, dielectric
                )
                dielectric_fraction = (
                    0.12 if dielectric_source == DielectricSource.ASSUMED_SCAN else 0.06
                )
                uncertainty = timing_uncertainty + thickness * dielectric_fraction / 2.0
                low, high = max(0.0, thickness - uncertainty), thickness + uncertainty
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
                    interpolated=any(item.interpolated for item in candidates),
                    anomaly=any(item.anomaly for item in candidates),
                )
            )
            if bottom >= 0:
                previous_bottom = bottom
            previous_interface_resolved = interface_resolved
    return output


def _issue_from_group(group: list[InterfacePick]) -> ReviewIssue:
    weakest = min(group, key=lambda item: item.confidence)
    reasons: list[str] = []
    if any(item.visibility == VisibilityState.NOT_VISIBLE for item in group):
        reasons.append("No reliable reflector candidate")
    if any(item.provenance == TrackingProvenance.DESIGN_CONFLICT for item in group):
        reasons.append("Signal-only and design-guided paths disagree")
    if any(item.interpolated for item in group):
        reasons.append("Short evidence gap was interpolated")
    if min(item.confidence for item in group) < 0.5:
        reasons.append("Low calibrated path confidence")
    if not reasons:
        reasons.append("Conflicting reflector evidence")
    length = max(0.0, group[-1].chainage_m - group[0].chainage_m)
    unresolved_fraction = sum(item.status == PickStatus.UNRESOLVED for item in group) / len(group)
    conflict_fraction = sum(
        item.provenance == TrackingProvenance.DESIGN_CONFLICT for item in group
    ) / len(group)
    priority = min(
        1.0,
        0.45 * (1.0 - weakest.confidence)
        + 0.30 * unresolved_fraction
        + 0.15 * conflict_fraction
        + 0.10 * min(1.0, length / 50.0),
    )
    return ReviewIssue(
        issue_id=str(uuid4()),
        layer_order=group[0].layer_order,
        layer_name=group[0].layer_name,
        start_chainage_m=group[0].chainage_m,
        end_chainage_m=group[-1].chainage_m,
        reasons=reasons,
        suggested_action=(
            "Inspect the suggested station, then correct, mark not visible, or accept."
        ),
        priority=priority,
        suggested_chainage_m=weakest.chainage_m,
    )


def _review_issues(picks: list[InterfacePick], bin_width_m: float = 10.0) -> list[ReviewIssue]:
    output: list[ReviewIssue] = []
    for layer_order in sorted({item.layer_order for item in picks}):
        layer = [item for item in picks if item.layer_order == layer_order]
        open_group: list[InterfacePick] = []
        for item in layer:
            uncertain = item.status not in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}
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
    output.sort(key=lambda item: (-item.priority, item.start_chainage_m, item.layer_order))
    return output


def _detect_anomalies(
    chainage: np.ndarray,
    score: np.ndarray | None,
) -> tuple[np.ndarray, list[AnomalyRegion]]:
    """Group strong non-horizontal/disturbed-energy evidence into structural gaps."""
    mask = np.zeros(len(chainage), dtype=bool)
    if score is None or len(score) != len(chainage) or len(chainage) < 3:
        return mask, []
    values = np.asarray(score, dtype=float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        return mask, []
    threshold = max(0.78, float(np.percentile(values[finite], 97.5)))
    raw = finite & (values >= threshold)
    groups, group_count = label(raw)
    for group_id in range(1, group_count + 1):
        rows = np.flatnonzero(groups == group_id)
        if len(rows) >= 2:
            mask[rows] = True
    mask = binary_dilation(mask, iterations=1)
    groups, group_count = label(mask)
    regions: list[AnomalyRegion] = []
    for group_id in range(1, group_count + 1):
        rows = np.flatnonzero(groups == group_id)
        if not len(rows):
            continue
        regions.append(
            AnomalyRegion(
                start_chainage_m=float(chainage[rows[0]]),
                end_chainage_m=float(chainage[rows[-1]]),
                score=float(np.max(values[rows])),
            )
        )
    return mask, regions


def _profile_points(
    thickness: list[ThicknessResult],
    layers: list[LayerSpec],
    designs: list[LayerDesign],
    segments: list[DesignSegment],
    anomaly_regions: list[AnomalyRegion],
) -> list[LayerProfilePoint]:
    by_chainage: dict[float, list[ThicknessResult]] = {}
    for item in thickness:
        by_chainage.setdefault(item.chainage_m, []).append(item)
    design_by_order = {item.layer_order: item for item in designs}
    layer_by_order = {item.order: item for item in layers}
    output: list[LayerProfilePoint] = []
    for chainage, items in sorted(by_chainage.items()):
        cumulative = 0.0
        cumulative_low = 0.0
        cumulative_high = 0.0
        cumulative_valid = True
        anomaly = any(
            region.start_chainage_m <= chainage <= region.end_chainage_m
            for region in anomaly_regions
        )
        for item in sorted(items, key=lambda value: value.layer_order):
            if item.thickness_mm is None or item.status == PickStatus.UNRESOLVED:
                cumulative_valid = False
            elif cumulative_valid:
                cumulative += item.thickness_mm
                cumulative_low += item.uncertainty_low_mm or item.thickness_mm
                cumulative_high += item.uncertainty_high_mm or item.thickness_mm
            layer = layer_by_order.get(item.layer_order)
            design = design_by_order.get(item.layer_order)
            segment = _segment_at(
                segments,
                layer.name if layer else item.layer_name,
                chainage,
            )
            design_mm = (
                segment.design_thickness_mm
                if segment is not None
                else design.thickness_mm
                if design is not None
                else None
            )
            output.append(
                LayerProfilePoint(
                    layer_order=item.layer_order,
                    layer_name=item.layer_name,
                    chainage_m=chainage,
                    cumulative_depth_mm=(cumulative if cumulative_valid and not anomaly else None),
                    individual_thickness_mm=(item.thickness_mm if not anomaly else None),
                    cumulative_low_mm=(
                        cumulative_low if cumulative_valid and not anomaly else None
                    ),
                    cumulative_high_mm=(
                        cumulative_high if cumulative_valid and not anomaly else None
                    ),
                    individual_low_mm=(item.uncertainty_low_mm if not anomaly else None),
                    individual_high_mm=(item.uncertainty_high_mm if not anomaly else None),
                    confidence=item.confidence,
                    status=(PickStatus.UNRESOLVED if anomaly else item.status),
                    interpolated=item.interpolated,
                    anomaly=anomaly,
                    design_thickness_mm=design_mm,
                )
            )
    return output


def _candidate_events(
    paths: dict[int, PickPath],
    radargram: np.ndarray,
    chainage: np.ndarray,
    corridors: dict[int, SearchCorridor],
    limit: int = 12,
) -> list[CandidateEvent]:
    """Materialize the strongest separated radar events used by review and export."""
    output: list[CandidateEvent] = []
    for order, path in paths.items():
        local_maxima = path.feature >= maximum_filter1d(
            path.feature, size=5, axis=1, mode="nearest"
        )
        corridor = corridors.get(order)
        for row in range(len(chainage)):
            candidates = np.flatnonzero(local_maxima[row])
            if corridor is not None and np.isfinite(corridor.lower_sample[row]):
                candidates = candidates[
                    (candidates >= corridor.lower_sample[row])
                    & (candidates <= corridor.upper_sample[row])
                ]
            if not len(candidates):
                continue
            ranked = candidates[np.argsort(path.feature[row, candidates])[::-1]]
            selected: list[int] = []
            for sample in ranked:
                if any(abs(int(sample) - prior) < 4 for prior in selected):
                    continue
                selected.append(int(sample))
                if len(selected) >= limit:
                    break
            for rank, sample in enumerate(selected, 1):
                design_score = 0.0
                if corridor is not None and np.isfinite(corridor.centre_sample[row]):
                    half_width = max(
                        (corridor.upper_sample[row] - corridor.lower_sample[row]) / 2.0,
                        1.0,
                    )
                    design_score = float(
                        np.exp(-0.5 * ((sample - corridor.centre_sample[row]) / half_width) ** 2)
                    )
                components = path.candidate_components

                def component(
                    name: str,
                    component_maps=components,
                    row_index=row,
                    sample_index=sample,
                ) -> float:
                    values = component_maps.get(name)
                    return (
                        float(values[row_index, sample_index])
                        if values is not None
                        else 0.0
                    )

                phase = component("analytic_phase_rad")
                phase_class = int(
                    np.floor(((phase + np.pi) % (2.0 * np.pi)) * 8.0 / (2.0 * np.pi))
                )
                branch_names = (
                    "residual_envelope",
                    "vertical_gradient",
                    "oriented_coherence",
                    "reflectivity_strength",
                    "signed_seed_correlation",
                    "absolute_seed_correlation",
                    "stripped_gain",
                )
                output.append(
                    CandidateEvent(
                        layer_order=order,
                        chainage_m=float(chainage[row]),
                        sample_index=sample,
                        rank=rank,
                        radar_score=float(path.feature[row, sample]),
                        design_tiebreak=design_score,
                        polarity=int(np.sign(radargram[row, sample])),
                        signed_amplitude=float(radargram[row, sample]),
                        analytic_phase_rad=phase,
                        phase_class=phase_class,
                        reflectivity_strength=component("reflectivity_strength"),
                        lateral_semblance=component("oriented_coherence"),
                        residual_improvement=component("stripped_gain"),
                        waveform_correlation=component("absolute_seed_correlation"),
                        signed_waveform_correlation=component(
                            "signed_seed_correlation"
                        ),
                        branch_scores={name: component(name) for name in branch_names},
                    )
                )
    return output


def _additional_seed_rows(
    chainage: np.ndarray,
    issues: list[ReviewIssue],
    stations: list[SeedStation],
    limit: int,
) -> np.ndarray:
    if limit <= 0 or not len(chainage):
        return np.empty(0, dtype=np.int32)
    existing = [float(item.chainage_m) for item in stations]
    selected: list[float] = []
    span = float(chainage[-1] - chainage[0]) if len(chainage) > 1 else 0.0
    minimum_separation = max(25.0, span / 12.0)
    for issue in issues:
        candidate = issue.suggested_chainage_m
        if candidate is None:
            candidate = (issue.start_chainage_m + issue.end_chainage_m) / 2.0
        if any(abs(candidate - value) < minimum_separation for value in [*existing, *selected]):
            continue
        selected.append(float(candidate))
        if len(selected) >= limit:
            break
    return np.asarray(
        [int(np.argmin(np.abs(chainage - value))) for value in selected],
        dtype=np.int32,
    )


def _fine_segment_replacements(
    result: AnalysisResult,
    options: AnalysisOptions,
    start_chainage_m: float,
    end_chainage_m: float,
    context_m: float,
) -> tuple[dict[tuple[int, float], InterfacePick], dict[str, float | int]] | None:
    """Reprocess raw traces around a coarse segment and map them onto coarse outputs."""
    fine_stack = max(1, result.stack_size // 4)
    if fine_stack >= result.stack_size:
        return None
    road_path = Path(result.source.dzt_path)
    if not road_path.exists():
        return None
    coordinate_layer = min(
        (item.layer_order for item in result.picks),
        default=None,
    )
    coordinate_picks = sorted(
        (item for item in result.picks if item.layer_order == coordinate_layer),
        key=lambda item: item.chainage_m,
    )
    if len(coordinate_picks) < 2:
        return None
    coarse_chainage = np.asarray([item.chainage_m for item in coordinate_picks], dtype=float)
    coarse_traces = np.asarray([item.trace_index for item in coordinate_picks], dtype=float)
    road = DZTFile(road_path)
    support_start = max(float(coarse_chainage[0]), start_chainage_m - context_m)
    support_end = min(float(coarse_chainage[-1]), end_chainage_m + context_m)
    start_trace = max(
        0,
        int(math.floor(np.interp(support_start, coarse_chainage, coarse_traces))) - fine_stack,
    )
    stop_trace = min(
        road.header.trace_count,
        int(math.ceil(np.interp(support_end, coarse_chainage, coarse_traces))) + fine_stack + 1,
    )
    if stop_trace - start_trace < fine_stack:
        return None
    plate_path_value = result.parameters.get("plate_path")
    plate_path = Path(plate_path_value) if plate_path_value else None
    plate = DZTFile(plate_path) if plate_path and plate_path.exists() else None
    calibrated = calibrate(
        road,
        plate,
        stack_size=fine_stack,
        start_trace=start_trace,
        stop_trace=stop_trace,
    )
    dzx = (
        read_dzx(result.source.dzx_path)
        if result.source.dzx_path and Path(result.source.dzx_path).exists()
        else None
    )
    fine_chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    reference_shift = result.reference_surface_sample - calibrated.reference_surface_sample
    calibrated.radargram = _shift_sample_axis(calibrated.radargram, reference_shift)
    calibrated.plate_template = _shift_sample_axis(
        calibrated.plate_template[None, :], reference_shift
    )[0]
    calibrated.reference_surface_sample = result.reference_surface_sample
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        result.reference_surface_sample,
        calibrated.plate_template,
        options.preprocessing,
    )
    calibrated.radargram = interpreted.radargram

    user_anchors = _anchor_rows(_all_anchor_values(options), fine_chainage)
    tracker_anchors = {order: dict(values) for order, values in user_anchors.items()}
    for layer in (item for item in options.layer_specs if item.analysis_enabled):
        existing = [
            item
            for item in result.picks
            if item.layer_order == layer.order and item.sample_index >= 0
        ]
        if not existing:
            continue
        for row, distance in ((0, fine_chainage[0]), (len(fine_chainage) - 1, fine_chainage[-1])):
            if row in tracker_anchors.get(layer.order, {}):
                continue
            boundary = min(existing, key=lambda item: abs(item.chainage_m - distance))
            tracker_anchors.setdefault(layer.order, {})[row] = int(round(boundary.sample_index))
    break_rows = {
        int(np.argmin(np.abs(fine_chainage - distance)))
        for distance in options.structural_breaks_m
        if fine_chainage[0] <= distance <= fine_chainage[-1]
    }
    dielectric_by_layer = _dielectric_from_result(result)
    corridors = build_search_corridors(
        fine_chainage,
        result.reference_surface_sample,
        road.header.sample_interval_ns,
        options.layer_specs,
        options.layer_designs,
        options.design_segments,
        dielectric_by_layer,
        options.seed_stations,
    )
    anomaly_mask, _ = _detect_anomalies(
        fine_chainage, interpreted.feature_branches.get("anomaly_score")
    )
    fine_bin_width = float(np.median(np.diff(fine_chainage))) if len(fine_chainage) > 1 else 1.0
    paths = pick_interfaces(
        calibrated.radargram,
        result.reference_surface_sample,
        options.layer_specs,
        anchor_samples=tracker_anchors,
        matched_template=interpreted.matched_template,
        feature_branches=interpreted.feature_branches,
        design_weight=options.design_weight,
        search_corridors=corridors,
        break_rows=break_rows,
        anomaly_mask=anomaly_mask,
        max_interpolation_rows=max(1, int(round(1.0 / max(fine_bin_width, 1e-6)))),
        horizontal_step_m=fine_bin_width,
        method=options.tracker_method,
    )
    fine_picks = _interface_picks(
        paths,
        options.layer_specs,
        calibrated.radargram,
        result.reference_surface_sample,
        result.header.sample_interval_ns,
        calibrated.trace_centres,
        fine_chainage,
        user_anchors,
        options.confidence_threshold,
        options.layer_confidence_thresholds,
        search_corridors=corridors,
        anomaly_mask=anomaly_mask,
    )
    by_layer = {
        order: sorted(
            (item for item in fine_picks if item.layer_order == order),
            key=lambda item: item.chainage_m,
        )
        for order in {item.layer_order for item in fine_picks}
    }
    replacements: dict[tuple[int, float], InterfacePick] = {}
    for coarse in result.picks:
        if not start_chainage_m <= coarse.chainage_m <= end_chainage_m:
            continue
        candidates = by_layer.get(coarse.layer_order, [])
        if not candidates:
            continue
        local = min(candidates, key=lambda item: abs(item.chainage_m - coarse.chainage_m))
        replacements[(coarse.layer_order, coarse.chainage_m)] = replace(
            local,
            trace_index=coarse.trace_index,
            chainage_m=coarse.chainage_m,
            latitude=coarse.latitude,
            longitude=coarse.longitude,
        )
    metadata: dict[str, float | int] = {
        "start_chainage_m": float(start_chainage_m),
        "end_chainage_m": float(end_chainage_m),
        "context_m": float(context_m),
        "coarse_stack_size": int(result.stack_size),
        "fine_stack_size": int(fine_stack),
        "raw_start_trace": int(start_trace),
        "raw_stop_trace": int(stop_trace),
        "fine_bins": int(len(fine_chainage)),
    }
    return replacements, metadata


def _refresh_path_snapshots(result: AnalysisResult) -> None:
    result.signal_only_paths = {}
    result.design_guided_paths = {}
    for order in sorted({item.layer_order for item in result.picks}):
        layer = sorted(
            (item for item in result.picks if item.layer_order == order),
            key=lambda item: item.chainage_m,
        )
        result.signal_only_paths[order] = np.asarray(
            [
                item.signal_only_sample if item.signal_only_sample is not None else -1
                for item in layer
            ],
            dtype=np.int32,
        )
        if any(item.design_guided_sample is not None for item in layer):
            result.design_guided_paths[order] = np.asarray(
                [
                    item.design_guided_sample if item.design_guided_sample is not None else -1
                    for item in layer
                ],
                dtype=np.int32,
            )


def _automatic_fine_windows(
    result: AnalysisResult,
    maximum: int,
) -> list[tuple[float, float]]:
    if maximum <= 0 or not len(result.chainage_m):
        return []
    selected: list[float] = []
    for issue in result.review_issues:
        centre = issue.suggested_chainage_m
        if centre is None:
            centre = (issue.start_chainage_m + issue.end_chainage_m) / 2.0
        if any(abs(centre - prior) < 35.0 for prior in selected):
            continue
        selected.append(float(centre))
        if len(selected) >= maximum:
            break
    minimum = float(result.chainage_m[0])
    maximum_chainage = float(result.chainage_m[-1])
    return [
        (max(minimum, centre - 25.0), min(maximum_chainage, centre + 25.0)) for centre in selected
    ]


def analyze_acquisition(
    source: AcquisitionFileSet,
    plate_source: AcquisitionFileSet | None = None,
    options: AnalysisOptions | None = None,
    *,
    progress: Callable[[int, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> AnalysisResult:
    options = options or AnalysisOptions()
    training_stations = [
        station for station in options.seed_stations if station.role != "correction"
    ]
    if len(training_stations) > 5:
        raise ValueError("At most five model-training seed stations are allowed.")
    if options.tracker_method not in TRACKER_METHODS:
        raise ValueError(
            f"Unknown tracker method {options.tracker_method!r}; "
            f"choose one of {', '.join(TRACKER_METHODS)}."
        )

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
    update(7, "Fingerprinting source data")
    source.fingerprint = fingerprint_file(source.dzt_path)
    stack_size = _effective_stack(road.header.trace_count, options.stack_size)
    update(13, f"Calibrating and stacking {stack_size} traces per coarse bin")
    try:
        calibrated = calibrate(road, plate, stack_size=stack_size, cancel=cancel)
    except InterruptedError as exc:
        raise AnalysisCancelled(str(exc)) from exc
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    update(30, "Building interpretation and display branches")
    measurement_branch = calibrated.radargram.copy()
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
        "Interpretation and display enhancement are isolated from amplitude dielectric calibration."
    )
    update(39, "Attaching GPS observations")
    latitude = longitude = None
    if source.dzg_path and Path(source.dzg_path).exists():
        observations = read_dzg(source.dzg_path)
        latitude, longitude = interpolate_gps(observations, calibrated.trace_centres)
    reflection_dielectric = np.full(len(chainage), np.nan)
    if calibrated.diagnostics.valid_for_dielectric:
        reflection_dielectric = surface_reflection_dielectric(
            calibrated.surface_amplitudes, calibrated.diagnostics.plate_peak_amplitude
        )
    dielectric_by_layer: dict[int, tuple[float | None, DielectricSource]] = {}
    for layer in options.layer_specs:
        layer_design = next(
            (item for item in options.layer_designs if item.layer_order == layer.order),
            None,
        )
        reflected = (
            float(np.nanmedian(reflection_dielectric))
            if layer.order == 1 and np.count_nonzero(np.isfinite(reflection_dielectric))
            else None
        )
        dielectric_by_layer[layer.order] = resolve_dielectric(
            reflected,
            options.analyst_dielectric.get(
                layer.order,
                layer_design.dielectric
                if layer_design and layer_design.dielectric is not None
                else layer.dielectric,
            ),
            dzx.dielectric if dzx and dzx.dielectric else road.header.dielectric,
            True,
        )
        if dielectric_by_layer[layer.order][0] is None:
            dielectric_by_layer[layer.order] = (7.0, DielectricSource.ASSUMED_SCAN)
    anchors = _anchor_rows(_all_anchor_values(options), chainage)
    break_rows = {
        int(np.argmin(np.abs(chainage - distance))) for distance in options.structural_breaks_m
    }
    corridors = build_search_corridors(
        chainage,
        calibrated.reference_surface_sample,
        road.header.sample_interval_ns,
        options.layer_specs,
        options.layer_designs,
        options.design_segments,
        dielectric_by_layer,
        options.seed_stations,
    )
    anomaly_mask, anomaly_regions = _detect_anomalies(
        chainage, interpreted.feature_branches.get("anomaly_score")
    )
    for region in anomaly_regions:
        break_rows.add(int(np.argmin(np.abs(chainage - region.start_chainage_m))))
        break_rows.add(int(np.argmin(np.abs(chainage - region.end_chainage_m))))
    bin_width = float(np.median(np.diff(chainage))) if len(chainage) > 1 else 1.0
    interpolation_rows = max(1, int(round(1.0 / max(bin_width, 1e-6))))
    update(48, "Tracking all interfaces forward and backward")
    try:
        paths = pick_interfaces(
            calibrated.radargram,
            calibrated.reference_surface_sample,
            options.layer_specs,
            anchor_samples=anchors,
            matched_template=interpreted.matched_template,
            feature_branches=interpreted.feature_branches,
            design_weight=options.design_weight,
            search_corridors=corridors,
            break_rows=break_rows,
            anomaly_mask=anomaly_mask,
            max_interpolation_rows=interpolation_rows,
            horizontal_step_m=bin_width,
            method=options.tracker_method,
            cancel=cancel,
        )
    except InterruptedError as exc:
        raise AnalysisCancelled(str(exc)) from exc
    for layer in options.layer_specs:
        residual_key = f"layer_{layer.order}_stripped_residual"
        improvement_key = f"layer_{layer.order}_subtraction_improvement"
        if residual_key in interpreted.feature_branches:
            interpreted.display_views[f"After {layer.name} stripping"] = np.asarray(
                interpreted.feature_branches[residual_key], dtype=np.float32
            )
        if improvement_key in interpreted.feature_branches:
            interpreted.display_views[f"{layer.name} subtraction fit"] = np.asarray(
                interpreted.feature_branches[improvement_key], dtype=np.float32
            )
    update(76, "Calibrating confidence and evidence provenance")
    picks = _interface_picks(
        paths,
        options.layer_specs,
        calibrated.radargram,
        calibrated.reference_surface_sample,
        road.header.sample_interval_ns,
        calibrated.trace_centres,
        chainage,
        anchors,
        options.confidence_threshold,
        options.layer_confidence_thresholds,
        latitude,
        longitude,
        corridors,
        anomaly_mask,
    )
    candidate_events = _candidate_events(paths, calibrated.radargram, chainage, corridors)
    candidate_lookup: dict[tuple[int, float], list[CandidateEvent]] = {}
    for event in candidate_events:
        candidate_lookup.setdefault((event.layer_order, event.chainage_m), []).append(event)
    for pick in picks:
        if pick.sample_index < 0:
            continue
        events = candidate_lookup.get((pick.layer_order, pick.chainage_m), [])
        if events:
            nearest = min(events, key=lambda event: abs(event.sample_index - pick.sample_index))
            if abs(nearest.sample_index - pick.sample_index) <= 4:
                pick.selected_candidate_rank = nearest.rank
    thickness = _aggregate_results(
        picks,
        options.layer_specs,
        road.header,
        options.report_interval_m,
        dielectric_by_layer,
        calibrated.reference_surface_sample,
    )
    issues = _review_issues(picks)
    candidate_feature = interpreted.feature_branches.get("candidate", np.abs(calibrated.radargram))
    known_design_orders = {
        item.layer_order for item in options.layer_designs if item.thickness_mm is not None
    }
    known_design_orders.update(
        layer.order
        for layer in options.layer_specs
        if any(
            segment.layer_name.casefold() == layer.name.casefold()
            for segment in options.design_segments
        )
    )
    unknown_design_orders = {
        layer.order
        for layer in options.layer_specs
        if layer.analysis_enabled and layer.order not in known_design_orders
    }
    required_seed_orders: set[int] = set()
    required_station_count = 0
    required_unknown_seeds = 0
    for order in unknown_design_orders:
        samples = [
            station.visible_sample(order)
            for station in options.seed_stations
            if station.visible_sample(order) is not None
        ]
        observation_count = sum(
            station.visible_sample(order) is not None
            or station.visibility.get(order)
            in {VisibilityState.NOT_VISIBLE, VisibilityState.ABSENT}
            for station in options.seed_stations
        )
        target = 2
        if len(samples) == 2:
            disagreement = abs(float(samples[1]) - float(samples[0]))
            if disagreement > max(14.0, 0.20 * float(np.median(samples))):
                target = 3
        if observation_count < target:
            required_seed_orders.add(order)
            required_station_count = max(required_station_count, target)
            required_unknown_seeds = max(
                required_unknown_seeds, target - observation_count
            )
    required_known_seeds = 0
    for order in known_design_orders:
        if order not in {layer.order for layer in options.layer_specs if layer.analysis_enabled}:
            continue
        layer_picks = [item for item in picks if item.layer_order == order]
        visible_fraction = (
            sum(item.sample_index >= 0 for item in layer_picks) / len(layer_picks)
            if layer_picks
            else 0.0
        )
        observations = sum(
            station.visible_sample(order) is not None
            or station.visibility.get(order)
            in {VisibilityState.NOT_VISIBLE, VisibilityState.ABSENT}
            for station in options.seed_stations
        )
        if visible_fraction < 0.70:
            target = 3
            if observations < target:
                required_seed_orders.add(order)
                required_station_count = max(required_station_count, target)
                required_known_seeds = max(required_known_seeds, target - observations)
    requested_seed_count = max(required_unknown_seeds, required_known_seeds)
    if requested_seed_count:
        proposed_rows = propose_seed_rows(candidate_feature, count=requested_seed_count)
    else:
        proposed_rows = _additional_seed_rows(
            chainage,
            issues,
            options.seed_stations,
            limit=max(0, 5 - len(options.seed_stations)),
        )
    update(94, "Preparing reproducible result")
    calibrated.radargram[:] = gaussian_filter1d(calibrated.radargram, sigma=0.35, axis=1)
    result = AnalysisResult(
        source=source,
        header=road.header,
        stack_size=stack_size,
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
            "survey_id": options.survey_id or source.dzt_path.stem,
            "tracker_method": options.tracker_method,
            "stack_size": stack_size,
            "stack_policy": "adaptive" if options.stack_size <= 0 else "fixed",
            "plate_path": str(plate_source.dzt_path) if plate_source else None,
            "report_interval_m": options.report_interval_m,
            "confidence_threshold": options.confidence_threshold,
            "layer_confidence_thresholds": options.layer_confidence_thresholds,
            "preprocessing": asdict(options.preprocessing),
            "accept_scan_dielectric": options.accept_scan_dielectric,
            "design_weight": min(0.10, max(0.0, options.design_weight)),
            "design_guided": bool(corridors),
            "layer_designs": [asdict(item) for item in options.layer_designs],
            "required_seed_orders": sorted(required_seed_orders),
            "required_seed_count": required_station_count,
            "structural_breaks_m": options.structural_breaks_m,
            "auto_fine_retrack": options.auto_fine_retrack,
            "max_auto_fine_regions": options.max_auto_fine_regions,
            "fine_retracked_segments": [],
            "dielectric_by_layer": {
                str(order): {"value": value, "source": str(source_type)}
                for order, (value, source_type) in dielectric_by_layer.items()
            },
        },
        interpretation_input_radargram=measurement_branch,
        matched_template=interpreted.matched_template,
        display_radargrams=interpreted.display_views,
        seed_stations=list(options.seed_stations),
        proposed_seed_chainages=[float(chainage[row]) for row in proposed_rows],
        signal_only_paths={
            order: path.signal_only_samples.copy()
            for order, path in paths.items()
            if path.signal_only_samples is not None
        },
        design_guided_paths={
            order: path.design_guided_samples.copy()
            for order, path in paths.items()
            if path.design_guided_samples is not None
        },
        search_corridors=corridors,
        anomaly_regions=anomaly_regions,
        candidate_events=candidate_events,
    )
    result.profile = _profile_points(
        result.thickness,
        options.layer_specs,
        options.layer_designs,
        options.design_segments,
        result.anomaly_regions,
    )
    if (
        options.auto_fine_retrack
        and stack_size > 1
        and (
            bool(corridors) or _complete_seed_count(options.seed_stations, options.layer_specs) >= 2
        )
    ):
        windows = _automatic_fine_windows(result, options.max_auto_fine_regions)
        for index, (start, end) in enumerate(windows, 1):
            update(
                min(99, 94 + index),
                f"Fine retracking uncertain region {index} of {len(windows)}",
            )
            retrack_segment(
                result,
                options,
                start,
                end,
                context_m=10.0,
                preserve_accepted=True,
            )
        proposed_rows = _additional_seed_rows(
            result.chainage_m,
            result.review_issues,
            options.seed_stations,
            limit=max(0, 5 - len(options.seed_stations)),
        )
        result.proposed_seed_chainages = [float(result.chainage_m[row]) for row in proposed_rows]
    update(100, "Analysis complete")
    return result


def retrack_segment(
    result: AnalysisResult,
    options: AnalysisOptions,
    start_chainage_m: float,
    end_chainage_m: float,
    *,
    context_m: float = 10.0,
    preserve_accepted: bool = False,
) -> AnalysisResult:
    """Re-track a bounded raw-data segment while preserving every outside pick."""
    core = (result.chainage_m >= start_chainage_m) & (result.chainage_m <= end_chainage_m)
    context = (result.chainage_m >= start_chainage_m - context_m) & (
        result.chainage_m <= end_chainage_m + context_m
    )
    context_indices = np.flatnonzero(context)
    if not len(context_indices) or not np.any(core):
        return result
    fine = _fine_segment_replacements(
        result,
        options,
        start_chainage_m,
        end_chainage_m,
        context_m,
    )
    if fine is not None:
        replacement_lookup, metadata = fine
        result.parameters.setdefault("fine_retracked_segments", []).append(metadata)
    else:
        subset_chainage = result.chainage_m[context_indices]
        user_anchors = _anchor_rows(_all_anchor_values(options), subset_chainage)
        tracker_anchors = {order: dict(values) for order, values in user_anchors.items()}
        for layer in (item for item in options.layer_specs if item.analysis_enabled):
            existing = [
                item
                for item in result.picks
                if item.layer_order == layer.order and item.sample_index >= 0
            ]
            if not existing:
                continue
            for row, distance in (
                (0, subset_chainage[0]),
                (len(subset_chainage) - 1, subset_chainage[-1]),
            ):
                if row in tracker_anchors.get(layer.order, {}):
                    continue
                boundary = min(existing, key=lambda item: abs(item.chainage_m - distance))
                tracker_anchors.setdefault(layer.order, {})[row] = int(round(boundary.sample_index))
        break_rows = {
            int(np.argmin(np.abs(subset_chainage - distance)))
            for distance in options.structural_breaks_m
            if subset_chainage[0] <= distance <= subset_chainage[-1]
        }
        dielectric_by_layer = _dielectric_from_result(result)
        corridors = build_search_corridors(
            subset_chainage,
            result.reference_surface_sample,
            result.header.sample_interval_ns,
            options.layer_specs,
            options.layer_designs,
            options.design_segments,
            dielectric_by_layer,
            options.seed_stations,
        )
        subset_interpreted = preprocess_for_interpretation(
            result.calibrated_radargram[context_indices],
            result.reference_surface_sample,
            result.matched_template,
            options.preprocessing,
        )
        anomaly_mask, _ = _detect_anomalies(
            subset_chainage,
            subset_interpreted.feature_branches.get("anomaly_score"),
        )
        subset_bin_width = (
            float(np.median(np.diff(subset_chainage))) if len(subset_chainage) > 1 else 1.0
        )
        paths = pick_interfaces(
            subset_interpreted.radargram,
            result.reference_surface_sample,
            options.layer_specs,
            anchor_samples=tracker_anchors,
            matched_template=result.matched_template,
            feature_branches=subset_interpreted.feature_branches,
            design_weight=options.design_weight,
            search_corridors=corridors,
            break_rows=break_rows,
            anomaly_mask=anomaly_mask,
            max_interpolation_rows=max(1, int(round(1.0 / max(subset_bin_width, 1e-6)))),
            horizontal_step_m=subset_bin_width,
            method=options.tracker_method,
        )
        trace_by_chainage = {
            item.chainage_m: item.trace_index
            for item in result.picks
            if item.layer_order == options.layer_specs[0].order
        }
        subset_centres = np.asarray(
            [trace_by_chainage[float(result.chainage_m[index])] for index in context_indices],
            dtype=float,
        )
        subset_latitude = (
            result.gps_latitude[context_indices] if result.gps_latitude is not None else None
        )
        subset_longitude = (
            result.gps_longitude[context_indices] if result.gps_longitude is not None else None
        )
        replacements = _interface_picks(
            paths,
            options.layer_specs,
            result.calibrated_radargram[context_indices],
            result.reference_surface_sample,
            result.header.sample_interval_ns,
            subset_centres,
            subset_chainage,
            user_anchors,
            options.confidence_threshold,
            options.layer_confidence_thresholds,
            subset_latitude,
            subset_longitude,
            corridors,
            anomaly_mask,
        )
        core_chainages = set(float(value) for value in result.chainage_m[core])
        replacement_lookup = {
            (item.layer_order, item.chainage_m): item
            for item in replacements
            if item.chainage_m in core_chainages
        }
    result.picks = [
        item
        if preserve_accepted
        and item.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}
        else replacement_lookup.get((item.layer_order, item.chainage_m), item)
        for item in result.picks
    ]
    dielectric_by_layer = _dielectric_from_result(result)
    result.thickness = _aggregate_results(
        result.picks,
        options.layer_specs,
        result.header,
        options.report_interval_m,
        dielectric_by_layer,
        result.reference_surface_sample,
    )
    result.review_issues = _review_issues(result.picks)
    result.profile = _profile_points(
        result.thickness,
        options.layer_specs,
        options.layer_designs,
        options.design_segments,
        result.anomaly_regions,
    )
    result.seed_stations = list(options.seed_stations)
    _refresh_path_snapshots(result)
    return result


def resolve_review_issue(
    result: AnalysisResult,
    options: AnalysisOptions,
    issue_id: str,
    action: str,
) -> AnalysisResult:
    """Apply an explicit analyst decision to one grouped review region."""
    issue = next((item for item in result.review_issues if item.issue_id == issue_id), None)
    if issue is None:
        raise ValueError(f"Unknown review issue: {issue_id}")
    allowed = {"accept", "not_visible", "absent", "anomaly"}
    if action not in allowed:
        raise ValueError(f"Review action must be one of: {', '.join(sorted(allowed))}")
    for item in result.picks:
        if not (
            item.layer_order == issue.layer_order
            and issue.start_chainage_m <= item.chainage_m <= issue.end_chainage_m
        ):
            continue
        item.status = PickStatus.ACCEPTED
        item.provenance = TrackingProvenance.MANUAL_CORRECTION
        if action == "anomaly":
            item.anomaly = True
            item.visibility = VisibilityState.NOT_VISIBLE
            item.sample_index = -1.0
            item.twtt_ns = float("nan")
            item.amplitude = float("nan")
            item.polarity = 0
        elif action == "not_visible":
            item.visibility = VisibilityState.NOT_VISIBLE
            item.sample_index = -1.0
            item.twtt_ns = float("nan")
            item.amplitude = float("nan")
            item.polarity = 0
        elif action == "absent":
            item.visibility = VisibilityState.ABSENT
            item.sample_index = -1.0
            item.twtt_ns = float("nan")
            item.amplitude = float("nan")
            item.polarity = 0
        else:
            item.visibility = (
                VisibilityState.VISIBLE if item.sample_index >= 0 else VisibilityState.NOT_VISIBLE
            )
    if action == "anomaly" and not any(
        region.start_chainage_m <= issue.start_chainage_m
        and region.end_chainage_m >= issue.end_chainage_m
        for region in result.anomaly_regions
    ):
        result.anomaly_regions.append(
            AnomalyRegion(
                start_chainage_m=issue.start_chainage_m,
                end_chainage_m=issue.end_chainage_m,
                score=1.0,
                status="confirmed",
            )
        )
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
    result.profile = _profile_points(
        result.thickness,
        options.layer_specs,
        options.layer_designs,
        options.design_segments,
        result.anomaly_regions,
    )
    return result
