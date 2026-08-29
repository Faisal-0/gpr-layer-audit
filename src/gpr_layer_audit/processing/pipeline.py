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
    SeedRequest,
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
from .preprocessing import (
    PreprocessingOptions,
    measurement_packet_support,
    preprocess_for_interpretation,
)
from .reliability import apply_seed_dropout_check
from .tracker import TRACKER_METHODS, PickPath, pick_interfaces, propose_seed_rows


class AnalysisCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class AnalysisOptions:
    # Zero selects a physical 0.4 m grid, independent of survey length.
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
    max_auto_fine_regions: int | None = None
    # Manual interface seeds are valid operational evidence. Leave-one-seed-out
    # analysis remains available as an optional diagnostic, but must not erase
    # useful seed-assisted tracking in the normal workflow.
    validate_seed_dropout: bool = False
    anchors: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    preprocessing: PreprocessingOptions = field(default_factory=PreprocessingOptions)


def _seed_dropout_options(options: AnalysisOptions, station: SeedStation) -> AnalysisOptions:
    """Refit the same workflow without the station or any duplicate anchor.

    Turning off fine refinement only in the withheld run confounds seed
    sensitivity with a change of algorithm. Only recursive dropout is disabled;
    both sides use the requested resolution and refinement policy.
    """
    duplicate_picks = stations_as_anchors([station])
    anchors = {
        order: [
            (chainage, sample) for chainage, sample in values
            if not any(
                abs(chainage - held_chainage) < 1e-6 and abs(sample - held_sample) < 1e-6
                for held_chainage, held_sample in duplicate_picks.get(order, [])
            )
        ]
        for order, values in options.anchors.items()
    }
    return replace(
        options,
        seed_stations=[s for s in options.seed_stations if s.station_id != station.station_id],
        anchors=anchors,
        validate_seed_dropout=False,
    )


def _effective_stack(
    trace_count: int, requested: int, distance_per_trace_m: float | None = None
) -> int:
    if requested > 0:
        return requested
    spatial_stack = (
        math.ceil(0.4 / distance_per_trace_m)
        if distance_per_trace_m is not None and distance_per_trace_m > 0
        else 1
    )
    return max(1, spatial_stack)


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
            if not _in_chainage_extent(distance, chainage):
                continue
            row = int(np.argmin(np.abs(chainage - distance)))
            output.setdefault(layer, {})[row] = int(round(sample))
    return output


def _in_chainage_extent(distance: float, chainage: np.ndarray) -> bool:
    """Include the endpoint bin, never clamp distant seeds onto a local window."""
    if not len(chainage):
        return False
    half_bin = float(np.median(np.abs(np.diff(chainage)))) / 2 if len(chainage) > 1 else 0.0
    return bool(chainage[0] - half_bin <= distance <= chainage[-1] + half_bin)


def _seed_metadata_rows(
    stations: list[SeedStation], chainage: np.ndarray
) -> dict[int, dict[int, dict[str, object]]]:
    output: dict[int, dict[int, dict[str, object]]] = {}
    for station in stations:
        if not _in_chainage_extent(station.chainage_m, chainage):
            continue
        row = int(np.argmin(np.abs(chainage - station.chainage_m)))
        for order, sample in station.samples.items():
            if station.visible_sample(order) is None:
                continue
            output.setdefault(order, {})[row] = {
                "station_id": station.station_id,
                "sample_index": float(sample),
                "phase_class": station.phase_class.get(order),
                "analytic_phase_rad": station.analytic_phase_rad.get(order),
                "polarity": station.polarity.get(order),
                "selected_lobe": station.selected_lobe.get(order),
                "canonical_sample_index": station.canonical_samples.get(order),
                "pulse_width_samples": station.pulse_width_samples.get(order),
                "event_id": station.event_ids.get(order),
                "family_id": station.family_ids.get(order),
                "competing_family_id": station.competing_family_ids.get(order),
                "regime_id": station.regime_ids.get(order, "default"),
                "competing_samples": station.competing_samples.get(order, []),
            }
    return output


def _seed_regime_break_rows(
    stations: list[SeedStation], chainage: np.ndarray
) -> set[int]:
    output: set[int] = set()
    layer_orders = sorted({order for station in stations for order in station.samples})
    for order in layer_orders:
        visible = sorted(
            (
                station
                for station in stations
                if station.visible_sample(order) is not None
            ),
            key=lambda station: station.chainage_m,
        )
        for left, right in zip(visible, visible[1:], strict=False):
            if left.regime_ids.get(order, "default") == right.regime_ids.get(
                order, "default"
            ):
                continue
            midpoint = 0.5 * (left.chainage_m + right.chainage_m)
            if _in_chainage_extent(midpoint, chainage):
                output.add(int(np.argmin(np.abs(chainage - midpoint))))
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
        preprocessing_agreement=values.get("preprocessing_agreement", 0.0),
        measurement_support=values.get("measurement_support", 0.0),
        measurement_support_gate=values.get("measurement_support_gate", 0.0),
        design_tiebreak=values.get("design_tiebreak", 0.0),
        hypothesis_agreement=values.get("hypothesis_agreement", 0.0),
        neighborhood_support=values.get("neighborhood_support", 0.0),
        residual_improvement=values.get("residual_improvement", 0.0),
        waveform_similarity=values.get("waveform_similarity", 0.0),
        reflectivity_strength=values.get("reflectivity_strength", 0.0),
        phase_cycle_agreement=values.get("phase_cycle_agreement", 0.0),
        path_margin=values.get("path_margin", 0.0),
        edge_condition=values.get("edge_condition", 0.0),
        canonical_event_sample=values.get("canonical_event_sample", -1.0),
        selected_lobe_code=values.get("selected_lobe_code", 0.0),
        alternative_cycle_margin=values.get("alternative_cycle_margin", 0.0),
        seed_distance_support=values.get("seed_distance_support", 0.0),
        drop_seed_stability=values.get("drop_seed_stability", 0.0),
        joint_hypothesis_support=values.get("joint_hypothesis_support", 0.0),
        tracklet_support=values.get("tracklet_support", 0.0),
        cycle_slip_risk=values.get("cycle_slip_risk", 0.0),
        branch_multimodality=values.get("branch_multimodality", 0.0),
        event_family_index=values.get("event_family_index", -1.0),
        regime_index=values.get("regime_index", -1.0),
        graph_selected_sample=values.get("graph_selected_sample", -1.0),
        pre_gate_confidence=values.get("pre_gate_confidence", 0.0),
        spatial_lineage_index=values.get("spatial_lineage_index", -1.0),
        seed_reachable=values.get("seed_reachable", 0.0),
        lineage_break=values.get("lineage_break", 0.0),
        seed_position_conflict=values.get("seed_position_conflict", 0.0),
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
            corridor = search_corridors.get(order)
            seed_regime_conflict = bool(
                corridor
                and any(
                    abs(float(chainage[index]) - station) <= 25.0
                    for station in corridor.seed_outlier_chainages_m
                )
            )
            conflict = (
                bool(path.design_conflict[index]) if path.design_conflict is not None else False
            ) or seed_regime_conflict
            graph_supported = (
                evidence.joint_hypothesis_support >= 0.70
                and evidence.forward_backward_agreement >= math.exp(-1.0)
                and evidence.neighborhood_support >= 0.44
                and evidence.signal_score >= 0.16
                and evidence.cycle_slip_risk < 0.55
                and evidence.branch_multimodality < 0.55
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
                        evidence.preprocessing_agreement >= 0.50
                        and evidence.forward_backward_agreement >= math.exp(-1.0)
                        and (
                            evidence.local_snr >= 1.6
                            or evidence.seed_correlation >= 0.52
                            or evidence.tracklet_support >= 0.55
                        )
                        and (
                            evidence.alternative_cycle_margin >= 0.45
                            or evidence.tracklet_support >= 0.68
                        )
                    )
                )
                and not conflict
                and evidence.lineage_break < 0.5
                and evidence.seed_position_conflict < 0.5
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
            elif (
                path.evidence.get("selected_design_path") is not None
                and path.evidence["selected_design_path"][index] >= 0.5
            ):
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
            canonical_sample = (
                float(evidence.canonical_event_sample)
                if valid_sample and evidence.canonical_event_sample >= 0
                else float(sample)
            )
            valid_observation = 0 <= canonical_sample < radargram.shape[1]
            amplitude = float(radargram[index, sample]) if valid_sample else float("nan")
            output.append(
                InterfacePick(
                    layer_order=order,
                    layer_name=layer.name,
                    trace_index=int(round(trace_centres[index])),
                    chainage_m=float(chainage[index]),
                    # Canonical packet time is the physical observation used
                    # for TWTT/thickness.  The clicked lobe remains separately
                    # available for drawing the tracker over the radargram.
                    sample_index=canonical_sample,
                    twtt_ns=(
                        (canonical_sample - reference_surface_sample) * sample_interval_ns
                        if valid_observation
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
                    canonical_event_sample=(
                        canonical_sample
                        if valid_observation
                        else None
                    ),
                    selected_lobe_sample=float(sample) if valid_sample else None,
                    selected_lobe=(
                        "negative_trough"
                        if evidence.selected_lobe_code == 1
                        else "positive_peak"
                        if evidence.selected_lobe_code == 2
                        else None
                    ),
                    event_family_id=(
                        f"L{order}:F{int(evidence.event_family_index)}"
                        if evidence.event_family_index >= 0
                        else None
                    ),
                    competing_family_sample=(
                        float(path.alternate_samples[index])
                        if path.alternate_samples[index] >= 0
                        else None
                    ),
                    regime_id=(
                        f"regime-{int(evidence.regime_index)}"
                        if evidence.regime_index >= 0
                        else "default"
                    ),
                    alternative_cycle_margin=evidence.alternative_cycle_margin,
                    branch_agreement=evidence.preprocessing_agreement,
                    drop_seed_stability=evidence.drop_seed_stability,
                    review_reason=("Local reflector continuation breaks here"
                                   if evidence.lineage_break >= 0.5 else None),
                )
            )
    output.sort(key=lambda item: (item.layer_order, item.chainage_m))
    return output


def _apply_seed_visibility(
    picks: list[InterfacePick], stations: list[SeedStation], chainage: np.ndarray,
) -> None:
    """Honor explicit absence/non-visibility at the recorded station only."""
    decisions: dict[tuple[int, float], VisibilityState] = {}
    for station in stations:
        if not _in_chainage_extent(station.chainage_m, chainage):
            continue
        row = int(np.argmin(np.abs(chainage - station.chainage_m)))
        for order, visibility in station.visibility.items():
            if (
                station.user_confirmed.get(order, False)
                and visibility in {VisibilityState.NOT_VISIBLE, VisibilityState.ABSENT}
            ):
                decisions[(order, float(chainage[row]))] = visibility
    for pick in picks:
        visibility = decisions.get((pick.layer_order, pick.chainage_m))
        if visibility is None:
            continue
        pick.sample_index = -1.0
        pick.selected_lobe_sample = None
        pick.canonical_event_sample = None
        pick.selected_candidate_rank = None
        pick.twtt_ns = float("nan")
        pick.amplitude = float("nan")
        pick.polarity = 0
        pick.visibility = visibility
        pick.status = PickStatus.ACCEPTED
        pick.source = PickSource.SEED
        pick.provenance = TrackingProvenance.MANUAL_CORRECTION
        pick.interpolated = False
        pick.confidence = 1.0  # confidence in the explicit decision, not a layer measurement


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
            if not resolved or any(
                item.status == PickStatus.UNRESOLVED or item.sample_index < 0
                for item in candidates
            ):
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
    information_pick = max(
        group,
        key=lambda item: (
            0.38 * item.evidence.branch_multimodality
            + 0.24 * item.evidence.cycle_slip_risk
            + 0.20 * (1.0 - item.evidence.alternative_cycle_margin)
            + 0.18 * (1.0 - item.evidence.drop_seed_stability)
            + 0.28
            * (
                item.evidence.seed_distance_support > 0
                and item.evidence.graph_selected_sample >= 0
                and item.evidence.seed_reachable < 0.5
            )
        ),
    )
    reasons: list[str] = []
    if any(item.visibility == VisibilityState.NOT_VISIBLE for item in group):
        reasons.append("No reliable reflector candidate")
    if any(item.provenance == TrackingProvenance.DESIGN_CONFLICT for item in group):
        reasons.append("Two persistent reflector families remain plausible")
    if max(item.evidence.cycle_slip_risk for item in group) >= 0.55:
        reasons.append("Possible phase-cycle or construction-regime transition")
    if any(item.evidence.lineage_break >= 0.5 for item in group):
        reasons.append("Local reflector continuation is broken; confirm the event family")
    if any(item.evidence.seed_position_conflict >= 0.5 for item in group):
        reasons.append("A stronger reflector competes with the seed-position preference")
    disconnected = [
        item
        for item in group
        if item.evidence.seed_distance_support > 0
        and item.evidence.graph_selected_sample >= 0
        and item.evidence.seed_reachable < 0.5
    ]
    if disconnected:
        reasons.append("Candidate reflector is not connected to a confirmed seed")
    if min(item.evidence.drop_seed_stability for item in group) < 0.25:
        reasons.append("Path is sensitive to removing one seed prototype")
    if any(item.interpolated for item in group):
        reasons.append("Short evidence gap was interpolated")
    if min(item.confidence for item in group) < 0.5:
        reasons.append("Low survey-normalized evidence support")
    if not reasons:
        reasons.append("Conflicting reflector evidence")
    length = max(0.0, group[-1].chainage_m - group[0].chainage_m)
    unresolved_fraction = sum(item.status == PickStatus.UNRESOLVED for item in group) / len(group)
    conflict_fraction = sum(
        item.provenance == TrackingProvenance.DESIGN_CONFLICT for item in group
    ) / len(group)
    disconnected_fraction = len(disconnected) / len(group)
    priority = min(
        1.0,
        0.45 * (1.0 - weakest.confidence)
        + 0.30 * unresolved_fraction
        + 0.15 * conflict_fraction
        + 0.15 * disconnected_fraction
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
        suggested_chainage_m=information_pick.chainage_m,
    )


def _review_issues(picks: list[InterfacePick], bin_width_m: float = 10.0) -> list[ReviewIssue]:
    output: list[ReviewIssue] = []
    for layer_order in sorted({item.layer_order for item in picks}):
        layer = sorted(
            (item for item in picks if item.layer_order == layer_order),
            key=lambda item: item.chainage_m,
        )
        uncertain = np.asarray(
            [
                item.status not in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}
                for item in layer
            ],
            dtype=bool,
        )
        # A few confident bins between two uncertain stretches do not justify
        # dozens of separate analyst questions.  Merge them into one branch-
        # level review span while leaving their accepted pick status unchanged.
        row = 0
        while row < len(uncertain):
            if uncertain[row]:
                row += 1
                continue
            start = row
            while row < len(uncertain) and not uncertain[row]:
                row += 1
            bounded = start > 0 and row < len(uncertain)
            length = layer[row - 1].chainage_m - layer[start].chainage_m
            if bounded and length <= 25.0:
                uncertain[start:row] = True
        open_group: list[InterfacePick] = []
        for item, needs_review in zip(layer, uncertain, strict=True):
            contiguous = (
                not open_group or item.chainage_m - open_group[-1].chainage_m <= bin_width_m
            )
            if needs_review and contiguous:
                open_group.append(item)
                continue
            if open_group:
                output.append(_issue_from_group(open_group))
                open_group = []
            if needs_review:
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
    # Absolute robust-excess threshold: a clean road is allowed to contain no
    # anomaly.  A percentile rule would always manufacture a fixed tail.
    raw = finite & (values >= 0.72)
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
    """Export the actual solver candidates, without a second pruning/ranking pass."""
    output: list[CandidateEvent] = []
    for order, path in paths.items():
        local_maxima = path.feature >= maximum_filter1d(
            path.feature, size=5, axis=1, mode="nearest"
        )
        corridor = corridors.get(order)
        for row in range(len(chainage)):
            packet_map = path.candidate_components.get("event_canonical_sample")
            if packet_map is not None:
                candidates = np.flatnonzero(np.isfinite(packet_map[row]))
            else:
                candidates = np.flatnonzero(local_maxima[row])
            if (
                packet_map is None and corridor is not None
                and np.isfinite(corridor.lower_sample[row])
            ):
                candidates = candidates[
                    (candidates >= corridor.lower_sample[row])
                    & (candidates <= corridor.upper_sample[row])
                ]
            if not len(candidates):
                continue
            rank_map = path.candidate_components.get("audit_candidate_rank")
            if rank_map is not None:
                candidates = candidates[np.isfinite(rank_map[row, candidates])]
                ranked = candidates[np.argsort(rank_map[row, candidates], kind="stable")]
            else:
                ranked = candidates[np.argsort(-path.feature[row, candidates], kind="stable")]
            selected: list[int] = []
            for sample in ranked:
                if packet_map is None and any(abs(int(sample) - prior) < 4 for prior in selected):
                    continue
                selected.append(int(sample))
                if packet_map is None and len(selected) >= limit:
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
                    default: float = 0.0,
                    component_maps=components,
                    row_index=row,
                    sample_index=sample,
                ) -> float:
                    values = component_maps.get(name)
                    return (
                        float(values[row_index, sample_index])
                        if values is not None
                        else default
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
                    "preprocessing_agreement",
                    "measurement_support",
                )
                canonical = component("event_canonical_sample", float(sample))
                if not np.isfinite(canonical):
                    canonical = float(sample)
                pulse_width = component("event_pulse_width")
                lobe_code = int(round(component("event_lobe_code")))
                selected_lobe = {1: "negative_trough", 2: "positive_peak", 3: "zero"}.get(
                    lobe_code, "unknown"
                )
                event_id = f"L{order}:R{row}:C{canonical:.3f}"
                lobe_samples = [
                    int(round(value))
                    for index in range(8)
                    if np.isfinite(value := component(f"event_lobe_{index}", float("nan")))
                ]
                family_index = int(round(component("event_family_index", -1.0)))
                family_id = (
                    f"L{order}:family-{family_index}" if family_index >= 0 else None
                )
                competing = [
                    f"L{order}:R{row}:C{other_canonical:.3f}"
                    for other in selected
                    if other != sample
                    and abs(other - sample) <= max(2.0 * pulse_width, 12.0)
                    and np.isfinite(
                        other_canonical := (
                            float(packet_map[row, other])
                            if packet_map is not None
                            else float(other)
                        )
                    )
                ]
                output.append(
                    CandidateEvent(
                        layer_order=order,
                        chainage_m=float(chainage[row]),
                        sample_index=sample,
                        rank=rank,
                        radar_score=component(
                            "audit_candidate_score", float(path.feature[row, sample])
                        ),
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
                        canonical_sample_index=canonical,
                        event_id=event_id,
                        selected_lobe=selected_lobe,
                        pulse_width_samples=pulse_width,
                        prototype_id=(
                            f"prototype-{int(component('event_prototype_index'))}"
                            if component("event_prototype_index", -1.0) >= 0
                            else None
                        ),
                        competing_event_ids=competing,
                        branch_scores={name: component(name) for name in branch_names},
                        lobe_samples=lobe_samples,
                        lobe_offsets=[float(value - canonical) for value in lobe_samples],
                        event_family_id=family_id,
                        alternative_cycle_margin=float(
                            path.evidence.get("alternative_cycle_margin", np.zeros(len(chainage)))[
                                row
                            ]
                        ),
                        branch_agreement=component("preprocessing_agreement"),
                        graph_selected=bool(component("audit_graph_selected")),
                        joint_hypothesis_count=int(component("audit_hypothesis_count")),
                        packet_id=(
                            f"L{order}:R{row}:P{component('event_packet_centre'):.3f}"
                            if "event_packet_centre" in components else event_id
                        ),
                        spatial_lineage_id=(
                            int(component("spatial_lineage_index"))
                            if component("spatial_lineage_index", -1) >= 0 else None
                        ),
                        seed_reachable=(
                            bool(component("seed_reachable"))
                            if component("seed_reachability_required") else None
                        ),
                    )
                )
    grouped: dict[tuple[int, float], list[CandidateEvent]] = {}
    for event in output:
        grouped.setdefault((event.layer_order, event.chainage_m), []).append(event)
    for events in grouped.values():
        for event in events:
            alternative = next(
                (
                    item
                    for item in sorted(events, key=lambda value: value.rank)
                    if item.event_id != event.event_id
                    and item.event_family_id != event.event_family_id
                ),
                None,
            )
            if alternative is not None:
                event.competing_family_id = alternative.event_family_id or alternative.event_id
    return output


def _additional_seed_requests(
    chainage: np.ndarray,
    issues: list[ReviewIssue],
    stations: list[SeedStation],
    limit: int,
) -> list[SeedRequest]:
    if limit <= 0 or not len(chainage):
        return []
    existing = [float(item.chainage_m) for item in stations]
    selected: list[SeedRequest] = []
    span = float(chainage[-1] - chainage[0]) if len(chainage) > 1 else 0.0
    minimum_separation = max(25.0, span / 12.0)
    for issue in issues:
        candidate = issue.suggested_chainage_m
        if candidate is None:
            candidate = (issue.start_chainage_m + issue.end_chainage_m) / 2.0
        if any(
            abs(candidate - value) < minimum_separation
            for value in [*existing, *(item.chainage_m for item in selected)]
        ):
            continue
        selected.append(
            SeedRequest(
                chainage_m=float(chainage[int(np.argmin(np.abs(chainage - candidate)))]),
                layer_orders=[int(issue.layer_order)],
                reason=" | ".join(issue.reasons),
                priority=float(issue.priority),
                source_issue_id=issue.issue_id,
            )
        )
        if len(selected) >= limit:
            break
    return selected


def _initial_seed_requests(
    chainage: np.ndarray,
    candidate_feature: np.ndarray,
    required_orders: set[int],
    count: int,
) -> list[SeedRequest]:
    rows = propose_seed_rows(candidate_feature, count=count)
    names = {item.order: item.name for item in LayerSpec.defaults()}
    layers = [names.get(order, f"Layer {order}") for order in sorted(required_orders)]
    reason = "Establish radar identity for " + ", ".join(layers)
    return [
        SeedRequest(
            chainage_m=float(chainage[row]),
            layer_orders=sorted(required_orders),
            reason=reason,
            priority=1.0,
        )
        for row in rows
    ]


def _attach_candidate_metadata(picks: list[InterfacePick], events: list[CandidateEvent]) -> None:
    lookup: dict[tuple[int, float], list[CandidateEvent]] = {}
    for event in events:
        lookup.setdefault((event.layer_order, event.chainage_m), []).append(event)
    for pick in picks:
        if pick.sample_index < 0:
            continue
        candidates = lookup.get((pick.layer_order, pick.chainage_m), [])
        display = pick.selected_lobe_sample
        if not candidates or display is None:
            continue
        nearest = min(candidates, key=lambda event: abs(event.sample_index - display))
        if abs(nearest.sample_index - display) > 2:
            continue
        pick.selected_candidate_rank = nearest.rank
        pick.event_family_id = nearest.event_family_id or pick.event_family_id
        pick.competing_family_id = nearest.competing_family_id
        pick.alternative_cycle_margin = nearest.alternative_cycle_margin
        pick.branch_agreement = nearest.branch_agreement


def _boundary_conditions(
    result: AnalysisResult,
    chainage: np.ndarray,
    anchors: dict[int, dict[int, int]],
    metadata: dict[int, dict[int, dict[str, object]]],
) -> None:
    """Carry a nearby accepted display lobe and its canonical offset into refinement."""
    spacing = (
        float(np.median(np.diff(result.chainage_m))) if len(result.chainage_m) > 1 else 0.0
    )
    for order in {pick.layer_order for pick in result.picks}:
        existing = [
            pick for pick in result.picks
            if pick.layer_order == order and pick.sample_index >= 0
            and pick.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}
            and not pick.anomaly and not pick.interpolated
        ]
        if not existing:
            continue
        for row in (0, len(chainage) - 1):
            if row in anchors.get(order, {}):
                continue
            boundary = min(existing, key=lambda pick: abs(pick.chainage_m - chainage[row]))
            if abs(boundary.chainage_m - chainage[row]) > max(spacing, 1e-6):
                continue
            display = (
                boundary.selected_lobe_sample
                if boundary.selected_lobe_sample is not None else boundary.sample_index
            )
            anchors.setdefault(order, {})[row] = int(round(display))
            metadata.setdefault(order, {})[row] = {
                "station_id": f"boundary-L{order}-{chainage[row]:.4f}",
                "sample_index": display,
                "canonical_sample_index": boundary.sample_index,
                "polarity": boundary.polarity,
                "selected_lobe": boundary.selected_lobe,
                "family_id": boundary.event_family_id,
                "regime_id": boundary.regime_id,
            }


def _fine_segment_replacements(
    result: AnalysisResult,
    options: AnalysisOptions,
    start_chainage_m: float,
    end_chainage_m: float,
    context_m: float,
) -> tuple[
    dict[tuple[int, float], InterfacePick], dict[str, float | int], list[CandidateEvent]
] | None:
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
    calibrated.measurement_radargram = _shift_sample_axis(
        calibrated.measurement_radargram, reference_shift
    )
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
    interpreted.feature_branches["measurement_support"] = measurement_packet_support(
        calibrated.measurement_radargram
    )
    calibrated.radargram = interpreted.radargram

    user_anchors = _anchor_rows(_all_anchor_values(options), fine_chainage)
    tracker_anchors = {order: dict(values) for order, values in user_anchors.items()}
    seed_metadata = _seed_metadata_rows(options.seed_stations, fine_chainage)
    _boundary_conditions(result, fine_chainage, tracker_anchors, seed_metadata)
    break_rows = {
        int(np.argmin(np.abs(fine_chainage - distance)))
        for distance in options.structural_breaks_m
        if fine_chainage[0] <= distance <= fine_chainage[-1]
    }
    break_rows.update(_seed_regime_break_rows(options.seed_stations, fine_chainage))
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
        seed_metadata=seed_metadata,
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
    fine_events = _candidate_events(paths, calibrated.radargram, fine_chainage, corridors)
    _attach_candidate_metadata(fine_picks, fine_events)
    event_lookup: dict[tuple[int, float], list[CandidateEvent]] = {}
    for event in fine_events:
        event_lookup.setdefault((event.layer_order, event.chainage_m), []).append(event)
    by_layer = {
        order: sorted(
            (item for item in fine_picks if item.layer_order == order),
            key=lambda item: item.chainage_m,
        )
        for order in {item.layer_order for item in fine_picks}
    }
    replacements: dict[tuple[int, float], InterfacePick] = {}
    replacement_events: list[CandidateEvent] = []
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
        replacement_events.extend(
            replace(event, chainage_m=coarse.chainage_m)
            for event in event_lookup.get((local.layer_order, local.chainage_m), [])
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
    return replacements, metadata, replacement_events


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
    maximum: int | None,
) -> list[tuple[float, float]]:
    if (maximum is not None and maximum <= 0) or not len(result.chainage_m):
        return []
    if maximum is None:
        # Accuracy-first default: cover the whole doubtful span, not just three
        # suggested clicks. Tile with overlap; merge repeated layer requests.
        spans = sorted((issue.start_chainage_m, issue.end_chainage_m)
                       for issue in result.review_issues)
        merged: list[list[float]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1] + 5.0:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        windows = []
        for start, end in merged:
            count = max(1, math.ceil(max(0.0, end - start - 50.0) / 40.0) + 1)
            for left in np.linspace(start, max(start, end - 50.0), count):
                windows.append((float(left), min(float(left + 50.0), end)))
        return windows
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
    stack_size = _effective_stack(
        road.header.trace_count,
        options.stack_size,
        road.header.distance_per_trace_m,
    )
    update(13, f"Calibrating and stacking {stack_size} traces per coarse bin")
    try:
        calibrated = calibrate(road, plate, stack_size=stack_size, cancel=cancel)
    except InterruptedError as exc:
        raise AnalysisCancelled(str(exc)) from exc
    chainage = _chainage(road.header, calibrated.trace_centres, dzx)
    update(30, "Building interpretation and display branches")
    measurement_branch = calibrated.measurement_radargram.copy()
    interpreted = preprocess_for_interpretation(
        calibrated.radargram,
        calibrated.reference_surface_sample,
        calibrated.plate_template,
        options.preprocessing,
    )
    interpreted.feature_branches["measurement_support"] = measurement_packet_support(
        measurement_branch
    )
    interpreted.display_views["Pre-subtraction measurement"] = measurement_branch
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
    break_rows.update(_seed_regime_break_rows(options.seed_stations, chainage))
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
            seed_metadata=_seed_metadata_rows(options.seed_stations, chainage),
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
        tracklet_key = f"layer_{layer.order}_tracklet_support"
        slip_key = f"layer_{layer.order}_cycle_slip_risk"
        if residual_key in interpreted.feature_branches:
            interpreted.display_views[f"After {layer.name} stripping"] = np.asarray(
                interpreted.feature_branches[residual_key], dtype=np.float32
            )
        if improvement_key in interpreted.feature_branches:
            interpreted.display_views[f"{layer.name} subtraction fit"] = np.asarray(
                interpreted.feature_branches[improvement_key], dtype=np.float32
            )
        if tracklet_key in interpreted.feature_branches:
            interpreted.display_views[f"{layer.name} phase-locked tracklets"] = np.asarray(
                interpreted.feature_branches[tracklet_key], dtype=np.float32
            )
        if slip_key in interpreted.feature_branches:
            interpreted.display_views[f"{layer.name} cycle-slip risk"] = np.asarray(
                interpreted.feature_branches[slip_key], dtype=np.float32
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
    _attach_candidate_metadata(picks, candidate_events)
    _apply_seed_visibility(picks, options.seed_stations, chainage)
    thickness = _aggregate_results(
        picks,
        options.layer_specs,
        road.header,
        options.report_interval_m,
        dielectric_by_layer,
        calibrated.reference_surface_sample,
    )
    issues = _review_issues(picks)
    for issue in issues:
        reason = " | ".join(issue.reasons)
        for pick in picks:
            if (
                pick.layer_order == issue.layer_order
                and issue.start_chainage_m <= pick.chainage_m <= issue.end_chainage_m
            ):
                pick.review_reason = reason
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
        proposed_seed_requests = _initial_seed_requests(
            chainage,
            candidate_feature,
            required_seed_orders,
            requested_seed_count,
        )
    else:
        proposed_seed_requests = _additional_seed_requests(
            chainage,
            issues,
            options.seed_stations,
            limit=max(0, 5 - len(training_stations)),
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
            "global_tracking_resolution_target_m": 0.4,
            "plate_path": str(plate_source.dzt_path) if plate_source else None,
            "report_interval_m": options.report_interval_m,
            "confidence_threshold": options.confidence_threshold,
            "layer_confidence_thresholds": options.layer_confidence_thresholds,
            "preprocessing": asdict(options.preprocessing),
            "accept_scan_dielectric": options.accept_scan_dielectric,
            "design_weight": min(0.10, max(0.0, options.design_weight)),
            "design_guided": bool(corridors),
            "event_family_tracker": {
                "phase_locked_tracklets": True,
                "adaptive_prototypes": True,
                "layer_aware_seed_ranker": False,
                "fixed_detection_identity_score": True,
                "hard_negative_neighbor_cycles": True,
                "maximum_event_packets_per_bin": 12,
                "stripping_hypotheses_per_layer": 3,
                "unstripped_deep_branch_retained": True,
                "seed_assisted_local_segment_growth": True,
                "soft_manual_seed_position_constraint": True,
                "seed_position_conflicts_require_review": True,
                "family_identity_beyond_tracklet_reach": True,
                "joint_forward_backward": True,
                "survey_normalized_reliability": True,
                "drop_seed_measure": "independent_seed_prototype_consensus",
                "confidence_is_calibrated_probability": False,
                "pre_subtraction_measurement_support": True,
            },
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
        proposed_seed_chainages=[
            item.chainage_m for item in proposed_seed_requests
        ],
        proposed_seed_requests=proposed_seed_requests,
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
        proposed_seed_requests = _additional_seed_requests(
            result.chainage_m,
            result.review_issues,
            options.seed_stations,
            limit=max(0, 5 - len(training_stations)),
        )
        result.proposed_seed_requests = proposed_seed_requests
        result.proposed_seed_chainages = [
            item.chainage_m for item in proposed_seed_requests
        ]
    if options.validate_seed_dropout and len(training_stations) >= 2:
        for index, station in enumerate(training_stations, 1):
            update(99, f"Checking seed independence: station {index}/{len(training_stations)}")
            withheld = analyze_acquisition(
                source, plate_source,
                _seed_dropout_options(options, station),
                cancel=cancel,
            )
            apply_seed_dropout_check(result, withheld, station.station_id)
        result.review_issues = _review_issues(result.picks)
        result.thickness = _aggregate_results(
            result.picks, options.layer_specs, result.header, options.report_interval_m,
            dielectric_by_layer, result.reference_surface_sample,
        )
        result.profile = _profile_points(
            result.thickness, options.layer_specs, options.layer_designs,
            options.design_segments, result.anomaly_regions,
        )
        proposed_seed_requests = _additional_seed_requests(
            result.chainage_m, result.review_issues, options.seed_stations,
            limit=max(0, 5 - len(training_stations)),
        )
        if proposed_seed_requests:
            result.proposed_seed_requests = proposed_seed_requests
            result.proposed_seed_chainages = [
                item.chainage_m for item in proposed_seed_requests
            ]
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
        replacement_lookup, metadata, replacement_events = fine
        result.parameters.setdefault("fine_retracked_segments", []).append(metadata)
    else:
        subset_chainage = result.chainage_m[context_indices]
        user_anchors = _anchor_rows(_all_anchor_values(options), subset_chainage)
        tracker_anchors = {order: dict(values) for order, values in user_anchors.items()}
        seed_metadata = _seed_metadata_rows(options.seed_stations, subset_chainage)
        _boundary_conditions(result, subset_chainage, tracker_anchors, seed_metadata)
        break_rows = {
            int(np.argmin(np.abs(subset_chainage - distance)))
            for distance in options.structural_breaks_m
            if subset_chainage[0] <= distance <= subset_chainage[-1]
        }
        break_rows.update(_seed_regime_break_rows(options.seed_stations, subset_chainage))
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
        # Rebuild features from the saved plate-corrected input, not from the
        # already enhanced final radargram. Reprocessing the latter compounds
        # gain, filtering, and denoising every time a bounded retrack falls
        # back from raw-file fine resolution.
        tracking_input = result.display_radargrams.get(
            "Raw", result.calibrated_radargram
        )
        subset_interpreted = preprocess_for_interpretation(
            tracking_input[context_indices],
            result.reference_surface_sample,
            None,
            options.preprocessing,
        )
        if result.interpretation_input_radargram is not None:
            subset_interpreted.feature_branches["measurement_support"] = (
                measurement_packet_support(
                    result.interpretation_input_radargram[context_indices]
                )
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
            seed_metadata=seed_metadata,
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
            subset_interpreted.radargram,
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
        replacement_events = _candidate_events(
            paths, subset_interpreted.radargram, subset_chainage, corridors
        )
        _attach_candidate_metadata(replacements, replacement_events)
    applied_keys = {
        (item.layer_order, item.chainage_m) for item in result.picks
        if (item.layer_order, item.chainage_m) in replacement_lookup
        and not (preserve_accepted and item.status in {
            PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED
        })
    }
    result.picks = [
        item
        if preserve_accepted
        and item.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}
        else replacement_lookup.get((item.layer_order, item.chainage_m), item)
        for item in result.picks
    ]
    _apply_seed_visibility(result.picks, options.seed_stations, result.chainage_m)
    # Review/A-scan candidates and retention diagnostics must refer to the
    # same fit as the displayed pick, not the stale global candidate table.
    result.candidate_events = [
        event for event in result.candidate_events
        if (event.layer_order, event.chainage_m) not in applied_keys
    ] + [
        event for event in replacement_events
        if (event.layer_order, event.chainage_m) in applied_keys
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
