from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import fftconvolve, hilbert

from gpr_layer_audit.models import (
    LayerSpec,
    PhaseLockedTracklet,
    SearchCorridor,
    WaveformPrototype,
)

from .preprocessing import subtract_tracked_reflection


@dataclass(slots=True)
class SeedConditionedPath:
    samples: NDArray[np.int32]
    confidence: NDArray[np.float64]
    feature: NDArray[np.float32]
    alternate_samples: NDArray[np.int32]
    visible: NDArray[np.bool_]
    interpolated: NDArray[np.bool_]
    evidence: dict[str, NDArray[np.float64]]
    signal_only_samples: NDArray[np.int32]
    design_guided_samples: NDArray[np.int32]
    design_conflict: NDArray[np.bool_]
    design_constrained: bool
    candidate_components: dict[str, NDArray[np.float32]] = field(default_factory=dict)


@dataclass(slots=True)
class _CandidateTable:
    samples: NDArray[np.int32]
    valid: NDArray[np.bool_]
    features: NDArray[np.float32]
    waveforms: NDArray[np.float32]
    canonical_samples: NDArray[np.float32]
    phase_classes: NDArray[np.int8]
    polarities: NDArray[np.int8]
    pulse_widths: NDArray[np.float32]
    prototype_indices: NDArray[np.int16]
    selected_lobes: NDArray[np.int8]
    dense_radar_score: NDArray[np.float32]
    component_maps: dict[str, NDArray[np.float32]]
    feature_names: tuple[str, ...]
    tracklet_support: NDArray[np.float32]
    cycle_slip_risk: NDArray[np.float32]
    seed_distance_support: NDArray[np.float32]
    family_indices: NDArray[np.int16]
    regime_indices: NDArray[np.int16]


@dataclass(slots=True)
class _LayerWorkspace:
    layer: LayerSpec
    table: _CandidateTable
    radar_score: NDArray[np.float32]
    emissions: NDArray[np.float32]
    anchors: dict[int, int]
    seed_conflicts: set[int]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    hypotheses: list[NDArray[np.int32]]
    hypothesis_scores: NDArray[np.float64]
    backward: NDArray[np.int32]
    tracklets: list[PhaseLockedTracklet]


_FEATURE_NAMES = (
    "signed_seed_correlation",
    "absolute_seed_correlation",
    "phase_cycle_agreement",
    "polarity_agreement",
    "oriented_coherence",
    "reflectivity_strength",
    "residual_envelope",
    "vertical_gradient",
    "absolute_strength",
    "stripped_gain",
    "preprocessing_agreement",
    "tracklet_support",
    "seed_distance_support",
    "cycle_slip_safety",
    "drop_seed_stability",
    "generic_radar_score",
    "design_tiebreak",
)


def _normalise_rows(values: NDArray[np.floating]) -> NDArray[np.float32]:
    data = np.asarray(values, dtype=np.float32)
    low = np.percentile(data, 20.0, axis=1, keepdims=True)
    high = np.percentile(data, 97.5, axis=1, keepdims=True)
    return np.asarray(
        np.clip((data - low) / np.maximum(high - low, 1e-6), 0.0, 1.0),
        dtype=np.float32,
    )


def _normalise_waveform(values: NDArray[np.floating]) -> NDArray[np.float32] | None:
    waveform = np.asarray(values, dtype=np.float32).copy()
    waveform -= float(np.mean(waveform))
    norm = float(np.linalg.norm(waveform))
    return waveform / norm if norm > 1e-7 else None


def _phase_class(angle: NDArray[np.floating] | float) -> NDArray[np.int8] | int:
    value = np.mod(np.asarray(angle) + np.pi, 2.0 * np.pi)
    classes = np.floor(value * 8.0 / (2.0 * np.pi)).astype(np.int8)
    return int(classes) if classes.ndim == 0 else classes


def _template_bank(
    data: NDArray[np.float32],
    anchors: dict[int, int],
    fallback_radius: int,
    metadata: dict[int, dict[str, object]] | None = None,
) -> list[WaveformPrototype]:
    analytic = hilbert(data, axis=1)
    output: list[WaveformPrototype] = []
    for row, sample in sorted(anchors.items()):
        item = (metadata or {}).get(row, {})
        pulse_width = float(item.get("pulse_width_samples") or fallback_radius / 1.5)
        radius = max(5, int(round(1.5 * pulse_width)))
        if not 0 <= row < len(data) or sample < radius or sample + radius >= data.shape[1]:
            continue
        real = _normalise_waveform(data[row, sample - radius : sample + radius + 1])
        if real is None:
            continue
        quadrature = _normalise_waveform(np.imag(hilbert(real)))
        if quadrature is None:
            continue
        output.append(
            WaveformPrototype(
                layer_order=0,
                station_id=str(item.get("station_id") or f"row-{row}"),
                chainage_m=float(row),
                sample_index=int(sample),
                real_waveform=real,
                quadrature_waveform=quadrature,
                polarity=int(item.get("polarity") or np.sign(data[row, sample])),
                phase_class=int(
                    item.get("phase_class")
                    if item.get("phase_class") is not None
                    else _phase_class(np.angle(analytic[row, sample]))
                ),
                radius_samples=radius,
                analytic_phase_rad=float(
                    item.get("analytic_phase_rad")
                    if item.get("analytic_phase_rad") is not None
                    else np.angle(analytic[row, sample])
                ),
                selected_lobe=str(item.get("selected_lobe") or "unknown"),
                canonical_offset_samples=float(
                    (item.get("canonical_sample_index") or sample) - sample
                ),
                event_id=(str(item["event_id"]) if item.get("event_id") else None),
                regime_id=str(item.get("regime_id") or "default"),
            )
        )
    return output


def _template_feature_maps(
    data: NDArray[np.float32], prototypes: list[WaveformPrototype]
) -> tuple[dict[str, NDArray[np.float32]], list[NDArray[np.float32]]]:
    analytic = hilbert(data, axis=1)
    phase_classes = _phase_class(np.angle(analytic))
    per_prototype: list[NDArray[np.float32]] = []
    for prototype in prototypes:
        template = np.asarray(prototype.real_waveform, dtype=np.float32)
        correlation = fftconvolve(data, template[None, ::-1], mode="same", axes=1)
        energy = np.sqrt(
            np.maximum(
                uniform_filter1d(
                    np.square(data, dtype=np.float64),
                    size=len(template),
                    axis=1,
                    mode="nearest",
                )
                * len(template),
                1e-10,
            )
        )
        correlation = np.asarray(np.clip(correlation / energy, -1.0, 1.0), dtype=np.float32)
        per_prototype.append(correlation)
    if prototypes:
        correlations = np.stack(per_prototype, axis=0)
        best_prototype = np.argmax(correlations, axis=0).astype(np.int16)
        best_signed = np.take_along_axis(
            correlations, best_prototype[None, ...], axis=0
        )[0]
        signed = np.asarray(np.clip(best_signed, 0.0, 1.0), dtype=np.float32)
        absolute = np.asarray(np.max(np.abs(correlations), axis=0), dtype=np.float32)
        positive_correlations = np.clip(correlations, 0.0, 1.0)
        if len(prototypes) >= 2:
            second = np.partition(positive_correlations, -2, axis=0)[-2]
            drop_seed_stability = np.asarray(second, dtype=np.float32)
        else:
            drop_seed_stability = np.asarray(signed, dtype=np.float32)
        prototype_phases = np.asarray(
            [prototype.phase_class for prototype in prototypes], dtype=np.int8
        )
        selected_phase = prototype_phases[best_prototype]
        distance = np.abs(phase_classes.astype(int) - selected_phase.astype(int))
        distance = np.minimum(distance, 8 - distance)
        phase_agreement = np.asarray(
            0.5 + 0.5 * np.cos(distance * np.pi / 4.0), dtype=np.float32
        )
        prototype_polarities = np.asarray(
            [prototype.polarity for prototype in prototypes], dtype=np.int8
        )
        selected_polarity = prototype_polarities[best_prototype]
        polarity_agreement = np.asarray(
            np.sign(data) == selected_polarity, dtype=np.float32
        )
        prototype_offsets = np.asarray(
            [prototype.canonical_offset_samples for prototype in prototypes],
            dtype=np.float32,
        )
        canonical_offset = prototype_offsets[best_prototype]
    else:
        signed = np.zeros_like(data, dtype=np.float32)
        absolute = np.zeros_like(data, dtype=np.float32)
        phase_agreement = np.full_like(data, 0.5, dtype=np.float32)
        polarity_agreement = np.full_like(data, 0.5, dtype=np.float32)
        best_prototype = np.full(data.shape, -1, dtype=np.int16)
        drop_seed_stability = np.zeros_like(data, dtype=np.float32)
        canonical_offset = np.zeros_like(data, dtype=np.float32)
    return (
        {
            "signed_seed_correlation": signed,
            "absolute_seed_correlation": absolute,
            "phase_cycle_agreement": phase_agreement,
            "polarity_agreement": polarity_agreement,
            "prototype_index": best_prototype,
            "drop_seed_stability": drop_seed_stability,
            "canonical_offset_samples": canonical_offset,
        },
        per_prototype,
    )


def _branch(
    branches: dict[str, NDArray[np.floating]] | None,
    name: str,
    fallback: NDArray[np.floating],
) -> NDArray[np.float32]:
    if branches is not None and name in branches:
        return np.asarray(branches[name], dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32)


def _component_maps(
    original: NDArray[np.float32],
    residual: NDArray[np.float32],
    branches: dict[str, NDArray[np.floating]] | None,
    prototypes: list[WaveformPrototype],
    pulse_width: float,
) -> tuple[dict[str, NDArray[np.float32]], list[NDArray[np.float32]]]:
    analytic_residual = hilbert(residual, axis=1)
    residual_envelope_raw = np.abs(analytic_residual).astype(np.float32)
    original_envelope_raw = np.abs(hilbert(original, axis=1)).astype(np.float32)
    residual_envelope = _normalise_rows(residual_envelope_raw)
    gradient = _normalise_rows(np.abs(np.gradient(residual, axis=1)))
    reflectivity_raw = np.abs(_branch(branches, "reflectivity", residual))
    reflectivity = _normalise_rows(reflectivity_raw)
    coherent_sum = np.abs(
        uniform_filter1d(analytic_residual, size=7, axis=0, mode="nearest")
    )
    coherent_amplitude = uniform_filter1d(
        np.abs(analytic_residual), size=7, axis=0, mode="nearest"
    )
    coherence_fallback = np.asarray(
        coherent_sum / np.maximum(coherent_amplitude, 1e-6), dtype=np.float32
    )
    oriented = _branch(branches, "oriented_coherence", coherence_fallback)
    residual_reference = max(float(np.percentile(residual_envelope_raw, 97.5)), 1e-7)
    original_reference = max(float(np.percentile(original_envelope_raw, 97.5)), 1e-7)
    residual_absolute = residual_envelope_raw / residual_reference
    original_absolute = original_envelope_raw / original_reference
    trace_strength = np.clip(
        np.percentile(original_envelope_raw, 97.0, axis=1) / original_reference,
        0.0,
        1.0,
    )
    original_trace_relative = (
        0.25 * _normalise_rows(original_envelope_raw) * trace_strength[:, None]
    )
    # Visibility is radar evidence, not subtraction evidence.  Stripping may
    # reveal a deep interface, but it must never erase an interface that is
    # plainly supported in the original calibrated trace.
    absolute = np.asarray(
        np.clip(
            np.maximum.reduce(
                (residual_absolute, original_absolute, original_trace_relative)
            ),
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    stripped_gain = np.asarray(
        np.clip(
            _normalise_rows(residual_envelope_raw)
            - _normalise_rows(original_envelope_raw)
            + 0.5,
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )
    independent_branches = [residual]
    if branches is not None:
        independent_branches.extend(
            np.asarray(branches[name], dtype=np.float32)
            for name in ("background_full", "wavelet_amplitude", "deconvolved")
            if name in branches
        )
    # Agreement is defined at reflection-event scale.  Comparing amplitudes at
    # the exact selected lobe made agreement identically zero whenever a branch
    # shifted the peak by only a few samples.
    event_window = max(3, int(round(2.0 * pulse_width)) | 1)
    if len(independent_branches) >= 3:
        branch_support = []
        for values in independent_branches:
            normalised = _normalise_rows(np.abs(values))
            local_peak = normalised >= maximum_filter1d(
                normalised, size=max(3, int(round(0.7 * pulse_width)) | 1), axis=1
            )
            strong_peak = local_peak & (normalised >= 0.14)
            branch_support.append(
                maximum_filter1d(
                    strong_peak.astype(np.float32), size=event_window, axis=1
                )
            )
        branch_support = np.stack(branch_support, axis=0)
        preprocessing_agreement = np.mean(branch_support, axis=0, dtype=np.float32)
    else:
        preprocessing_agreement = np.zeros_like(residual, dtype=np.float32)
    # Event-family identity must not depend on one stripping hypothesis.  In
    # particular, a slightly misplaced asphalt subtraction can distort the
    # deeper base waveform even though the original radar still contains the
    # correct event.  Use the original branch as the phase/polarity reference
    # and let the stripped residual add support when it improves correlation.
    original_template_maps, original_per_prototype = _template_feature_maps(
        original, prototypes
    )
    residual_template_maps, residual_per_prototype = _template_feature_maps(
        residual, prototypes
    )
    prefer_original = (
        original_template_maps["signed_seed_correlation"]
        >= residual_template_maps["signed_seed_correlation"]
    )
    template_maps = {
        "signed_seed_correlation": np.maximum(
            original_template_maps["signed_seed_correlation"],
            residual_template_maps["signed_seed_correlation"],
        ),
        "absolute_seed_correlation": np.maximum(
            original_template_maps["absolute_seed_correlation"],
            residual_template_maps["absolute_seed_correlation"],
        ),
        "phase_cycle_agreement": original_template_maps["phase_cycle_agreement"],
        "polarity_agreement": original_template_maps["polarity_agreement"],
        "prototype_index": np.where(
            prefer_original,
            original_template_maps["prototype_index"],
            residual_template_maps["prototype_index"],
        ).astype(np.int16),
        "drop_seed_stability": np.maximum(
            original_template_maps["drop_seed_stability"],
            residual_template_maps["drop_seed_stability"],
        ),
        "canonical_offset_samples": np.where(
            prefer_original,
            original_template_maps["canonical_offset_samples"],
            residual_template_maps["canonical_offset_samples"],
        ).astype(np.float32),
    }
    per_prototype = [
        np.maximum(original_values, residual_values)
        for original_values, residual_values in zip(
            original_per_prototype, residual_per_prototype, strict=True
        )
    ]
    morphology = np.asarray(
        0.20 * residual_envelope
        + 0.14 * gradient
        + 0.22 * reflectivity
        + 0.24 * oriented
        + 0.12 * absolute
        + 0.05 * stripped_gain
        + 0.03 * preprocessing_agreement,
        dtype=np.float32,
    )
    if prototypes:
        # Unsigned morphology locates a reflection packet; it must not decide
        # which half-cycle/lobe represents the user-confirmed interface.
        identity = np.asarray(
            0.58 * template_maps["signed_seed_correlation"]
            + 0.27 * template_maps["phase_cycle_agreement"]
            + 0.15 * template_maps["polarity_agreement"],
            dtype=np.float32,
        )
        polarity_gate = 0.10 + 0.90 * template_maps["polarity_agreement"]
        morphology = morphology * polarity_gate
        generic = 0.72 * identity + 0.28 * morphology * (0.15 + 0.85 * identity)
    else:
        generic = morphology
    maps = {
        **template_maps,
        "oriented_coherence": np.clip(oriented, 0.0, 1.0),
        "reflectivity_strength": reflectivity,
        "residual_envelope": residual_envelope,
        "vertical_gradient": gradient,
        "absolute_strength": absolute,
        "stripped_gain": stripped_gain,
        "preprocessing_agreement": preprocessing_agreement,
        "tracklet_support": np.zeros_like(residual, dtype=np.float32),
        "seed_distance_support": np.zeros_like(residual, dtype=np.float32),
        "cycle_slip_safety": np.full_like(residual, 0.5, dtype=np.float32),
        "cycle_slip_risk": np.full_like(residual, 0.5, dtype=np.float32),
        "generic_radar_score": np.clip(generic, 0.0, 1.0),
        "analytic_phase_rad": np.asarray(
            np.angle(hilbert(original, axis=1)), dtype=np.float32
        ),
    }
    return maps, per_prototype


def _filled_path(path: NDArray[np.integer] | None, default: float) -> NDArray[np.float64]:
    if path is None:
        return np.full(0, default, dtype=float)
    values = np.asarray(path, dtype=float)
    finite = values >= 0
    if np.count_nonzero(finite) < 2:
        fill = float(values[finite][0]) if np.any(finite) else default
        return np.full(len(values), fill, dtype=float)
    rows = np.arange(len(values), dtype=float)
    return np.interp(rows, rows[finite], values[finite])


def _search_bounds(
    rows: int,
    samples: int,
    reference_surface_sample: int,
    layer: LayerSpec,
    previous: NDArray[np.integer] | None,
    corridor: SearchCorridor | None,
    anchors: dict[int, int],
    pulse_width: float,
    horizontal_step_m: float,
    previous_candidate_bounds: tuple[NDArray[np.floating], NDArray[np.floating]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    lower = np.full(rows, reference_surface_sample + layer.min_offset_samples, dtype=float)
    upper = np.full(
        rows,
        min(samples - 1, reference_surface_sample + layer.max_offset_samples),
        dtype=float,
    )
    centre = (lower + upper) / 2.0
    previous_guide = (
        _filled_path(previous, reference_surface_sample)
        if previous is not None
        else np.full(rows, reference_surface_sample, dtype=float)
    )
    previous_lower = previous_guide
    previous_upper = previous_guide
    if previous_candidate_bounds is not None:
        previous_lower = np.asarray(previous_candidate_bounds[0], dtype=float)
        previous_upper = np.asarray(previous_candidate_bounds[1], dtype=float)
    if previous is not None or previous_candidate_bounds is not None:
        lower = np.maximum(lower, previous_lower + layer.min_gap_samples)
    if corridor is not None:
        finite = (
            np.isfinite(corridor.gap_lower_samples)
            & np.isfinite(corridor.gap_centre_samples)
            & np.isfinite(corridor.gap_upper_samples)
        )
        candidate_lower = previous_lower + corridor.gap_lower_samples
        candidate_centre = (
            (previous_lower + previous_upper) / 2.0 + corridor.gap_centre_samples
        )
        candidate_upper = previous_upper + corridor.gap_upper_samples
        lower[finite] = np.maximum(lower[finite], candidate_lower[finite])
        upper[finite] = np.minimum(upper[finite], candidate_upper[finite])
        centre[finite] = candidate_centre[finite]
    local_rows = max(2, int(round(10.0 / max(horizontal_step_m, 1e-3))))
    for anchor_row, anchor_sample in anchors.items():
        start = max(0, anchor_row - local_rows)
        stop = min(rows, anchor_row + local_rows + 1)
        distance = np.abs(np.arange(start, stop) - anchor_row)
        reach = 2.0 * pulse_width + 1.5 * distance
        lower[start:stop] = np.minimum(lower[start:stop], anchor_sample - reach)
        upper[start:stop] = np.maximum(upper[start:stop], anchor_sample + reach)
        centre[anchor_row] = anchor_sample
    lower = np.clip(lower, 0, samples - 1)
    upper = np.clip(np.maximum(upper, lower + 1), 0, samples - 1)
    return lower, centre, upper


def _estimate_local_pulse_width(
    envelope: NDArray[np.float32], sample: int, fallback: float
) -> float:
    radius = max(3, int(round(2.0 * fallback)))
    start = max(0, sample - radius)
    stop = min(len(envelope), sample + radius + 1)
    local = envelope[start:stop]
    if not len(local):
        return float(fallback)
    peak = float(envelope[sample])
    if peak <= 1e-7:
        return float(fallback)
    above = np.flatnonzero(local >= 0.5 * peak)
    if not len(above):
        return float(fallback)
    width = float(above[-1] - above[0] + 1)
    return float(np.clip(width, 0.55 * fallback, 2.0 * fallback))


def _lobe_code(value: float) -> int:
    if value < 0:
        return 1
    if value > 0:
        return 2
    return 3


def _packet_representatives(
    candidates: NDArray[np.int64],
    generic: NDArray[np.float32],
    reflectivity: NDArray[np.float32],
    identity: NDArray[np.float32],
    anchor_sample: int | None,
    pulse_width: float,
) -> list[tuple[int, int, list[int]]]:
    """Group observable lobes into reflection packets.

    The returned tuples contain selected lobe, canonical reflectivity time, and
    every candidate sample belonging to the packet.  Greedy packet centres are
    used instead of transitive clustering so a run of weak intermediate pixels
    cannot merge two distinct reflections.
    """
    remaining = {int(sample) for sample in candidates}
    packets: list[tuple[int, int, list[int]]] = []
    # One event packet spans the complete emitted pulse (both extrema and zero
    # crossings), not merely the distance to the adjacent pixel peak.
    packet_radius = max(3, int(round(1.7 * pulse_width)))
    while remaining:
        centre = max(remaining, key=lambda sample: float(generic[sample]))
        members = sorted(
            sample for sample in remaining if abs(sample - centre) <= packet_radius
        )
        for sample in members:
            remaining.discard(sample)
        canonical = max(members, key=lambda sample: float(reflectivity[sample]))
        selected = max(members, key=lambda sample: float(identity[sample]))
        if anchor_sample is not None and abs(anchor_sample - canonical) <= packet_radius:
            selected = int(anchor_sample)
            if selected not in members:
                members.append(selected)
                members.sort()
        packets.append((selected, canonical, members))
    return packets


def _candidate_table(
    original: NDArray[np.float32],
    residual: NDArray[np.float32],
    maps: dict[str, NDArray[np.float32]],
    lower: NDArray[np.float64],
    centre: NDArray[np.float64],
    upper: NDArray[np.float64],
    anchors: dict[int, int],
    anomaly_mask: NDArray[np.bool_],
    pulse_width: float,
    limit: int = 12,
) -> _CandidateTable:
    rows, sample_count = residual.shape
    candidate_sources = (
        np.abs(original),
        np.abs(residual),
        maps["residual_envelope"],
        maps["vertical_gradient"],
        maps["reflectivity_strength"],
        maps["oriented_coherence"],
        maps["signed_seed_correlation"],
    )
    maxima = [source >= maximum_filter1d(source, size=3, axis=1) for source in candidate_sources]
    candidate_union = np.logical_or.reduce(maxima)
    sample_axis = np.arange(sample_count)[None, :]
    corridor_mask = (sample_axis >= lower[:, None]) & (sample_axis <= upper[:, None])
    candidate_union &= corridor_mask
    candidate_union[anomaly_mask] = False
    for row, sample in anchors.items():
        if 0 <= row < rows and 0 <= sample < sample_count:
            candidate_union[row, sample] = True
    samples = np.full((rows, limit + 1), -1, dtype=np.int32)
    valid = np.zeros((rows, limit + 1), dtype=bool)
    feature_values = np.zeros((rows, limit + 1, len(_FEATURE_NAMES)), dtype=np.float32)
    radius = max(5, int(round(1.5 * pulse_width)))
    waveform_length = 2 * radius + 1
    waveforms = np.zeros((rows, limit + 1, waveform_length), dtype=np.float32)
    canonical_samples = np.full((rows, limit + 1), -1.0, dtype=np.float32)
    phase_classes = np.full((rows, limit + 1), -1, dtype=np.int8)
    polarities = np.zeros((rows, limit + 1), dtype=np.int8)
    pulse_widths = np.zeros((rows, limit + 1), dtype=np.float32)
    prototype_indices = np.full((rows, limit + 1), -1, dtype=np.int16)
    selected_lobes = np.zeros((rows, limit + 1), dtype=np.int8)
    tracklet_support = np.zeros((rows, limit + 1), dtype=np.float32)
    cycle_slip_risk = np.full((rows, limit + 1), 0.5, dtype=np.float32)
    seed_distance_support = np.zeros((rows, limit + 1), dtype=np.float32)
    family_indices = np.full((rows, limit + 1), -1, dtype=np.int16)
    regime_indices = np.full((rows, limit + 1), -1, dtype=np.int16)
    design_map = np.zeros_like(residual, dtype=np.float32)
    half_width = np.maximum((upper - lower) / 2.0, 1.0)
    finite_centre = np.isfinite(centre)
    design_map[finite_centre] = np.exp(
        -0.5
        * np.square(
            (sample_axis[0] - centre[finite_centre, None])
            / np.maximum(0.55 * half_width[finite_centre, None], 1.0)
        )
    )
    maps = {**maps, "design_tiebreak": design_map}
    generic = maps["generic_radar_score"]
    for row in range(rows):
        if anomaly_mask[row]:
            valid[row, -1] = True
            continue
        candidates = np.flatnonzero(candidate_union[row])
        if not len(candidates):
            in_corridor = np.flatnonzero(corridor_mask[row])
            if len(in_corridor):
                candidates = np.asarray(
                    [in_corridor[int(np.argmax(generic[row, in_corridor]))]], dtype=int
                )
        anchor_sample = anchors.get(row)
        identity = np.asarray(
            0.62 * maps["signed_seed_correlation"][row]
            + 0.23 * maps["phase_cycle_agreement"][row]
            + 0.15 * maps["polarity_agreement"][row],
            dtype=np.float32,
        )
        if not np.any(maps["signed_seed_correlation"][row] > 0):
            identity = generic[row]
        packets = _packet_representatives(
            np.asarray(candidates, dtype=np.int64),
            generic[row],
            maps["reflectivity_strength"][row],
            identity,
            anchor_sample,
            pulse_width,
        )
        if packets:
            strongest_packet = max(float(generic[row, packet[0]]) for packet in packets)
            packets = [
                packet
                for packet in packets
                if packet[0] == anchor_sample
                or float(generic[row, packet[0]]) >= max(0.08, 0.35 * strongest_packet)
            ]
        packets.sort(
            key=lambda packet: (
                anchor_sample is not None and packet[0] == anchor_sample,
                float(generic[row, packet[0]]),
            ),
            reverse=True,
        )
        for index, (sample, canonical, _members) in enumerate(packets[:limit]):
            prototype_index = int(maps["prototype_index"][row, sample])
            if (
                prototype_index >= 0
                and maps["signed_seed_correlation"][row, sample] >= 0.18
            ):
                canonical = int(
                    round(sample + maps["canonical_offset_samples"][row, sample])
                )
            samples[row, index] = sample
            valid[row, index] = True
            feature_values[row, index] = [maps[name][row, sample] for name in _FEATURE_NAMES]
            canonical_samples[row, index] = float(canonical)
            phase_classes[row, index] = int(
                _phase_class(maps["analytic_phase_rad"][row, sample])
            )
            polarities[row, index] = int(np.sign(original[row, sample]))
            pulse_widths[row, index] = _estimate_local_pulse_width(
                maps["residual_envelope"][row], canonical, pulse_width
            )
            prototype_indices[row, index] = int(maps["prototype_index"][row, sample])
            selected_lobes[row, index] = _lobe_code(float(residual[row, sample]))
            if radius <= sample < sample_count - radius:
                waveform = _normalise_waveform(
                    original[row, sample - radius : sample + radius + 1]
                )
                if waveform is not None:
                    waveforms[row, index] = waveform
        valid[row, -1] = True
    packet_canonical_map = np.full_like(residual, np.nan, dtype=np.float32)
    packet_width_map = np.zeros_like(residual, dtype=np.float32)
    packet_lobe_map = np.zeros_like(residual, dtype=np.float32)
    packet_prototype_map = np.full_like(residual, -1.0, dtype=np.float32)
    for row in range(rows):
        for index in np.flatnonzero(valid[row, :-1]):
            sample = int(samples[row, index])
            packet_canonical_map[row, sample] = canonical_samples[row, index]
            packet_width_map[row, sample] = pulse_widths[row, index]
            packet_lobe_map[row, sample] = selected_lobes[row, index]
            packet_prototype_map[row, sample] = prototype_indices[row, index]
    maps = {
        **maps,
        "event_canonical_sample": packet_canonical_map,
        "event_pulse_width": packet_width_map,
        "event_lobe_code": packet_lobe_map,
        "event_prototype_index": packet_prototype_map,
    }
    return _CandidateTable(
        samples=samples,
        valid=valid,
        features=feature_values,
        waveforms=waveforms,
        canonical_samples=canonical_samples,
        phase_classes=phase_classes,
        polarities=polarities,
        pulse_widths=pulse_widths,
        prototype_indices=prototype_indices,
        selected_lobes=selected_lobes,
        dense_radar_score=generic,
        component_maps=maps,
        feature_names=_FEATURE_NAMES,
        tracklet_support=tracklet_support,
        cycle_slip_risk=cycle_slip_risk,
        seed_distance_support=seed_distance_support,
        family_indices=family_indices,
        regime_indices=regime_indices,
    )


def _propagated_positives(
    table: _CandidateTable,
    anchors: dict[int, int],
    pulse_width: float,
    horizontal_step_m: float,
) -> dict[int, int]:
    output = dict(anchors)
    correlation_index = table.feature_names.index("signed_seed_correlation")
    phase_index = table.feature_names.index("phase_cycle_agreement")
    coherence_index = table.feature_names.index("oriented_coherence")
    maximum_rows = max(1, int(round(10.0 / max(horizontal_step_m, 1e-3))))
    maximum_step = max(2, int(round(0.65 * pulse_width)))
    for anchor_row, anchor_sample in anchors.items():
        for direction in (-1, 1):
            previous_sample = anchor_sample
            for distance in range(1, maximum_rows + 1):
                row = anchor_row + direction * distance
                if not 0 <= row < len(table.samples):
                    break
                indices = np.flatnonzero(table.valid[row, :-1])
                if not len(indices):
                    break
                candidate_samples = table.samples[row, indices]
                indices = indices[np.abs(candidate_samples - previous_sample) <= maximum_step]
                if not len(indices):
                    break
                features = table.features[row, indices]
                score = (
                    0.65 * features[:, correlation_index]
                    + 0.20 * features[:, phase_index]
                    + 0.15 * features[:, coherence_index]
                )
                selected = int(indices[int(np.argmax(score))])
                if (
                    table.features[row, selected, correlation_index] < 0.28
                    or float(np.max(score)) < 0.40
                ):
                    break
                previous_sample = int(table.samples[row, selected])
                output[row] = previous_sample
    return output


def _phase_distance(left: int, right: int) -> float:
    if left < 0 or right < 0:
        return 1.0
    distance = abs(int(left) - int(right))
    return float(min(distance, 8 - distance) / 4.0)


def _grow_phase_locked_tracklets(
    table: _CandidateTable,
    anchors: dict[int, int],
    metadata: dict[int, dict[str, object]] | None,
    layer_order: int,
    pulse_width: float,
    horizontal_step_m: float,
) -> list[PhaseLockedTracklet]:
    """Grow conservative, adaptive event-family continuations from confirmed seeds.

    Template updates are accepted only after a high-support, well-separated
    continuation.  A polarity/phase-cycle change terminates the tracklet so a
    graph transition cannot silently walk to the adjacent wavelet lobe.
    """
    if not anchors:
        return []
    waveform_index = table.feature_names.index("signed_seed_correlation")
    coherence_index = table.feature_names.index("oriented_coherence")
    generic_index = table.feature_names.index("generic_radar_score")
    tracklet_feature = table.feature_names.index("tracklet_support")
    distance_feature = table.feature_names.index("seed_distance_support")
    safety_feature = table.feature_names.index("cycle_slip_safety")
    dx = max(float(horizontal_step_m), 1e-3)
    family_lookup: dict[tuple[str, str, int, int], int] = {}
    regime_lookup: dict[str, int] = {}
    output: list[PhaseLockedTracklet] = []

    for anchor_row, _anchor_sample in sorted(anchors.items()):
        item = (metadata or {}).get(anchor_row, {})
        regime_id = str(item.get("regime_id") or "default")
        phase = int(item.get("phase_class") if item.get("phase_class") is not None else -1)
        polarity = int(item.get("polarity") or 0)
        selected_lobe = str(item.get("selected_lobe") or "unknown")
        family_lookup.setdefault(
            (regime_id, selected_lobe, phase, polarity), len(family_lookup)
        )
        regime_lookup.setdefault(regime_id, len(regime_lookup))

    for anchor_row, anchor_sample in sorted(anchors.items()):
        seed_indices = np.flatnonzero(table.valid[anchor_row, :-1])
        if not len(seed_indices):
            continue
        seed_index = int(
            seed_indices[
                int(np.argmin(np.abs(table.samples[anchor_row, seed_indices] - anchor_sample)))
            ]
        )
        item = (metadata or {}).get(anchor_row, {})
        local_pulse_width = float(item.get("pulse_width_samples") or pulse_width)
        maximum_step = max(2.0, 0.65 * local_pulse_width)
        station_id = str(item.get("station_id") or f"row-{anchor_row}")
        regime_id = str(item.get("regime_id") or "default")
        regime_index = regime_lookup.setdefault(regime_id, len(regime_lookup))
        seed_phase = int(
            item.get("phase_class")
            if item.get("phase_class") is not None
            else table.phase_classes[anchor_row, seed_index]
        )
        seed_polarity = int(
            item.get("polarity")
            if item.get("polarity") is not None
            else table.polarities[anchor_row, seed_index]
        )
        selected_lobe = str(item.get("selected_lobe") or "unknown")
        family_key = (regime_id, selected_lobe, seed_phase, seed_polarity)
        family_index = int(table.family_indices[anchor_row, seed_index])
        if family_index < 0:
            family_index = family_lookup.setdefault(family_key, len(family_lookup))
        seed_waveform = table.waveforms[anchor_row, seed_index].copy()
        if float(np.linalg.norm(seed_waveform)) < 1e-6:
            continue

        selected: dict[int, tuple[int, float, float]] = {
            anchor_row: (seed_index, 1.0, 0.0)
        }
        stop_reasons: list[str] = []
        for direction in (-1, 1):
            template = seed_waveform.copy()
            previous_sample = float(anchor_sample)
            previous_slope = 0.0
            gap_rows = 0
            stopped_reason = "road boundary"
            row = anchor_row + direction
            while 0 <= row < len(table.samples):
                candidates = np.flatnonzero(table.valid[row, :-1])
                if not len(candidates):
                    gap_rows += 1
                    if gap_rows > 2:
                        stopped_reason = "evidence gap"
                        break
                    row += direction
                    continue
                expected = previous_sample + direction * previous_slope
                reach = maximum_step * (1.0 + 0.45 * gap_rows)
                candidates = candidates[
                    np.abs(table.samples[row, candidates] - expected) <= reach
                ]
                if not len(candidates):
                    gap_rows += 1
                    if gap_rows > 2:
                        stopped_reason = "unreachable event family"
                        break
                    row += direction
                    continue
                waveforms = table.waveforms[row, candidates]
                similarities = _waveform_similarity_matrix(
                    waveforms, template[None, :], maximum_shift=2
                )[:, 0]
                phase_distances = np.asarray(
                    [
                        _phase_distance(table.phase_classes[row, index], seed_phase)
                        for index in candidates
                    ]
                )
                polarity_ok = table.polarities[row, candidates] == seed_polarity
                # Waveform character and analytic phase may rotate when thin
                # reflections overlap.  The selected lobe polarity remains
                # hard; phase and shape contribute to support and slip risk.
                # Canonical packet timing can survive a displayed-lobe reversal
                # caused by thin-layer interference.  Keep it as a penalised
                # hypothesis and expose the reversal as cycle-slip risk.
                valid = np.ones_like(polarity_ok, dtype=bool)
                if not np.any(valid):
                    stopped_reason = "phase-cycle or polarity boundary"
                    break
                candidates = candidates[valid]
                similarities = similarities[valid]
                phase_distances = phase_distances[valid]
                features = table.features[row, candidates]
                slope_cost = np.abs(table.samples[row, candidates] - expected) / max(reach, 1.0)
                score = (
                    0.38 * np.clip((similarities + 1.0) / 2.0, 0.0, 1.0)
                    + 0.23 * features[:, waveform_index]
                    + 0.15 * (1.0 - phase_distances)
                    + 0.10 * features[:, coherence_index]
                    + 0.09 * features[:, generic_index]
                    - 0.05 * slope_cost
                    - 0.12 * (~polarity_ok[valid])
                )
                ranking = np.argsort(score)[::-1]
                chosen_position = int(ranking[0])
                chosen = int(candidates[chosen_position])
                best_score = float(score[chosen_position])
                second_score = float(score[ranking[1]]) if len(ranking) > 1 else 0.0
                margin = best_score - second_score
                minimum_support = 0.26 if layer_order >= 2 else 0.34
                if best_score < minimum_support:
                    stopped_reason = "weak event-family support"
                    break
                similarity = float(similarities[chosen_position])
                slip_risk = float(
                    np.clip(
                        0.28 * phase_distances[chosen_position]
                        + 0.22 * (1.0 - np.clip((similarity + 1.0) / 2.0, 0.0, 1.0))
                        + 0.18 * np.clip((0.10 - margin) / 0.10, 0.0, 1.0)
                        + 0.45 * (not bool(polarity_ok[valid][chosen_position])),
                        0.0,
                        1.0,
                    )
                )
                support = float(np.clip(best_score * (1.0 - 0.55 * slip_risk), 0.0, 1.0))
                selected[row] = (chosen, support, slip_risk)
                new_sample = float(table.samples[row, chosen])
                # Preserve the signed derivative in increasing-row
                # coordinates.  Using abs() here forced forward growth to
                # deepen and backward growth to shallow, so a real undulating
                # interface broke whenever its slope changed sign.
                observed_slope = direction * (new_sample - previous_sample)
                previous_slope = 0.75 * previous_slope + 0.25 * observed_slope
                previous_sample = new_sample
                gap_rows = 0
                # Conservative local adaptation: a doubtful update never
                # replaces the last stable prototype.
                if (
                    support >= 0.62
                    and margin >= 0.05
                    and bool(polarity_ok[valid][chosen_position])
                ):
                    candidate_waveform = table.waveforms[row, chosen]
                    adaptation = 0.05 if layer_order >= 2 else 0.08
                    updated = (1.0 - adaptation) * template + adaptation * candidate_waveform
                    norm = float(np.linalg.norm(updated))
                    if norm > 1e-7:
                        template = np.asarray(updated / norm, dtype=np.float32)
                row += direction
            stop_reasons.append(stopped_reason)

        rows = np.asarray(sorted(selected), dtype=np.int32)
        indices = np.asarray([selected[int(row)][0] for row in rows], dtype=np.int16)
        support = np.asarray([selected[int(row)][1] for row in rows], dtype=np.float32)
        slip = np.asarray([selected[int(row)][2] for row in rows], dtype=np.float32)
        samples = table.samples[rows, indices].astype(np.int32)
        canonical = table.canonical_samples[rows, indices].astype(np.float32)
        tracklet_id = f"L{layer_order}:{regime_id}:{station_id}"
        for row, index, value, risk in zip(rows, indices, support, slip, strict=True):
            row_i, index_i = int(row), int(index)
            if value >= table.tracklet_support[row_i, index_i]:
                table.tracklet_support[row_i, index_i] = value
                table.cycle_slip_risk[row_i, index_i] = risk
                distance = np.exp(-abs(row_i - anchor_row) * dx / 300.0)
                table.seed_distance_support[row_i, index_i] = float(distance)
                table.family_indices[row_i, index_i] = family_index
                table.regime_indices[row_i, index_i] = regime_index
                table.features[row_i, index_i, tracklet_feature] = value
                table.features[row_i, index_i, distance_feature] = float(distance)
                table.features[row_i, index_i, safety_feature] = 1.0 - risk
        output.append(
            PhaseLockedTracklet(
                tracklet_id=tracklet_id,
                layer_order=layer_order,
                seed_station_id=station_id,
                regime_id=regime_id,
                rows=rows,
                samples=samples,
                canonical_samples=canonical,
                support=support,
                phase_classes=table.phase_classes[rows, indices].copy(),
                polarities=table.polarities[rows, indices].copy(),
                cycle_slip_risk=slip,
                stopped_reason="; ".join(stop_reasons),
            )
        )

    family_map = np.full_like(table.dense_radar_score, -1.0, dtype=np.float32)
    regime_map = np.full_like(table.dense_radar_score, -1.0, dtype=np.float32)
    for row in range(len(table.samples)):
        for index in np.flatnonzero(table.valid[row, :-1]):
            sample = int(table.samples[row, index])
            family_map[row, sample] = float(table.family_indices[row, index])
            regime_map[row, sample] = float(table.regime_indices[row, index])
    table.component_maps["event_family_index"] = family_map
    table.component_maps["event_regime_index"] = regime_map
    for name, values in (
        ("tracklet_support", table.tracklet_support),
        ("seed_distance_support", table.seed_distance_support),
        ("cycle_slip_risk", table.cycle_slip_risk),
    ):
        dense = np.zeros_like(table.dense_radar_score, dtype=np.float32)
        for row in range(len(table.samples)):
            for index in np.flatnonzero(table.valid[row, :-1]):
                dense[row, table.samples[row, index]] = values[row, index]
        table.component_maps[name] = dense
    return output


def _fixed_radar_score(
    features: NDArray[np.float32], seeded: bool, layer_order: int
) -> NDArray[np.float32]:
    del layer_order
    if seeded:
        weights = np.asarray(
            [
                0.25,
                0.00,
                0.15,
                0.05,
                0.06,
                0.05,
                0.04,
                0.03,
                0.02,
                0.03,
                0.07,
                0.08,
                0.04,
                0.03,
                0.05,
                0.05,
            ],
            dtype=np.float32,
        )
    else:
        weights = np.asarray(
            [
                0.00,
                0.00,
                0.00,
                0.00,
                0.20,
                0.16,
                0.14,
                0.10,
                0.14,
                0.04,
                0.12,
                0.00,
                0.00,
                0.00,
                0.00,
                0.10,
            ],
            dtype=np.float32,
        )
    return np.asarray(np.clip(features[..., :-1] @ weights, 0.0, 1.0), dtype=np.float32)


def _fit_candidate_ranker(
    table: _CandidateTable,
    positives: dict[int, int],
    fixed: NDArray[np.float32],
    pulse_width: float,
) -> NDArray[np.float32]:
    positive_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    for row, sample in positives.items():
        indices = np.flatnonzero(table.valid[row, :-1])
        if not len(indices):
            continue
        selected = int(indices[int(np.argmin(np.abs(table.samples[row, indices] - sample)))])
        if abs(int(table.samples[row, selected]) - sample) > 2:
            continue
        positive_rows.append(table.features[row, selected, :-1])
        negative_indices = indices[
            np.abs(table.samples[row, indices] - sample) >= max(3, int(0.6 * pulse_width))
        ]
        if len(negative_indices):
            ranked = negative_indices[np.argsort(fixed[row, negative_indices])[::-1]][:4]
            negative_rows.extend(table.features[row, ranked, :-1])
    if len(positive_rows) < 6 or len(negative_rows) < 12:
        return fixed
    positive = np.asarray(positive_rows, dtype=float)
    negative = np.asarray(negative_rows, dtype=float)
    x = np.vstack((positive, negative))
    y = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    x = np.column_stack((np.ones(len(x)), x))
    sample_weight = np.concatenate(
        (
            np.full(len(positive), 0.5 / len(positive)),
            np.full(len(negative), 0.5 / len(negative)),
        )
    )
    coefficients = np.zeros(x.shape[1], dtype=float)
    regularization = np.eye(x.shape[1]) * 1.5
    regularization[0, 0] = 0.2
    for _ in range(18):
        probability = 1.0 / (1.0 + np.exp(-np.clip(x @ coefficients, -20.0, 20.0)))
        variance = np.maximum(probability * (1.0 - probability), 1e-4)
        gradient = x.T @ (sample_weight * (y - probability)) - regularization @ coefficients
        hessian = x.T @ ((sample_weight * variance)[:, None] * x) + regularization
        update = np.linalg.solve(hessian, gradient)
        coefficients += update
        if float(np.linalg.norm(update)) < 1e-5:
            break
    all_features = table.features[..., :-1].astype(float)
    logit = coefficients[0] + np.tensordot(all_features, coefficients[1:], axes=([-1], [0]))
    learned = 1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0)))
    return np.asarray(np.clip(0.55 * fixed + 0.45 * learned, 0.0, 1.0), dtype=np.float32)


def _seed_conflicts(
    table: _CandidateTable,
    anchors: dict[int, int],
    per_prototype: list[NDArray[np.float32]],
    pulse_width: float,
) -> set[int]:
    if len(anchors) < 3 or len(per_prototype) < 3:
        return set()
    anchor_rows = sorted(anchors)
    output: set[int] = set()
    for prototype_index, row in enumerate(anchor_rows[: len(per_prototype)]):
        sample = anchors[row]
        other = [
            values
            for index, values in enumerate(per_prototype)
            if index != prototype_index
        ]
        if len(other) < 2:
            continue
        # Polarity may genuinely reverse along a road; conflict detection is
        # about wavelet-cycle identity, so use magnitude across *other* seeds.
        support = max(abs(float(values[row, sample])) for values in other)
        indices = np.flatnonzero(table.valid[row, :-1])
        indices = indices[
            np.abs(table.samples[row, indices] - sample) >= max(3, int(pulse_width))
        ]
        alternative = (
            max(
                float(np.max(np.abs(values[row, table.samples[row, indices]])))
                for values in other
            )
            if len(indices)
            else support
        )
        if support < 0.12 and alternative > support + 0.28:
            output.add(row)
    return output


def _event_emissions(
    table: _CandidateTable,
    radar_score: NDArray[np.float32],
    anchors: dict[int, int],
    propagated_positives: dict[int, int],
    seed_conflicts: set[int],
    anomaly_mask: NDArray[np.bool_],
    design_weight: float,
) -> NDArray[np.float32]:
    weight = float(np.clip(design_weight, 0.0, 0.10))
    design = table.features[..., -1]
    combined = (1.0 - weight) * radar_score + weight * design
    emissions = np.asarray(1.55 * combined - 0.75, dtype=np.float32)
    emissions[:, :-1] += 0.25 * table.tracklet_support[:, :-1]
    emissions[~table.valid] = -np.inf
    # Confirmed seeds justify carrying a weak-but-coherent event hypothesis
    # through ordinary pavement.  Confidence still decides whether it is
    # accepted; the graph no longer prefers no-pick merely because one trace is
    # locally weak.
    # For a seeded layer, no-pick is a visibility state—not permission to
    # forget the latent event family and restart on another ringing cycle.  The
    # graph therefore carries a continuous provisional hypothesis across weak
    # ordinary traces; the final evidence gate may still blank those rows.
    # Confirmed anomalies remain explicit no-pick states below.
    emissions[:, -1] = -2.0 if anchors else -0.030
    emissions[anomaly_mask, :-1] = -np.inf
    emissions[anomaly_mask, -1] = 0.10
    for row, sample in propagated_positives.items():
        if row in anchors or anomaly_mask[row]:
            continue
        indices = np.flatnonzero(table.valid[row, :-1])
        if not len(indices):
            continue
        selected = int(indices[int(np.argmin(np.abs(table.samples[row, indices] - sample)))])
        if abs(int(table.samples[row, selected]) - sample) <= 2:
            emissions[row, selected] += 0.28
    for row, sample in anchors.items():
        indices = np.flatnonzero(table.valid[row, :-1])
        if not len(indices):
            continue
        selected = int(indices[int(np.argmin(np.abs(table.samples[row, indices] - sample)))])
        emissions[row] = -np.inf
        emissions[row, selected] = 100.0
    return emissions


def _shift_waveforms(values: NDArray[np.float32], shift: int) -> NDArray[np.float32]:
    if shift == 0:
        return values
    output = np.zeros_like(values)
    if shift > 0:
        output[:, shift:] = values[:, :-shift]
    else:
        output[:, :shift] = values[:, -shift:]
    return output


def _waveform_similarity_matrix(
    left: NDArray[np.float32],
    right: NDArray[np.float32],
    maximum_shift: int = 2,
) -> NDArray[np.float32]:
    """Signed waveform similarity with limited timing accommodation.

    A candidate is already phase/lobe locked, but thin-layer interference can
    move the apparent peak by one or two samples.  Taking the best small shift
    preserves the event family without granting enough freedom to jump to an
    adjacent half-cycle.
    """
    left_norm = np.linalg.norm(left, axis=1)[:, None]
    best = np.full((len(left), len(right)), -1.0, dtype=np.float32)
    for shift in range(-maximum_shift, maximum_shift + 1):
        shifted = _shift_waveforms(right, shift)
        denominator = left_norm * np.linalg.norm(shifted, axis=1)[None, :]
        similarity = (left @ shifted.T) / np.maximum(denominator, 1e-8)
        best = np.maximum(best, similarity.astype(np.float32))
    return np.asarray(np.clip(best, 0.0, 1.0), dtype=np.float32)


def _transition_cube(
    oldest: NDArray[np.int32],
    middle: NDArray[np.int32],
    current: NDArray[np.int32],
    waveform_similarity: NDArray[np.float32],
    middle_phase: NDArray[np.int8],
    current_phase: NDArray[np.int8],
    middle_polarity: NDArray[np.int8],
    current_polarity: NDArray[np.int8],
    structural_break: bool,
    horizontal_step_m: float,
) -> NDArray[np.float32]:
    old = oldest[:, None, None].astype(np.float32)
    prior = middle[None, :, None].astype(np.float32)
    selected = current[None, None, :].astype(np.float32)
    dx = max(float(horizontal_step_m), 1e-3)
    slope = (selected - prior) / dx
    slope_cost = (0.008 if structural_break else 0.030) * np.abs(slope)
    previous_slope = (prior - old) / dx
    curvature = (slope - previous_slope) / dx
    curvature_cost = (0.002 if structural_break else 0.008) * np.abs(curvature)
    curvature_cost = np.where(old >= 0, curvature_cost, 0.0)
    waveform_cost = 0.12 * (1.0 - waveform_similarity[None, :, :])
    middle_phase_values = middle_phase[:, None].astype(int)
    current_phase_values = current_phase[None, :].astype(int)
    phase_distance = np.abs(middle_phase_values - current_phase_values)
    phase_distance = np.minimum(phase_distance, 8 - phase_distance) / 4.0
    phase_valid = (middle_phase_values >= 0) & (current_phase_values >= 0)
    phase_cost = np.where(
        phase_valid,
        (0.035 if structural_break else 0.14) * phase_distance,
        0.0,
    )
    polarity_change = (
        (middle_polarity[:, None] != 0)
        & (current_polarity[None, :] != 0)
        & (middle_polarity[:, None] != current_polarity[None, :])
    )
    polarity_cost = (0.025 if structural_break else 0.10) * polarity_change
    transition = -(
        slope_cost
        + curvature_cost
        + waveform_cost
        + phase_cost[None, :, :]
        + polarity_cost[None, :, :]
    )
    both_null = (prior < 0) & (selected < 0)
    one_null = (prior < 0) ^ (selected < 0)
    transition = np.where(both_null, 0.0, transition)
    # Entering no-pick must not be a cheap way to forget the current event
    # family and restart on another ringing cycle one row later.  Long weak
    # spans can still overcome this finite cost; short gaps retain continuity.
    transition = np.where(one_null, -1.10 if not structural_break else -0.22, transition)
    return np.asarray(transition, dtype=np.float32)


def _graph_hypotheses(
    table: _CandidateTable,
    emissions: NDArray[np.float32],
    break_rows: set[int],
    *,
    top_n: int = 32,
    beam_size: int = 128,
    horizontal_step_m: float = 0.4,
    cancel: Callable[[], bool] | None = None,
) -> tuple[list[NDArray[np.int32]], NDArray[np.float64]]:
    rows, states = table.samples.shape
    if rows == 0:
        return [], np.empty(0, dtype=float)
    if rows == 1:
        indices = np.argsort(emissions[0])[::-1][:top_n]
        paths = [np.asarray([table.samples[0, index]], dtype=np.int32) for index in indices]
        return paths, emissions[0, indices].astype(float)
    initial_waveform_similarity = _waveform_similarity_matrix(
        table.waveforms[0], table.waveforms[1]
    )
    initial_transition = _transition_cube(
        np.full(states, -1, dtype=np.int32),
        table.samples[0],
        table.samples[1],
        initial_waveform_similarity,
        table.phase_classes[0],
        table.phase_classes[1],
        table.polarities[0],
        table.polarities[1],
        1 in break_rows,
        horizontal_step_m,
    )[0]
    previous = (
        emissions[0, :, None] + emissions[1, None, :] + initial_transition
    ).astype(np.float32)
    back = np.full((rows, states, states), -1, dtype=np.int8)
    for row in range(2, rows):
        if cancel and row % 64 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        waveform_similarity = _waveform_similarity_matrix(
            table.waveforms[row - 1], table.waveforms[row]
        )
        transitions = _transition_cube(
            table.samples[row - 2],
            table.samples[row - 1],
            table.samples[row],
            waveform_similarity,
            table.phase_classes[row - 1],
            table.phase_classes[row],
            table.polarities[row - 1],
            table.polarities[row],
            row in break_rows,
            horizontal_step_m,
        )
        values = previous[:, :, None] + transitions
        choices = np.argmax(values, axis=0).astype(np.int8)
        current_scores = np.max(values, axis=0) + emissions[row][None, :]
        back[row] = choices
        finite = np.flatnonzero(np.isfinite(current_scores.ravel()))
        if len(finite) > beam_size:
            keep = finite[np.argpartition(current_scores.ravel()[finite], -beam_size)[-beam_size:]]
            keep_mask = np.zeros(current_scores.size, dtype=bool)
            keep_mask[keep] = True
            current_scores.ravel()[~keep_mask] = -np.inf
        previous = current_scores
    final_flat = np.flatnonzero(np.isfinite(previous.ravel()))
    if not len(final_flat):
        return [np.full(rows, -1, dtype=np.int32)], np.asarray([-np.inf])
    ranked = final_flat[np.argsort(previous.ravel()[final_flat])[::-1]][:top_n]
    paths: list[NDArray[np.int32]] = []
    scores: list[float] = []
    for flat in ranked:
        middle, current = np.unravel_index(int(flat), previous.shape)
        indices = np.full(rows, states - 1, dtype=np.int16)
        indices[-2] = middle
        indices[-1] = current
        for row in range(rows - 1, 1, -1):
            old = int(back[row, indices[row - 1], indices[row]])
            if old < 0:
                old = states - 1
            indices[row - 2] = old
        path = table.samples[np.arange(rows), indices].astype(np.int32)
        paths.append(path)
        scores.append(float(previous.ravel()[flat]))
    return paths, np.asarray(scores, dtype=float)


def _reverse_table(table: _CandidateTable) -> _CandidateTable:
    return _CandidateTable(
        samples=table.samples[::-1].copy(),
        valid=table.valid[::-1].copy(),
        features=table.features[::-1].copy(),
        waveforms=table.waveforms[::-1].copy(),
        canonical_samples=table.canonical_samples[::-1].copy(),
        phase_classes=table.phase_classes[::-1].copy(),
        polarities=table.polarities[::-1].copy(),
        pulse_widths=table.pulse_widths[::-1].copy(),
        prototype_indices=table.prototype_indices[::-1].copy(),
        selected_lobes=table.selected_lobes[::-1].copy(),
        dense_radar_score=table.dense_radar_score[::-1].copy(),
        component_maps={name: values[::-1].copy() for name, values in table.component_maps.items()},
        feature_names=table.feature_names,
        tracklet_support=table.tracklet_support[::-1].copy(),
        cycle_slip_risk=table.cycle_slip_risk[::-1].copy(),
        seed_distance_support=table.seed_distance_support[::-1].copy(),
        family_indices=table.family_indices[::-1].copy(),
        regime_indices=table.regime_indices[::-1].copy(),
    )


def _path_confidence(
    table: _CandidateTable,
    selected: NDArray[np.int32],
    backward: NDArray[np.int32],
    hypotheses: list[NDArray[np.int32]],
    hypothesis_scores: NDArray[np.float64],
    radar_score: NDArray[np.float32],
    anchors: dict[int, int],
    seed_conflicts: set[int],
    pulse_width: float,
    joint_hypothesis_support: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    rows = len(selected)
    indices = np.full(rows, -1, dtype=int)
    for row, sample in enumerate(selected):
        candidates = np.flatnonzero(table.valid[row, :-1])
        if sample >= 0 and len(candidates):
            indices[row] = int(
                candidates[int(np.argmin(np.abs(table.samples[row, candidates] - sample)))]
            )
    valid = indices >= 0
    selected_score = np.zeros(rows, dtype=float)
    selected_score[valid] = radar_score[np.arange(rows)[valid], indices[valid]]
    margin = np.zeros(rows, dtype=float)
    local_snr = np.zeros(rows, dtype=float)
    for row in np.flatnonzero(valid):
        candidates = np.flatnonzero(table.valid[row, :-1])
        values = radar_score[row, candidates]
        chosen = selected_score[row]
        alternatives = values[
            np.abs(table.samples[row, candidates] - selected[row]) > max(2, pulse_width / 2)
        ]
        alternate = float(np.max(alternatives)) if len(alternatives) else 0.0
        margin[row] = np.clip((chosen - alternate + 0.10) / 0.25, 0.0, 1.0)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) + 1e-5
        local_snr[row] = max(0.0, (chosen - median) / mad)
    if len(hypothesis_scores) and np.isfinite(hypothesis_scores[0]):
        weights = np.exp(np.clip((hypothesis_scores - hypothesis_scores[0]) / 0.20, -30.0, 0.0))
        weights /= max(float(np.sum(weights)), 1e-9)
    else:
        weights = np.ones(max(len(hypotheses), 1), dtype=float)
        weights /= len(weights)
    hypothesis_agreement = np.zeros(rows, dtype=float)
    for weight, path in zip(weights, hypotheses, strict=False):
        hypothesis_agreement += weight * (
            (path >= 0)
            & (selected >= 0)
            & (np.abs(path - selected) <= pulse_width)
        )
    forward_backward = np.zeros(rows, dtype=float)
    both = (selected >= 0) & (backward >= 0)
    forward_backward[both] = np.exp(
        -np.abs(selected[both] - backward[both]) / max(pulse_width, 1.0)
    )
    def selected_component(name: str) -> NDArray[np.float64]:
        output = np.zeros(rows, dtype=float)
        output[valid] = table.features[
            np.arange(rows)[valid], indices[valid], table.feature_names.index(name)
        ]
        return output

    correlation = selected_component("signed_seed_correlation")
    phase = selected_component("phase_cycle_agreement")
    coherence = selected_component("oriented_coherence")
    reflectivity = selected_component("reflectivity_strength")
    absolute = selected_component("absolute_strength")
    stripped_gain = selected_component("stripped_gain")
    preprocessing_agreement = selected_component("preprocessing_agreement")
    tracklet_support = selected_component("tracklet_support")
    drop_seed_stability = selected_component("drop_seed_stability")
    cycle_slip_safety = selected_component("cycle_slip_safety")
    cycle_slip_risk = 1.0 - cycle_slip_safety
    seed_distance_support = np.zeros(rows, dtype=float)
    if anchors:
        anchor_rows = np.asarray(sorted(anchors), dtype=float)
        row_axis = np.arange(rows, dtype=float)
        nearest = np.min(np.abs(row_axis[:, None] - anchor_rows[None, :]), axis=1)
        # A seed is direct evidence for event identity, not for a straight
        # depth guide.  The slow decay only communicates extrapolation risk.
        seed_distance_support = np.exp(-nearest / max(rows / 3.0, 1.0))
    joint_support = (
        np.asarray(joint_hypothesis_support, dtype=float)
        if joint_hypothesis_support is not None
        else hypothesis_agreement.copy()
    )
    continuation = np.zeros(rows, dtype=float)
    continuation_count = np.zeros(rows, dtype=float)
    for row in range(1, rows):
        if indices[row - 1] < 0 or indices[row] < 0:
            continue
        left = table.waveforms[row - 1, indices[row - 1]]
        right = table.waveforms[row, indices[row]]
        similarity = float(
            _waveform_similarity_matrix(
                left[None, :], right[None, :], maximum_shift=2
            )[0, 0]
        )
        continuation[row - 1] += similarity
        continuation[row] += similarity
        continuation_count[row - 1] += 1.0
        continuation_count[row] += 1.0
    continuation /= np.maximum(continuation_count, 1.0)
    continuation = uniform_filter1d(continuation, size=5, mode="nearest")
    neighborhood = uniform_filter1d(
        (
            0.35 * hypothesis_agreement
            + 0.25 * coherence
            + 0.40 * continuation
        ).astype(float),
        size=7,
        mode="nearest",
    )
    def survey_quality(values: NDArray[np.float64]) -> NDArray[np.float64]:
        usable = values[valid & np.isfinite(values)]
        if len(usable) < 8:
            return np.clip(values, 0.0, 1.0)
        low, high = np.percentile(usable, [15.0, 85.0])
        if high - low < 1e-6:
            return np.clip(values, 0.0, 1.0)
        return np.clip((values - low) / (high - low), 0.0, 1.0)

    snr_quality = 1.0 / (1.0 + np.exp(-(local_snr - 1.2)))
    identity = (
        0.34 * correlation
        + 0.20 * phase
        + 0.18 * tracklet_support
        + 0.14 * drop_seed_stability
        + 0.08 * seed_distance_support
        + 0.06 * cycle_slip_safety
    )
    consensus = (
        0.19 * hypothesis_agreement
        + 0.18 * joint_support
        + 0.16 * forward_backward
        + 0.16 * preprocessing_agreement
        + 0.16 * neighborhood
        + 0.15 * margin
    )
    radar_quality = (
        0.31 * survey_quality(selected_score)
        + 0.18 * survey_quality(coherence)
        + 0.14 * survey_quality(reflectivity)
        + 0.13 * snr_quality
        + 0.12 * survey_quality(absolute)
        + 0.12 * survey_quality(continuation)
    )
    confidence = 0.42 * identity + 0.38 * consensus + 0.20 * radar_quality
    # Survey normalisation cannot turn pure noise into evidence.  At least one
    # absolute identity/continuation cue must support the relative ranking.
    absolute_gate = np.clip(
        np.maximum.reduce(
            (
                correlation * phase,
                tracklet_support,
                preprocessing_agreement * coherence,
                selected_score * continuation,
            )
        )
        / 0.42,
        0.0,
        1.0,
    )
    confidence *= 0.55 + 0.45 * absolute_gate
    for row in anchors:
        if row not in seed_conflicts and 0 <= row < rows:
            confidence[row] = 1.0
    for row in seed_conflicts:
        if 0 <= row < rows:
            confidence[row] = 0.0
    confidence[~valid] = 0.0
    evidence = {
        "signal_score": selected_score,
        "absolute_strength": absolute,
        "seed_correlation": correlation,
        "phase_score": phase,
        "coherence_score": coherence,
        "candidate_margin": margin,
        "forward_backward_agreement": forward_backward,
        "perturbation_stability": hypothesis_agreement,
        "local_snr": local_snr,
        "ensemble_agreement": preprocessing_agreement,
        "preprocessing_agreement": preprocessing_agreement,
        "hypothesis_agreement": hypothesis_agreement,
        "neighborhood_support": neighborhood,
        "residual_improvement": stripped_gain,
        "waveform_similarity": np.maximum(correlation, continuation),
        "candidate_continuation": continuation,
        "reflectivity_strength": reflectivity,
        "phase_cycle_agreement": phase,
        "path_margin": margin,
        "alternative_cycle_margin": margin,
        "seed_distance_support": seed_distance_support,
        "drop_seed_stability": drop_seed_stability,
        "joint_hypothesis_support": joint_support,
        "tracklet_support": tracklet_support,
        "cycle_slip_risk": cycle_slip_risk,
        "branch_multimodality": uniform_filter1d(
            (
                (margin < 0.28)
                & (hypothesis_agreement < 0.82)
                & (forward_backward < 0.82)
            ).astype(float),
            size=9,
            mode="nearest",
        ),
        "canonical_event_sample": np.where(
            valid, table.canonical_samples[np.arange(rows), np.maximum(indices, 0)], -1.0
        ),
        "selected_lobe_code": np.where(
            valid, table.selected_lobes[np.arange(rows), np.maximum(indices, 0)], 0.0
        ),
        "regime_index": np.where(
            valid, table.regime_indices[np.arange(rows), np.maximum(indices, 0)], -1.0
        ),
        "event_family_index": np.where(
            valid, table.family_indices[np.arange(rows), np.maximum(indices, 0)], -1.0
        ),
        "edge_condition": np.zeros(rows, dtype=float),
        "seed_conflict": np.asarray(
            [1.0 if row in seed_conflicts else 0.0 for row in range(rows)], dtype=float
        ),
        "design_tiebreak": selected_component("design_tiebreak"),
        "design_score": selected_component("design_tiebreak"),
    }
    return np.clip(confidence, 0.0, 1.0), evidence


def _interpolate_short_gaps(
    path: NDArray[np.int32], maximum_rows: int, maximum_jump: int
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    output = path.copy()
    interpolated = np.zeros(len(path), dtype=bool)
    row = 0
    while row < len(output):
        if output[row] >= 0:
            row += 1
            continue
        start = row
        while row < len(output) and output[row] < 0:
            row += 1
        if (
            row - start <= maximum_rows
            and start > 0
            and row < len(output)
            and abs(int(output[row]) - int(output[start - 1]))
            <= maximum_jump * (row - start + 1)
        ):
            output[start:row] = np.rint(
                np.linspace(output[start - 1], output[row], row - start + 2)[1:-1]
            ).astype(np.int32)
            interpolated[start:row] = True
    return output, interpolated


def _remove_short_visible_runs(
    visible: NDArray[np.bool_], minimum_rows: int, anchors: dict[int, int]
) -> NDArray[np.bool_]:
    output = visible.copy()
    row = 0
    while row < len(output):
        if not output[row]:
            row += 1
            continue
        start = row
        while row < len(output) and output[row]:
            row += 1
        anchored = any(start <= anchor < row for anchor in anchors)
        if row - start < minimum_rows and not anchored:
            output[start:row] = False
    return output


def _finalize_workspace_path(
    workspace: _LayerWorkspace,
    selected_samples: NDArray[np.int32],
    anomaly_mask: NDArray[np.bool_],
    pulse_width_samples: float,
    max_interpolation_rows: int,
    design_constrained: bool,
    joint_backward: NDArray[np.int32] | None = None,
    joint_hypothesis_support: NDArray[np.floating] | None = None,
) -> SeedConditionedPath:
    selected = np.asarray(selected_samples, dtype=np.int32).copy()
    confidence, evidence = _path_confidence(
        workspace.table,
        selected,
        workspace.backward if joint_backward is None else joint_backward,
        workspace.hypotheses,
        workspace.hypothesis_scores,
        workspace.radar_score,
        workspace.anchors,
        workspace.seed_conflicts,
        pulse_width_samples,
        joint_hypothesis_support,
    )
    edge = (
        (selected >= 0)
        & (
            (np.abs(selected - workspace.lower) <= pulse_width_samples)
            | (np.abs(selected - workspace.upper) <= pulse_width_samples)
        )
        if design_constrained
        else np.zeros(len(selected), dtype=bool)
    )
    evidence["edge_condition"] = edge.astype(float)
    continuation = evidence["candidate_continuation"]
    if workspace.anchors:
        deeper = workspace.layer.order >= 2
        connected_family = evidence["event_family_index"] >= 0
        correlation_threshold = 0.18 if deeper else 0.24
        phase_threshold = 0.40 if deeper else 0.50
        tracklet_threshold = 0.38 if deeper else 0.47
        continuation_threshold = 0.34 if deeper else 0.42
        coherence_threshold = 0.26 if deeper else 0.34
        adequate = (
            (evidence["signal_score"] >= 0.16)
            & (
                (
                    (evidence["seed_correlation"] >= correlation_threshold)
                    & (evidence["phase_score"] >= phase_threshold)
                    & ((not deeper) | connected_family)
                )
                | (
                    (evidence["tracklet_support"] >= tracklet_threshold)
                    & (evidence["cycle_slip_risk"] <= 0.52)
                )
                | (
                    (continuation >= continuation_threshold)
                    & (evidence["coherence_score"] >= coherence_threshold)
                    & (evidence["preprocessing_agreement"] >= 0.50)
                    & ((not deeper) | connected_family)
                )
            )
        ) | (
            (evidence["joint_hypothesis_support"] >= 0.78)
            & (evidence["neighborhood_support"] >= 0.52)
            & (evidence["signal_score"] >= 0.17)
            & ((not deeper) | connected_family)
        )
    else:
        adequate = (
            (evidence["signal_score"] >= 0.30)
            & (continuation >= 0.58)
            & (evidence["coherence_score"] >= 0.58)
            & (evidence["absolute_strength"] >= 0.12)
        )
    adequate &= (
        (evidence["absolute_strength"] >= 0.025)
        | (evidence["tracklet_support"] >= 0.45)
        | (evidence["preprocessing_agreement"] >= 0.50)
        | (
            (evidence["drop_seed_stability"] >= 0.42)
            & (evidence["seed_correlation"] >= 0.62)
            & (continuation >= 0.52)
            & (evidence["neighborhood_support"] >= 0.52)
            & (evidence["absolute_strength"] >= 0.012)
        )
    )
    visible = (
        (selected >= 0)
        & (confidence >= 0.24)
        & adequate
        & ~anomaly_mask
        & ~edge
    )
    visible = _remove_short_visible_runs(
        visible, 6 if workspace.anchors else 12, workspace.anchors
    )
    # Do not let a smooth graph protrude a couple of weak rows into a verified
    # long evidence gap.  Strong phase-locked tracklets and seeds are exempt.
    gap_start = 0
    while gap_start < len(visible):
        if visible[gap_start]:
            gap_start += 1
            continue
        gap_end = gap_start
        while gap_end < len(visible) and not visible[gap_end]:
            gap_end += 1
        if gap_end - gap_start >= 6:
            for row in range(max(0, gap_start - 2), gap_start):
                if row not in workspace.anchors and evidence["tracklet_support"][row] < 0.55:
                    visible[row] = False
            for row in range(gap_end, min(len(visible), gap_end + 2)):
                if row not in workspace.anchors and evidence["tracklet_support"][row] < 0.55:
                    visible[row] = False
        gap_start = gap_end
    for row, sample in workspace.anchors.items():
        selected[row] = sample
        visible[row] = True
        confidence[row] = 0.19 if row in workspace.seed_conflicts else 1.0
    selected[~visible] = -1
    confidence[~visible] = 0.0
    selected, interpolated = _interpolate_short_gaps(
        selected, max(0, max_interpolation_rows), maximum_jump=7
    )
    interpolated[anomaly_mask] = False
    selected[anomaly_mask] = -1
    confidence[interpolated] = np.minimum(confidence[interpolated], 0.45)
    conflict = edge | (evidence["seed_conflict"] > 0.5)
    return SeedConditionedPath(
        samples=selected,
        confidence=confidence,
        feature=workspace.table.dense_radar_score,
        alternate_samples=workspace.backward,
        visible=selected >= 0,
        interpolated=interpolated,
        evidence=evidence,
        signal_only_samples=selected.copy(),
        design_guided_samples=selected.copy(),
        design_conflict=conflict,
        design_constrained=design_constrained,
        candidate_components=workspace.table.component_maps,
    )


def _joint_states_at_row(
    workspaces: list[_LayerWorkspace], row: int, maximum_states: int = 128
) -> tuple[NDArray[np.int16], NDArray[np.float32]]:
    choices: list[NDArray[np.int16]] = []
    for workspace in workspaces:
        finite = np.flatnonzero(np.isfinite(workspace.emissions[row]))
        if not len(finite):
            finite = np.asarray([workspace.table.samples.shape[1] - 1], dtype=int)
        null_index = workspace.table.samples.shape[1] - 1
        event_indices = finite[finite != null_index]
        if row in workspace.anchors:
            sample = workspace.anchors[row]
            event_indices = np.asarray(
                [
                    int(
                        event_indices[
                            np.argmin(np.abs(workspace.table.samples[row, event_indices] - sample))
                        ]
                    )
                ],
                dtype=int,
            )
        elif len(event_indices) > 6:
            # Do not prune a confirmed event family merely because a stronger
            # isolated lobe wins the local emission score.  Reserve two slots
            # for seed-family candidates and use the remaining slots for the
            # strongest unrestricted radar alternatives.
            known = event_indices[
                workspace.table.family_indices[row, event_indices] >= 0
            ]
            known = known[
                np.argsort(workspace.emissions[row, known])[::-1][:2]
            ]
            unrestricted = event_indices[
                np.argsort(workspace.emissions[row, event_indices])[::-1][:6]
            ]
            event_indices = np.unique(np.concatenate((known, unrestricted)))
            if len(event_indices) > 6:
                priority = workspace.emissions[row, event_indices] + 0.18 * (
                    workspace.table.family_indices[row, event_indices] >= 0
                )
                event_indices = event_indices[np.argsort(priority)[::-1][:6]]
        layer_choices = np.unique(np.append(event_indices, null_index)).astype(np.int16)
        choices.append(layer_choices)

    states: list[tuple[int, ...]] = []
    scores: list[float] = []
    for combination in product(*choices):
        valid = True
        previous_sample: int | None = None
        for layer_index, (workspace, candidate_index) in enumerate(
            zip(workspaces, combination, strict=True)
        ):
            sample = int(workspace.table.samples[row, candidate_index])
            if sample >= 0 and previous_sample is not None:
                minimum_gap = workspaces[layer_index].layer.min_gap_samples
                if sample < previous_sample + minimum_gap:
                    valid = False
                    break
            if sample >= 0:
                previous_sample = sample
        if not valid:
            continue
        states.append(tuple(int(index) for index in combination))
        scores.append(
            float(
                sum(
                    workspace.emissions[row, candidate_index]
                    for workspace, candidate_index in zip(
                        workspaces, combination, strict=True
                    )
                )
            )
        )
    if not states:
        states = [tuple(workspace.table.samples.shape[1] - 1 for workspace in workspaces)]
        scores = [0.0]
    if len(states) > maximum_states:
        score_array = np.asarray(scores, dtype=np.float32)
        keep = np.argpartition(score_array, -maximum_states)[-maximum_states:]
        states = [states[int(index)] for index in keep]
        scores = [scores[int(index)] for index in keep]
    return np.asarray(states, dtype=np.int16), np.asarray(scores, dtype=np.float32)


def _joint_transition_matrix(
    workspaces: list[_LayerWorkspace],
    row: int,
    previous_states: NDArray[np.int16],
    current_states: NDArray[np.int16],
    horizontal_step_m: float,
    structural_break: bool,
) -> NDArray[np.float32]:
    dx = max(float(horizontal_step_m), 1e-3)
    transition = np.zeros((len(previous_states), len(current_states)), dtype=np.float32)
    previous_samples_by_layer: list[NDArray[np.int32]] = []
    current_samples_by_layer: list[NDArray[np.int32]] = []
    for layer_index, workspace in enumerate(workspaces):
        previous_indices = previous_states[:, layer_index]
        current_indices = current_states[:, layer_index]
        previous_samples = workspace.table.samples[row - 1, previous_indices]
        current_samples = workspace.table.samples[row, current_indices]
        previous_samples_by_layer.append(previous_samples)
        current_samples_by_layer.append(current_samples)
        both = (previous_samples[:, None] >= 0) & (current_samples[None, :] >= 0)
        one_missing = (previous_samples[:, None] >= 0) ^ (current_samples[None, :] >= 0)
        slope = np.abs(current_samples[None, :] - previous_samples[:, None]) / dx
        transition -= np.where(
            both, (0.007 if structural_break else 0.018) * slope, 0.0
        ).astype(np.float32)
        previous_phase = workspace.table.phase_classes[row - 1, previous_indices]
        current_phase = workspace.table.phase_classes[row, current_indices]
        phase_distance = np.abs(previous_phase[:, None].astype(int) - current_phase[None, :])
        phase_distance = np.minimum(phase_distance, 8 - phase_distance) / 4.0
        phase_valid = (previous_phase[:, None] >= 0) & (current_phase[None, :] >= 0)
        transition -= np.where(
            both & phase_valid,
            (0.035 if structural_break else 0.14) * phase_distance,
            0.0,
        ).astype(np.float32)
        previous_waveforms = workspace.table.waveforms[row - 1, previous_indices]
        current_waveforms = workspace.table.waveforms[row, current_indices]
        similarity = np.clip(previous_waveforms @ current_waveforms.T, 0.0, 1.0)
        transition -= np.where(both, 0.12 * (1.0 - similarity), 0.0).astype(np.float32)
        previous_family = workspace.table.family_indices[row - 1, previous_indices]
        current_family = workspace.table.family_indices[row, current_indices]
        family_known = (previous_family[:, None] >= 0) & (current_family[None, :] >= 0)
        family_change = family_known & (previous_family[:, None] != current_family[None, :])
        transition -= np.where(
            both & family_change, 0.06 if structural_break else 0.45, 0.0
        ).astype(np.float32)
        one_family_known = (previous_family[:, None] >= 0) ^ (
            current_family[None, :] >= 0
        )
        transition -= np.where(
            both & one_family_known, 0.01 if structural_break else 0.08, 0.0
        ).astype(np.float32)
        previous_regime = workspace.table.regime_indices[row - 1, previous_indices]
        current_regime = workspace.table.regime_indices[row, current_indices]
        regime_known = (previous_regime[:, None] >= 0) & (current_regime[None, :] >= 0)
        regime_change = regime_known & (previous_regime[:, None] != current_regime[None, :])
        transition -= np.where(
            both & regime_change, 0.04 if structural_break else 0.50, 0.0
        ).astype(np.float32)
        transition -= np.where(one_missing, 0.16, 0.0).astype(np.float32)

    for layer_index in range(1, len(workspaces)):
        previous_top = previous_samples_by_layer[layer_index - 1]
        previous_bottom = previous_samples_by_layer[layer_index]
        current_top = current_samples_by_layer[layer_index - 1]
        current_bottom = current_samples_by_layer[layer_index]
        previous_valid = (previous_top >= 0) & (previous_bottom >= 0)
        current_valid = (current_top >= 0) & (current_bottom >= 0)
        gap_change = np.abs(
            (current_bottom - current_top)[None, :]
            - (previous_bottom - previous_top)[:, None]
        ) / dx
        transition -= np.where(
            previous_valid[:, None] & current_valid[None, :],
            (0.003 if structural_break else 0.010) * gap_change,
            0.0,
        ).astype(np.float32)
    return transition


def _joint_directional_hypotheses(
    workspaces: list[_LayerWorkspace],
    break_rows: set[int],
    horizontal_step_m: float,
    cancel: Callable[[], bool] | None,
    top_n: int = 8,
) -> tuple[list[dict[int, NDArray[np.int32]]], NDArray[np.float64]]:
    rows = len(workspaces[0].table.samples)
    states_by_row: list[NDArray[np.int16]] = []
    back_by_row: list[NDArray[np.int16]] = [np.empty(0, dtype=np.int16)]
    states, emissions = _joint_states_at_row(workspaces, 0)
    states_by_row.append(states)
    scores = emissions
    for row in range(1, rows):
        if cancel and row % 64 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        current_states, current_emissions = _joint_states_at_row(workspaces, row)
        transition = _joint_transition_matrix(
            workspaces,
            row,
            states,
            current_states,
            horizontal_step_m,
            row in break_rows,
        )
        values = scores[:, None] + transition
        back = np.argmax(values, axis=0).astype(np.int16)
        scores = np.max(values, axis=0) + current_emissions
        states = current_states
        states_by_row.append(states)
        back_by_row.append(back)
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return (
            [
                {
                    workspace.layer.order: np.full(rows, -1, dtype=np.int32)
                    for workspace in workspaces
                }
            ],
            np.asarray([-np.inf]),
        )
    terminal = finite[np.argsort(scores[finite])[::-1]][:top_n]
    hypotheses: list[dict[int, NDArray[np.int32]]] = []
    hypothesis_scores: list[float] = []
    for selected_state in terminal:
        state_path = np.zeros(rows, dtype=np.int16)
        state_path[-1] = int(selected_state)
        for row in range(rows - 1, 0, -1):
            state_path[row - 1] = back_by_row[row][state_path[row]]
        output: dict[int, NDArray[np.int32]] = {}
        for layer_index, workspace in enumerate(workspaces):
            candidate_indices = np.asarray(
                [states_by_row[row][state_path[row], layer_index] for row in range(rows)],
                dtype=np.int16,
            )
            output[workspace.layer.order] = workspace.table.samples[
                np.arange(rows), candidate_indices
            ].astype(np.int32)
        hypotheses.append(output)
        hypothesis_scores.append(float(scores[selected_state]))
    return hypotheses, np.asarray(hypothesis_scores, dtype=float)


def _reverse_workspace(workspace: _LayerWorkspace) -> _LayerWorkspace:
    rows = len(workspace.table.samples)
    return _LayerWorkspace(
        layer=workspace.layer,
        table=_reverse_table(workspace.table),
        radar_score=workspace.radar_score[::-1].copy(),
        emissions=workspace.emissions[::-1].copy(),
        anchors={rows - 1 - row: sample for row, sample in workspace.anchors.items()},
        seed_conflicts={rows - 1 - row for row in workspace.seed_conflicts},
        lower=workspace.lower[::-1].copy(),
        upper=workspace.upper[::-1].copy(),
        hypotheses=[path[::-1].copy() for path in workspace.hypotheses],
        hypothesis_scores=workspace.hypothesis_scores.copy(),
        backward=workspace.backward[::-1].copy(),
        tracklets=[],
    )


def _joint_multilayer_paths(
    workspaces: list[_LayerWorkspace],
    break_rows: set[int],
    horizontal_step_m: float,
    pulse_width_samples: float,
    cancel: Callable[[], bool] | None,
) -> tuple[
    dict[int, NDArray[np.int32]],
    dict[int, NDArray[np.int32]],
    dict[int, NDArray[np.float64]],
]:
    if len(workspaces) <= 1:
        workspace = workspaces[0]
        support = np.zeros(len(workspace.table.samples), dtype=float)
        for path in workspace.hypotheses[:8]:
            support += (
                (path >= 0)
                & (workspace.hypotheses[0] >= 0)
                & (np.abs(path - workspace.hypotheses[0]) <= pulse_width_samples)
            )
        support /= max(min(len(workspace.hypotheses), 8), 1)
        return (
            {workspace.layer.order: workspace.hypotheses[0].copy()},
            {workspace.layer.order: workspace.backward.copy()},
            {workspace.layer.order: support},
        )
    rows = len(workspaces[0].table.samples)
    del horizontal_step_m
    # Each layer workspace already contains second-order, phase-aware graph
    # hypotheses.  Jointly select among those complete hypotheses instead of
    # replacing them with a first-order row solver that can shed curvature and
    # event-family memory through a no-pick state.
    choices = [range(min(8, len(workspace.hypotheses))) for workspace in workspaces]
    combinations: list[tuple[float, tuple[int, ...]]] = []
    break_mask = np.zeros(rows, dtype=bool)
    for row in break_rows:
        if 0 <= row < rows:
            break_mask[max(0, row - 1) : min(rows, row + 2)] = True
    for combination in product(*choices):
        if cancel and len(combinations) % 64 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        score = 0.0
        selected_paths: list[NDArray[np.int32]] = []
        for workspace, index in zip(workspaces, combination, strict=True):
            selected_paths.append(workspace.hypotheses[index])
            scale = max(0.20, rows * 0.015)
            score += float(
                (workspace.hypothesis_scores[index] - workspace.hypothesis_scores[0])
                / scale
            )
        for layer_index in range(1, len(selected_paths)):
            top = selected_paths[layer_index - 1]
            bottom = selected_paths[layer_index]
            valid = (top >= 0) & (bottom >= 0)
            minimum_gap = workspaces[layer_index].layer.min_gap_samples
            violations = valid & (bottom < top + minimum_gap)
            score -= 3.0 * float(np.count_nonzero(violations))
            gap = bottom.astype(float) - top.astype(float)
            adjacent = valid[1:] & valid[:-1] & ~break_mask[1:]
            if np.any(adjacent):
                score -= 0.025 * float(np.mean(np.abs(np.diff(gap)[adjacent])))
        combinations.append((score, tuple(int(index) for index in combination)))
    combinations.sort(key=lambda item: item[0], reverse=True)
    retained = combinations[:32]
    best_indices = retained[0][1]
    selected = {
        workspace.layer.order: workspace.hypotheses[index].copy()
        for workspace, index in zip(workspaces, best_indices, strict=True)
    }
    backward = {
        workspace.layer.order: workspace.backward.copy() for workspace in workspaces
    }
    retained_scores = np.asarray([item[0] for item in retained], dtype=float)
    weights = np.exp(np.clip(retained_scores - retained_scores[0], -30.0, 0.0))
    weights /= max(float(np.sum(weights)), 1e-9)
    support: dict[int, NDArray[np.float64]] = {}
    for layer_index, workspace in enumerate(workspaces):
        order = workspace.layer.order
        selected_path = selected[order]
        values = np.zeros(rows, dtype=float)
        for weight, (_score, indices) in zip(weights, retained, strict=True):
            path = workspace.hypotheses[indices[layer_index]]
            values += weight * (
                (selected_path >= 0)
                & (path >= 0)
                & (np.abs(selected_path - path) <= pulse_width_samples)
            )
        support[order] = values
    return selected, backward, support


def _pick_seed_conditioned_pass(
    radargram: NDArray[np.floating],
    reference_surface_sample: int,
    layers: list[LayerSpec],
    *,
    anchor_samples: dict[int, dict[int, int]],
    seed_metadata: dict[int, dict[int, dict[str, object]]] | None = None,
    feature_branches: dict[str, NDArray[np.floating]] | None,
    search_corridors: dict[int, SearchCorridor],
    design_weight: float,
    pulse_width_samples: float,
    break_rows: set[int],
    anomaly_mask: NDArray[np.bool_],
    max_interpolation_rows: int,
    horizontal_step_m: float = 0.4,
    cancel: Callable[[], bool] | None = None,
) -> dict[int, SeedConditionedPath]:
    original = np.asarray(radargram, dtype=np.float32)
    rows, sample_count = original.shape
    surface_path = np.full(rows, int(reference_surface_sample), dtype=np.int32)
    residual, _, _ = subtract_tracked_reflection(
        original, surface_path, pulse_width_samples=pulse_width_samples
    )
    output: dict[int, SeedConditionedPath] = {}
    workspaces: list[_LayerWorkspace] = []
    previous: NDArray[np.int32] | None = None
    for layer in sorted(
        (item for item in layers if item.analysis_enabled),
        key=lambda item: item.order,
    ):
        anchors = anchor_samples.get(layer.order, {})
        prototype_radius = max(6, int(round(1.5 * pulse_width_samples)))
        prototypes = _template_bank(
            original,
            anchors,
            prototype_radius,
            (seed_metadata or {}).get(layer.order),
        )
        for prototype in prototypes:
            prototype.layer_order = layer.order
        maps, per_prototype = _component_maps(
            original, residual, feature_branches, prototypes, pulse_width_samples
        )
        previous_candidate_bounds = None
        if workspaces:
            previous_table = workspaces[-1].table
            previous_result = output[workspaces[-1].layer.order]
            candidate_low = np.empty(rows, dtype=float)
            candidate_high = np.empty(rows, dtype=float)
            fallback = _filled_path(previous, reference_surface_sample)
            for row in range(rows):
                indices = np.flatnonzero(previous_table.valid[row, :-1])
                values = previous_table.samples[row, indices]
                values = values[values >= 0]
                # Raw min/max candidate bounds may span multiple ringing
                # cycles.  When the overlying interface is unresolved, center
                # the next-layer corridor on the nearest confident
                # interpolation and keep a deliberately broad margin.
                candidate_low[row] = fallback[row] - 3.0 * pulse_width_samples
                candidate_high[row] = fallback[row] + 3.0 * pulse_width_samples
                if previous_result.samples[row] >= 0:
                    guide = float(previous_result.samples[row])
                    confidence = previous_result.confidence[row]
                    width = pulse_width_samples
                    if confidence < 0.55:
                        width = 2.0 * pulse_width_samples
                    if confidence < 0.40:
                        width = 3.0 * pulse_width_samples
                    candidate_low[row] = guide - width
                    candidate_high[row] = guide + width
            previous_candidate_bounds = (candidate_low, candidate_high)
        lower, centre, upper = _search_bounds(
            rows,
            sample_count,
            reference_surface_sample,
            layer,
            previous,
            search_corridors.get(layer.order),
            anchors,
            pulse_width_samples,
            horizontal_step_m,
            previous_candidate_bounds,
        )
        table = _candidate_table(
            original,
            residual,
            maps,
            lower,
            centre,
            upper,
            anchors,
            anomaly_mask,
            pulse_width_samples,
        )
        tracklets = _grow_phase_locked_tracklets(
            table,
            anchors,
            (seed_metadata or {}).get(layer.order),
            layer.order,
            pulse_width_samples,
            horizontal_step_m,
        )
        if feature_branches is not None:
            feature_branches[f"layer_{layer.order}_tracklet_support"] = table.component_maps[
                "tracklet_support"
            ]
            feature_branches[f"layer_{layer.order}_cycle_slip_risk"] = table.component_maps[
                "cycle_slip_risk"
            ]
        fixed = _fixed_radar_score(table.features, bool(prototypes), layer.order)
        positives = _propagated_positives(
            table, anchors, pulse_width_samples, horizontal_step_m
        )
        for tracklet in tracklets:
            for row, sample, support in zip(
                tracklet.rows, tracklet.samples, tracklet.support, strict=True
            ):
                if support >= 0.58:
                    positives[int(row)] = int(sample)
        radar_score = _fit_candidate_ranker(table, positives, fixed, pulse_width_samples)
        seed_conflicts = _seed_conflicts(
            table, anchors, per_prototype, pulse_width_samples
        )
        emissions = _event_emissions(
            table,
            radar_score,
            anchors,
            positives,
            seed_conflicts,
            anomaly_mask,
            design_weight,
        )
        hypotheses, hypothesis_scores = _graph_hypotheses(
            table,
            emissions,
            break_rows,
            top_n=32,
            beam_size=128,
            horizontal_step_m=horizontal_step_m,
            cancel=cancel,
        )
        selected = hypotheses[0].copy()
        reverse_table = _reverse_table(table)
        reverse_emissions = emissions[::-1].copy()
        reverse_breaks = {rows - row for row in break_rows if 0 < row < rows}
        backward_hypotheses, _ = _graph_hypotheses(
            reverse_table,
            reverse_emissions,
            reverse_breaks,
            top_n=1,
            beam_size=128,
            horizontal_step_m=horizontal_step_m,
            cancel=cancel,
        )
        backward = backward_hypotheses[0][::-1]
        workspace = _LayerWorkspace(
            layer=layer,
            table=table,
            radar_score=radar_score,
            emissions=emissions,
            anchors=anchors,
            seed_conflicts=seed_conflicts,
            lower=lower,
            upper=upper,
            hypotheses=hypotheses,
            hypothesis_scores=hypothesis_scores,
            backward=backward,
            tracklets=tracklets,
        )
        workspaces.append(workspace)
        output[layer.order] = _finalize_workspace_path(
            workspace,
            selected,
            anomaly_mask,
            pulse_width_samples,
            max_interpolation_rows,
            layer.order in search_corridors,
        )
        previous = output[layer.order].samples
        strip_path = output[layer.order].samples.copy()
        strip_path[output[layer.order].confidence < 0.18] = -1
        residual, improvement, _ = subtract_tracked_reflection(
            residual, strip_path, pulse_width_samples=pulse_width_samples
        )
        if feature_branches is not None:
            feature_branches[f"layer_{layer.order}_stripped_residual"] = residual
            feature_branches[f"layer_{layer.order}_subtraction_improvement"] = improvement
    joint_paths, joint_backward, joint_support = _joint_multilayer_paths(
        workspaces,
        break_rows,
        horizontal_step_m,
        pulse_width_samples,
        cancel,
    )
    for workspace in workspaces:
        output[workspace.layer.order] = _finalize_workspace_path(
            workspace,
            joint_paths[workspace.layer.order],
            anomaly_mask,
            pulse_width_samples,
            max_interpolation_rows,
            workspace.layer.order in search_corridors,
            joint_backward[workspace.layer.order],
            joint_support[workspace.layer.order],
        )
    return output


def pick_seed_conditioned_interfaces(
    radargram: NDArray[np.floating],
    reference_surface_sample: int,
    layers: list[LayerSpec],
    *,
    anchor_samples: dict[int, dict[int, int]],
    seed_metadata: dict[int, dict[int, dict[str, object]]] | None = None,
    feature_branches: dict[str, NDArray[np.floating]] | None,
    search_corridors: dict[int, SearchCorridor],
    design_weight: float,
    pulse_width_samples: float,
    break_rows: set[int],
    anomaly_mask: NDArray[np.bool_],
    max_interpolation_rows: int,
    horizontal_step_m: float = 0.4,
    cancel: Callable[[], bool] | None = None,
) -> dict[int, SeedConditionedPath]:
    """Run independent radar-only and design-guided event graphs.

    Design never changes the signal-only observation or radar confidence.  A
    disagreement wider than one pulse is preserved as two hypotheses and sent
    to review instead of silently selecting the design-favoured event.
    """
    signal_branches = dict(feature_branches or {})
    signal = _pick_seed_conditioned_pass(
        radargram,
        reference_surface_sample,
        layers,
        anchor_samples=anchor_samples,
        seed_metadata=seed_metadata,
        feature_branches=signal_branches,
        search_corridors={},
        design_weight=0.0,
        pulse_width_samples=pulse_width_samples,
        break_rows=break_rows,
        anomaly_mask=anomaly_mask,
        max_interpolation_rows=max_interpolation_rows,
        horizontal_step_m=horizontal_step_m,
        cancel=cancel,
    )
    if not search_corridors or design_weight <= 0.0:
        for path in signal.values():
            path.signal_only_samples = path.samples.copy()
            path.design_guided_samples = path.samples.copy()
            path.design_constrained = False
        if feature_branches is not None:
            feature_branches.update(signal_branches)
        return signal

    guided_branches = dict(feature_branches or {})
    guided = _pick_seed_conditioned_pass(
        radargram,
        reference_surface_sample,
        layers,
        anchor_samples=anchor_samples,
        seed_metadata=seed_metadata,
        feature_branches=guided_branches,
        search_corridors=search_corridors,
        design_weight=min(0.10, max(0.0, design_weight)),
        pulse_width_samples=pulse_width_samples,
        break_rows=break_rows,
        anomaly_mask=anomaly_mask,
        max_interpolation_rows=max_interpolation_rows,
        horizontal_step_m=horizontal_step_m,
        cancel=cancel,
    )
    combined: dict[int, SeedConditionedPath] = {}
    for order in sorted(signal):
        radar_path = signal[order]
        design_path = guided[order]
        radar_visible = radar_path.samples >= 0
        design_visible = design_path.samples >= 0
        both = radar_visible & design_visible
        disagreement = both & (
            np.abs(radar_path.samples - design_path.samples) > pulse_width_samples
        )
        def support(path: SeedConditionedPath) -> NDArray[np.float64]:
            evidence = path.evidence
            return np.asarray(
                0.38 * path.confidence
                + 0.17 * evidence.get("tracklet_support", 0.0)
                + 0.14 * evidence.get("joint_hypothesis_support", 0.0)
                + 0.12 * evidence.get("seed_correlation", 0.0)
                + 0.10 * evidence.get("preprocessing_agreement", 0.0)
                + 0.09 * evidence.get("alternative_cycle_margin", 0.0),
                dtype=float,
            )

        merge_window = 9
        radar_support = uniform_filter1d(
            support(radar_path), size=merge_window, mode="nearest"
        )
        design_support = uniform_filter1d(
            support(design_path), size=merge_window, mode="nearest"
        )
        seeded_layer = bool(anchor_samples.get(order))
        use_design = design_visible & (
            (not seeded_layer)
            | ~radar_visible
            | (design_support > radar_support + 0.08)
        )
        selected = radar_path.samples.copy()
        selected[use_design] = design_path.samples[use_design]
        confidence = radar_path.confidence.copy()
        confidence[use_design] = design_path.confidence[use_design]
        near_equal = np.abs(radar_support - design_support) < 0.10
        both_supported = (radar_support >= 0.38) & (design_support >= 0.38)
        raw_multimodal = disagreement & near_equal & both_supported
        persistent_multimodal = (
            uniform_filter1d(
                raw_multimodal.astype(float), size=merge_window, mode="nearest"
            )
            >= 0.45
        )
        conflict = (
            persistent_multimodal
            | radar_path.design_conflict
            | design_path.design_conflict
        )
        confidence[conflict] = np.clip(confidence[conflict], 0.20, 0.49)
        evidence: dict[str, NDArray[np.float64]] = {}
        for name in set(radar_path.evidence) | set(design_path.evidence):
            radar_values = np.asarray(
                radar_path.evidence.get(name, np.zeros_like(confidence)), dtype=float
            )
            design_values = np.asarray(
                design_path.evidence.get(name, np.zeros_like(confidence)), dtype=float
            )
            evidence[name] = np.where(use_design, design_values, radar_values)
        evidence["signal_design_agreement"] = np.where(
            both,
            np.exp(
                -np.abs(radar_path.samples - design_path.samples)
                / max(pulse_width_samples, 1.0)
            ),
            0.0,
        )
        evidence["design_conflict"] = conflict.astype(float)
        evidence["branch_multimodality"] = np.maximum(
            evidence.get("branch_multimodality", np.zeros_like(confidence)),
            persistent_multimodal.astype(float),
        )
        evidence["selected_design_path"] = use_design.astype(float)
        evidence["radar_family_support"] = radar_support
        evidence["design_family_support"] = design_support
        combined[order] = SeedConditionedPath(
            samples=selected,
            confidence=confidence,
            feature=design_path.feature,
            alternate_samples=design_path.alternate_samples,
            visible=selected >= 0,
            interpolated=radar_path.interpolated | design_path.interpolated,
            evidence=evidence,
            signal_only_samples=radar_path.samples.copy(),
            design_guided_samples=design_path.samples.copy(),
            design_conflict=conflict,
            design_constrained=True,
            candidate_components=design_path.candidate_components,
        )
    if feature_branches is not None:
        feature_branches.update(guided_branches)
    return combined
