from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import fftconvolve, hilbert

from gpr_layer_audit.models import LayerSpec, SearchCorridor, WaveformPrototype

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
    dense_radar_score: NDArray[np.float32]
    component_maps: dict[str, NDArray[np.float32]]
    feature_names: tuple[str, ...]


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
    data: NDArray[np.float32], anchors: dict[int, int], radius: int
) -> list[WaveformPrototype]:
    analytic = hilbert(data, axis=1)
    output: list[WaveformPrototype] = []
    for row, sample in sorted(anchors.items()):
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
                station_id=f"row-{row}",
                chainage_m=float(row),
                sample_index=int(sample),
                real_waveform=real,
                quadrature_waveform=quadrature,
                polarity=int(np.sign(data[row, sample])),
                phase_class=int(_phase_class(np.angle(analytic[row, sample]))),
                radius_samples=radius,
            )
        )
    return output


def _template_feature_maps(
    data: NDArray[np.float32], prototypes: list[WaveformPrototype]
) -> tuple[dict[str, NDArray[np.float32]], list[NDArray[np.float32]]]:
    analytic = hilbert(data, axis=1)
    phase_classes = _phase_class(np.angle(analytic))
    signed = np.zeros_like(data, dtype=np.float32)
    absolute = np.zeros_like(data, dtype=np.float32)
    phase_agreement = np.zeros_like(data, dtype=np.float32)
    polarity_agreement = np.zeros_like(data, dtype=np.float32)
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
        signed = np.maximum(signed, np.clip(correlation, 0.0, 1.0))
        absolute = np.maximum(absolute, np.abs(correlation))
        distance = np.abs(phase_classes.astype(int) - prototype.phase_class)
        distance = np.minimum(distance, 8 - distance)
        phase_agreement = np.maximum(
            phase_agreement,
            np.asarray(0.5 + 0.5 * np.cos(distance * np.pi / 4.0), dtype=np.float32),
        )
        same_polarity = np.sign(data) == prototype.polarity
        polarity_agreement = np.maximum(polarity_agreement, same_polarity.astype(np.float32))
    return (
        {
            "signed_seed_correlation": signed,
            "absolute_seed_correlation": absolute,
            "phase_cycle_agreement": phase_agreement,
            "polarity_agreement": polarity_agreement,
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
    global_reference = max(float(np.percentile(residual_envelope_raw, 97.5)), 1e-7)
    absolute = np.asarray(
        np.clip(residual_envelope_raw / global_reference, 0.0, 1.0), dtype=np.float32
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
    template_maps, per_prototype = _template_feature_maps(residual, prototypes)
    if not prototypes:
        template_maps["phase_cycle_agreement"][:] = 0.5
        template_maps["polarity_agreement"][:] = 0.5
    generic = np.asarray(
        0.17 * residual_envelope
        + 0.13 * gradient
        + 0.18 * reflectivity
        + 0.22 * oriented
        + 0.14 * template_maps["signed_seed_correlation"]
        + 0.06 * template_maps["absolute_seed_correlation"]
        + 0.10 * absolute,
        dtype=np.float32,
    )
    maps = {
        **template_maps,
        "oriented_coherence": np.clip(oriented, 0.0, 1.0),
        "reflectivity_strength": reflectivity,
        "residual_envelope": residual_envelope,
        "vertical_gradient": gradient,
        "absolute_strength": absolute,
        "stripped_gain": stripped_gain,
        "generic_radar_score": np.clip(generic, 0.0, 1.0),
        "analytic_phase_rad": np.asarray(np.angle(analytic_residual), dtype=np.float32),
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
    if previous is not None:
        lower = np.maximum(lower, previous_guide + layer.min_gap_samples)
    if corridor is not None:
        finite = (
            np.isfinite(corridor.gap_lower_samples)
            & np.isfinite(corridor.gap_centre_samples)
            & np.isfinite(corridor.gap_upper_samples)
        )
        candidate_lower = previous_guide + corridor.gap_lower_samples
        candidate_centre = previous_guide + corridor.gap_centre_samples
        candidate_upper = previous_guide + corridor.gap_upper_samples
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
        ranked = candidates[np.argsort(generic[row, candidates])[::-1]] if len(candidates) else []
        selected: list[int] = []
        anchor_sample = anchors.get(row)
        if anchor_sample is not None:
            selected.append(int(anchor_sample))
        for sample in ranked:
            if sample in selected or any(abs(int(sample) - prior) < 2 for prior in selected):
                continue
            selected.append(int(sample))
            if len(selected) >= limit:
                break
        for index, sample in enumerate(selected[:limit]):
            samples[row, index] = sample
            valid[row, index] = True
            feature_values[row, index] = [maps[name][row, sample] for name in _FEATURE_NAMES]
            if radius <= sample < sample_count - radius:
                waveform = _normalise_waveform(
                    residual[row, sample - radius : sample + radius + 1]
                )
                if waveform is not None:
                    waveforms[row, index] = waveform
        valid[row, -1] = True
    return _CandidateTable(
        samples=samples,
        valid=valid,
        features=feature_values,
        waveforms=waveforms,
        dense_radar_score=generic,
        component_maps=maps,
        feature_names=_FEATURE_NAMES,
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


def _fixed_radar_score(
    features: NDArray[np.float32], seeded: bool, layer_order: int
) -> NDArray[np.float32]:
    if seeded and layer_order == 2:
        # Frozen from Talagang development blocks only (100 m block modulo five
        # 0-2).  Calibration/test blocks are never used to fit these weights.
        # Negative semblance weight is deliberate: shallow antenna/asphalt
        # ringing is often more laterally coherent than the selected base event.
        coefficients = np.asarray(
            [
                0.1897,
                0.4680,
                0.1713,
                -0.0707,
                -0.6403,
                0.0534,
                0.7826,
                0.7054,
                0.0500,
                0.1418,
                0.2359,
            ],
            dtype=np.float32,
        )
        raw = np.asarray(features[..., :-1] @ coefficients, dtype=np.float32)
        low = np.percentile(raw, 10.0, axis=1, keepdims=True)
        high = np.percentile(raw, 95.0, axis=1, keepdims=True)
        return np.asarray(
            np.clip((raw - low) / np.maximum(high - low, 1e-5), 0.0, 1.0),
            dtype=np.float32,
        )
    if seeded:
        weights = np.asarray(
            [0.25, 0.02, 0.10, 0.04, 0.14, 0.11, 0.09, 0.06, 0.08, 0.03, 0.08],
            dtype=np.float32,
        )
    else:
        weights = np.asarray(
            [0.00, 0.00, 0.00, 0.00, 0.24, 0.18, 0.16, 0.12, 0.16, 0.04, 0.10],
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
    emissions[~table.valid] = -np.inf
    emissions[:, -1] = -0.030
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


def _waveform_similarity_matrix(
    left: NDArray[np.float32], right: NDArray[np.float32]
) -> NDArray[np.float32]:
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(
        right, axis=1
    )[None, :]
    return np.asarray(
        np.clip(numerator / np.maximum(denominator, 1e-8), 0.0, 1.0),
        dtype=np.float32,
    )


def _transition_cube(
    oldest: NDArray[np.int32],
    middle: NDArray[np.int32],
    current: NDArray[np.int32],
    waveform_similarity: NDArray[np.float32],
    structural_break: bool,
) -> NDArray[np.float32]:
    old = oldest[:, None, None].astype(np.float32)
    prior = middle[None, :, None].astype(np.float32)
    selected = current[None, None, :].astype(np.float32)
    slope = selected - prior
    slope_cost = (0.020 if structural_break else 0.045) * np.abs(slope)
    curvature = slope - (prior - old)
    curvature_cost = (0.010 if structural_break else 0.032) * np.abs(curvature)
    curvature_cost = np.where(old >= 0, curvature_cost, 0.0)
    waveform_cost = 0.12 * (1.0 - waveform_similarity[None, :, :])
    transition = -(slope_cost + curvature_cost + waveform_cost)
    both_null = (prior < 0) & (selected < 0)
    one_null = (prior < 0) ^ (selected < 0)
    transition = np.where(both_null, 0.0, transition)
    transition = np.where(one_null, -0.16, transition)
    return np.asarray(transition, dtype=np.float32)


def _graph_hypotheses(
    table: _CandidateTable,
    emissions: NDArray[np.float32],
    break_rows: set[int],
    *,
    top_n: int = 32,
    beam_size: int = 128,
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
        1 in break_rows,
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
            row in break_rows,
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
        dense_radar_score=table.dense_radar_score[::-1].copy(),
        component_maps={name: values[::-1].copy() for name, values in table.component_maps.items()},
        feature_names=table.feature_names,
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
    continuation = np.zeros(rows, dtype=float)
    continuation_count = np.zeros(rows, dtype=float)
    for row in range(1, rows):
        if indices[row - 1] < 0 or indices[row] < 0:
            continue
        left = table.waveforms[row - 1, indices[row - 1]]
        right = table.waveforms[row, indices[row]]
        norm = float(np.linalg.norm(left) * np.linalg.norm(right))
        similarity = max(float(np.dot(left, right) / norm), 0.0) if norm > 1e-8 else 0.0
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
    snr_quality = 1.0 / (1.0 + np.exp(-(local_snr - 1.4)))
    confidence = (
        0.20 * selected_score
        + 0.17 * hypothesis_agreement
        + 0.13 * forward_backward
        + 0.12 * neighborhood
        + 0.10 * margin
        + 0.10 * coherence
        + 0.08 * correlation
        + 0.05 * reflectivity
        + 0.05 * snr_quality
    )
    confidence *= np.clip(0.45 + 0.55 * absolute, 0.0, 1.0)
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
        "ensemble_agreement": hypothesis_agreement,
        "hypothesis_agreement": hypothesis_agreement,
        "neighborhood_support": neighborhood,
        "residual_improvement": stripped_gain,
        "waveform_similarity": np.maximum(correlation, continuation),
        "candidate_continuation": continuation,
        "reflectivity_strength": reflectivity,
        "phase_cycle_agreement": phase,
        "path_margin": margin,
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


def pick_seed_conditioned_interfaces(
    radargram: NDArray[np.floating],
    reference_surface_sample: int,
    layers: list[LayerSpec],
    *,
    anchor_samples: dict[int, dict[int, int]],
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
    previous: NDArray[np.int32] | None = None
    for layer in sorted(
        (item for item in layers if item.analysis_enabled),
        key=lambda item: item.order,
    ):
        anchors = anchor_samples.get(layer.order, {})
        prototype_radius = max(6, int(round(1.5 * pulse_width_samples)))
        prototypes = _template_bank(residual, anchors, prototype_radius)
        for prototype in prototypes:
            prototype.layer_order = layer.order
        maps, per_prototype = _component_maps(original, residual, feature_branches, prototypes)
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
        fixed = _fixed_radar_score(table.features, bool(prototypes), layer.order)
        positives = _propagated_positives(
            table, anchors, pulse_width_samples, horizontal_step_m
        )
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
            cancel=cancel,
        )
        backward = backward_hypotheses[0][::-1]
        confidence, evidence = _path_confidence(
            table,
            selected,
            backward,
            hypotheses,
            hypothesis_scores,
            radar_score,
            anchors,
            seed_conflicts,
            pulse_width_samples,
        )
        edge = (selected >= 0) & (
            (np.abs(selected - lower) <= pulse_width_samples)
            | (np.abs(selected - upper) <= pulse_width_samples)
        )
        evidence["edge_condition"] = edge.astype(float)
        continuation = evidence["candidate_continuation"]
        if anchors:
            adequate = (
                (evidence["signal_score"] >= 0.22)
                & (
                    (evidence["seed_correlation"] >= 0.36)
                    | (
                        (continuation >= 0.48)
                        & (evidence["coherence_score"] >= 0.42)
                        & (evidence["absolute_strength"] >= 0.08)
                    )
                )
            ) | (
                (evidence["hypothesis_agreement"] >= 0.86)
                & (evidence["neighborhood_support"] >= 0.58)
                & (evidence["signal_score"] >= 0.20)
            )
        else:
            adequate = (
                (evidence["signal_score"] >= 0.30)
                & (continuation >= 0.58)
                & (evidence["coherence_score"] >= 0.58)
                & (evidence["absolute_strength"] >= 0.12)
            )
        visible = (
            (selected >= 0)
            & (confidence >= 0.18)
            & adequate
            & ~anomaly_mask
            & ~edge
        )
        visible = _remove_short_visible_runs(visible, 6 if anchors else 12, anchors)
        for row, sample in anchors.items():
            selected[row] = sample
            visible[row] = True
            confidence[row] = 0.19 if row in seed_conflicts else 1.0
        selected[~visible] = -1
        confidence[~visible] = 0.0
        selected, interpolated = _interpolate_short_gaps(
            selected, max(0, max_interpolation_rows), maximum_jump=7
        )
        interpolated[anomaly_mask] = False
        selected[anomaly_mask] = -1
        confidence[interpolated] = np.minimum(confidence[interpolated], 0.45)
        design_constrained = layer.order in search_corridors
        conflict = edge | (evidence["seed_conflict"] > 0.5)
        output[layer.order] = SeedConditionedPath(
            samples=selected,
            confidence=confidence,
            feature=table.dense_radar_score,
            alternate_samples=backward,
            visible=selected >= 0,
            interpolated=interpolated,
            evidence=evidence,
            signal_only_samples=selected.copy(),
            design_guided_samples=selected.copy(),
            design_conflict=conflict,
            design_constrained=design_constrained,
            candidate_components=table.component_maps,
        )
        previous = selected
        strip_path = selected.copy()
        strip_path[confidence < 0.25] = -1
        residual, improvement, _ = subtract_tracked_reflection(
            residual, strip_path, pulse_width_samples=pulse_width_samples
        )
        if feature_branches is not None:
            feature_branches[f"layer_{layer.order}_stripped_residual"] = residual
            feature_branches[f"layer_{layer.order}_subtraction_improvement"] = improvement
    # Final hard joint ordering check against the original observations.
    previous = None
    layer_lookup = {item.order: item for item in layers}
    for order in sorted(output):
        path = output[order]
        if previous is not None:
            invalid = (
                (path.samples >= 0)
                & (previous >= 0)
                & (path.samples < previous + layer_lookup[order].min_gap_samples)
            )
            path.samples[invalid] = -1
            path.visible[invalid] = False
            path.confidence[invalid] = 0.0
        previous = path.samples
    return output
