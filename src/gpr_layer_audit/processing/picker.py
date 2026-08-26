from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import fftconvolve, hilbert

from gpr_layer_audit.models import LayerSpec

TRACKER_METHODS = (
    "joint_seed_adaptive",
    "current_baseline",
    "enhanced_current",
    "deconvolution_ablation",
    "phase_coherence_ablation",
)


@dataclass(slots=True)
class PickPath:
    samples: NDArray[np.int32]
    confidence: NDArray[np.float64]
    feature: NDArray[np.float32]
    alternate_samples: NDArray[np.int32]
    visible: NDArray[np.bool_]
    interpolated: NDArray[np.bool_]
    evidence: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    signal_only_samples: NDArray[np.int32] | None = None
    design_guided_samples: NDArray[np.int32] | None = None
    design_conflict: NDArray[np.bool_] | None = None


def _normalise_rows(values: NDArray[np.floating]) -> NDArray[np.float32]:
    data = np.asarray(values, dtype=np.float32)
    low = np.percentile(data, 10, axis=1, keepdims=True)
    high = np.percentile(data, 95, axis=1, keepdims=True)
    return np.asarray(np.clip((data - low) / np.maximum(high - low, 1e-6), 0.0, 1.0))


def feature_maps(
    radargram: NDArray[np.floating],
    matched_template: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float32], ...]:
    """Compatibility feature set used by tests and the no-seed baseline."""
    data = np.asarray(radargram, dtype=np.float32)
    envelope = gaussian_filter(np.abs(hilbert(data, axis=1)), sigma=(1.2, 0.9))
    gradient = gaussian_filter(np.abs(np.gradient(data, axis=1)), sigma=(1.0, 0.7))
    combined = 0.58 * _normalise_rows(envelope) + 0.27 * _normalise_rows(gradient)
    if matched_template is not None and len(matched_template) >= 5:
        matched = np.abs(
            fftconvolve(data, np.asarray(matched_template)[None, ::-1], mode="same", axes=1)
        )
        combined += 0.15 * _normalise_rows(matched)
    return (
        np.asarray(envelope, dtype=np.float32),
        np.asarray(gradient, dtype=np.float32),
        np.asarray(combined, dtype=np.float32),
    )


def _seed_template(
    data: NDArray[np.float32], anchors: dict[int, int], radius: int = 10
) -> NDArray[np.float32] | None:
    snippets: list[NDArray[np.float32]] = []
    for row, sample in sorted(anchors.items()):
        if not 0 <= row < data.shape[0] or not radius <= sample < data.shape[1] - radius:
            continue
        snippet = data[row, sample - radius : sample + radius + 1].astype(np.float32).copy()
        snippet -= float(np.mean(snippet))
        norm = float(np.linalg.norm(snippet))
        if norm <= 1e-8:
            continue
        snippet /= norm
        if snippets and float(np.dot(snippet, snippets[0])) < 0:
            snippet *= -1.0
        snippets.append(snippet)
    if not snippets:
        return None
    template = np.median(np.stack(snippets), axis=0).astype(np.float32)
    template -= float(np.mean(template))
    norm = float(np.linalg.norm(template))
    return template / norm if norm > 1e-8 else None


def _template_correlation(
    data: NDArray[np.float32], template: NDArray[np.float32] | None
) -> NDArray[np.float32]:
    if template is None:
        return np.zeros_like(data, dtype=np.float32)
    correlation = fftconvolve(data, template[None, ::-1], mode="same", axes=1)
    energy = np.sqrt(
        np.maximum(
            uniform_filter1d(np.square(data, dtype=np.float64), len(template), axis=1),
            1e-12,
        )
        * len(template)
    )
    # Magnitude is intentional: a polarity reversal remains the same interface,
    # while uncorrelated noise should score near zero rather than the old 0.5.
    return np.asarray(np.clip(np.abs(correlation / energy), 0.0, 1.0), dtype=np.float32)


def _branch(
    branches: dict[str, NDArray[np.floating]] | None,
    name: str,
    fallback: NDArray[np.floating],
) -> NDArray[np.float32]:
    if branches and name in branches:
        return np.asarray(branches[name], dtype=np.float32)
    return np.asarray(fallback, dtype=np.float32)


def _layer_score(
    data: NDArray[np.float32],
    anchors: dict[int, int],
    feature_branches: dict[str, NDArray[np.floating]] | None,
    matched_template: NDArray[np.floating] | None,
    method: str,
) -> tuple[NDArray[np.float32], dict[str, NDArray[np.float32]]]:
    envelope0, gradient0, _ = feature_maps(data, matched_template)
    row_peaks = np.percentile(envelope0, 99, axis=1)
    reference_peak = max(float(np.percentile(row_peaks, 70)), 1e-6)
    absolute_strength = np.asarray(
        np.clip(envelope0 / (0.35 * reference_peak), 0.0, 1.0),
        dtype=np.float32,
    )
    envelope = _normalise_rows(_branch(feature_branches, "envelope", envelope0))
    gradient = _normalise_rows(_branch(feature_branches, "gradient", gradient0))
    phase = np.abs(_branch(feature_branches, "phase", np.cos(np.angle(hilbert(data, axis=1)))))
    lateral = gaussian_filter(data, sigma=(1.2, 0.0), mode="nearest")
    lateral_envelope = np.abs(hilbert(lateral, axis=1))
    coherence0 = np.clip(lateral_envelope / np.maximum(envelope0, 1e-6), 0.0, 1.0)
    coherence = _branch(feature_branches, "coherence", coherence0)
    deconvolved = _normalise_rows(_branch(feature_branches, "deconvolved", data))
    template = _seed_template(data, anchors)
    if template is None and matched_template is not None and len(matched_template) >= 5:
        template = np.asarray(matched_template, dtype=np.float32)
        template -= float(np.mean(template))
        norm = float(np.linalg.norm(template))
        template = template / norm if norm > 1e-8 else None
    correlation = _template_correlation(data, template)
    local_peak = maximum_filter1d(envelope, size=5, axis=1, mode="nearest")
    peakness = np.clip(1.0 - np.abs(envelope - local_peak), 0.0, 1.0)
    if method == "current_baseline":
        score = 0.55 * envelope + 0.30 * gradient + 0.15 * peakness
    elif method == "enhanced_current":
        score = (
            0.35 * envelope
            + 0.25 * gradient
            + 0.15 * phase
            + 0.10 * coherence
            + 0.15 * correlation
        )
    elif method == "deconvolution_ablation":
        score = (
            0.27 * envelope
            + 0.18 * gradient
            + 0.15 * phase
            + 0.14 * coherence
            + 0.22 * correlation
            + 0.04 * peakness
        )
    elif method == "phase_coherence_ablation":
        score = (
            0.31 * envelope
            + 0.24 * gradient
            + 0.27 * correlation
            + 0.12 * deconvolved
            + 0.06 * peakness
        )
    elif method == "joint_seed_adaptive":
        score = (
            0.24 * envelope
            + 0.17 * gradient
            + 0.14 * phase
            + 0.13 * coherence
            + 0.22 * correlation
            + 0.06 * deconvolved
            + 0.04 * peakness
        )
    else:
        raise ValueError(
            f"Unknown tracker method {method!r}; choose one of {', '.join(TRACKER_METHODS)}."
        )
    # Per-trace normalization is useful for seeing weak reflectors, but it can
    # also turn pure noise into an apparently excellent candidate. Preserve an
    # absolute, cross-trace evidence gate so missing layers remain missing.
    score *= 0.22 + 0.78 * absolute_strength
    score = gaussian_filter(score, sigma=(0.65, 0.35), mode="nearest")
    components = {
        "envelope": envelope,
        "gradient": gradient,
        "phase": phase,
        "coherence": coherence,
        "correlation": correlation,
        "deconvolved": deconvolved,
        "absolute_strength": absolute_strength,
    }
    return np.asarray(np.clip(score, 0.0, 1.0), dtype=np.float32), components


_FROM_NULL = np.int16(30_000)


def _viterbi_optional(
    score: NDArray[np.floating],
    valid_mask: NDArray[np.bool_],
    *,
    max_jump: int,
    smooth_penalty: float,
    anchors: dict[int, int] | None = None,
    minimum_evidence: float = 0.26,
    enter_exit_penalty: float = 0.32,
    cancel: Callable[[], bool] | None = None,
) -> NDArray[np.int32]:
    values = np.asarray(score, dtype=np.float32)
    rows, columns = values.shape
    anchors = anchors or {}
    deltas = np.arange(-max_jump, max_jump + 1, dtype=np.int16)
    back = np.zeros((rows, columns), dtype=np.int16)
    null_from = np.full(rows, -1, dtype=np.int32)
    previous = values[0] - minimum_evidence
    previous = np.where(valid_mask[0], previous, -1e9)
    null_previous = np.float32(0.0)
    if 0 in anchors:
        anchor = int(np.clip(anchors[0], 0, columns - 1))
        previous[:] = -1e9
        previous[anchor] = values[0, anchor] + 1e4
        null_previous = np.float32(-1e9)
    for row in range(1, rows):
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        candidates = np.full((len(deltas), columns), -1e9, dtype=np.float32)
        for index, delta0 in enumerate(deltas):
            delta = int(delta0)
            if delta < 0:
                candidates[index, -delta:] = previous[: columns + delta]
            elif delta > 0:
                candidates[index, : columns - delta] = previous[delta:]
            else:
                candidates[index] = previous
            candidates[index] -= smooth_penalty * abs(delta)
        choice = np.argmax(candidates, axis=0)
        best = candidates[choice, np.arange(columns)]
        from_null = null_previous - enter_exit_penalty > best
        best[from_null] = null_previous - enter_exit_penalty
        current = values[row] - minimum_evidence + best
        current[~valid_mask[row]] = -1e9
        back[row] = deltas[choice]
        back[row, from_null] = _FROM_NULL
        best_sample = int(np.argmax(previous))
        exit_score = previous[best_sample] - enter_exit_penalty
        if exit_score > null_previous:
            null_current = exit_score
            null_from[row] = best_sample
        else:
            null_current = null_previous
        if row in anchors:
            anchor = int(np.clip(anchors[row], 0, columns - 1))
            keep = current[anchor]
            current[:] = -1e9
            current[anchor] = keep + 1e4
            null_current = np.float32(-1e9)
        previous = current
        null_previous = np.float32(null_current)
    state = int(np.argmax(previous)) if float(np.max(previous)) >= float(null_previous) else -1
    path = np.full(rows, -1, dtype=np.int32)
    path[-1] = state
    for row in range(rows - 1, 0, -1):
        if state < 0:
            state = int(null_from[row])
        else:
            delta = int(back[row, state])
            state = -1 if delta == int(_FROM_NULL) else int(np.clip(state + delta, 0, columns - 1))
        path[row - 1] = state
    return path


def viterbi_path(
    score: NDArray[np.floating],
    valid_mask: NDArray[np.bool_],
    *,
    max_jump: int = 5,
    smooth_penalty: float = 0.16,
    anchors: dict[int, int] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> NDArray[np.int32]:
    """Public optional-state Viterbi path; -1 represents no reliable pick."""
    return _viterbi_optional(
        score,
        valid_mask,
        max_jump=max_jump,
        smooth_penalty=smooth_penalty,
        anchors=anchors,
        cancel=cancel,
    )


def _reverse_anchors(anchors: dict[int, int], rows: int) -> dict[int, int]:
    return {rows - 1 - row: sample for row, sample in anchors.items()}


def _bidirectional_path(
    score: NDArray[np.float32],
    mask: NDArray[np.bool_],
    anchors: dict[int, int],
    *,
    max_jump: int,
    smooth_penalty: float,
    minimum_evidence: float,
    break_rows: set[int] | None,
    cancel: Callable[[], bool] | None,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    breaks = sorted(row for row in (break_rows or set()) if 0 < row < len(score))
    if breaks:
        selected = np.full(len(score), -1, dtype=np.int32)
        alternate = np.full(len(score), -1, dtype=np.int32)
        boundaries = [0, *breaks, len(score)]
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=False):
            local_anchors = {
                row - start: sample
                for row, sample in anchors.items()
                if start <= row < stop
            }
            local_selected, local_alternate = _bidirectional_path(
                score[start:stop],
                mask[start:stop],
                local_anchors,
                max_jump=max_jump,
                smooth_penalty=smooth_penalty,
                minimum_evidence=minimum_evidence,
                break_rows=None,
                cancel=cancel,
            )
            selected[start:stop] = local_selected
            alternate[start:stop] = local_alternate
        return selected, alternate
    forward = _viterbi_optional(
        score,
        mask,
        max_jump=max_jump,
        smooth_penalty=smooth_penalty,
        minimum_evidence=minimum_evidence,
        anchors=anchors,
        cancel=cancel,
    )
    backward = _viterbi_optional(
        score[::-1],
        mask[::-1],
        max_jump=max_jump,
        smooth_penalty=smooth_penalty,
        minimum_evidence=minimum_evidence,
        anchors=_reverse_anchors(anchors, len(score)),
        cancel=cancel,
    )[::-1]
    rows = np.arange(len(forward))
    selected = forward.copy()
    both = (forward >= 0) & (backward >= 0)
    disagree = both & (forward != backward)
    forward_values = np.full(len(rows), -np.inf, dtype=float)
    backward_values = np.full(len(rows), -np.inf, dtype=float)
    valid_forward = forward >= 0
    valid_backward = backward >= 0
    forward_values[valid_forward] = score[rows[valid_forward], forward[valid_forward]]
    backward_values[valid_backward] = score[rows[valid_backward], backward[valid_backward]]
    use_backward = (~valid_forward & valid_backward) | (
        disagree & (backward_values > forward_values)
    )
    selected[use_backward] = backward[use_backward]
    for row, sample in anchors.items():
        if 0 <= row < len(selected):
            selected[row] = sample
    return selected, backward


def _fill_guide(path: NDArray[np.int32], default: int) -> NDArray[np.float64]:
    x = np.arange(len(path))
    finite = path >= 0
    if np.count_nonzero(finite) < 2:
        return np.full(len(path), default, dtype=float)
    return np.interp(x, x[finite], path[finite]).astype(float)


def _seed_geometry(
    row_count: int,
    anchors: dict[int, int],
    pulse_width: float,
) -> tuple[NDArray[np.float64] | None, float]:
    """Return a conservative whole-line depth guide learned only from seed clicks."""
    if not anchors:
        return None, 0.0
    locations = np.asarray(sorted(anchors), dtype=float)
    samples = np.asarray([anchors[int(row)] for row in locations], dtype=float)
    guide = np.interp(np.arange(row_count, dtype=float), locations, samples)
    if len(samples) == 1:
        radius = max(6.0 * pulse_width, 28.0)
    else:
        # A wider family is allowed when the analyst's clicks show meaningful
        # depth variation.  Outside that family the tracker emits no-pick and
        # asks for another seed instead of snapping to an unrelated reflector.
        radius = max(3.0 * pulse_width, 0.60 * float(np.ptp(samples)), 16.0)
    return guide, float(radius)


def _interpolate_short_gaps(
    path: NDArray[np.int32], max_gap_rows: int, max_jump: int
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    result = path.copy()
    interpolated = np.zeros(len(path), dtype=bool)
    index = 0
    while index < len(path):
        if result[index] >= 0:
            index += 1
            continue
        start = index
        while index < len(path) and result[index] < 0:
            index += 1
        stop = index
        gap = stop - start
        if (
            gap <= max_gap_rows
            and start > 0
            and stop < len(path)
            and abs(int(result[stop]) - int(result[start - 1])) <= max_jump * (gap + 1)
        ):
            values = np.linspace(result[start - 1], result[stop], gap + 2)[1:-1]
            result[start:stop] = np.rint(values).astype(np.int32)
            interpolated[start:stop] = True
    return result, interpolated


def _confidence(
    score: NDArray[np.float32],
    components: dict[str, NDArray[np.float32]],
    path: NDArray[np.int32],
    alternate: NDArray[np.int32],
    perturbation: NDArray[np.int32],
    mask: NDArray[np.bool_],
    anchors: dict[int, int],
    pulse_width: float,
) -> tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    rows = np.arange(len(path))
    valid = path >= 0
    signal = np.zeros(len(path), dtype=float)
    signal[valid] = score[rows[valid], path[valid]]
    local_snr = np.zeros(len(path), dtype=float)
    margin = np.zeros(len(path), dtype=float)
    coherence = np.zeros(len(path), dtype=float)
    correlation = np.zeros(len(path), dtype=float)
    absolute_strength = np.zeros(len(path), dtype=float)
    for row in rows[valid]:
        sample = int(path[row])
        values = score[row, mask[row]]
        median = float(np.median(values)) if len(values) else 0.0
        spread = float(np.median(np.abs(values - median))) + 1e-6 if len(values) else 1.0
        local_snr[row] = 1.0 / (1.0 + np.exp(-(signal[row] - median) / (2.0 * spread)))
        excluded = mask[row].copy()
        excluded[max(0, sample - 4) : sample + 5] = False
        alternate_score = float(np.max(score[row, excluded])) if np.any(excluded) else 0.0
        margin[row] = np.clip((signal[row] - alternate_score + 0.15) / 0.30, 0.0, 1.0)
        coherence[row] = float(components["coherence"][row, sample])
        correlation[row] = float(components["correlation"][row, sample])
        absolute_strength[row] = float(components["absolute_strength"][row, sample])
    agreement = np.zeros(len(path), dtype=float)
    both = valid & (alternate >= 0)
    agreement[both] = np.exp(-np.abs(path[both] - alternate[both]) / max(pulse_width, 1.0))
    perturbation_stability = np.zeros(len(path), dtype=float)
    perturbation_both = valid & (perturbation >= 0)
    perturbation_stability[perturbation_both] = np.exp(
        -np.abs(path[perturbation_both] - perturbation[perturbation_both])
        / max(pulse_width, 1.0)
    )
    continuity = np.zeros(len(path), dtype=float)
    guide = _fill_guide(path, 0)
    continuity[valid] = np.exp(-np.abs(np.gradient(guide))[valid] / 4.0)
    if anchors:
        locations = np.asarray(sorted(anchors), dtype=float)
        seed_distance = np.min(np.abs(rows[:, None] - locations[None, :]), axis=1)
        seed_support = np.exp(-seed_distance / max(10.0, len(path) * 0.25))
    else:
        seed_support = np.full(len(path), 0.45)
    confidence = (
        0.28 * local_snr
        + 0.18 * margin
        + 0.11 * agreement
        + 0.07 * perturbation_stability
        + 0.12 * continuity
        + 0.10 * coherence
        + 0.09 * correlation
        + 0.05 * seed_support
    )
    confidence *= np.clip((signal - 0.16) / 0.14, 0.0, 1.0)
    confidence *= np.clip(absolute_strength / 0.08, 0.0, 1.0)
    confidence *= 0.55 + 0.45 * correlation
    confidence = gaussian_filter1d(confidence, sigma=1.3, mode="nearest")
    if not anchors:
        # A visually strong reflector is not enough to identify which pavement
        # interface the analyst intends. Seedless paths are hypotheses only.
        confidence = np.minimum(confidence, 0.65)
    else:
        # Seed clicks are hard observations.  Smoothing the confidence field can
        # otherwise push a valid, low-amplitude seed below the visibility gate.
        # That silently discards the user's explicit evidence and can also
        # underconstrain the next (deeper) interface.
        for row in anchors:
            if 0 <= row < len(confidence):
                confidence[row] = 1.0
    confidence[~valid] = 0.0
    evidence = {
        "signal_score": signal,
        "seed_correlation": correlation,
        "phase_score": np.where(
            valid,
            components["phase"][rows, np.maximum(path, 0)],
            0.0,
        ),
        "coherence_score": coherence,
        "candidate_margin": margin,
        "forward_backward_agreement": agreement,
        "perturbation_stability": perturbation_stability,
        "local_snr": local_snr,
        "seed_support": seed_support,
        "absolute_strength": absolute_strength,
    }
    return np.clip(confidence, 0.0, 1.0), evidence


def propose_seed_rows(
    candidate_feature: NDArray[np.floating], count: int = 3
) -> NDArray[np.int32]:
    """Choose distributed, high-information stations without consulting labels."""
    rows = candidate_feature.shape[0]
    if rows == 0 or count <= 0:
        return np.empty(0, dtype=np.int32)
    quality = np.percentile(candidate_feature, 98, axis=1) - np.median(candidate_feature, axis=1)
    quality = gaussian_filter1d(quality.astype(float), sigma=max(1.0, rows / 400.0))
    selected: list[int] = []
    edges = np.linspace(0, rows, min(count, rows) + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:], strict=False):
        if stop <= start:
            continue
        selected.append(start + int(np.argmax(quality[start:stop])))
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def _validate_anchors(
    anchor_samples: dict[int, dict[int, int]],
    layers: list[LayerSpec],
    row_count: int,
    sample_count: int,
) -> None:
    enabled = {
        item.order: item
        for item in layers
        if item.analysis_enabled
    }
    by_row: dict[int, dict[int, int]] = {}
    for order, anchors in anchor_samples.items():
        if order not in enabled:
            continue
        for row, sample in anchors.items():
            if not 0 <= row < row_count or not 0 <= sample < sample_count:
                raise ValueError(
                    f"Layer {order} seed ({row}, {sample}) lies outside the radargram."
                )
            by_row.setdefault(row, {})[order] = sample
    for row, values in by_row.items():
        previous_sample: int | None = None
        previous_order: int | None = None
        for order in sorted(values):
            sample = values[order]
            if previous_sample is not None:
                minimum = enabled[order].min_gap_samples
                if sample < previous_sample + minimum:
                    raise ValueError(
                        "Seed interfaces are out of order at stacked row "
                        f"{row}: layer {order} must be at least {minimum} samples "
                        f"below layer {previous_order}."
                    )
            previous_order = order
            previous_sample = sample


def pick_interfaces(
    radargram: NDArray[np.floating],
    reference_surface_sample: int,
    layers: list[LayerSpec],
    *,
    anchor_samples: dict[int, dict[int, int]] | None = None,
    matched_template: NDArray[np.floating] | None = None,
    feature_branches: dict[str, NDArray[np.floating]] | None = None,
    design_prior_samples: dict[int, NDArray[np.floating]] | None = None,
    design_prior_widths: dict[int, NDArray[np.floating]] | None = None,
    design_weight: float = 0.20,
    pulse_width_samples: float = 7.0,
    break_rows: set[int] | None = None,
    method: str = "joint_seed_adaptive",
    cancel: Callable[[], bool] | None = None,
) -> dict[int, PickPath]:
    data = np.asarray(radargram, dtype=np.float32)
    rows, samples = data.shape
    anchor_samples = anchor_samples or {}
    design_prior_samples = design_prior_samples or {}
    design_prior_widths = design_prior_widths or {}
    if method not in TRACKER_METHODS:
        raise ValueError(
            f"Unknown tracker method {method!r}; choose one of {', '.join(TRACKER_METHODS)}."
        )
    _validate_anchors(anchor_samples, layers, rows, samples)
    output: dict[int, PickPath] = {}
    previous: NDArray[np.int32] | None = None
    sample_axis = np.arange(samples)[None, :]
    for layer in sorted((item for item in layers if item.analysis_enabled), key=lambda x: x.order):
        anchors = anchor_samples.get(layer.order, {})
        score, components = _layer_score(
            data, anchors, feature_branches, matched_template, method
        )
        seed_guide, seed_radius = _seed_geometry(rows, anchors, pulse_width_samples)
        lower = reference_surface_sample + layer.min_offset_samples
        upper = min(samples - 1, reference_surface_sample + layer.max_offset_samples)
        if seed_guide is not None:
            anchor_values = np.asarray(list(anchors.values()), dtype=float)
            if len(anchor_values) == 1:
                seed_margin = max(5.0 * pulse_width_samples, 24.0)
            else:
                seed_margin = max(
                    1.5 * pulse_width_samples,
                    0.25 * float(np.ptp(anchor_values)),
                    8.0,
                )
            lower = max(0, int(np.ceil(np.min(anchor_values) - seed_margin)))
            upper = min(samples - 1, int(np.floor(np.max(anchor_values) + seed_margin)))
        mask = np.broadcast_to(
            (sample_axis >= lower) & (sample_axis <= upper), (rows, samples)
        ).copy()
        if seed_guide is not None:
            distance = np.abs(sample_axis - seed_guide[:, None])
            mask &= distance <= seed_radius
            geometry_score = np.exp(
                -0.5 * np.square(distance / max(seed_radius * 0.55, 1.0))
            ).astype(np.float32)
            # Seed geometry expresses interface identity, not radar strength;
            # keep its contribution small enough that weak data remains weak.
            score = np.asarray(0.86 * score + 0.14 * geometry_score, dtype=np.float32)
        else:
            # Before the analyst identifies the reflector family, prefer the
            # first credible event in each ordered search window. This prevents
            # a strong deeper interface from being reused as every layer in the
            # seed-proposal preview.
            depth_fraction = np.clip(
                (sample_axis - lower) / max(float(upper - lower), 1.0),
                0.0,
                1.0,
            )
            score = np.asarray(np.clip(score - 0.12 * depth_fraction, 0.0, 1.0))
        edge_distance = np.minimum(sample_axis - lower, upper - sample_axis)
        edge_ramp = np.clip(
            edge_distance / max(1.5 * pulse_width_samples, 1.0),
            0.0,
            1.0,
        )
        score *= np.asarray(0.55 + 0.45 * edge_ramp, dtype=np.float32)
        if previous is not None:
            guide = _fill_guide(previous, lower - layer.min_gap_samples)
            mask &= sample_axis >= (guide[:, None] + layer.min_gap_samples)
        for row, sample in anchors.items():
            if 0 <= row < rows and 0 <= sample < samples:
                mask[row, sample] = True
        underconstrained = np.count_nonzero(mask, axis=1) < 3
        for row in np.flatnonzero(underconstrained):
            if row not in anchors:
                mask[row] = False
        signal_path, alternate = _bidirectional_path(
            score,
            mask,
            anchors,
            max_jump=7,
            smooth_penalty=0.045,
            minimum_evidence=0.12 if anchors else 0.26,
            break_rows=break_rows,
            cancel=cancel,
        )
        perturbation_path = alternate
        if feature_branches and "ensemble_amplitude" in feature_branches:
            ensemble_branches = {
                "phase": feature_branches.get("ensemble_phase"),
                "coherence": feature_branches.get("ensemble_coherence"),
                "deconvolved": feature_branches["ensemble_amplitude"],
            }
            ensemble_branches = {
                key: value for key, value in ensemble_branches.items() if value is not None
            }
            ensemble_score, _ = _layer_score(
                np.asarray(feature_branches["ensemble_amplitude"], dtype=np.float32),
                anchors,
                ensemble_branches,
                matched_template,
                method,
            )
            if seed_guide is not None:
                ensemble_score = np.asarray(
                    0.86 * ensemble_score + 0.14 * geometry_score,
                    dtype=np.float32,
                )
            else:
                ensemble_score = np.asarray(
                    np.clip(ensemble_score - 0.12 * depth_fraction, 0.0, 1.0),
                    dtype=np.float32,
                )
            ensemble_score *= np.asarray(0.55 + 0.45 * edge_ramp, dtype=np.float32)
            perturbation_path = _viterbi_optional(
                ensemble_score,
                mask,
                max_jump=7,
                smooth_penalty=0.045,
                anchors=anchors,
                minimum_evidence=0.12 if anchors else 0.26,
                cancel=cancel,
            )
        prior = design_prior_samples.get(layer.order)
        design_path: NDArray[np.int32] | None = None
        conflict = np.zeros(rows, dtype=bool)
        design_score_values = np.zeros(rows, dtype=float)
        if prior is not None:
            expected = np.asarray(prior, dtype=float)
            valid_prior = np.isfinite(expected)
            raw_width = design_prior_widths.get(layer.order)
            width = (
                np.maximum(np.asarray(raw_width, dtype=float), pulse_width_samples)
                if raw_width is not None
                else np.full(rows, max(pulse_width_samples, 1.0))
            )
            prior_score = np.exp(
                -0.5 * np.square((sample_axis - expected[:, None]) / width[:, None])
            ).astype(np.float32)
            prior_score[~valid_prior] = 0.0
            weight = float(np.clip(design_weight, 0.0, 0.20))
            guided_score = np.asarray(
                (1.0 - weight) * score + weight * prior_score, dtype=np.float32
            )
            design_path, _ = _bidirectional_path(
                guided_score,
                mask,
                anchors,
                max_jump=7,
                smooth_penalty=0.045,
                minimum_evidence=0.12 if anchors else 0.26,
                break_rows=break_rows,
                cancel=cancel,
            )
            both = (signal_path >= 0) & (design_path >= 0)
            conflict = ((signal_path >= 0) != (design_path >= 0)) | (
                both & (np.abs(signal_path - design_path) > pulse_width_samples)
            )
            valid_design = design_path >= 0
            design_score_values[valid_design] = prior_score[
                np.arange(rows)[valid_design], design_path[valid_design]
            ]
        final_path = signal_path.copy()
        confidence, evidence = _confidence(
            score,
            components,
            final_path,
            alternate,
            perturbation_path,
            mask,
            anchors,
            pulse_width_samples,
        )
        evidence["design_score"] = design_score_values
        visible = (final_path >= 0) & (confidence >= 0.20)
        for row, sample in anchors.items():
            if 0 <= row < rows and 0 <= sample < samples:
                final_path[row] = sample
                visible[row] = True
                confidence[row] = 1.0
        final_path[~visible] = -1
        confidence[~visible] = 0.0
        final_path, interpolated = _interpolate_short_gaps(final_path, 10, 7)
        visible = final_path >= 0
        confidence[interpolated] = np.minimum(confidence[interpolated], 0.45)
        confidence[conflict] = np.minimum(confidence[conflict], 0.49)
        output[layer.order] = PickPath(
            samples=final_path,
            confidence=confidence,
            feature=score,
            alternate_samples=alternate,
            visible=visible,
            interpolated=interpolated,
            evidence=evidence,
            signal_only_samples=signal_path,
            design_guided_samples=design_path,
            design_conflict=conflict,
        )
        previous = final_path
    return output
