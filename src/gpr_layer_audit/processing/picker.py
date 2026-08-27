from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter1d, uniform_filter1d
from scipy.signal import fftconvolve, hilbert

from gpr_layer_audit.models import LayerSpec, SearchCorridor

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
    design_constrained: bool = False
    candidate_components: dict[str, NDArray[np.float32]] = field(default_factory=dict)


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


def _dtw_candidate_similarity(
    data: NDArray[np.float32],
    anchors: dict[int, int],
    score: NDArray[np.float32],
    radius: int = 10,
    candidates_per_row: int = 6,
) -> NDArray[np.float32]:
    """Constrained DTW on only the strongest fine-region candidates."""
    template = _seed_template(data, anchors, radius=radius)
    output = np.zeros_like(data, dtype=np.float32)
    if template is None or data.shape[0] > 1_500:
        return output
    length = len(template)
    band = 2
    for row in range(len(data)):
        count = min(candidates_per_row, data.shape[1])
        candidates = np.argpartition(score[row], -count)[-count:]
        for sample in candidates:
            start = int(sample) - radius
            stop = start + length
            if start < 0 or stop > data.shape[1]:
                continue
            waveform = data[row, start:stop].astype(np.float32).copy()
            waveform -= float(np.mean(waveform))
            norm = float(np.linalg.norm(waveform))
            if norm <= 1e-8:
                continue
            waveform /= norm
            cost = np.full((length + 1, length + 1), np.inf, dtype=np.float32)
            cost[0, 0] = 0.0
            for left in range(1, length + 1):
                for right in range(max(1, left - band), min(length, left + band) + 1):
                    difference = abs(float(template[left - 1] - waveform[right - 1]))
                    cost[left, right] = difference + min(
                        cost[left - 1, right],
                        cost[left, right - 1],
                        cost[left - 1, right - 1],
                    )
            output[row, sample] = float(np.exp(-cost[length, length] / length))
    return output


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
    lateral_mean = uniform_filter1d(data, size=5, axis=0, mode="nearest")
    lateral_power = uniform_filter1d(
        np.square(data, dtype=np.float64), size=5, axis=0, mode="nearest"
    )
    coherence0 = np.clip(
        np.square(lateral_mean, dtype=np.float64) / np.maximum(lateral_power, 1e-12),
        0.0,
        1.0,
    )
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
            0.35 * envelope + 0.25 * gradient + 0.15 * phase + 0.10 * coherence + 0.15 * correlation
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
                row - start: sample for row, sample in anchors.items() if start <= row < stop
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


def _remove_short_visible_runs(
    visible: NDArray[np.bool_], minimum_rows: int, anchors: dict[int, int]
) -> NDArray[np.bool_]:
    result = visible.copy()
    index = 0
    while index < len(result):
        if not result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and result[index]:
            index += 1
        if index - start < minimum_rows and not any(start <= row < index for row in anchors):
            result[start:index] = False
    return result


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
    snr_quality = np.zeros(len(path), dtype=float)
    margin = np.zeros(len(path), dtype=float)
    coherence = np.zeros(len(path), dtype=float)
    correlation = np.zeros(len(path), dtype=float)
    absolute_strength = np.zeros(len(path), dtype=float)
    for row in rows[valid]:
        sample = int(path[row])
        values = score[row, mask[row]]
        median = float(np.median(values)) if len(values) else 0.0
        spread = float(np.median(np.abs(values - median))) + 1e-6 if len(values) else 1.0
        local_snr[row] = max(0.0, (signal[row] - median) / spread)
        snr_quality[row] = 1.0 / (1.0 + np.exp(-(local_snr[row] - 2.0)))
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
        -np.abs(path[perturbation_both] - perturbation[perturbation_both]) / max(pulse_width, 1.0)
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
        0.28 * snr_quality
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


def propose_seed_rows(candidate_feature: NDArray[np.floating], count: int = 3) -> NDArray[np.int32]:
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
        margin = int(0.15 * (stop - start))
        search_start = min(stop - 1, start + margin)
        search_stop = max(search_start + 1, stop - margin)
        selected.append(
            search_start + int(np.argmax(quality[search_start:search_stop]))
        )
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def _validate_anchors(
    anchor_samples: dict[int, dict[int, int]],
    layers: list[LayerSpec],
    row_count: int,
    sample_count: int,
) -> None:
    enabled = {item.order: item for item in layers if item.analysis_enabled}
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
    seed_metadata: dict[int, dict[int, dict[str, object]]] | None = None,
    matched_template: NDArray[np.floating] | None = None,
    feature_branches: dict[str, NDArray[np.floating]] | None = None,
    design_prior_samples: dict[int, NDArray[np.floating]] | None = None,
    design_prior_widths: dict[int, NDArray[np.floating]] | None = None,
    design_weight: float = 0.20,
    search_corridors: dict[int, SearchCorridor] | None = None,
    pulse_width_samples: float = 7.0,
    break_rows: set[int] | None = None,
    anomaly_mask: NDArray[np.bool_] | None = None,
    max_interpolation_rows: int = 2,
    horizontal_step_m: float = 0.4,
    method: str = "joint_seed_adaptive",
    cancel: Callable[[], bool] | None = None,
) -> dict[int, PickPath]:
    data = np.asarray(radargram, dtype=np.float32)
    rows, samples = data.shape
    anchor_samples = anchor_samples or {}
    design_prior_samples = design_prior_samples or {}
    design_prior_widths = design_prior_widths or {}
    search_corridors = search_corridors or {}
    anomaly_mask = (
        np.asarray(anomaly_mask, dtype=bool)
        if anomaly_mask is not None
        else np.zeros(rows, dtype=bool)
    )
    if anomaly_mask.shape != (rows,):
        raise ValueError("anomaly_mask must contain one value per horizontal bin")
    if method not in TRACKER_METHODS:
        raise ValueError(
            f"Unknown tracker method {method!r}; choose one of {', '.join(TRACKER_METHODS)}."
        )
    _validate_anchors(anchor_samples, layers, rows, samples)
    if method == "joint_seed_adaptive":
        from .seed_graph import pick_seed_conditioned_interfaces

        replacement = pick_seed_conditioned_interfaces(
            data,
            reference_surface_sample,
            layers,
            anchor_samples=anchor_samples,
            seed_metadata=seed_metadata,
            feature_branches=feature_branches,
            search_corridors=search_corridors,
            design_weight=design_weight,
            pulse_width_samples=pulse_width_samples,
            break_rows=break_rows or set(),
            anomaly_mask=anomaly_mask,
            max_interpolation_rows=max_interpolation_rows,
            horizontal_step_m=horizontal_step_m,
            cancel=cancel,
        )
        return {
            order: PickPath(
                samples=path.samples,
                confidence=path.confidence,
                feature=path.feature,
                alternate_samples=path.alternate_samples,
                visible=path.visible,
                interpolated=path.interpolated,
                evidence=path.evidence,
                signal_only_samples=path.signal_only_samples,
                design_guided_samples=path.design_guided_samples,
                design_conflict=path.design_conflict,
                design_constrained=path.design_constrained,
                candidate_components=path.candidate_components,
            )
            for order, path in replacement.items()
        }
    output: dict[int, PickPath] = {}
    previous: NDArray[np.int32] | None = None
    sample_axis = np.arange(samples)[None, :]
    for layer in sorted((item for item in layers if item.analysis_enabled), key=lambda x: x.order):
        anchors = anchor_samples.get(layer.order, {})
        raw_score, components = _layer_score(
            data, anchors, feature_branches, matched_template, method
        )
        if anchors:
            dtw = _dtw_candidate_similarity(data, anchors, raw_score)
            if np.any(dtw):
                raw_score = np.asarray(0.94 * raw_score + 0.06 * dtw, dtype=np.float32)
                components["correlation"] = np.maximum(components["correlation"], dtw)
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
        broad_mask = np.broadcast_to(
            (sample_axis >= lower) & (sample_axis <= upper), (rows, samples)
        ).copy()
        if seed_guide is not None:
            distance = np.abs(sample_axis - seed_guide[:, None])
            broad_mask &= distance <= seed_radius
            geometry_score = np.exp(
                -0.5 * np.square(distance / max(seed_radius * 0.55, 1.0))
            ).astype(np.float32)
            # Seed geometry expresses interface identity, not radar strength;
            # keep its contribution small enough that weak data remains weak.
            identity_score = np.asarray(0.86 * raw_score + 0.14 * geometry_score, dtype=np.float32)
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
            identity_score = np.asarray(np.clip(raw_score - 0.12 * depth_fraction, 0.0, 1.0))
        edge_distance = np.minimum(sample_axis - lower, upper - sample_axis)
        edge_ramp = np.clip(
            edge_distance / max(1.5 * pulse_width_samples, 1.0),
            0.0,
            1.0,
        )
        identity_score *= np.asarray(0.55 + 0.45 * edge_ramp, dtype=np.float32)
        if previous is not None:
            guide = _fill_guide(previous, lower - layer.min_gap_samples)
            broad_mask &= sample_axis >= (guide[:, None] + layer.min_gap_samples)
        for row, sample in anchors.items():
            if 0 <= row < rows and 0 <= sample < samples:
                broad_mask[row, sample] = True
        underconstrained = np.count_nonzero(broad_mask, axis=1) < 3
        for row in np.flatnonzero(underconstrained):
            if row not in anchors:
                broad_mask[row] = False
        signal_path, alternate = _bidirectional_path(
            identity_score,
            broad_mask,
            anchors,
            max_jump=7,
            smooth_penalty=0.045,
            minimum_evidence=0.12 if anchors else 0.26,
            break_rows=break_rows,
            cancel=cancel,
        )
        mask = broad_mask.copy()
        score = identity_score.copy()
        prior = design_prior_samples.get(layer.order)
        design_path: NDArray[np.int32] | None = None
        conflict = np.zeros(rows, dtype=bool)
        design_score_values = np.zeros(rows, dtype=float)
        corridor = search_corridors.get(layer.order)
        design_constrained = corridor is not None
        proximity: NDArray[np.float32] | None = None
        if corridor is not None:
            if len(corridor.centre_sample) != rows:
                raise ValueError(
                    f"Search corridor for layer {layer.order} has the wrong row count."
                )
            if previous is None:
                base = np.full(rows, reference_surface_sample, dtype=float)
            else:
                base = _fill_guide(previous, reference_surface_sample)
            expected = base + np.asarray(corridor.gap_centre_samples, dtype=float)
            corridor_lower = base + np.asarray(corridor.gap_lower_samples, dtype=float)
            corridor_upper = base + np.asarray(corridor.gap_upper_samples, dtype=float)
            for row, anchor_sample in anchors.items():
                if not 0 <= row < rows:
                    continue
                if anchor_sample < corridor_lower[row] or anchor_sample > corridor_upper[row]:
                    local_gap = float(anchor_sample - base[row])
                    corridor.gap_centre_samples[row] = local_gap
                    corridor.gap_lower_samples[row] = min(
                        corridor.gap_lower_samples[row],
                        local_gap - 2.0 * pulse_width_samples,
                    )
                    corridor.gap_upper_samples[row] = max(
                        corridor.gap_upper_samples[row],
                        local_gap + 2.0 * pulse_width_samples,
                    )
                    corridor.centre_sample[row] = anchor_sample
                    corridor.lower_sample[row] = base[row] + corridor.gap_lower_samples[row]
                    corridor.upper_sample[row] = base[row] + corridor.gap_upper_samples[row]
                    expected[row] = anchor_sample
                    corridor_lower[row] = corridor.lower_sample[row]
                    corridor_upper[row] = corridor.upper_sample[row]
            valid_prior = (
                np.isfinite(expected) & np.isfinite(corridor_lower) & np.isfinite(corridor_upper)
            )
            corridor_mask = (sample_axis >= corridor_lower[:, None]) & (
                sample_axis <= corridor_upper[:, None]
            )
            mask[valid_prior] &= corridor_mask[valid_prior]
            half_width = np.maximum((corridor_upper - corridor_lower) / 2.0, 1.0)
            proximity = np.exp(
                -0.5
                * np.square(
                    (sample_axis - expected[:, None]) / np.maximum(0.55 * half_width[:, None], 1.0)
                )
            ).astype(np.float32)
            proximity[~valid_prior] = 0.0
            # Design identifies the reflector family but cannot supply evidence.
            # Its deliberately small contribution only breaks radar-score ties.
            weight = float(np.clip(design_weight, 0.0, 0.10))
            score = np.asarray(
                (1.0 - weight) * identity_score + weight * proximity,
                dtype=np.float32,
            )
        elif prior is not None:
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
            weight = float(np.clip(design_weight, 0.0, 0.10))
            score = np.asarray(
                (1.0 - weight) * identity_score + weight * prior_score,
                dtype=np.float32,
            )
            proximity = prior_score
            design_constrained = True

        mask[anomaly_mask] = False
        for row, sample in anchors.items():
            if 0 <= row < rows and 0 <= sample < samples:
                # A user seed is evidence and may expand an incorrect tentative
                # design locally instead of being silently rejected.
                mask[row, sample] = True

        design_path, design_alternate = _bidirectional_path(
            score,
            mask,
            anchors,
            max_jump=7,
            smooth_penalty=0.045,
            minimum_evidence=0.12 if anchors else 0.26,
            break_rows=break_rows,
            cancel=cancel,
        )
        if corridor is not None:
            valid_design = design_path >= 0
            boundary = valid_design & (
                (np.abs(design_path - corridor_lower) <= pulse_width_samples)
                | (np.abs(design_path - corridor_upper) <= pulse_width_samples)
            )
            coherent_boundary = boundary & (
                identity_score[np.arange(rows), np.maximum(design_path, 0)] >= 0.26
            )
            if np.count_nonzero(coherent_boundary) >= max(2, int(0.04 * rows)):
                half_width = np.maximum((corridor_upper - corridor_lower) / 2.0, 1.0)
                corridor_lower = expected - 1.5 * half_width
                corridor_upper = expected + 1.5 * half_width
                expanded = broad_mask.copy()
                expanded[valid_prior] &= (
                    (sample_axis >= corridor_lower[:, None])
                    & (sample_axis <= corridor_upper[:, None])
                )[valid_prior]
                expanded[anomaly_mask] = False
                for row, anchor_sample in anchors.items():
                    expanded[row, anchor_sample] = True
                mask = expanded
                design_path, design_alternate = _bidirectional_path(
                    score,
                    mask,
                    anchors,
                    max_jump=7,
                    smooth_penalty=0.045,
                    minimum_evidence=0.12 if anchors else 0.26,
                    break_rows=break_rows,
                    cancel=cancel,
                )
                corridor.expanded = True
                corridor.gap_lower_samples = corridor_lower - base
                corridor.gap_upper_samples = corridor_upper - base
                corridor.lower_sample = corridor_lower.copy()
                corridor.upper_sample = corridor_upper.copy()
        if design_constrained:
            both = (signal_path >= 0) & (design_path >= 0)
            conflict = ((signal_path >= 0) != (design_path >= 0)) | (
                both & (np.abs(signal_path - design_path) > pulse_width_samples)
            )
            if proximity is not None:
                valid_design = design_path >= 0
                design_score_values[valid_design] = proximity[
                    np.arange(rows)[valid_design], design_path[valid_design]
                ]
            if corridor is not None and corridor.expanded:
                continued_edge = (design_path >= 0) & (
                    (np.abs(design_path - corridor_lower) <= pulse_width_samples)
                    | (np.abs(design_path - corridor_upper) <= pulse_width_samples)
                )
                conflict |= continued_edge
        else:
            # Without design this is the same unrestricted radar solution.
            design_path = signal_path.copy()
            design_alternate = alternate

        ensemble_paths: list[NDArray[np.int32]] = [design_path]
        ensemble_specs = (
            ("wavelet_amplitude", "wavelet_phase", "wavelet_coherence"),
            ("deconvolved", "deconvolved_phase", "deconvolved_coherence"),
            ("ensemble_amplitude", "ensemble_phase", "ensemble_coherence"),
        )
        for amplitude_name, phase_name, coherence_name in ensemble_specs:
            if not feature_branches or amplitude_name not in feature_branches:
                continue
            branch_map = {
                "phase": feature_branches.get(phase_name),
                "coherence": feature_branches.get(coherence_name),
                "deconvolved": feature_branches.get("deconvolved"),
            }
            branch_map = {key: value for key, value in branch_map.items() if value is not None}
            variant_raw, _ = _layer_score(
                np.asarray(feature_branches[amplitude_name], dtype=np.float32),
                anchors,
                branch_map,
                matched_template,
                method,
            )
            if seed_guide is not None:
                variant = np.asarray(0.86 * variant_raw + 0.14 * geometry_score, dtype=np.float32)
            else:
                variant = np.asarray(
                    np.clip(variant_raw - 0.12 * depth_fraction, 0.0, 1.0),
                    dtype=np.float32,
                )
            variant *= np.asarray(0.55 + 0.45 * edge_ramp, dtype=np.float32)
            if proximity is not None:
                variant = np.asarray(
                    (1.0 - weight) * variant + weight * proximity,
                    dtype=np.float32,
                )
            ensemble_paths.append(
                _viterbi_optional(
                    variant,
                    mask,
                    max_jump=7,
                    smooth_penalty=0.045,
                    anchors=anchors,
                    minimum_evidence=0.12 if anchors else 0.26,
                    cancel=cancel,
                )
            )

        final_path = design_path.copy()
        perturbation_path = ensemble_paths[1] if len(ensemble_paths) > 1 else design_alternate
        confidence, evidence = _confidence(
            identity_score,
            components,
            final_path,
            design_alternate,
            perturbation_path,
            mask,
            anchors,
            pulse_width_samples,
        )
        ensemble_agreement = np.zeros(rows, dtype=float)
        for row in range(rows):
            selected = int(final_path[row])
            if selected < 0:
                continue
            agreeing = sum(
                int(path[row] >= 0 and abs(int(path[row]) - selected) <= pulse_width_samples)
                for path in ensemble_paths
            )
            ensemble_agreement[row] = agreeing / max(len(ensemble_paths), 1)
        radar_gate = np.clip((evidence["signal_score"] - 0.16) / 0.14, 0.0, 1.0)
        radar_gate *= np.clip(evidence["absolute_strength"] / 0.08, 0.0, 1.0)
        confidence = np.asarray(
            0.78 * confidence + 0.22 * ensemble_agreement * radar_gate * (final_path >= 0),
            dtype=float,
        )
        evidence["ensemble_agreement"] = ensemble_agreement
        evidence["design_tiebreak"] = design_score_values
        evidence["design_score"] = design_score_values

        adequate_radar = (
            (
                (evidence["local_snr"] >= 2.0)
                & (evidence["coherence_score"] >= 0.35)
                & (evidence["signal_score"] >= 0.22)
            )
            | (evidence["seed_correlation"] >= 0.60)
        )
        visible = (
            (final_path >= 0)
            & (confidence >= 0.20)
            & adequate_radar
            & ~anomaly_mask
        )
        visible = _remove_short_visible_runs(
            visible, 6 if anchors else 12, anchors
        )
        for row, sample in anchors.items():
            if 0 <= row < rows and 0 <= sample < samples:
                final_path[row] = sample
                visible[row] = True
                confidence[row] = 1.0
        final_path[~visible] = -1
        confidence[~visible] = 0.0
        final_path, interpolated = _interpolate_short_gaps(
            final_path, max(0, int(max_interpolation_rows)), 7
        )
        interpolated[anomaly_mask] = False
        final_path[anomaly_mask] = -1
        visible = final_path >= 0
        confidence[interpolated] = np.minimum(confidence[interpolated], 0.45)
        # Design/signal disagreement is review-worthy only where radar evidence
        # is not already strong and stable; it does not discard a valid corridor path.
        weak_conflict = conflict & (confidence < 0.60)
        confidence[weak_conflict] = np.minimum(confidence[weak_conflict], 0.49)
        output[layer.order] = PickPath(
            samples=final_path,
            confidence=confidence,
            feature=identity_score,
            alternate_samples=design_alternate,
            visible=visible,
            interpolated=interpolated,
            evidence=evidence,
            signal_only_samples=signal_path,
            design_guided_samples=design_path,
            design_conflict=conflict,
            design_constrained=design_constrained,
        )
        previous = final_path
    return output
