from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import fftconvolve, hilbert

from gpr_layer_audit.models import LayerSpec


@dataclass(slots=True)
class PickPath:
    samples: NDArray[np.int32]
    confidence: NDArray[np.float64]
    feature: NDArray[np.float32]
    alternate_samples: NDArray[np.int32]


def feature_maps(
    radargram: NDArray[np.floating],
    matched_template: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float32], ...]:
    data = np.asarray(radargram, dtype=np.float32)
    envelope = np.abs(hilbert(data, axis=1)).astype(np.float32)
    gradient = np.abs(np.gradient(data, axis=1)).astype(np.float32)
    envelope = gaussian_filter(envelope, sigma=(1.2, 1.0))
    gradient = gaussian_filter(gradient, sigma=(1.2, 0.8))
    if matched_template is not None and len(matched_template) >= 5:
        wavelet = np.asarray(matched_template, dtype=np.float32)[None, ::-1]
        matched = np.abs(fftconvolve(data, wavelet, mode="same", axes=1)).astype(np.float32)
        matched = gaussian_filter(matched, sigma=(1.0, 0.7))
        combined = 0.44 * envelope + 0.20 * gradient + 0.36 * matched
        features = (envelope, gradient, matched, combined)
    else:
        combined = 0.72 * envelope + 0.28 * gradient
        features = (envelope, gradient, combined)
    for feature in features:
        scale = np.percentile(feature, 95, axis=1, keepdims=True)
        feature /= np.maximum(scale, 1e-6)
    return envelope, gradient, combined


def viterbi_path(
    score: NDArray[np.floating],
    valid_mask: NDArray[np.bool_],
    *,
    max_jump: int = 5,
    smooth_penalty: float = 0.16,
    anchors: dict[int, int] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> NDArray[np.int32]:
    values = np.asarray(score, dtype=np.float32).copy()
    values[~valid_mask] = -1e9
    rows, columns = values.shape
    anchors = anchors or {}
    if 0 in anchors:
        anchor = anchors[0]
        values[0] = -1e9
        values[0, anchor] = score[0, anchor] + 1e4
    previous = values[0].copy()
    back = np.zeros((rows, columns), dtype=np.int8)
    deltas = np.arange(-max_jump, max_jump + 1, dtype=int)
    for row in range(1, rows):
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        candidates = np.full((len(deltas), columns), -1e9, dtype=np.float32)
        for index, delta in enumerate(deltas):
            # ``delta`` is predecessor - current.  Keeping that definition
            # explicit avoids a subtle sign reversal during backtracking.
            if delta < 0:
                candidates[index, -delta:] = previous[: columns + delta]
            elif delta > 0:
                candidates[index, : columns - delta] = previous[delta:]
            else:
                candidates[index] = previous
            candidates[index] -= smooth_penalty * abs(delta)
        choice = np.argmax(candidates, axis=0)
        current = values[row] + candidates[choice, np.arange(columns)]
        if row in anchors:
            anchor = anchors[row]
            keep = current[anchor]
            current[:] = -1e9
            current[anchor] = keep + 1e4
        back[row] = deltas[choice]
        previous = current
    path = np.empty(rows, dtype=np.int32)
    path[-1] = int(np.argmax(previous))
    for row in range(rows - 1, 0, -1):
        path[row - 1] = np.clip(path[row] + int(back[row, path[row]]), 0, columns - 1)
    return path


def _confidence(
    feature: NDArray[np.floating],
    path: NDArray[np.integer],
    alternate: NDArray[np.integer],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    rows = np.arange(len(path))
    selected = feature[rows, path]
    valid_rows = valid_mask.any(axis=1)
    median = np.zeros(feature.shape[0], dtype=np.float64)
    spread = np.ones(feature.shape[0], dtype=np.float64)
    masked = np.where(valid_mask, feature, np.nan)
    if np.any(valid_rows):
        valid_values = masked[valid_rows]
        valid_median = np.nanmedian(valid_values, axis=1)
        median[valid_rows] = valid_median
        spread[valid_rows] = (
            np.nanmedian(np.abs(valid_values - valid_median[:, None]), axis=1) + 1e-6
        )
    signal_score = 1.0 / (1.0 + np.exp(-(selected - median) / (2.5 * spread)))
    agreement = np.exp(-np.abs(path - alternate) / 7.0)
    jumps = np.abs(np.gradient(path.astype(float)))
    continuity = np.exp(-jumps / 4.0)
    lower = np.argmax(valid_mask, axis=1)
    upper = valid_mask.shape[1] - 1 - np.argmax(valid_mask[:, ::-1], axis=1)
    boundary_distance = np.minimum(path - lower, upper - path)
    boundary_score = np.clip(boundary_distance / 4.0, 0.0, 1.0)
    # A smooth, mutually-agreeing path is not sufficient evidence when the
    # reflection itself is weak.  Signal evidence is therefore multiplicative.
    confidence = signal_score * (0.55 + 0.30 * agreement + 0.15 * continuity) * boundary_score
    confidence = np.clip(gaussian_filter1d(confidence, sigma=2), 0.0, 1.0)
    confidence[~valid_rows] = 0.0
    return confidence


def pick_interfaces(
    radargram: NDArray[np.floating],
    reference_surface_sample: int,
    layers: list[LayerSpec],
    *,
    anchor_samples: dict[int, dict[int, int]] | None = None,
    matched_template: NDArray[np.floating] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[int, PickPath]:
    envelope, gradient, combined = feature_maps(radargram, matched_template)
    rows, samples = radargram.shape
    anchor_samples = anchor_samples or {}
    output: dict[int, PickPath] = {}
    previous: NDArray[np.int32] | None = None
    sample_axis = np.arange(samples)[None, :]
    for layer in sorted((item for item in layers if item.analysis_enabled), key=lambda x: x.order):
        lower = reference_surface_sample + layer.min_offset_samples
        upper = min(samples - 1, reference_surface_sample + layer.max_offset_samples)
        mask = np.broadcast_to(
            (sample_axis >= lower) & (sample_axis <= upper), (rows, samples)
        ).copy()
        if previous is not None:
            mask &= sample_axis >= (previous[:, None] + layer.min_gap_samples)
        anchors = anchor_samples.get(layer.order, {})
        primary = viterbi_path(
            combined,
            mask,
            max_jump=6,
            smooth_penalty=0.18,
            anchors=anchors,
            cancel=cancel,
        )
        alternate = viterbi_path(
            envelope,
            mask,
            max_jump=7,
            smooth_penalty=0.14,
            anchors=anchors,
            cancel=cancel,
        )
        confidence = _confidence(combined, primary, alternate, mask)
        output[layer.order] = PickPath(primary, confidence, combined, alternate)
        previous = primary
    return output
