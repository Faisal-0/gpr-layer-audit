"""Banded whole-trace registration, with bounded local stretching.

Unlike independently re-centred candidate snippets, one registration preserves
the order of all reflectors in the time window. It supplies correspondence only;
it neither changes measured signal nor establishes interface visibility.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def balance_waveforms(data, pulse_width, valid=None):
    data = np.asarray(data, np.float32)
    if valid is not None:
        from .conventional_signal import numerical_extension

        data = numerical_extension(data, valid)
    energy = uniform_filter1d(data.astype(np.float64) ** 2,
                             max(3, round(2 * pulse_width) | 1), axis=1, mode="nearest")
    floor = .05 * np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=1, keepdims=True))
    if valid is not None:
        window = max(3, round(2 * pulse_width) | 1)
        weight = uniform_filter1d(valid.astype(float), window, axis=1, mode="nearest")
        energy = uniform_filter1d(np.where(valid, data, 0).astype(float) ** 2,
                                  window, axis=1, mode="nearest")
        energy = np.divide(energy, weight, out=np.zeros_like(energy), where=weight > 0)
        floor = .05 * np.sqrt(
            np.sum(np.where(valid, data, 0).astype(float)**2, axis=1, keepdims=True)
            / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        )
    scale = np.maximum(np.sqrt(np.maximum(energy, 0)), floor)
    return np.divide(data, scale, out=np.zeros_like(data), where=scale > 0)


def register_pairs(left, right, band, pulse_width, cancel=None):
    """Register equally sampled pairs in a bounded batch.

    Allowed path steps are (1,1), (1,2), (2,1). Arbitrarily long horizontal or
    vertical runs cannot collapse several ringing cycles onto one sample.
    Returns forward and inverse coordinate maps, plus local waveform agreement.
    Arrays are (pairs, samples). Caller bounds batch size to limit memory.
    """
    left, right = np.asarray(left, np.float32), np.asarray(right, np.float32)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 2:
        raise ValueError("Registration needs equally shaped (pairs, samples) arrays")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Registration waveforms must be finite")
    if not isinstance(band, int) or band < 1 or not np.isfinite(pulse_width) or pulse_width <= 0:
        raise ValueError("Positive integer warp band and pulse width are required")
    pairs, samples = left.shape
    band = min(band, samples - 1)
    offsets = np.arange(-band, band + 1)
    width = len(offsets)
    parents = np.zeros((pairs, samples, width), dtype=np.int8)
    previous = np.full((pairs, width), np.inf, np.float32)
    before_previous = previous.copy()
    previous_error = previous.copy()
    for row in range(samples):
        if cancel and row % 32 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        columns = row + offsets
        valid = (columns >= 0) & (columns < samples)
        error = (left[:, row, None] - right[:, np.clip(columns, 0, samples - 1)]) ** 2
        error[:, ~valid] = np.inf
        diagonal = previous
        stretch_right = np.full_like(previous, np.inf)
        stretch_left = np.full_like(previous, np.inf)
        stretch_right[:, 1:] = previous[:, :-1] + error[:, :-1] + .02
        stretch_left[:, :-1] = before_previous[:, 1:] + previous_error[:, 1:] + .02
        costs = np.stack((diagonal, stretch_right, stretch_left))
        choice = np.argmin(costs, axis=0)
        current = np.take_along_axis(costs, choice[None], axis=0)[0] + error
        if row == 0:
            current[:, band] = error[:, band]
        parents[:, row] = choice
        before_previous, previous = previous, current
        previous_error = error
    forward = np.zeros(left.shape, np.float32)
    inverse = np.zeros(left.shape, np.float32)
    for pair in range(pairs):
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        counts_left, counts_right = np.zeros(samples), np.zeros(samples)
        row = column = samples - 1
        while row >= 0 and column >= 0:
            forward[pair, row] += column
            inverse[pair, column] += row
            counts_left[row] += 1
            counts_right[column] += 1
            if row == 0 and column == 0:
                break
            direction = parents[pair, row, column - row + band]
            if direction == 1:
                forward[pair, row] += column - 1
                inverse[pair, column - 1] += row
                counts_left[row] += 1
                counts_right[column - 1] += 1
                row, column = row - 1, column - 2
            elif direction == 2:
                forward[pair, row - 1] += column
                inverse[pair, column] += row - 1
                counts_left[row - 1] += 1
                counts_right[column] += 1
                row, column = row - 2, column - 1
            else:
                row, column = row - 1, column - 1
        forward[pair] /= np.maximum(counts_left, 1)
        inverse[pair] /= np.maximum(counts_right, 1)
    target = np.arange(samples)
    error_left, error_right = np.zeros_like(left), np.zeros_like(right)
    for pair in range(pairs):
        error_left[pair] = (left[pair] - np.interp(forward[pair], target, right[pair])) ** 2
        error_right[pair] = (right[pair] - np.interp(inverse[pair], target, left[pair])) ** 2
    window = max(3, round(2 * pulse_width) | 1)
    agreement_left = np.exp(-.5 * uniform_filter1d(error_left, window, axis=1))
    agreement_right = np.exp(-.5 * uniform_filter1d(error_right, window, axis=1))
    return forward, inverse, agreement_left, agreement_right


def sparse_registration(
    measurement, candidates, distances, regions, pulse_width, step, cancel=None, *, valid=None
):
    """Cache whole-profile coordinate maps for the sparse correspondence pairs."""
    rows, samples = measurement.shape
    selected = candidates[candidates >= 0]
    if not len(selected):
        return {}
    low = max(0, int(np.min(selected)) - round(4 * pulse_width))
    high = min(samples, int(np.max(selected)) + round(4 * pulse_width) + 1)
    if high - low < 2:
        return {}
    balanced = balance_waveforms(measurement, pulse_width, valid)[:, low:high]
    result = {}
    for gap in distances:
        if gap >= rows:
            continue
        source = np.arange(rows - gap)
        source = source[regions[source] == regions[source + gap]]
        if not len(source):
            continue
        forward, inverse, quality, reverse_quality = (
            np.zeros((rows, high - low), np.float32) for _ in range(4)
        )
        band = max(1, int(np.ceil(pulse_width * (1 + np.sqrt(gap * step) / 2))))
        for offset in range(0, len(source), 64):
            batch = source[offset:offset + 64]
            f, _, q, _ = register_pairs(balanced[batch], balanced[batch + gap],
                                       band, pulse_width, cancel)
            # Reversing the first warp is not independent reciprocal evidence.
            b, _, r, _ = register_pairs(balanced[batch + gap], balanced[batch],
                                       band, pulse_width, cancel)
            forward[batch] = f + low
            inverse[batch + gap] = b + low
            quality[batch] = q
            reverse_quality[batch + gap] = r
        if valid is not None:
            quality[~valid[:, low:high]] = 0
            reverse_quality[~valid[:, low:high]] = 0
        result[gap] = (low, forward, inverse, quality, reverse_quality)
    return result
