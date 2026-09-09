"""Batched banded DTW with independent reciprocal alignment and centre-slip audit."""

import numpy as np


def batch_dtw(left, right, band):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("DTW batch waveforms must have matching shapes")
    pairs, n = left.shape
    if not pairs:
        return np.empty(0), np.empty(0)
    costs = np.full((pairs, n + 1, n + 1), np.inf)
    costs[:, 0, 0] = 0
    parent = np.zeros((pairs, n, n), np.int8)
    for diagonal in range(2 * n - 1):
        i = np.arange(max(0, diagonal - n + 1), min(n - 1, diagonal) + 1)
        j = diagonal - i
        valid = abs(i - j) <= band
        i, j = i[valid], j[valid]
        alternatives = np.stack((costs[:, i, j], costs[:, i, j + 1], costs[:, i + 1, j]))
        choice = np.argmin(alternatives, axis=0)
        parent[:, i, j] = choice
        costs[:, i + 1, j + 1] = (left[:, i] - right[:, j]) ** 2 + np.min(alternatives, axis=0)
    i = np.full(pairs, n - 1)
    j = i.copy()
    centre_sum, centre_count = np.zeros(pairs), np.zeros(pairs)
    for _ in range(2 * n):
        active = np.flatnonzero((i >= 0) & (j >= 0))
        if not len(active):
            break
        centre = active[i[active] == n // 2]
        centre_sum[centre] += j[centre] - n // 2
        centre_count[centre] += 1
        direction = parent[active, i[active], j[active]]
        i[active] -= direction != 2
        j[active] -= direction != 1
    return np.exp(-costs[:, -1, -1]), centre_sum / np.maximum(centre_count, 1)


def reciprocal_dtw(left, right, band):
    agreement, shift = batch_dtw(left, right, band)
    reverse, reverse_shift = batch_dtw(right, left, band)
    return np.minimum(agreement, reverse), np.maximum(abs(shift), abs(reverse_shift))
