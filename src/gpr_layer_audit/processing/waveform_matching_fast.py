"""Exact optional compiled implementation of the batched DTW numerical contract.

This module does not select candidates, alter validity masks, or reuse a forward
alignment as its reverse. The caller owns these contracts exactly as before.
Numba (BSD-2-Clause) compiles original project code, with no fast-math or parallel
reassociation. Accumulation and tie precedence match ``waveform_matching``;
NumPy performs the final exponential outside the compiled kernel.
"""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False, nogil=True)
def _batch_costs_shifts(left, right, band):
    pairs, n = left.shape
    final_costs = np.empty(pairs, dtype=np.float64)
    shifts = np.empty(pairs, dtype=np.float64)
    # Pair-local workspace avoids a pairs * n * n float64 cost cube.
    previous = np.empty(n + 1, dtype=np.float64)
    current = np.empty(n + 1, dtype=np.float64)
    parent = np.zeros((n, n), dtype=np.int8)
    for pair in range(pairs):
        previous[:] = np.inf
        previous[0] = 0.0
        for i in range(n):
            current[:] = np.inf
            for j in range(n):
                # Comparing directly also preserves the reference's fractional,
                # negative, NaN and infinite scalar band semantics.
                if not abs(i - j) <= band:
                    parent[i, j] = 0
                    continue
                diagonal = previous[j]
                up = previous[j + 1]
                leftward = current[j]
                choice = 0
                minimum = diagonal
                # np.argmin selects the first NaN, otherwise the first minimum:
                # diagonal, then up, then left. Strict < preserves tied paths.
                if not np.isnan(minimum) and (np.isnan(up) or up < minimum):
                    choice = 1
                    minimum = up
                if not np.isnan(minimum) and (
                    np.isnan(leftward) or leftward < minimum
                ):
                    choice = 2
                    minimum = leftward
                difference = left[pair, i] - right[pair, j]
                current[j + 1] = difference * difference + minimum
                parent[i, j] = choice
            previous, current = current, previous
        final_costs[pair] = previous[n]
        i = n - 1
        j = n - 1
        centre_sum = 0.0
        centre_count = 0
        for _ in range(2 * n):
            if i < 0 or j < 0:
                break
            if i == n // 2:
                centre_sum += j - n // 2
                centre_count += 1
            direction = parent[i, j]
            i -= direction != 2
            j -= direction != 1
        shifts[pair] = centre_sum / max(centre_count, 1)
    return final_costs, shifts


def batch_dtw_fast(left, right, band):
    """Drop-in ``batch_dtw`` result: agreement and signed centre displacement.

    Arrays must have the same two-dimensional shape. Empty pairs and zero-length
    packets follow the existing implementation. Inputs are never mutated. A
    contiguous float64 conversion keeps the compiled signature stable for slices
    and the float32 packet arrays used by correspondence inference.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("DTW batch waveforms must have matching shapes")
    if not left.shape[0]:
        return np.empty(0), np.empty(0)
    costs, shifts = _batch_costs_shifts(
        np.ascontiguousarray(left), np.ascontiguousarray(right), float(band)
    )
    return np.exp(-costs), shifts


def reciprocal_dtw_fast(left, right, band):
    """Independently align both orientations, as in the reference function."""
    agreement, shift = batch_dtw_fast(left, right, band)
    reverse, reverse_shift = batch_dtw_fast(right, left, band)
    return np.minimum(agreement, reverse), np.maximum(abs(shift), abs(reverse_shift))
