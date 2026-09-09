"""Research adapter for HorizonTracker's full-trace DTW correspondence.

Upstream: ajbugge/HorizonTracker, bf550594bd55d7741a0fcbfdef0e4ede2567d48b,
MIT, Copyright (c) 2019 Aina Juell Bugge. This module calls tslearn rather than
copying upstream code. The experiment loads and differential-tests the actual
upstream function. No interpolation or upstream horizon interpolation is used.
This adapter is not selected by application dispatch.
"""

from __future__ import annotations

import numpy as np


def valid_runs(mask):
    """Half-open contiguous observation ranges; never bridge a signal gap."""
    values = np.asarray(mask, bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, values, False].astype(int)))
    return list(zip(boundaries[::2], boundaries[1::2], strict=True))


def profile_paths(left, right, valid_left=None, valid_right=None):
    """Independent forward/reverse unconstrained DTW on common valid runs.

    Coordinates are native sample indices. Arrays are individual A-scans;
    callers convert repository (trace, sample) arrays to upstream
    (sample, inline, crossline) only when invoking its reference function.
    Unlike upstream trailing-zero trimming, explicit masks govern validity.
    """
    from tslearn.metrics import dtw_path

    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Equal-length one-dimensional traces are required")
    valid = np.isfinite(left) & np.isfinite(right)
    for mask in (valid_left, valid_right):
        if mask is not None:
            if np.shape(mask) != left.shape:
                raise ValueError("Validity must have the trace shape")
            valid &= np.asarray(mask, bool)
    forward, reverse = [], []
    for low, high in valid_runs(valid):
        if high - low < 2 or not np.any(left[low:high]) or not np.any(right[low:high]):
            continue
        first, _ = dtw_path(left[low:high], right[low:high])
        second, _ = dtw_path(right[low:high], left[low:high])
        forward.extend((a + low, b + low) for a, b in first)
        reverse.extend((a + low, b + low) for a, b in second)
    return np.asarray(forward, int).reshape(-1, 2), np.asarray(reverse, int).reshape(-1, 2)


def warped_context_cosine(left, right, path, at_left, at_right, radius, tolerance):
    """Cosine over observed aligned pairs, preserving all mapped centre samples."""
    if not len(path):
        return -1.0
    centres = path[path[:, 0] == at_left, 1]
    if not len(centres) or np.min(abs(centres - at_right)) > tolerance:
        return -1.0
    selected = (abs(path[:, 0] - at_left) <= radius) & (
        abs(path[:, 1] - at_right) <= radius
    )
    points = path[selected]
    if len(points) < 3:
        return -1.0
    a, b = np.asarray(left)[points[:, 0]], np.asarray(right)[points[:, 1]]
    a, b = a - a.mean(), b - b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / scale, -1, 1)) if scale > 0 else -1.0


class ProfileCorrespondence:
    """One-mechanism experimental replacement for raw packet cosine only."""

    def __init__(self, measurement, valid=None):
        self.measurement = np.asarray(measurement)
        self.valid = np.ones_like(measurement, bool) if valid is None else np.asarray(valid, bool)
        self.counts = {"trace_pairs": 0, "candidate_pairs": 0, "cosine_rescues": 0}

    def similarity(self, left, right, ii, js, table, matching_waveforms, pulse_width):
        raw = matching_waveforms[left, ii] @ matching_waveforms[right, js].T
        before, after = self.measurement[left], self.measurement[right]
        forward, reverse = profile_paths(before, after, self.valid[left], self.valid[right])
        radius = matching_waveforms.shape[-1] // 2
        tolerance = max(2, pulse_width / 4)
        result = raw.copy()
        self.counts["trace_pairs"] += 1
        self.counts["candidate_pairs"] += result.size
        for a, source in enumerate(table.samples[left, ii]):
            targets = forward[forward[:, 0] == source, 1]
            if not len(targets):
                continue
            for b in np.flatnonzero(
                np.min(abs(table.samples[right, js, None] - targets), axis=1) <= tolerance
            ):
                target = table.samples[right, js[b]]
                value = min(
                    warped_context_cosine(
                        before, after, forward, source, target, radius, tolerance
                    ),
                    warped_context_cosine(
                        after, before, reverse, target, source, radius, tolerance
                    ),
                )
                result[a, b] = max(raw[a, b], value)
        self.counts["cosine_rescues"] += int(np.count_nonzero((raw < .60) & (result >= .60)))
        return result
