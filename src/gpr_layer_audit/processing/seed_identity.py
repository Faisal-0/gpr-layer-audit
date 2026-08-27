"""Keep a seeded interface from being replaced by a separate persistent packet.

Horizontal continuity alone is not a defect. This guard only excludes a
persistent, strong competing packet inside an interval bounded by two manual
seeds that both select other packets. It neither generates picks nor declares
the competitor to be an artifact. Outside bracketing seeds the tracker must use
ordinary radar continuation and explicit gaps; a single seed is not authority
for a road-length hard exclusion.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks


def stationary_competitor_mask(
    radargram: np.ndarray,
    candidates: np.ndarray,
    anchors: dict[int, int],
    lower: np.ndarray,
    upper: np.ndarray,
    pulse_width_samples: float,
    horizontal_step_m: float,
    break_rows: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return excluded candidates and a dense review map, without depth priors.

    Bands are measured on the unstripped interpretation radar, independently
    of graph scores. A peak must persist within a narrow sample tolerance on
    >=80% of a span and have substantial relative amplitude. Both bracketing
    seeds must be separated from it; a seed on/near the packet protects it.
    Structural breaks reset seed ownership. Fewer than two seeds in a
    structural segment means no exclusion in that segment.
    """
    blocked = np.zeros(candidates.shape, dtype=bool)
    dense = np.zeros(radargram.shape, dtype=np.float32)
    if len(anchors) < 2 or not len(radargram):
        return blocked, dense
    rows, sample_count = radargram.shape
    width = max(1.0, float(pulse_width_samples))
    drift = max(2, int(round(0.25 * width)))
    packet_radius = max(3, int(np.ceil(0.60 * width)))
    seed_clearance = packet_radius + max(2, int(np.ceil(0.25 * width)))
    minimum_rows = max(12, int(np.ceil(10.0 / max(horizontal_step_m, 1e-3))))
    amplitude = np.nan_to_num(np.abs(radargram), nan=0.0, posinf=0.0, neginf=0.0)
    columns = np.arange(sample_count)
    in_bounds = (columns[None, :] >= lower[:, None]) & (columns[None, :] <= upper[:, None])
    amplitude = np.where(in_bounds, amplitude, 0.0)
    scale = np.max(amplitude, axis=1, keepdims=True)
    normalized = np.divide(amplitude, scale, out=np.zeros_like(amplitude), where=scale > 0)
    peaks = (amplitude == maximum_filter1d(amplitude, 5, axis=1)) & (amplitude > 0)
    peak_strength = np.where(peaks, normalized, 0.0)
    nearby_strength = maximum_filter1d(peak_strength, 2 * drift + 1, axis=1)
    boundaries = sorted({0, rows, *(r for r in break_rows if 0 < r < rows)})
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        owned = sorted(r for r in anchors if start <= r < stop)
        if len(owned) < 2:
            continue
        # A hard identity decision is justified only between consecutive
        # manual observations. Do not project it to either road boundary.
        for first, last in zip(owned[:-1], owned[1:], strict=True):
            if last - first + 1 < minimum_rows:
                continue
            seed_samples = np.array([anchors[first], anchors[last]])
            local = nearby_strength[first : last + 1]
            support = np.mean(local >= 0.20, axis=0)
            strength = np.median(local, axis=0)
            score = support * strength
            # Include boundaries in peak discovery, then use actual extrema
            # to locate the packet centre (not a plateau's arbitrary edge).
            centres, _ = find_peaks(np.pad(score, 1), distance=max(3, int(width / 2)))
            centres = centres - 1
            occupied: list[int] = []
            for centre in sorted(centres, key=lambda c: score[c], reverse=True):
                if support[centre] < 0.80 or strength[centre] < 0.35:
                    continue
                low, high = max(0, centre - drift), min(sample_count, centre + drift + 1)
                weights = peak_strength[first : last + 1, low:high].sum(axis=0)
                band = low + int(np.argmax(weights))
                if any(abs(band - previous) <= packet_radius for previous in occupied):
                    continue
                occupied.append(band)
                if np.any(abs(seed_samples - band) <= seed_clearance):
                    continue
                # This is a persistent exclusion, including null-state gaps:
                # restarting after a fade cannot reacquire this other packet.
                begin, end = (
                    max(0, band - packet_radius),
                    min(sample_count, band + packet_radius + 1),
                )
                dense[first : last + 1, begin:end] = 1.0
    # Anchors are authoritative even at an interval boundary or regime change.
    for row in anchors:
        dense[row] = 0.0
    valid = (candidates >= 0) & (candidates < sample_count)
    indices = np.clip(candidates, 0, sample_count - 1).astype(int)
    blocked = valid & (dense[np.arange(rows)[:, None], indices] > 0)
    return blocked, dense
