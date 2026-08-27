"""Radar-derived local continuation, separate from seed-template resemblance.

Matching a seed's waveform identifies a plausible event type, not its spatial
lineage. These maps register adjacent analytic traces with signed correlation;
candidate components then describe which events can actually be continued to
a confirmed seed without teleporting between similar-looking wavelet cycles.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert


def _shift_columns(data: np.ndarray, shift: int) -> np.ndarray:
    output = np.zeros_like(data)
    if shift > 0:
        output[:, :-shift] = data[:, shift:]
    elif shift < 0:
        output[:, -shift:] = data[:, :shift]
    else:
        output[:] = data
    return output


def local_phase_motion(data: np.ndarray, pulse_width: float, cancel=None) -> dict[str, np.ndarray]:
    """Measure forward/backward displacement with no seeds, design or labels.

    Search at most one pulse per adjacent trace bin. Larger changes require a
    break rather than claiming continuation across an unobserved wavelet cycle.
    The real part of complex correlation preserves phase and polarity; its
    magnitude alone would incorrectly identify opposite lobes as equivalent.
    """
    data = np.asarray(data, dtype=np.float32)
    radius = max(2, int(np.ceil(pulse_width)))
    window = 2 * radius + 1
    analytic = hilbert(data, axis=1)
    shifts = np.arange(-radius, radius + 1)
    maps = {}
    for direction in ("forward", "backward"):
        source = analytic[:-1] if direction == "forward" else analytic[1:]
        target = analytic[1:] if direction == "forward" else analytic[:-1]
        displacement, strength, margin = (np.zeros_like(data) for _ in range(3))
        # Chunking bounds temporary allocations without omitting any samples.
        for start in range(0, len(source), 256):
            if cancel and cancel():
                raise InterruptedError("Analysis cancelled")
            left, right = source[start : start + 256], target[start : start + 256]
            energy_left = uniform_filter1d(abs(left) ** 2, window, axis=1, mode="constant")
            scores = []
            for shift in shifts:
                shifted = _shift_columns(right, int(shift))
                cross = uniform_filter1d(np.conj(left) * shifted, window, axis=1, mode="constant")
                energy_right = uniform_filter1d(abs(shifted) ** 2, window, axis=1, mode="constant")
                denominator = np.sqrt(np.maximum(energy_left * energy_right, 1e-20))
                score = np.clip(cross.real / denominator, -1, 1).astype(np.float32)
                score[:, :radius] = -1
                score[:, -radius:] = -1
                scores.append(score)
            scores = np.stack(scores)
            # Stable tie handling prefers zero displacement, not the most
            # negative shift on zero-energy or exactly periodic data.
            priority = scores - 1e-6 * abs(shifts[:, None, None])
            winner = np.argmax(priority, axis=0)
            best = np.take_along_axis(scores, winner[None, ...], axis=0)[0]
            competing = abs(shifts[:, None, None] - shifts[winner]) > max(1, radius // 3)
            runner_up = np.max(np.where(competing, scores, -1), axis=0)
            offset = start if direction == "forward" else start + 1
            stop = offset + len(left)
            displacement[offset:stop] = shifts[winner]
            strength[offset:stop] = np.clip(best, 0, 1)
            margin[offset:stop] = np.maximum(best - runner_up, 0)
        maps[f"motion_{direction}_shift"] = displacement
        maps[f"motion_{direction}_support"] = strength
        maps[f"motion_{direction}_margin"] = margin
    return maps


def _advect(samples, first, last, maps, direction):
    predicted = np.asarray(samples, dtype=float).copy()
    support = np.ones(len(samples))
    step = 1 if direction == "forward" else -1
    shifts = maps[f"motion_{direction}_shift"]
    strength = maps[f"motion_{direction}_support"]
    for row in range(first, last, step):
        columns = np.clip(np.rint(predicted).astype(int), 0, shifts.shape[1] - 1)
        support = np.minimum(support, strength[row, columns])
        predicted += shifts[row, columns]
    return predicted, support


def candidate_links(table, left: int, right: int, pulse_width: float) -> np.ndarray:
    """Reciprocal phase-motion compatibility, including short candidate holes."""
    maps = table.component_maps
    previous = table.samples[left]
    current = table.samples[right]
    valid = (previous[:, None] >= 0) & (current[None, :] >= 0)
    valid &= table.valid[left, :, None] & table.valid[right, None, :]
    valid &= table.polarities[left, :, None] == table.polarities[right, None, :]
    predicted, forward_support = _advect(previous, left, right, maps, "forward")
    reversed_prediction, backward_support = _advect(current, right, left, maps, "backward")
    tolerance = max(2.0, 0.35 * pulse_width)
    forward_error = abs(current[None, :] - predicted[:, None])
    backward_error = abs(previous[:, None] - reversed_prediction[None, :])
    # Snippets are centred on the selected lobe. Keep the sign and require
    # waveform support as well as a locally registered displacement.
    similarity = table.waveforms[left] @ table.waveforms[right].T
    strength = np.minimum(forward_support[:, None], backward_support[None, :])
    return (
        valid
        & (forward_error <= tolerance)
        & (backward_error <= tolerance)
        & (similarity >= 0.35)
        & (strength >= 0.50)
    )


def attach_spatial_lineage(table, anchors, break_rows, pulse_width, horizontal_step_m, cancel=None):
    """Find seed-connected event components without using their ranker scores.

    Strong local evidence can bridge a candidate-table hole of at most one
    metre, but not a zero-signal gap or structural break. Reachability is an
    identity constraint, never a confidence boost or additional training label.
    """
    if "motion_forward_shift" not in table.component_maps:
        return
    rows, count = table.samples.shape
    parent = np.arange(rows * count)
    rank = np.zeros(len(parent), dtype=np.int8)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return int(value)

    def union(left, right):
        a, b = find(left), find(right)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    links = np.zeros((rows, count, count), dtype=bool)
    maximum_skip = max(1, int(np.floor(1.0 / max(horizontal_step_m, 1e-3))) + 1)
    for right in range(1, rows):
        if cancel and right % 64 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        for gap in range(1, min(maximum_skip, right) + 1):
            left = right - gap
            if any(row in break_rows for row in range(left + 1, right + 1)):
                continue
            # A fully invalid row is a forced gap/anomaly, not a dropped
            # candidate inside an otherwise observable radar trace.
            if any(not np.any(table.valid[row, :-1]) for row in range(left, right + 1)):
                continue
            connected = candidate_links(table, left, right, pulse_width)
            if gap == 1:
                links[right] = connected
            for previous, current in zip(*np.nonzero(connected), strict=True):
                union(left * count + previous, right * count + current)
    lineage = np.full(table.samples.shape, -1, dtype=np.int32)
    for row, index in zip(*np.nonzero(table.valid & (table.samples >= 0)), strict=True):
        lineage[row, index] = find(row * count + index)
    seed_roots = set()
    for row, sample in anchors.items():
        matches = np.flatnonzero(table.valid[row] & (table.samples[row] == sample))
        if len(matches):
            seed_roots.add(int(lineage[row, matches[0]]))
    reachable = np.isin(lineage, list(seed_roots)) & (lineage >= 0)
    table.continuation_links = links
    table.spatial_lineage_indices = lineage
    table.seed_reachable = reachable
    table.component_maps["seed_reachability_required"] = np.full_like(
        table.dense_radar_score, float(bool(anchors))
    )
    for name, values in (("spatial_lineage_index", lineage), ("seed_reachable", reachable)):
        dense = np.full_like(table.dense_radar_score, -1 if "index" in name else 0)
        for row in range(rows):
            valid = table.valid[row] & (table.samples[row] >= 0)
            dense[row, table.samples[row, valid]] = values[row, valid]
        table.component_maps[name] = dense
