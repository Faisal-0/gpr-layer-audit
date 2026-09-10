"""Experimental dense native-depth decoding, independent of the legacy graph.

Every valid native sample is a state. A separate gap state can stop and restart
the path, so missing evidence is never filled by an interpolated measurement.
The continuity cost is soft and physical; there is no seed corridor or candidate
pruning. Scores and entropy are model evidence, not correctness probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DenseDecoderConfig:
    continuity_cost_per_ns: float = 2.0
    gap_probability: float = 0.01
    gap_transition_cost: float = 1.0
    acceptance_threshold: float = 0.0


@dataclass
class DensePathResult:
    samples: np.ndarray
    proposed_samples: np.ndarray
    accepted: np.ndarray
    manual: np.ndarray
    confidence: np.ndarray
    entropy: np.ndarray
    gap_mask: np.ndarray
    diagnostics: dict


def _seeds(seeds, shape):
    items = seeds.items() if isinstance(seeds, Mapping) else seeds
    result = {}
    for row, sample in items:
        if int(row) != row or not 0 <= row < shape[0]:
            raise ValueError("Seed trace is not an exact retained native-grid row")
        if not np.isfinite(sample) or not 0 <= sample <= shape[1] - 1:
            raise ValueError("Seed sample is outside the native depth grid")
        row, sample = int(row), float(sample)
        if row in result and result[row] != sample:
            raise ValueError(f"Contradictory selected-interface seeds at trace {row}")
        result[row] = sample
    return result


def prepare_probabilities(probabilities, sample_valid=None):
    """Validate depth probabilities and normalize over valid native samples."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or 0 in p.shape:
        raise ValueError("Expected nonempty probabilities [native trace, native sample]")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Probabilities must be finite and nonnegative")
    valid = np.ones(p.shape, bool) if sample_valid is None else np.asarray(sample_valid, bool)
    if valid.shape != p.shape:
        raise ValueError("Sample validity shape differs from evidence")
    p = np.where(valid, p, 0)
    totals = p.sum(axis=1, keepdims=True)
    p = np.divide(p, totals, out=np.zeros_like(p), where=totals > 0)
    logp = np.full(p.shape, -np.inf)
    np.log(p, out=logp, where=p > 0)
    entropy = -np.sum(np.where(p > 0, p * np.where(p > 0, logp, 0), 0), axis=1)
    normalizer = np.log(np.maximum(np.count_nonzero(valid, axis=1), 2))
    return p, valid, logp, entropy / normalizer


def direct_predictions(probabilities, *, sample_valid=None, acceptance_threshold=0.0):
    """Stage A: depth argmax, with no seed override, smoothing, or old graph."""
    if not np.isfinite(acceptance_threshold) or acceptance_threshold < 0:
        raise ValueError("Acceptance threshold must be finite and nonnegative")
    p, _, _, entropy = prepare_probabilities(probabilities, sample_valid)
    proposal = np.argmax(p, axis=1).astype(float)
    confidence = np.max(p, axis=1)
    proposal[confidence == 0] = -1
    accepted = (proposal >= 0) & (confidence >= acceptance_threshold)
    return DensePathResult(
        np.where(accepted, proposal, -1),
        proposal,
        accepted,
        np.zeros(len(p), bool),
        confidence,
        entropy,
        proposal < 0,
        {"stage": "direct_native_depth_argmax", "acceptance_threshold": acceptance_threshold},
    )


def _l1_transition(previous, penalty):
    """Exact max_j previous[j] - penalty*abs(i-j), including its argmax."""
    grid = np.arange(len(previous))
    left_input = previous + penalty * grid
    left = np.maximum.accumulate(left_input)
    left_index = np.maximum.accumulate(np.where(left_input == left, grid, -1))
    right_input = (previous - penalty * grid)[::-1]
    right = np.maximum.accumulate(right_input)
    reverse_index = np.maximum.accumulate(np.where(right_input == right, grid, -1))
    right_index = len(previous) - 1 - reverse_index[::-1]
    left = left - penalty * grid
    right = right[::-1] + penalty * grid
    use_right = right > left
    return np.where(use_right, right, left), np.where(use_right, right_index, left_index)


def decode_dense_path(
    probabilities,
    seeds,
    *,
    sample_valid=None,
    dx_m,
    dt_ns,
    config=None,
    break_rows=(),
    cancel=None,
):
    """Stage B: exact-seed dense Viterbi with explicit unresolved gap states.

    ``seeds`` use retained row indices and native *floating* sample coordinates.
    A fractional observation is represented at its exact coordinate for incoming
    and outgoing transitions, and exported unchanged. No waveform snapping occurs.
    ``break_rows`` are explicit unknown/discontinuity rows, never bridged; callers
    must not manufacture breaks from withheld reference coverage.

    The sum of per-trace log evidence minus total variation in ns is optimized.
    Dividing the entire objective by road length cannot affect the solution.
    Acceptance uses local selected-depth evidence, never an accumulated margin.
    """
    config = config or DenseDecoderConfig()
    if not np.isfinite(dx_m) or dx_m <= 0 or not np.isfinite(dt_ns) or dt_ns <= 0:
        raise ValueError("Positive physical trace/sample spacing is required")
    if (
        not np.isfinite(config.continuity_cost_per_ns)
        or config.continuity_cost_per_ns < 0
        or not 0 < config.gap_probability <= 1
        or not np.isfinite(config.gap_transition_cost)
        or config.gap_transition_cost < 0
        or not np.isfinite(config.acceptance_threshold)
        or config.acceptance_threshold < 0
    ):
        raise ValueError("Invalid dense decoder configuration")
    p, valid, emissions, entropy = prepare_probabilities(probabilities, sample_valid)
    n, depth = p.shape
    anchors = _seeds(seeds, p.shape)
    breaks = np.zeros(n, bool)
    for row in break_rows:
        if int(row) != row or not 0 <= row < n:
            raise ValueError("Break must be an exact retained native trace")
        if row in anchors:
            raise ValueError("A trace cannot be both an explicit break and an exact seed")
        breaks[int(row)] = True
    emissions[breaks] = -np.inf
    gap = depth
    pointer = np.full((n, depth + 1), -1, dtype=np.int32)
    previous = np.zeros(depth + 1)
    gap_log = float(np.log(config.gap_probability))
    penalty = config.continuity_cost_per_ns * dt_ns
    positions = np.arange(depth, dtype=float)
    for row in range(n):
        if cancel is not None and row % 128 == 0 and cancel():
            raise InterruptedError("Dense path decoding cancelled")
        if row == 0:
            best = np.zeros(depth)
            indices = np.full(depth, -1, np.int32)
        elif row - 1 in anchors:
            index = int(round(anchors[row - 1]))
            best = previous[index] - penalty * abs(positions - anchors[row - 1])
            indices = np.full(depth, index, np.int32)
        else:
            best, indices = _l1_transition(previous[:depth], penalty)
        if row > 0:
            from_gap = previous[gap] - config.gap_transition_cost
            use_gap = from_gap > best
            best = np.where(use_gap, from_gap, best)
            indices = np.where(use_gap, gap, indices)
        current = np.full(depth + 1, -np.inf)
        current[:depth] = best + emissions[row]
        pointer[row, :depth] = indices
        if row == 0:
            current[gap] = gap_log
        else:
            best_index = int(np.argmax(previous[:depth]))
            best_start = previous[best_index] - config.gap_transition_cost
            if previous[gap] >= best_start:
                best_start, best_index = previous[gap], gap
            current[gap] = best_start + gap_log
            pointer[row, gap] = best_index
        if row in anchors:
            sample = anchors[row]
            index = int(round(sample))  # storage slot only; geometry remains exact
            if row == 0:
                score, ancestor = 0.0, -1
            else:
                previous_positions = positions.copy()
                if row - 1 in anchors:
                    previous_positions[int(round(anchors[row - 1]))] = anchors[row - 1]
                options = previous[:depth] - penalty * abs(previous_positions - sample)
                ancestor = int(np.argmax(options))
                score = options[ancestor]
                if previous[gap] - config.gap_transition_cost > score:
                    score, ancestor = previous[gap] - config.gap_transition_cost, gap
            current[:] = -np.inf
            current[index] = score
            pointer[row, index] = ancestor
        previous = current
    state = int(np.argmax(previous))
    proposal = np.full(n, -1.0)
    for row in range(n - 1, -1, -1):
        if state != gap:
            proposal[row] = anchors.get(row, float(state))
        state = int(pointer[row, state])
    manual = np.zeros(n, bool)
    manual[list(anchors)] = True
    confidence = np.zeros(n)
    chosen = proposal >= 0
    confidence[chosen] = p[np.flatnonzero(chosen), np.rint(proposal[chosen]).astype(int)]
    accepted = chosen & (confidence >= config.acceptance_threshold)
    accepted[manual] = True
    gap_mask = proposal < 0
    starts = np.flatnonzero(gap_mask & np.r_[True, ~gap_mask[:-1]])
    stops = np.flatnonzero(gap_mask & np.r_[~gap_mask[1:], True])
    return DensePathResult(
        np.where(accepted, proposal, -1),
        proposal,
        accepted,
        manual,
        confidence,
        entropy,
        gap_mask,
        {
            "stage": "dense_native_soft_continuity_with_gaps",
            "configuration": asdict(config),
            "native_dx_m": dx_m,
            "dt_ns": dt_ns,
            "all_native_depths_eligible": True,
            "hard_displacement_bound": None,
            "continuity": "soft total variation in native time; no seed corridor",
            "resolution_note": (
                "emissions summed per retained trace; smoothing depends on lateral stride; "
                "compare configurations on a fixed grid"
            ),
            "seed_rows": sorted(anchors),
            "break_rows": np.flatnonzero(breaks).tolist(),
            "manual_seed_invalid_signal_rows": [
                row for row, value in anchors.items() if not valid[row, int(round(value))]
            ],
            "gap_runs": [
                {
                    "start_row": int(a),
                    "stop_row": int(b),
                    "observed_grid_footprint_m": float((b - a + 1) * dx_m),
                }
                for a, b in zip(starts, stops, strict=True)
            ],
            "acceptance_is_calibrated_correctness_probability": False,
        },
    )


def seed_interpolation(n_traces, seeds, *, extrapolate_tails=True):
    """Radar-free control. Tails hold the nearest seed; never a measured path."""
    if not seeds:
        return np.full(n_traces, -1.0)
    items = sorted(seeds.items())
    rows, samples = np.asarray(items, dtype=float).T
    if np.any(rows != rows.astype(int)) or rows[0] < 0 or rows[-1] >= n_traces:
        raise ValueError("Interpolation seeds must be exact retained native rows")
    if np.any(~np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("Interpolation seeds must have valid native samples")
    values = np.interp(np.arange(n_traces), rows, samples)
    if not extrapolate_tails:
        values[(np.arange(n_traces) < rows[0]) | (np.arange(n_traces) > rows[-1])] = -1
    return values
