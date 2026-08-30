"""Second-order joint event search; no confidence or reference labels enter it."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .seed_graph import _LayerWorkspace


def _stable_top_indices(values: np.ndarray, count: int) -> np.ndarray:
    """Return the same finite stable descending top-k as a full argsort.

    Joint transition tables can contain hundreds of thousands of values per
    radar row while the production beam retains only 128. Partitioning finds
    the cutoff in linear time; explicit cutoff-tie handling preserves the
    original flat-index order exactly, including repeated scores.
    """
    flat = np.asarray(values).ravel()
    finite = np.flatnonzero(np.isfinite(flat))
    if count <= 0 or not len(finite):
        return np.empty(0, dtype=int)
    if len(finite) <= count:
        return finite[np.argsort(-flat[finite], kind="stable")]
    finite_values = flat[finite]
    threshold = np.partition(finite_values, len(finite_values) - count)[
        len(finite_values) - count
    ]
    above = finite[finite_values > threshold]
    above = above[np.argsort(-flat[above], kind="stable")]
    equal = finite[finite_values == threshold]
    return np.concatenate((above, equal[: count - len(above)]))


def _retain_histories(values: np.ndarray, beam_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep destination diversity AND the best distinct predecessor histories.

    A destination-only beam is not a k-best path search: when two reflector
    families meet a confirmed anchor, it erases every alternative prefix. Keep
    a separate path-history budget in addition to the destination budget. The
    beam may therefore contain up to twice ``beam_size`` entries. This is a
    search approximation, not an exhaustive posterior over physical layers.
    """
    best_predecessor = np.argmax(values, axis=0)
    destinations = np.arange(values.shape[1])
    best_values = values[best_predecessor, destinations]
    destination_order = _stable_top_indices(best_values, beam_size)
    # The two histories need not differ in their final event. Their earlier
    # samples, slopes or missing-state memory may still differ materially.
    flat_order = _stable_top_indices(values, beam_size)
    history_back, history_dest = np.unravel_index(flat_order, values.shape)
    pairs = {(int(best_predecessor[d]), int(d)) for d in destination_order}
    pairs.update(zip(history_back.tolist(), history_dest.tolist(), strict=True))
    ranked = sorted(pairs, key=lambda pair: (-values[pair], pair[1], pair[0]))
    if not ranked:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    back, dest = np.asarray(ranked, dtype=int).T
    return back, dest


def joint_family_beam(
    workspaces: list[_LayerWorkspace],
    break_rows: set[int],
    horizontal_step_m: float,
    cancel: Callable[[], bool] | None = None,
    *,
    beam_size: int = 128,
    top_n: int = 32,
) -> tuple[list[dict[int, np.ndarray]], np.ndarray]:
    """Retain joint layer states and their slopes, including latent gap memory.

    Candidate combinations are pruned *after* transition costs. Independent
    whole-layer solutions cannot restrict the reachable joint paths. Separate
    destination and path-history budgets preserve alternatives after merging
    at an anchor, including their distinct curvature and missing-state memory.
    """
    from .seed_graph import _joint_states_at_row, _joint_transition_matrix

    rows = len(workspaces[0].table.samples)
    layer_count = len(workspaces)
    dx = max(horizontal_step_m, 1e-3)
    states, emissions = _joint_states_at_row(workspaces, 0, maximum_states=None)
    # Preserve the original initialization semantics, including placeholder
    # states when fewer than ``beam_size`` emissions are finite. This one-time
    # sort is small; the per-row transition tables are the optimization target.
    initial = np.argsort(-emissions, kind="stable")[:beam_size]
    states = states[initial]
    scores = emissions[initial].astype(float)
    histories = [states]
    backpointers = [np.full(len(states), -1, dtype=int)]
    last_sample = np.column_stack(
        [w.table.samples[0, states[:, i]] for i, w in enumerate(workspaces)]
    ).astype(float)
    last_family = np.column_stack(
        [w.table.family_indices[0, states[:, i]] for i, w in enumerate(workspaces)]
    )
    slopes = np.zeros_like(last_sample)
    gap_length = np.zeros_like(last_sample, dtype=int)

    for row in range(1, rows):
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        proposed, emissions = _joint_states_at_row(workspaces, row, maximum_states=None)
        cost = _joint_transition_matrix(
            workspaces, row, states, proposed, dx, row in break_rows
        ).astype(float)
        new_samples = np.column_stack(
            [w.table.samples[row, proposed[:, i]] for i, w in enumerate(workspaces)]
        ).astype(float)
        new_families = np.column_stack(
            [w.table.family_indices[row, proposed[:, i]] for i, w in enumerate(workspaces)]
        )
        if row not in break_rows:
            for layer_index in range(layer_count):
                prior = last_sample[:, layer_index, None]
                current = new_samples[None, :, layer_index]
                connected = (prior >= 0) & (current >= 0)
                distance = dx * (1 + gap_length[:, layer_index, None])
                slope = (current - prior) / distance
                curvature = np.abs(slope - slopes[:, layer_index, None]) / distance
                cost -= np.where(connected, 0.008 * curvature, 0.0)
                # A null row cannot erase the previous family and offer a
                # cost-free restart on a distant same-polarity ringing cycle.
                resumed = connected & (gap_length[:, layer_index, None] > 0)
                cost -= np.where(resumed, 0.030 * np.abs(slope), 0.0)
                prior_family = last_family[:, layer_index, None]
                current_family = new_families[None, :, layer_index]
                changed = (prior_family >= 0) & (current_family >= 0)
                changed &= prior_family != current_family
                cost -= np.where(resumed & changed, 0.45, 0.0)

        values = scores[:, None] + cost + emissions[None, :]
        back, dest = _retain_histories(values, beam_size)
        if not len(dest):
            raise ValueError(f"No feasible joint event path at row {row}; check seed ordering.")
        scores = values[back, dest]
        # A common offset preserves path comparisons and avoids float32 drift.
        scores -= np.max(scores)
        states = proposed[dest]
        histories.append(states)
        backpointers.append(back)
        current = new_samples[dest]
        prior = last_sample[back]
        distance = dx * (1 + gap_length[back])
        new_slopes = (current - prior) / distance
        slopes = np.where((current >= 0) & (prior >= 0), new_slopes, slopes[back])
        last_sample = np.where(current >= 0, current, prior)
        last_family = np.where(current >= 0, new_families[dest], last_family[back])
        gap_length = np.where(current >= 0, 0, gap_length[back] + 1)
        # Forced missing evidence (including anomalies) terminates the tracklet;
        # ordinary optional no-pick states retain memory above.
        for index, workspace in enumerate(workspaces):
            forced_gap = not np.any(workspace.table.valid[row, :-1])
            if forced_gap or row in break_rows:
                missing = current[:, index] < 0
                last_sample[missing, index] = -1
                last_family[missing, index] = -1
                slopes[:, index] = 0.0

    terminal = np.argsort(-scores, kind="stable")[:top_n]
    hypotheses = []
    for last in terminal:
        indices = np.zeros((rows, layer_count), dtype=int)
        state = int(last)
        for row in range(rows - 1, -1, -1):
            indices[row] = histories[row][state]
            state = int(backpointers[row][state])
        hypotheses.append(
            {
                w.layer.order: w.table.samples[np.arange(rows), indices[:, i]].copy()
                for i, w in enumerate(workspaces)
            }
        )
    return hypotheses, scores[terminal]
