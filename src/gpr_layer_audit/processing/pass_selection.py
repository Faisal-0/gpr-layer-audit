"""Joint reconciliation of independently computed signal/design hypotheses."""

from __future__ import annotations

from itertools import product

import numpy as np


def select_joint_passes(
    samples: np.ndarray,
    canonical: np.ndarray,
    lobe_codes: np.ndarray,
    support: np.ndarray,
    preferred: np.ndarray,
    minimum_gaps: list[int],
    pulse_width: float,
    horizontal_step_m: float,
    break_rows: set[int],
    cancel=None,
    *,
    required_samples: dict[int, dict[int, int]] | None = None,
) -> np.ndarray:
    """Return row/layer choices: 0 signal, 1 design, 2 explicit no-pick.

    Reconciliation cannot combine two valid joint paths into crossed layers.
    Switching between distant families or opposite lobes requires an explicit
    break or gap, never an unnoticed per-layer boolean selection. Support is
    unchanged by this selector; preferences do not become radar confidence.
    """
    rows, layers, branches = samples.shape
    if branches != 2 or canonical.shape != samples.shape or lobe_codes.shape != samples.shape:
        raise ValueError("Expected matching row/layer/two-branch observations")
    if rows == 0:
        return np.empty((0, layers), dtype=np.int8)
    choices = np.asarray(list(product(range(3), repeat=layers)), dtype=np.int8)
    axis = np.arange(layers)
    observed = np.concatenate((samples, np.full((rows, layers, 1), -1)), axis=2)
    times = np.concatenate((canonical, np.full((rows, layers, 1), -1)), axis=2)
    lobes = np.concatenate((lobe_codes, np.zeros((rows, layers, 1))), axis=2)
    scores = support + 0.08 * (np.arange(2)[None, None, :] == preferred[:, :, None])
    scores = np.where(samples >= 0, scores, -np.inf)
    scores = np.concatenate((scores, np.full((rows, layers, 1), -0.35)), axis=2)
    back = np.zeros((rows, len(choices)), dtype=np.int16)
    previous = np.zeros(len(choices))
    old_sample = np.full((len(choices), layers), -1.0)
    old_time = old_sample.copy()
    old_lobe = np.zeros_like(old_sample)
    dx = max(horizontal_step_m, 1e-3)
    for row in range(rows):
        if cancel and row % 128 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        selected = observed[row, axis[None, :], choices]
        selected_time = times[row, axis[None, :], choices]
        selected_lobe = lobes[row, axis[None, :], choices]
        emission = np.sum(scores[row, axis[None, :], choices], axis=1)
        for layer_index, anchors in (required_samples or {}).items():
            if row in anchors:
                emission[selected[:, layer_index] != anchors[row]] = -np.inf
        # Compare every visible pair, including across an independently missing
        # intermediate interface. Missing one layer never permits another to cross.
        for upper in range(layers):
            for lower in range(upper + 1, layers):
                both = (selected[:, upper] >= 0) & (selected[:, lower] >= 0)
                crossed = selected[:, lower] < selected[:, upper] + minimum_gaps[lower]
                crossed |= selected_time[:, lower] < selected_time[:, upper] + minimum_gaps[lower]
                emission[both & crossed] = -np.inf
        if row:
            transition = np.zeros((len(choices), len(choices)))
            structural = row in break_rows
            for layer in range(layers):
                both = (old_sample[:, layer, None] >= 0) & (selected[None, :, layer] >= 0)
                delta = np.abs(old_sample[:, layer, None] - selected[None, :, layer])
                time_delta = np.abs(old_time[:, layer, None] - selected_time[None, :, layer])
                changed = choices[:, layer, None] != choices[None, :, layer]
                flipped = old_lobe[:, layer, None] != selected_lobe[None, :, layer]
                flipped &= (old_lobe[:, layer, None] > 0) & (selected_lobe[None, :, layer] > 0)
                if not structural:
                    impossible = (
                        changed
                        & both
                        & ((delta > pulse_width) | (time_delta > pulse_width) | flipped)
                    )
                    transition[impossible] = -np.inf
                transition -= np.where(
                    both, (0.003 if structural else 0.018) * delta / dx + 0.08 * changed, 0
                )
                missing_switch = (old_sample[:, layer, None] >= 0) ^ (selected[None, :, layer] >= 0)
                transition -= 0.15 * missing_switch
            values = previous[:, None] + transition
            back[row] = np.argmax(values, axis=0)
            previous = np.max(values, axis=0) + emission
        else:
            previous = emission
        if not np.any(np.isfinite(previous)):
            raise ValueError(f"No feasible joint pass at row {row}; check confirmed seeds.")
        previous -= np.max(previous)
        old_sample, old_time, old_lobe = selected, selected_time, selected_lobe
    result = np.empty((rows, layers), dtype=np.int8)
    state = int(np.argmax(previous))
    for row in range(rows - 1, -1, -1):
        result[row] = choices[state]
        state = int(back[row, state])
    return result
