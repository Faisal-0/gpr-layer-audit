"""One bounded template update from unambiguous seed-supported path segments."""

from __future__ import annotations

import numpy as np


def adaptive_template_scores(
    table,
    selected,
    correspondence,
    path_margin,
    observable,
    anchors,
    breaks,
    step,
    config,
    *,
    cancel=None,
    allowed_rows=None,
):
    """Return a local score supplement and auditable, separate seed/regime templates.

    This function has no reference/evaluation-pick input. Original seed waveforms
    are immutable. Updates use only the first inference pass, never their own
    proposals. They cannot create candidates, graph links or measurements.
    """
    rows, count = table.samples.shape
    bonus = np.zeros((rows, count), np.float32)
    records = []
    if not anchors:
        return bonus, records
    region = np.searchsorted(sorted(breaks), np.arange(rows), side="right")
    seed_rows = np.array(sorted(anchors), dtype=int)
    distance = abs(np.arange(rows)[:, None] - seed_rows[None, :]) * step
    distance = np.where(region[:, None] == region[seed_rows][None, :], distance, np.inf)
    owner = seed_rows[np.argmin(distance, axis=1)]
    owner[~np.isfinite(distance).any(axis=1)] = -1
    columns = np.full(rows, -1, dtype=int)
    for row, sample in enumerate(selected):
        found = (
            np.flatnonzero(table.valid[row] & (table.samples[row] == sample)) if sample >= 0 else []
        )
        if len(found):
            columns[row] = found[0]
    eligible = (columns >= 0) & (path_margin >= config.template_update_path_margin)
    if allowed_rows is not None:
        eligible &= allowed_rows
    rr = np.flatnonzero(eligible)
    eligible[rr] &= (correspondence[rr, selected[rr]] >= config.template_update_support) & (
        observable[rr, selected[rr]] >= config.minimum_measurement_support
    )
    radius_m = max(config.matching_distances_m)
    for seed in seed_rows:
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        indices = np.flatnonzero(table.valid[seed] & (table.samples[seed] == anchors[seed]))
        if not len(indices):
            continue
        seed_col = int(indices[0])
        reference = table.waveforms[seed, seed_col]
        polarity = table.polarities[seed, seed_col]
        candidates = np.flatnonzero(eligible & (owner == seed))
        if not len(candidates):
            continue
        identity = table.waveforms[candidates, columns[candidates]] @ reference
        candidates = candidates[
            (identity >= config.template_minimum_seed_similarity)
            & (table.polarities[candidates, columns[candidates]] == polarity)
        ]
        groups = np.split(
            candidates,
            np.flatnonzero(np.diff(candidates) * step > config.template_update_gap_m) + 1,
        )
        for group in groups:
            # An original click may anchor an update, but repeated copies of a
            # single observation cannot create a supported waveform regime.
            if len(group) < config.template_update_min_rows or not np.any(group != seed):
                continue
            waveform = np.median(table.waveforms[group, columns[group]], axis=0)
            waveform = waveform - waveform.mean()
            norm = np.linalg.norm(waveform)
            if norm <= 1e-7:
                continue
            waveform = waveform / norm
            agreement = float(waveform @ reference)
            if agreement < config.template_minimum_seed_similarity:
                continue
            at_rows = np.flatnonzero(
                (owner == seed)
                & (
                    np.maximum(group[0] - np.arange(rows), np.arange(rows) - group[-1]) * step
                    <= radius_m
                )
            )
            local_distance = (
                np.maximum(0, np.maximum(group[0] - at_rows, at_rows - group[-1])) * step
            )
            original = table.waveforms[at_rows] @ reference
            adapted = table.waveforms[at_rows] @ waveform
            quality = np.maximum(0, adapted - original)
            quality *= np.exp(-local_distance[:, None] / radius_m)
            cols = np.maximum(table.samples[at_rows], 0)
            compatible = table.valid[at_rows] & (table.samples[at_rows] >= 0)
            compatible &= table.polarities[at_rows] == polarity
            compatible &= correspondence[at_rows[:, None], cols] > 0
            quality = np.where(compatible, quality * config.template_adaptation_weight, 0)
            bonus[at_rows] = np.maximum(bonus[at_rows], quality)
            records.append(
                {
                    "seed_row": int(seed),
                    "start_row": int(group[0]),
                    "stop_row": int(group[-1]),
                    "source_rows": group.tolist(),
                    "source_samples": selected[group].tolist(),
                    "reference_agreement": agreement,
                    "reference_waveform": reference.tolist(),
                    "waveform": waveform.tolist(),
                }
            )
    return bonus, records
