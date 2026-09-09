"""Exact distinct reflector alternatives on the observation DAG."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .path_objective import local_score, transition_penalty


def matching_candidates(table, reference, tolerance):
    """Timing agreement must also preserve the selected signed lobe."""
    samples = table.samples
    same = (reference[:, None] >= 0) & (samples >= 0)
    same &= abs(samples - reference[:, None]) <= tolerance
    lobe_ids = getattr(table, "component_maps", {}).get("hybrid_lobe_id")
    rr = np.arange(len(samples))[:, None]
    if lobe_ids is not None:
        candidate_ids = lobe_ids[rr, np.clip(samples, 0, lobe_ids.shape[1] - 1)]
        reference_ids = lobe_ids[
            np.arange(len(samples)), np.clip(reference, 0, lobe_ids.shape[1] - 1)
        ]
        same &= candidate_ids == reference_ids[:, None]
    elif hasattr(table, "polarities"):
        locations = samples == reference[:, None]
        index = np.argmax(locations, axis=1)
        polarity = table.polarities[np.arange(len(samples)), index]
        same &= table.polarities == polarity[:, None]
    return same


def distinct_complete_paths(
    segments, edges, table, scores, anchors, pulse_width, step, config, *, cancel=None
):
    """Select up to three best paths with distinct observed reflector identities.

    Each new solve carries a bit for disagreement with each retained path. All
    endpoints are mandatory. This prevents close timing variants of one lobe
    from exhausting a three-history beam before a distinct reflector can reach
    the endpoint. Gaps remain explicit; gap-only variants compete in max-marginal
    acceptance rather than consuming all reflector-hypothesis slots.
    """
    usable = [
        s
        for s in segments
        if not any(
            row in anchors and table.samples[row, col] != anchors[row] for row, col in s.nodes
        )
    ]
    ordered = sorted(usable, key=lambda s: (s.start, s.stop, s.index))
    by_index = {s.index: s for s in ordered}
    incoming = defaultdict(list)
    for (a, b), quality in edges.items():
        if a not in by_index or b not in by_index:
            continue
        if any(by_index[a].stop < row < by_index[b].start for row in anchors):
            continue
        penalty = transition_penalty(
            by_index[a],
            by_index[b],
            quality,
            step,
            config.transition_cost_per_m,
            config.gap_cost_per_m,
        )
        incoming[b].append((a, penalty))
    local = {s.index: local_score(s, scores, step, config.transition_cost_per_m) for s in ordered}
    result = []
    while len(result) < 3:
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        required_mask = (1 << len(result)) - 1
        matches = [matching_candidates(table, path, max(2, pulse_width / 4)) for path in result]
        histories = {}
        for position, segment in enumerate(ordered):
            if cancel and position % 128 == 0 and cancel():
                raise InterruptedError("Analysis cancelled")
            mask = 0
            for rank, (reference, matching) in enumerate(zip(result, matches, strict=True)):
                if any(
                    reference[row] >= 0 and not matching[row, col] for row, col in segment.nodes
                ):
                    mask |= 1 << rank
            seed_count = len(segment.seed_rows)
            states = {mask: (seed_count, local[segment.index], None)}
            for previous, penalty in incoming[segment.index]:
                for previous_mask, (count, total, _) in histories[previous].items():
                    combined = mask | previous_mask
                    candidate = (count + seed_count, total + local[segment.index] - penalty)
                    if combined not in states or candidate > states[combined][:2]:
                        states[combined] = (*candidate, (previous, previous_mask))
            histories[segment.index] = states
        complete = [
            (states[required_mask][1], index)
            for index, states in histories.items()
            if required_mask in states and states[required_mask][0] == len(anchors)
        ]
        if not complete or not anchors:
            break
        _, index = max(complete)
        state = required_mask
        proposed = np.full(len(table.samples), -1, np.int32)
        while index is not None:
            for row, col in by_index[index].nodes:
                proposed[row] = table.samples[row, col]
            parent = histories[index][state][2]
            if parent is None:
                break
            index, state = parent
        result.append(proposed)
    return result or [np.full(len(table.samples), -1, np.int32)]
