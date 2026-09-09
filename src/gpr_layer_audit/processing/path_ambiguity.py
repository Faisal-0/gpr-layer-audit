"""Exact seeded DAG max-marginals for reflector ambiguity diagnostics.

Compare complete correspondence routes, not the brightness of unrelated pixels.
Scores are not calibrated error probabilities. The old local candidate margin
remains separately available for audit.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def component_path_margins(
    segments, edges, table, scores, anchors, selected, pulse_width, *, step=1.0, config=None
):
    """Return per-row route margin and the best distinct-lobe competitor.

    Only paths satisfying every seed in this connected component can justify a
    positive margin. If no such path exists, identity stays unresolved. Paths
    missing a row also compete, so a forced proposal cannot gain confidence
    merely by being the sole retained lobe at that row.
    """
    count = len(selected)
    margin = np.zeros(count, dtype=float)
    alternate = np.full(count, -1, dtype=np.int32)
    by_index = {segment.index: segment for segment in segments}
    required = set().union(*(segment.seed_rows for segment in segments))
    if not required:
        return margin, alternate
    ordered = sorted(segments, key=lambda segment: (segment.start, segment.stop, segment.index))
    from .path_objective import local_score, transition_penalty

    local = {
        segment.index: local_score(
            segment, scores, step, config.transition_cost_per_m if config else 1.0
        )
        for segment in segments
    }
    incoming, outgoing = defaultdict(list), defaultdict(list)
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
            config.transition_cost_per_m if config else 1.0,
            config.gap_cost_per_m if config else 0.1,
        )
        if config is None:
            penalty = 1 - quality + 0.1 * (by_index[b].start - by_index[a].stop - 1)
        incoming[b].append((a, penalty))
        outgoing[a].append((b, penalty))
    forward, backward, forward_parent, backward_parent = {}, {}, {}, {}
    for order, neighbours, values, parents in (
        (ordered, incoming, forward, forward_parent),
        (reversed(ordered), outgoing, backward, backward_parent),
    ):
        for segment in order:
            index = segment.index
            seed_count = len(segment.seed_rows)
            choices = [(seed_count, local[index])]
            predecessors = [None]
            for neighbour, penalty in neighbours[index]:
                previous_count, previous_score = values[neighbour]
                choices.append(
                    (previous_count + seed_count, previous_score + local[index] - penalty)
                )
                predecessors.append(neighbour)
            winner = max(range(len(choices)), key=lambda i: choices[i])
            values[index] = choices[winner]
            parents[index] = predecessors[winner]
    optimum_count, optimum = max(forward.values())
    if optimum_count != len(required):
        return margin, alternate
    # Compare total route scores per horizontal observation in the component.
    # This conservative normalization avoids amplifying tiny floating-point ties.
    first_row = min(s.start for s in segments)
    last_row = max(segment.stop for segment in segments)
    span = last_row - first_row + 1
    selected_score = np.full(count, -np.inf)
    competitor_score = np.full(count, -np.inf)
    competitor_route = np.full(count, -1, dtype=int)
    routes = []
    tolerance = max(2, pulse_width / 4)
    selected_polarity = np.zeros(count)
    lobe_matches = None
    if config and config.distinct_path_inference:
        from .diverse_paths import matching_candidates

        lobe_matches = matching_candidates(table, selected, tolerance)
    if hasattr(table, "polarities"):
        for row, sample in enumerate(selected):
            indices = np.flatnonzero(table.samples[row] == sample) if sample >= 0 else []
            if len(indices):
                selected_polarity[row] = table.polarities[row, indices[0]]
    for segment in segments:
        index = segment.index
        seeds = forward[index][0] + backward[index][0] - len(segment.seed_rows)
        if seeds != optimum_count:
            continue
        total = forward[index][1] + backward[index][1] - local[index]
        route_id = len(routes)
        routes.append((index, index))
        for row, col in segment.nodes:
            sample = int(table.samples[row, col])
            same_lobe = not hasattr(table, "polarities") or (
                table.polarities[row, col] == selected_polarity[row]
            )
            if lobe_matches is not None:
                same_lobe = bool(lobe_matches[row, col])
            if selected[row] >= 0 and abs(sample - selected[row]) <= tolerance and same_lobe:
                selected_score[row] = max(selected_score[row], total)
            elif total > competitor_score[row]:
                competitor_score[row] = total
                alternate[row] = sample
                competitor_route[row] = route_id
    # Explicit skip routes compete with measured paths in the skipped interval.
    for a, transitions in outgoing.items():
        for b, penalty in transitions:
            if forward[a][0] + backward[b][0] != optimum_count:
                continue
            total = forward[a][1] + backward[b][1] - penalty
            route_id = len(routes)
            routes.append((a, b))
            start, stop = by_index[a].stop + 1, by_index[b].start
            better = total >= competitor_score[start:stop] - 1e-8
            competitor_score[start:stop][better] = total
            alternate[start:stop][better] = -1
            competitor_route[start:stop][better] = route_id
    if config is not None:
        # A route may begin or end without measurements outside its endpoint
        # seeds. Those missing prefixes/suffixes also compete with measured tails.
        for segment in segments:
            index = segment.index
            for values, start, stop, route in (
                (backward, first_row, segment.start, (None, index)),
                (forward, segment.stop + 1, last_row + 1, (index, None)),
            ):
                if values[index][0] != optimum_count or start >= stop:
                    continue
                total = values[index][1]
                better = total > competitor_score[start:stop]
                competitor_score[start:stop][better] = total
                alternate[start:stop][better] = -1
                competitor_route[start:stop][better] = len(routes)
                routes.append(route)
    # A selected event must itself lie on a globally best feasible seeded route.
    supported = (selected >= 0) & (selected_score >= optimum - 1e-6)
    denominator = np.full(count, span * step, dtype=float)
    if config is not None:
        # Recover each winning feasible competitor and measure its own interval
        # of disagreement. Unrelated skip alternatives elsewhere do not enlarge it.
        for route_id in np.unique(competitor_route[competitor_route >= 0]):
            left, right = routes[route_id]
            sequence = []
            while left is not None:
                sequence.append(left)
                left = forward_parent[left]
            while right is not None:
                sequence.append(right)
                right = backward_parent[right]
            rival = np.full(count, -1, dtype=int)
            rival_polarity = np.zeros(count)
            for index in sequence:
                for row, col in by_index[index].nodes:
                    rival[row] = table.samples[row, col]
                    if hasattr(table, "polarities"):
                        rival_polarity[row] = table.polarities[row, col]
            different = ((rival < 0) != (selected < 0)) | (abs(rival - selected) > tolerance)
            different |= (rival >= 0) & (selected >= 0) & (rival_polarity != selected_polarity)
            if lobe_matches is not None:
                for index in sequence:
                    for row, col in by_index[index].nodes:
                        if selected[row] >= 0 and not lobe_matches[row, col]:
                            different[row] = True
            contested = np.flatnonzero(different)
            for group in np.split(contested, np.flatnonzero(np.diff(contested) > 1) + 1):
                use = group[competitor_route[group] == route_id]
                denominator[use] = max(step, len(group) * step)
    gap = np.maximum(0, (optimum - competitor_score) / denominator)
    margin[supported] = np.minimum(1, gap[supported])
    return margin, alternate


def suggest_path_observations(
    hypotheses, measurement_support, anchors, step, pulse_width, *, limit=3
):
    """Rank observable route disagreements for an additional layer-specific click.

    Uses radar and graph proposals only. The contested span is a prioritization
    heuristic, not a claim that one click will resolve that entire distance.
    """
    if len(hypotheses) < 2 or limit <= 0:
        return []
    rows = len(hypotheses[0])
    row_axis = np.arange(rows)
    separation = np.zeros(rows)
    observability = np.zeros(rows)
    samples_by_row = [set() for _ in range(rows)]
    for index, left in enumerate(hypotheses):
        for right in hypotheses[index + 1 :]:
            valid = (left >= 0) & (right >= 0)
            valid &= abs(left - right) > max(2, pulse_width / 4)
            left_support = measurement_support[row_axis, np.maximum(left, 0)]
            right_support = measurement_support[row_axis, np.maximum(right, 0)]
            strength = np.minimum(left_support, right_support)
            valid &= strength >= 0.015
            separation[valid] = np.maximum(separation[valid], abs(left[valid] - right[valid]))
            observability[valid] = np.maximum(observability[valid], strength[valid])
            for row in np.flatnonzero(valid):
                samples_by_row[row].update((int(left[row]), int(right[row])))
    disagreements = np.flatnonzero(separation > 0)
    if not len(disagreements):
        return []
    # Keep short candidate holes inside an otherwise contested reflector span.
    groups = np.split(disagreements, np.flatnonzero(np.diff(disagreements) * step > 2) + 1)
    proposals = []
    for group in groups:
        distance = np.full(len(group), np.inf)
        if anchors:
            distance = np.min(abs(group[:, None] - np.array(list(anchors))[None, :]), axis=1) * step
        available = distance > 0
        if not np.any(available):
            continue
        candidates = group[available]
        # Prefer a representative interior observation with separated visible lobes.
        centre = (group[0] + group[-1]) / 2
        centrality = 1 - abs(candidates - centre) / max(1, len(group))
        quality = np.minimum(separation[candidates] / max(1, pulse_width), 3)
        quality *= np.sqrt(observability[candidates]) * np.maximum(0.1, centrality)
        row = int(candidates[np.argmax(quality)])
        proposals.append(
            {
                "row": row,
                "candidate_samples": sorted(samples_by_row[row]),
                "contested_start_row": int(group[0]),
                "contested_stop_row": int(group[-1]),
                "contested_observations": len(group),
                "contested_span_m": float((group[-1] - group[0] + 1) * step),
                "priority": float(len(group) * step * np.max(quality)),
            }
        )
    selected = []
    for proposal in sorted(proposals, key=lambda p: (-p["priority"], p["row"])):
        if all(proposal["row"] != old["row"] for old in selected):
            selected.append(proposal)
        if len(selected) >= limit:
            break
    return selected
