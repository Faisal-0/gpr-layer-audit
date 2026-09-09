"""Sparse seeded reflector-segment graph.

Algorithm references: Bugge et al. (2019), HorizonTracker (MIT), non-local
trace matching; Xiong et al. (2017), ARESELP (MIT), multiscale ridge tracing.
This is an independent 2-D implementation; no upstream code is vendored.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.signal import fftconvolve

from .hybrid_evidence import HybridEvidence


@dataclass(slots=True)
class ReflectorSegment:
    index: int
    nodes: list[tuple[int, int]] = field(default_factory=list)
    seed_rows: set[int] = field(default_factory=set)
    transition_cost: float = 0.0

    @property
    def start(self):
        return self.nodes[0][0]

    @property
    def stop(self):
        return self.nodes[-1][0]


def wavelet_ridges(data: np.ndarray, pulse_width: float) -> np.ndarray:
    """Scale-normalized Mexican-hat responses; original amplitudes are untouched."""
    result = np.zeros_like(data, dtype=np.float32)
    for scale in (0.5, 1.0, 2.0):
        width = max(1.0, pulse_width * scale / 3)
        x = np.arange(-int(np.ceil(4 * width)), int(np.ceil(4 * width)) + 1) / width
        kernel = (1 - x * x) * np.exp(-x * x / 2)
        kernel -= kernel.mean()
        kernel /= max(np.linalg.norm(kernel), 1e-9)
        values = np.abs(fftconvolve(data, kernel[None, :], mode="same", axes=1))
        values /= np.maximum(np.max(values, axis=1, keepdims=True), 1e-9)
        result = np.maximum(result, values)
    return result.astype(np.float32)


def constrained_dtw(left, right, band: int) -> tuple[float, float]:
    """Return normalized waveform agreement and the mapped centre displacement.

    The band limits pathological stretching and the centre displacement exposes
    cycle slips. Callers also compute the reverse alignment independently.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    n, m = len(left), len(right)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    parent = np.zeros((n, m), np.int8)
    for i in range(n):
        for j in range(max(0, i - band), min(m, i + band + 1)):
            previous = (cost[i, j], cost[i, j + 1], cost[i + 1, j])
            k = (
                0
                if previous[0] <= previous[1] and previous[0] <= previous[2]
                else 1
                if previous[1] <= previous[2]
                else 2
            )
            cost[i + 1, j + 1] = (left[i] - right[j]) ** 2 + previous[k]
            parent[i, j] = k
    if not np.isfinite(cost[-1, -1]):
        return 0.0, float("inf")
    i, j = n - 1, m - 1
    centres = []
    while i >= 0 and j >= 0:
        if i == n // 2:
            centres.append(j - m // 2)
        direction = parent[i, j]
        i -= direction != 2
        j -= direction != 1
    return float(np.exp(-cost[-1, -1])), float(np.mean(centres)) if centres else 0.0


@lru_cache(maxsize=4096)
def _cached_dtw(left_bytes, right_bytes, band):
    """Bounded waveform cache; exact content keys cannot reuse stale seed evidence."""
    left = np.frombuffer(left_bytes, dtype=np.float32)
    right = np.frombuffer(right_bytes, dtype=np.float32)
    forward, shift = constrained_dtw(left, right, band)
    backward, reverse_shift = constrained_dtw(right, left, band)
    return min(forward, backward), max(abs(shift), abs(reverse_shift))


def _contract_links(table, anchors, links, local_next, local_previous):
    """Contract only chain interiors, preserving entry/exit points of every route.

    Whole-chain contraction can discard a valid non-local edge when its source
    and destination chains overlap in trace coordinates. Splitting at junctions
    preserves the original observation DAG without manufacturing new edges.
    """
    chains, chain_of = [], {}
    for row in range(len(table.samples)):
        for candidate in np.flatnonzero(table.valid[row] & (table.samples[row] >= 0)):
            node = (row, int(candidate))
            if node in chain_of or node in local_previous:
                continue
            chain = []
            while node is not None:
                chain_of[node] = len(chains)
                chain.append(node)
                node = local_next.get(node)
            chains.append(chain)
    starts, stops = set(), set()
    link_quality = {(a, b): q for a, b, q, gap in links if gap == 1}
    for source, target, _quality, _gap in links:
        if chain_of[source] != chain_of[target]:
            stops.add(source)
            starts.add(target)
    # A contradictory click must not invalidate an entire otherwise useful
    # segment. Seed rows are decision points, including for competing chains.
    for node in chain_of:
        if node[0] in anchors:
            starts.add(node)
            stops.add(node)
    segments, membership = [], {}
    for chain in chains:
        current = None
        for node in chain:
            if current is None or node in starts:
                current = ReflectorSegment(len(segments))
                segments.append(current)
            membership[node] = current.index
            if current.nodes:
                current.transition_cost += 1 - link_quality[current.nodes[-1], node]
            current.nodes.append(node)
            row, col = node
            if anchors.get(row) == table.samples[row, col]:
                current.seed_rows.add(row)
            if node in stops:
                current = None
    return segments, membership


def seed_lobe_supported(measurement, row, sample, pulse_width):
    """Whether a clicked sample has local lobe amplitude usable for propagation."""
    radius = max(2, round(pulse_width / 3))
    neighbourhood = abs(measurement[row, max(0, sample - radius) : sample + radius + 1])
    peak = float(np.max(neighbourhood, initial=0))
    return peak > 0 and abs(measurement[row, sample]) >= 0.25 * peak


def _ranked_lobe_matches(scores, same_lobe, count=3):
    """Keep near-best distinct-lobe alternatives, not duplicate feature timings."""
    remaining = scores.copy()
    selected = np.zeros(scores.shape, dtype=bool)
    best = np.max(scores, axis=1)
    rows = np.arange(len(scores))
    for _ in range(count):
        winner = np.argmax(remaining, axis=1)
        value = remaining[rows, winner]
        valid = np.isfinite(value) & (value >= best - 0.15)
        selected[rows[valid], winner[valid]] = True
        remaining = np.where(same_lobe[winner], -np.inf, remaining)
    return selected


def correspondence_graph(
    table,
    anchors,
    breaks,
    pulse_width,
    step,
    cancel=None,
    *,
    measurement=None,
    use_registration=False,
    config=None,
    diagnostics=None,
    displacement_samples=None,
):
    """Reciprocal sparse links followed by unambiguous local segment contraction."""
    rows, _ = table.samples.shape
    distances = sorted(
        {
            1,
            *(
                max(1, round(x / step))
                for x in (config.matching_distances_m if config else (1, 2, 5, 10, 25))
            ),
        }
    )
    boundary = np.zeros(rows, dtype=int)
    for row in breaks:
        if 0 <= row < rows:
            boundary[row] = 1
    region = np.cumsum(boundary)
    registration = {}
    matching_waveforms = table.waveforms
    if config and config.waveform_context_radius_m > 0 and measurement is not None:
        from .motion_prediction import contextual_waveforms

        matching_waveforms = contextual_waveforms(
            table, measurement, table.component_maps, pulse_width, step,
            config.waveform_context_radius_m,
            valid=table.component_maps.get("sample_validity"), breaks=breaks, cancel=cancel,
            minimum_margin=config.minimum_motion_margin,
        )
    motion_predictions = {}
    if config and config.integrated_motion and "motion_forward_shift" in table.component_maps:
        from .motion_prediction import compose_motion

        motion_predictions = compose_motion(
            table.component_maps, [d for d in distances if d < rows],
            valid=table.component_maps.get("sample_validity"), cancel=cancel,
            minimum_margin=config.minimum_motion_margin,
        )
    if measurement is not None and use_registration:
        from .trace_registration import sparse_registration

        registration = sparse_registration(
            measurement, table.samples, distances, region, pulse_width, step, cancel,
            valid=table.component_maps.get("sample_validity"),
        )
    admissible = table.valid & (table.samples >= 0)
    # A sequence of individually small cycle slips must not walk arbitrarily
    # far from a nearby authoritative seed. Apply the same displacement budget
    # as a direct non-local match, over its 25 m neighbourhood. This is a
    # seed-relative correspondence constraint, not a design-thickness prior.
    # Separate structural regions impose separate constraints.
    nearest_distance = np.full(rows, np.inf)
    nearest_sample = np.zeros(rows)
    for seed_row, seed_sample in sorted(anchors.items()):
        distance = abs(np.arange(rows) - seed_row) * step
        closer = (distance < nearest_distance) & (region == region[seed_row])
        nearest_distance[closer] = distance[closer]
        nearest_sample[closer] = seed_sample
    nearby = (nearest_distance <= 25) & (not (config and config.directed_correspondence))
    allowance = pulse_width * (1 + np.sqrt(nearest_distance[nearby]) / 2)
    admissible[nearby] &= (
        abs(table.samples[nearby] - nearest_sample[nearby, None]) <= allowance[:, None]
    )
    # Contradictory observations remain exact; ordinary displacement and seed
    # barriers below still prohibit a forced connection between their lobes.
    for seed_row, seed_sample in anchors.items():
        admissible[seed_row] = table.valid[seed_row] & (table.samples[seed_row] == seed_sample)
    links = []
    local_next, local_previous = {}, {}
    tolerance = max(2.0, pulse_width / 4)
    # Several feature branches can propose neighbouring samples on one signed
    # lobe. They are timing alternatives, not evidence of a different reflector.
    # Keep every candidate (and the exact click), but compare correspondence
    # margins against distinct lobes. A zero/sign crossing always separates
    # lobes, even when both endpoints have the same polarity.
    lobe_ids = None
    if measurement is not None:
        signs = np.sign(measurement)
        starts = np.ones(measurement.shape, dtype=bool)
        starts[:, 1:] = (signs[:, 1:] != signs[:, :-1]) | (signs[:, 1:] == 0)
        lobe_ids = np.cumsum(starts, axis=1)
        if config and config.distinct_path_inference:
            table.component_maps["hybrid_lobe_id"] = lobe_ids.astype(np.int32)
    dtw_cache = {}
    for right in range(1, rows):
        if cancel and right % 16 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        right_valid = admissible[right].copy()
        if right in anchors:
            right_valid &= table.samples[right] == anchors[right]
        js = np.flatnonzero(right_valid)
        if not len(js):
            continue
        for gap in distances:
            left = right - gap
            if left < 0 or region[left] != region[right]:
                continue
            # A correspondence route cannot bypass an authoritative observation
            # and then spread its support through a contradictory lobe.
            if any(left < row < right for row in anchors):
                continue
            left_valid = admissible[left].copy()
            if left in anchors:
                left_valid &= table.samples[left] == anchors[left]
            ii = np.flatnonzero(left_valid)
            if not len(ii):
                continue
            similarity = matching_waveforms[left, ii] @ matching_waveforms[right, js].T
            displacement = table.samples[right, js][None, :] - table.samples[left, ii][:, None]
            # A bounded drift allowance, never a corridor based on design depth.
            predicted = 0.0
            if config and config.directed_correspondence:
                motion = motion_predictions.get(gap, table.component_maps)
                if "motion_forward_shift" in motion:
                    at = np.clip(
                        table.samples[left, ii], 0, motion["motion_forward_shift"].shape[1] - 1
                    )
                    predicted = motion["motion_forward_shift"][left, at][:, None]
                    predicted = predicted * (1 if gap in motion_predictions else gap)
                    reliable = motion["motion_forward_support"][left, at][:, None] >= 0.65
                    if gap not in motion_predictions and config.minimum_motion_margin > 0:
                        reliable &= motion["motion_forward_margin"][left, at][:, None] >= (
                            config.minimum_motion_margin
                        )
                    predicted = np.where(reliable, predicted, 0.0)
            budget = displacement_samples if displacement_samples is not None else pulse_width
            drift_radius = budget * (
                1 + np.sqrt(gap * step) / (4 if config and config.directed_correspondence else 2)
            )
            allowed = abs(displacement - predicted) <= drift_radius
            if (
                config
                and config.directed_correspondence
                and "motion_backward_shift" in table.component_maps
            ):
                motion = motion_predictions.get(gap, table.component_maps)
                at = np.clip(
                    table.samples[right, js], 0, motion["motion_backward_shift"].shape[1] - 1
                )
                reverse_prediction = -motion["motion_backward_shift"][right, at][None, :]
                reverse_prediction *= 1 if gap in motion_predictions else gap
                reverse_reliable = motion["motion_backward_support"][right, at][None, :] >= 0.65
                if gap not in motion_predictions and config.minimum_motion_margin > 0:
                    reverse_reliable &= motion["motion_backward_margin"][right, at][None, :] >= (
                        config.minimum_motion_margin
                    )
                reverse_prediction = np.where(reverse_reliable, reverse_prediction, 0.0)
                allowed &= abs(displacement - reverse_prediction) <= drift_radius
            allowed &= table.polarities[left, ii, None] == table.polarities[right, js][None, :]
            phase_distance = abs(
                table.phase_classes[left, ii, None].astype(int)
                - table.phase_classes[right, js][None, :].astype(int)
            )
            allowed &= np.minimum(phase_distance, 8 - phase_distance) <= 1
            if gap in registration:
                origin, mapped, inverse, fit, reverse_fit = registration[gap]
                left_positions = table.samples[left, ii] - origin
                right_positions = table.samples[right, js] - origin
                trusted = (fit[left, left_positions][:, None] >= 0.75) & (
                    reverse_fit[right, right_positions][None, :] >= 0.75
                )
                allowed &= ~trusted | (
                    (
                        abs(
                            table.samples[right, js][None, :]
                            - mapped[left, left_positions][:, None]
                        )
                        <= tolerance
                    )
                    & (
                        abs(
                            table.samples[left, ii][:, None]
                            - inverse[right, right_positions][None, :]
                        )
                        <= tolerance
                    )
                )
            motion = table.component_maps
            if gap == 1 and "motion_forward_shift" in motion:
                left_samples, right_samples = table.samples[left, ii], table.samples[right, js]
                forward = motion["motion_forward_shift"][left, left_samples][:, None]
                backward = motion["motion_backward_shift"][right, right_samples][None, :]
                reliable = (
                    motion["motion_forward_support"][left, left_samples][:, None] >= 0.65
                ) & (motion["motion_backward_support"][right, right_samples][None, :] >= 0.65)
                if config and config.minimum_motion_margin > 0:
                    reliable &= (
                        motion["motion_forward_margin"][left, left_samples][:, None]
                        >= config.minimum_motion_margin
                    ) & (
                        motion["motion_backward_margin"][right, right_samples][None, :]
                        >= config.minimum_motion_margin
                    )
                allowed &= ~reliable | (
                    (abs(displacement - forward) <= tolerance)
                    & (abs(displacement + backward) <= tolerance)
                )
            score = similarity - 0.08 * abs(displacement - predicted) / max(budget, 1)
            score[~allowed] = -np.inf
            left_samples, right_samples = table.samples[left, ii], table.samples[right, js]
            same_left, same_right = np.eye(len(ii), dtype=bool), np.eye(len(js), dtype=bool)
            if lobe_ids is not None:
                for same, at, samples_at in (
                    (same_left, left, left_samples),
                    (same_right, right, right_samples),
                ):
                    lobes = lobe_ids[at, samples_at]
                    same |= (abs(samples_at[:, None] - samples_at[None, :]) <= tolerance) & (
                        lobes[:, None] == lobes[None, :]
                    )
            contenders = _ranked_lobe_matches(score, same_right)
            contenders &= _ranked_lobe_matches(score.T, same_left).T & allowed
            # Run the independent forward/reverse alignments in bounded numerical
            # batches; avoid thousands of Python sample-by-sample DP loops.
            batch_values = {}
            eligible_pairs = contenders & (similarity >= 0.60)
            pair_count = np.count_nonzero(eligible_pairs)
            waveform_length = matching_waveforms.shape[-1]
            use_batch = (
                config is not None
                and pair_count >= (8 if waveform_length >= 45 else 24)
            )
            if use_batch:
                from .waveform_matching import reciprocal_dtw

                aa, bb = np.nonzero(eligible_pairs)
                batch_size = 32 if waveform_length >= 45 else 128
                for offset in range(0, len(aa), batch_size):
                    xa, xb = aa[offset : offset + batch_size], bb[offset : offset + batch_size]
                    agreements, slips = reciprocal_dtw(
                        matching_waveforms[left, ii[xa]],
                        matching_waveforms[right, js[xb]],
                        max(1, int(pulse_width / 4)),
                    )
                    for a0, b0, q0, s0 in zip(xa, xb, agreements, slips, strict=True):
                        batch_values[int(a0), int(b0)] = (q0, s0)
            for a, b in zip(*np.nonzero(contenders), strict=True):
                if similarity[a, b] < 0.60:
                    continue
                distinct_right = np.arange(len(js)) != b
                distinct_left = np.arange(len(ii)) != a
                if lobe_ids is not None:
                    left_samples = table.samples[left, ii]
                    right_samples = table.samples[right, js]
                    distinct_right &= (abs(right_samples - right_samples[b]) > tolerance) | (
                        lobe_ids[right, right_samples] != lobe_ids[right, right_samples[b]]
                    )
                    distinct_left &= (abs(left_samples - left_samples[a]) > tolerance) | (
                        lobe_ids[left, left_samples] != lobe_ids[left, left_samples[a]]
                    )
                alternatives = score[a, distinct_right]
                reverse_alternatives = score[distinct_left, b]
                margin = min(
                    score[a, b] - np.max(alternatives, initial=-np.inf),
                    score[a, b] - np.max(reverse_alternatives, initial=-np.inf),
                )
                previous, current = int(ii[a]), int(js[b])
                key = (left, previous, right, current)
                if use_batch:
                    dtw_cache[key] = batch_values[a, b]
                elif key not in dtw_cache:
                    before = matching_waveforms[left, previous]
                    after = matching_waveforms[right, current]
                    band = max(1, int(pulse_width / 4))
                    dtw_cache[key] = _cached_dtw(
                        np.asarray(before, np.float32).tobytes(),
                        np.asarray(after, np.float32).tobytes(),
                        band,
                    )
                agreement, slip = dtw_cache[key]
                if agreement < 0.75 or slip > tolerance:
                    continue
                quality = min(agreement, float(similarity[a, b]))
                quality *= 1 - min(0.1, max(0, 0.025 - margin))
                source, target = (left, previous), (right, current)
                links.append((source, target, quality, gap))
    if config and config.directed_correspondence:
        from .directed_support import reciprocal_route_filter

        links, rejected = reciprocal_route_filter(links, table.samples, tolerance)
        if diagnostics is not None:
            diagnostics["route_inconsistency_rejections"] = rejected
    outgoing_local, incoming_local = defaultdict(list), defaultdict(list)
    for source, target, _quality, gap in links:
        if gap == 1:
            outgoing_local[source].append(target)
            incoming_local[target].append(source)
    local_next, local_previous = {}, {}
    for source, targets in outgoing_local.items():
        if len(targets) == 1 and len(incoming_local[targets[0]]) == 1:
            local_next[source] = targets[0]
            local_previous[targets[0]] = source
    segments, membership = _contract_links(table, anchors, links, local_next, local_previous)
    edges = {}
    adjacency = defaultdict(list)
    for source, target, quality, _gap in links:
        adjacency[source].append((target, quality))
        adjacency[target].append((source, quality))
        a, b = membership[source], membership[target]
        if a != b and segments[a].stop < segments[b].start:
            edges[a, b] = max(edges.get((a, b), 0), quality)
    # Compute support on observations, not contracted segments. A weak local
    # edge must neither grant perfect support nor lower an entire long segment;
    # a stronger non-local route can bypass it without filling unobserved rows.
    import heapq

    node_support = {}
    queue = []
    for node in membership:
        row, col = node
        if anchors.get(row) == table.samples[row, col]:
            if measurement is not None and not seed_lobe_supported(
                measurement, row, table.samples[row, col], pulse_width
            ):
                # Preserve the click, but do not transfer cancellation/zero-crossing identity.
                continue
            node_support[node] = 1.0
            heapq.heappush(queue, (-1.0, node))
    if config and config.directed_correspondence:
        from .directed_support import directed_seed_support

        observable = {
            r: measurement is None or seed_lobe_supported(measurement, r, s, pulse_width)
            for r, s in anchors.items()
        }
        values, forward, backward = directed_seed_support(
            membership, links, anchors, table.samples, breaks, observable
        )
        node_support = {node: values[node] for node in membership}
        queue = []
        if diagnostics is not None:
            diagnostics.update(
                directed_supported_nodes=int(np.count_nonzero(values)),
                forward_supported_nodes=int(np.count_nonzero(forward)),
                backward_supported_nodes=int(np.count_nonzero(backward)),
            )
            diagnostics["_one_way_support"] = np.maximum(forward, backward)
    while queue:
        negative, source = heapq.heappop(queue)
        if -negative < node_support[source]:
            continue
        for target, quality in adjacency[source]:
            proposed = min(-negative, quality)
            if proposed > node_support.get(target, 0):
                node_support[target] = proposed
                heapq.heappush(queue, (-proposed, target))
    support = np.zeros(len(segments))
    dense = np.zeros_like(table.dense_radar_score)
    for node, index in membership.items():
        row, col = node
        quality = node_support.get(node, 0)
        dense[row, table.samples[row, col]] = quality
        support[index] = max(support[index], quality)
    return segments, edges, support, dense


def _solve_component(
    segments,
    edges,
    support,
    table,
    scores,
    anchors,
    breaks,
    pulse_width,
    *,
    step=1.0,
    config=None,
    require_all=False,
    cancel=None,
):
    """Top three distinct paths on a contracted DAG, preserving explicit holes."""
    if config and config.distinct_path_inference and require_all:
        from .diverse_paths import distinct_complete_paths

        return distinct_complete_paths(
            segments, edges, table, scores, anchors, pulse_width, step, config, cancel=cancel
        )
    incoming = defaultdict(list)
    for (a, b), quality in edges.items():
        incoming[b].append((a, quality))
    histories = {}
    by_index = {s.index: s for s in segments}
    seed_rows = sorted(anchors)
    for segment in sorted(segments, key=lambda s: (s.start, s.stop, s.index)):
        incompatible = any(
            row in anchors and table.samples[row, col] != anchors[row] for row, col in segment.nodes
        )
        if incompatible:
            continue
        from .path_objective import local_score, transition_penalty

        local = local_score(segment, scores, step, config.transition_cost_per_m if config else 1.0)
        seed_count = len(segment.seed_rows)
        choices = [(seed_count, local, (segment.index,))]
        for previous, quality in incoming[segment.index]:
            before = by_index[previous]
            if any(before.stop < r < segment.start for r in seed_rows):
                continue
            for count, total, path in histories.get(previous, []):
                penalty = transition_penalty(
                    before,
                    segment,
                    quality,
                    step,
                    config.transition_cost_per_m if config else 1.0,
                    config.gap_cost_per_m if config else 0.1,
                )
                if config is None:
                    penalty = 1 - quality + 0.1 * (segment.start - before.stop - 1)
                choices.append(
                    (
                        count + seed_count,
                        total + local - penalty,
                        (*path, segment.index),
                    )
                )
        histories[segment.index] = sorted(choices, reverse=True)[:3]
    all_paths = sorted((item for values in histories.values() for item in values), reverse=True)
    rows = len(table.samples)
    result = []
    for count, _score, sequence in all_paths:
        if count == 0 or (require_all and count != len(anchors)):
            continue
        proposed = np.full(rows, -1, np.int32)
        for index in sequence:
            for row, col in by_index[index].nodes:
                proposed[row] = table.samples[row, col]
        # Reject truncated variants of an already retained event family.
        if any(
            np.any((proposed >= 0) & (old >= 0))
            and np.all(
                abs(proposed[(proposed >= 0) & (old >= 0)] - old[(proposed >= 0) & (old >= 0)])
                <= max(2, pulse_width / 4)
            )
            for old in result
        ):
            continue
        result.append(proposed)
        if len(result) == 3:
            break
    if not result:
        result = [np.full(rows, -1, np.int32)]
    return result


def solve_complete_intervals(
    segments, edges, support, table, scores, anchors, breaks, pulse_width, step, config,
    diagnostics, *, cancel=None,
):
    """Solve each bracketing pair with both endpoints mandatory, retaining failed intervals."""
    from .path_ambiguity import component_path_margins

    rows = len(table.samples)
    result = [np.full(rows, -1, np.int32) for _ in range(3)]
    margins, alternate = np.zeros(rows), np.full(rows, -1, np.int32)
    unresolved = []
    boundaries = sorted({0, rows, *breaks})
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        seeds = sorted(r for r in anchors if start <= r < stop)
        if not seeds:
            continue
        intervals = [
            (start, seeds[0]),
            *zip(seeds[:-1], seeds[1:], strict=True),
            (seeds[-1], stop - 1),
        ]
        for left, right in intervals:
            if cancel and cancel():
                raise InterruptedError("Analysis cancelled")
            required = {r: s for r, s in anchors.items() if left <= r <= right}
            subset = [s for s in segments if s.start >= left and s.stop <= right]
            indices = {s.index for s in subset}
            connections = {
                (a, b): q for (a, b), q in edges.items() if a in indices and b in indices
            }
            paths = _solve_component(
                subset,
                connections,
                support,
                table,
                scores,
                required,
                breaks,
                pulse_width,
                step=step,
                config=config,
                require_all=True,
                cancel=cancel,
            )
            feasible = all(paths[0][r] == s for r, s in required.items())
            if not feasible:
                unresolved.append(
                    {
                        "start_row": left,
                        "stop_row": right,
                        "reason": "no_route_honours_both_endpoint_observations",
                    }
                )
                continue
            values, competitors = component_path_margins(
                subset,
                connections,
                table,
                scores,
                required,
                paths[0],
                pulse_width,
                step=step,
                config=config,
            )
            for rank, path in enumerate(result):
                path[left : right + 1] = paths[min(rank, len(paths) - 1)][left : right + 1]
            margins[left : right + 1] = values[left : right + 1]
            alternate[left : right + 1] = competitors[left : right + 1]
    for path in result:
        for row, sample in anchors.items():
            path[row] = sample
    diagnostics.update(
        path_margin=margins, path_alternate=alternate, unresolved_intervals=unresolved
    )
    distinct = []
    for path in result:
        if not any(np.array_equal(path, previous) for previous in distinct):
            distinct.append(path)
    return distinct


def solve_segments(
    segments, edges, support, table, scores, anchors, breaks, pulse_width, *, diagnostics=None
):
    """Solve disconnected seeded components without splicing competing reflectors.

    Overlapping components with contradictory seeds remain missing between clicks.
    Structural boundaries have already removed correspondence edges.
    """
    eligible = {
        s.index
        for s in segments
        if not any(r in anchors and table.samples[r, c] != anchors[r] for r, c in s.nodes)
    }
    adjacency = defaultdict(set)
    for a, b in edges:
        if a in eligible and b in eligible:
            adjacency[a].add(b)
            adjacency[b].add(a)
    pending = set(eligible)
    parts = []
    component_diagnostics = []
    while pending:
        root = min(pending)
        component, queue = set(), [root]
        while queue:
            index = queue.pop()
            if index in component:
                continue
            component.add(index)
            queue.extend(adjacency[index] - component)
        pending -= component
        if not any(segments[i].seed_rows for i in component):
            continue
        # Keep global indices: the component solver indexes predecessor segments.
        subset = [s for s in segments if s.index in component]
        selected_edges = {
            key: value
            for key, value in edges.items()
            if key[0] in component and key[1] in component
        }
        parts.append(
            _solve_component(
                subset, selected_edges, support, table, scores, anchors, breaks, pulse_width
            )
        )
        if diagnostics is not None:
            from .path_ambiguity import component_path_margins

            component_diagnostics.append(
                component_path_margins(
                    subset, selected_edges, table, scores, anchors, parts[-1][0], pulse_width
                )
            )
    rows = len(table.samples)
    result = [
        np.full(rows, -1, np.int32) for _ in range(max((len(part) for part in parts), default=1))
    ]
    for rank, path in enumerate(result):
        conflict = np.zeros(rows, bool)
        for part in parts:
            proposal = part[min(rank, len(part) - 1)]
            conflict |= (path >= 0) & (proposal >= 0) & (path != proposal)
            use = (path < 0) & (proposal >= 0)
            path[use] = proposal[use]
        path[conflict] = -1
        for row, sample in anchors.items():
            path[row] = sample
    if diagnostics is not None:
        margins = np.full(rows, np.inf)
        alternate = np.full(rows, -1, np.int32)
        for (values, alternatives), part in zip(component_diagnostics, parts, strict=True):
            use = (result[0] >= 0) & (result[0] == part[0]) & (values < margins)
            margins[use] = values[use]
            alternate[use] = alternatives[use]
        margins[~np.isfinite(margins)] = 0
        diagnostics.update(path_margin=margins, path_alternate=alternate)
    return result


def pick_hybrid_interfaces(
    radargram,
    surface,
    layers,
    *,
    anchor_samples,
    seed_metadata,
    feature_branches,
    search_corridors,
    design_weight,
    pulse_width_samples,
    break_rows,
    anomaly_mask,
    horizontal_step_m,
    cancel=None,
    evidence: HybridEvidence | None = None,
    config=None,
):
    from .conventional_config import resolve_config, resolve_pulse
    from .conventional_signal import numerical_extension
    from .preprocessing import measurement_packet_support
    from .seed_graph import (
        SeedConditionedPath,
        _candidate_table,
        _component_maps,
        _search_bounds,
        _template_bank,
        pick_seed_conditioned_interfaces,
    )

    config = resolve_config(config)
    data = np.asarray(radargram, np.float32)
    measurement = data if evidence is None else evidence.measurement
    if evidence is not None:
        evidence.validate(data.shape)
    valid = (
        evidence.valid
        if evidence is not None and evidence.valid is not None
        else np.isfinite(measurement)
    )
    if config.signal_validity:
        measurement = numerical_extension(measurement, valid)
        data = numerical_extension(data, valid)
    branches = dict(feature_branches or {})
    branches["measurement_support"] = measurement_packet_support(measurement)
    output = {}
    enabled = sorted((x for x in layers if x.analysis_enabled), key=lambda x: x.order)
    # Keep the established shallow-layer engine. Deep hypotheses cannot move it.
    shallow = [x for x in enabled if x.order not in config.hybrid_layers]
    if shallow:
        output.update(
            pick_seed_conditioned_interfaces(
                data,
                surface,
                shallow,
                anchor_samples=anchor_samples,
                seed_metadata=seed_metadata,
                feature_branches=branches,
                search_corridors={k: v for k, v in search_corridors.items() if k == 1},
                design_weight=design_weight,
                pulse_width_samples=pulse_width_samples,
                break_rows=break_rows,
                anomaly_mask=anomaly_mask,
                max_interpolation_rows=0,
                horizontal_step_m=horizontal_step_m,
                cancel=cancel,
            )
        )
    from .conventional_cache import radar_features

    rows = len(data)
    previous = output.get(1)
    for layer in (x for x in enabled if x.order in config.hybrid_layers):
        upper_orders = [order for order in output if order < layer.order]
        previous = output[max(upper_orders)] if upper_orders else None
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        anchors = anchor_samples.get(layer.order, {})
        identity_metadata = {
            r: dict(m)
            for r, m in (seed_metadata or {}).get(layer.order, {}).items()
            if r in anchors
        }
        pulse = resolve_pulse(
            measurement,
            valid,
            anchors,
            identity_metadata,
            evidence.sample_interval_ns if evidence else 1.0,
            config,
        )
        if evidence is not None:
            evidence.resolved_pulses[layer.order] = pulse.metadata()
        width = pulse.lobe_samples if config.physical_pulse else pulse_width_samples
        context = pulse.context_radius if config.physical_pulse else None
        ridges, motion, feature_cache_hit = radar_features(measurement, valid, width, cancel)
        branches["measurement_support"] = measurement_packet_support(
            measurement,
            # This function resolves its background/packet windows from a lobe
            # width. Passing packet width expands the background threefold and
            # lets a bright upper interface erase observable deeper reflections.
            pulse_width_samples=width,
            lateral_window_traces=max(3, round(config.lateral_context_m / horizontal_step_m)),
            valid=valid,
        )
        prototypes = _template_bank(
            measurement,
            anchors,
            max(6, round(1.5 * width)),
            identity_metadata,
            context_radius=context,
        )
        maps, per_prototype = _component_maps(
            measurement,
            data,
            branches,
            prototypes,
            width,
            [measurement, data],
            sample_validity=valid if config.signal_validity else None,
            lateral_window_traces=max(3, round(config.lateral_context_m / horizontal_step_m)) | 1,
        )
        maps.update(motion)
        maps["hybrid_source_wavelet"] = ridges
        learned = evidence.learned.get(layer.order) if evidence else None
        if learned:
            maps["hybrid_source_ml"] = np.where(
                learned.valid, learned.likelihood * learned.visibility[:, None], 0
            )
        upper_path = previous.samples if previous is not None else None
        lower, centre, upper = _search_bounds(
            rows,
            data.shape[1],
            surface,
            layer,
            None,
            None,
            anchors,
            width,
            horizontal_step_m,
        )
        if upper_path is not None:
            lower = np.maximum(
                lower, np.where(upper_path >= 0, upper_path + layer.min_gap_samples, surface)
            )
        table = _candidate_table(
            measurement,
            data,
            maps,
            lower,
            centre,
            upper,
            anchors,
            anomaly_mask,
            width,
            context_radius=context,
        )
        columns = np.clip(table.samples, 0, data.shape[1] - 1)
        if config.signal_validity:
            from scipy.ndimage import minimum_filter1d

            radius = context if context is not None else max(5, round(1.5 * width))
            context_valid = minimum_filter1d(
                valid.astype(np.uint8), size=2 * radius + 1, axis=1, mode="constant", cval=0
            ).astype(bool)
            table.valid &= context_valid[np.arange(rows)[:, None], columns]
        graph_diagnostics = {}
        segments, edges, support, correspondence = correspondence_graph(
            table,
            anchors,
            break_rows,
            width,
            horizontal_step_m,
            cancel,
            measurement=measurement,
            use_registration=bool(
                config.whole_trace_registration
                or (evidence and evidence.provenance.get("whole_trace_registration"))
            ),
            config=config,
            diagnostics=graph_diagnostics,
            displacement_samples=pulse.displacement_ns / pulse.dt_ns
            if config.physical_pulse
            else None,
        )
        one_way_support = graph_diagnostics.pop(
            "_one_way_support", np.zeros_like(table.samples, float)
        )
        cols = np.clip(table.samples, 0, data.shape[1] - 1)
        rr = np.arange(rows)[:, None]
        seed_scores = maps["signed_seed_correlation"][rr, cols]
        seed_score_diagnostics = {"policy": "strongest_seed_correlation"}
        if config.interval_seed_scoring:
            from .interval_templates import interval_seed_scores

            seed_scores, seed_score_diagnostics = interval_seed_scores(
                table, prototypes, per_prototype, anchors, break_rows, measurement, valid,
                horizontal_step_m, cancel=cancel,
            )
        scores = 0.55 * seed_scores + 0.25 * ridges[rr, cols]
        scores += 0.20 * maps["generic_radar_score"][rr, cols]
        # The margin must compare seeded identity, not just reflection brightness.
        # A disconnected identical-looking ringing train is not a competing seed path.
        scores += 0.55 * correspondence[rr, cols]
        if learned:
            scores += learned.weight * np.where(
                learned.valid[rr, cols],
                learned.likelihood[rr, cols] * learned.visibility[:, None] - 0.5,
                0,
            )
        route_diagnostics = {}
        solver = solve_complete_intervals if config.complete_paths else solve_segments
        solver_options = (
            {"step": horizontal_step_m, "config": config, "cancel": cancel}
            if config.complete_paths else {}
        )
        hypotheses = solver(
            segments,
            edges,
            support,
            table,
            scores,
            anchors,
            break_rows,
            width,
            diagnostics=route_diagnostics,
            **solver_options,
        )
        template_updates = []
        template_update_status = "inactive"
        if config.adapt_seed_templates:
            from .adaptive_templates import adaptive_template_scores

            allowed_updates = np.ones(rows, bool)
            for order, upper_result in output.items():
                if order >= layer.order:
                    continue
                allowed_updates &= (upper_result.samples < 0) | (
                    hypotheses[0] >= upper_result.samples + layer.min_gap_samples
                )
                alternatives = upper_result.provenance.get("hypothesis_samples", [])
                if alternatives:
                    compatible = np.any([
                        (np.asarray(path) < 0)
                        | (hypotheses[0] >= np.asarray(path) + layer.min_gap_samples)
                        for path in alternatives
                    ], axis=0)
                    allowed_updates &= (upper_result.samples >= 0) | compatible
            bonus, proposed_updates = adaptive_template_scores(
                table, hypotheses[0], correspondence, route_diagnostics["path_margin"],
                branches["measurement_support"], anchors, break_rows, horizontal_step_m,
                config, cancel=cancel, allowed_rows=allowed_updates,
            )
            template_update_status = "insufficient_unambiguous_segment_support"
            if proposed_updates and np.any(bonus):
                updated_diagnostics = {}
                updated = solver(
                    segments, edges, support, table, scores + bonus, anchors, break_rows,
                    width, diagnostics=updated_diagnostics, **solver_options,
                )
                # An update is invalid if it contradicts the observations that
                # justified it. Do not iterate or learn from its own new picks.
                source_rows = [r for item in proposed_updates for r in item["source_rows"]]
                if np.array_equal(updated[0][source_rows], hypotheses[0][source_rows]):
                    hypotheses, route_diagnostics = updated, updated_diagnostics
                    scores += bonus
                    template_updates = proposed_updates
                    template_update_status = "one_guarded_pass"
                else:
                    template_update_status = "discarded_source_path_changed"
        calibration = (
            evidence.provenance.get("acceptance_calibration", {})
            .get("layers", {})
            .get(str(layer.order), {})
            if evidence
            else {}
        )
        calibration = {**config.acceptance_by_layer.get(str(layer.order), {}), **calibration}
        minimum_support = float(
            calibration.get("minimum_correspondence", config.minimum_correspondence)
        )
        minimum_margin = float(calibration.get("minimum_path_margin", config.minimum_path_margin))
        if not 0.5 <= minimum_support <= 1 or not 0 < minimum_margin <= 1:
            raise ValueError("Invalid acceptance calibration thresholds")
        proposed = hypotheses[0]
        accepted = np.zeros(rows, bool)
        confidence = np.zeros(rows)
        canonical = np.full(rows, -1.0)
        family = np.full(rows, -1.0)
        membership = {node: s.index for s in segments for node in s.nodes}
        selected_support = np.zeros(rows)
        margin = np.zeros(rows)
        lobe = np.zeros(rows)
        for row, sample in enumerate(proposed):
            if sample < 0:
                continue
            indices = np.flatnonzero(table.valid[row] & (table.samples[row] == sample))
            if not len(indices):
                continue
            col = int(indices[0])
            segment_id = membership.get((row, col), -1)
            quality = float(correspondence[row, sample])
            selected_support[row] = quality
            canonical[row] = table.canonical_samples[row, col]
            lobe[row] = 1 if measurement[row, sample] < 0 else 2
            family[row] = segment_id
            competitors = table.valid[row] & (table.samples[row] >= 0)
            competitors &= abs(table.samples[row] - sample) > max(2, width / 2)
            winner = float(scores[row, col])
            runner = float(np.max(scores[row, competitors], initial=0))
            margin[row] = max(0, winner - runner)
            observable = (
                valid[row, sample]
                and branches["measurement_support"][row, sample]
                >= config.minimum_measurement_support
            )
            accepted[row] = (
                quality >= minimum_support
                and observable
                and route_diagnostics["path_margin"][row] >= minimum_margin
            )
            if not config.path_acceptance:
                accepted[row] = (
                    quality >= minimum_support
                    and observable
                    and margin[row] >= 0.06
                    and route_diagnostics["path_margin"][row] > 1e-8
                )
            confidence[row] = (
                min(quality, 0.50 + route_diagnostics["path_margin"][row]) if accepted[row] else 0
            )
            if row in anchors and valid[row, sample]:
                accepted[row], confidence[row] = True, 1
                canonical[row] = float(
                    (seed_metadata or {})
                    .get(layer.order, {})
                    .get(row, {})
                    .get("canonical_sample_index", canonical[row])
                )
        accepted &= ~anomaly_mask
        authoritative_rows = np.zeros(rows, bool)
        authoritative_rows[list(anchors)] = True
        if upper_path is not None:
            accepted &= (
                authoritative_rows
                | (upper_path < 0)
                | (proposed >= upper_path + layer.min_gap_samples)
            )
        # Keep compatible provisional upper hypotheses. A provisional winner
        # cannot veto a deeper measurement when another retained upper route fits.
        compatible_counts = np.ones(rows, dtype=int)
        if previous is not None:
            upper_hypotheses = previous.provenance.get("hypothesis_samples", [])
            if upper_hypotheses:
                compatible_counts = np.sum(
                    [
                        (np.asarray(h) < 0) | (proposed >= np.asarray(h) + layer.min_gap_samples)
                        for h in upper_hypotheses
                    ],
                    axis=0,
                )
                accepted &= authoritative_rows | (upper_path >= 0) | (compatible_counts > 0)
        samples = np.where(accepted, proposed, -1).astype(np.int32)
        # Uncalibrated support is not advertised as a probability of correctness.
        values = {
            "hybrid_backend": np.ones(rows),
            "hybrid_accepted": accepted.astype(float),
            "hybrid_correspondence": selected_support,
            "hybrid_path_margin": route_diagnostics["path_margin"],
            "hybrid_path_alternate": route_diagnostics["path_alternate"],
            "seed_reachable": selected_support,
            "deep_identity_support": accepted.astype(float),
            "seed_gap_support": accepted.astype(float),
            "canonical_event_sample": canonical,
            "selected_lobe_code": lobe,
            "event_family_index": family,
            "spatial_lineage_index": family,
            "graph_selected_sample": proposed.astype(float),
            "pre_gate_confidence": selected_support,
            "alternative_cycle_margin": margin,
            "candidate_margin": margin,
            "measurement_support": branches["measurement_support"][
                np.arange(rows), np.maximum(proposed, 0)
            ],
        }
        components = dict(table.component_maps)
        components["hybrid_correspondence"] = correspondence
        components["hybrid_wavelet"] = ridges
        for name in (
            "audit_candidate_rank",
            "audit_candidate_score",
            "event_family_index",
            "seed_reachable",
        ):
            components[name] = np.full_like(data, np.nan)
        for row in range(rows):
            indices = np.flatnonzero(table.valid[row] & (table.samples[row] >= 0))
            ranked = indices[np.argsort(-scores[row, indices], kind="stable")]
            for rank, col in enumerate(ranked, 1):
                sample = table.samples[row, col]
                segment = membership.get((row, int(col)), -1)
                components["audit_candidate_rank"][row, sample] = rank
                components["audit_candidate_score"][row, sample] = scores[row, col]
                components["event_family_index"][row, sample] = segment
                components["seed_reachable"][row, sample] = correspondence[row, sample]
        if learned:
            components["hybrid_likelihood"] = learned.likelihood
        from .path_ambiguity import suggest_path_observations

        observation_requests = suggest_path_observations(
            [*hypotheses, route_diagnostics["path_alternate"]],
            branches["measurement_support"],
            anchors,
            horizontal_step_m,
            width,
        )
        # Failed endpoint intervals still need actionable requests. Use observed
        # candidates reached from either endpoint, without promoting them to picks.
        for interval in route_diagnostics.get("unresolved_intervals", []):
            start, stop = interval["start_row"], interval["stop_row"]
            contested = []
            for row in range(start + 1, stop):
                if row in anchors:
                    continue
                cols_at = np.flatnonzero(
                    table.valid[row]
                    & (table.samples[row] >= 0)
                    & (one_way_support[row] >= minimum_support)
                )
                cols_at = [
                    c
                    for c in cols_at
                    if branches["measurement_support"][row, table.samples[row, c]]
                    >= config.minimum_measurement_support
                ]
                if cols_at:
                    contested.append((row, cols_at))
            if contested:
                row, cols_at = min(contested, key=lambda item: abs(item[0] - (start + stop) / 2))
                length = len(contested) * horizontal_step_m
                observation_requests.append(
                    {
                        "row": row,
                        "candidate_samples": sorted({int(table.samples[row, c]) for c in cols_at}),
                        "contested_start_row": start,
                        "contested_stop_row": stop,
                        "contested_span_m": float((stop - start) * horizontal_step_m),
                        "supported_ambiguous_length_m": float(length),
                        "priority": float(length),
                        "reason": (
                            "Endpoint observations have no compatible route; "
                            "add an observation near the supported frontier"
                        ),
                    }
                )
        observation_requests.sort(key=lambda item: (-item["priority"], item["row"]))
        output[layer.order] = SeedConditionedPath(
            samples=samples,
            confidence=confidence,
            feature=table.dense_radar_score,
            alternate_samples=hypotheses[1] if len(hypotheses) > 1 else np.full(rows, -1, np.int32),
            visible=accepted,
            interpolated=np.zeros(rows, bool),
            evidence=values,
            signal_only_samples=samples.copy(),
            design_guided_samples=samples.copy(),
            design_conflict=np.zeros(rows, bool),
            design_constrained=False,
            candidate_components=components,
            provisional_samples=proposed,
            provenance={
                "backend": "seed_hybrid",
                "correspondence_rules": "directed-reciprocal-routes-v2"
                if config.directed_correspondence
                else "legacy-undirected",
                "conventional_config": __import__("dataclasses").asdict(config),
                "configuration_sha256": config.fingerprint,
                "resolved_pulse": pulse.metadata(),
                "seed_identity_scoring": seed_score_diagnostics,
                "coordinate_transforms": evidence.coordinate_transforms if evidence else [],
                "seed_independent_feature_cache_hit": feature_cache_hit,
                "template_adaptation": {
                    "status": template_update_status, "updates": template_updates,
                    "original_seed_templates_preserved": True,
                },
                "correspondence_cache": "rebuilt_from_current_seeds",
                "graph_diagnostics": graph_diagnostics,
                "unresolved_intervals": route_diagnostics.get("unresolved_intervals", []),
                "compatible_upper_hypotheses": compatible_counts.tolist(),
                "sample_validity": {"invalid_samples": int(np.count_nonzero(~valid))},
                "whole_trace_registration": bool(
                    config.whole_trace_registration
                    or (evidence and evidence.provenance.get("whole_trace_registration"))
                ),
                "maximum_reciprocal_lobe_alternatives": 3,
                "seed_displacement_neighbourhood_m": 25.0,
                "suggested_observations": observation_requests,
                "segments": len(segments),
                "edges": len(edges),
                "hypotheses": len(hypotheses),
                "confidence_is_calibrated": False,
                "seed_identity_warnings": [
                    {
                        "row": int(row),
                        "sample": int(sample),
                        "reason": "weak_clicked_lobe_relative_to_neighbour; recapture reflector",
                    }
                    for row, sample in anchors.items()
                    if not valid[row, sample]
                    or not seed_lobe_supported(measurement, row, sample, width)
                ],
                "hypothesis_samples": [path.tolist() for path in hypotheses],
                "acceptance_calibration": calibration,
                "ml": learned.provenance if learned else {"status": "inactive"},
            },
        )
        previous = output[layer.order]
        if feature_branches is not None:
            feature_branches[f"Layer {layer.order} correspondence"] = correspondence
            feature_branches[f"Layer {layer.order} wavelet"] = ridges
            if learned:
                feature_branches[f"Layer {layer.order} ML likelihood"] = learned.likelihood
    for path in output.values():
        selected = np.maximum(path.samples, 0)
        invalid = ~valid[np.arange(rows), selected]
        if np.any(invalid):
            path.samples[invalid] = -1
            path.visible[invalid] = False
            path.confidence[invalid] = 0
    enforce_accepted_order(output, enabled, anchor_samples)
    return output


def enforce_accepted_order(paths, layers, anchors):
    """Check all accepted upper layers, including across an unknown intermediate layer.

    Authoritative deeper observations can invalidate an automatic upper pick;
    their exact coordinates are never shifted to accommodate that automatic pick.
    """
    ordered = sorted((layer for layer in layers if layer.order in paths), key=lambda x: x.order)
    for index, layer in enumerate(ordered):
        lower = paths[layer.order]
        for upper_layer in ordered[:index]:
            upper = paths[upper_layer.order]
            crossed = (
                (upper.samples >= 0)
                & (lower.samples >= 0)
                & (lower.samples < upper.samples + layer.min_gap_samples)
            )
            for row in np.flatnonzero(crossed):
                keep_lower = row in anchors.get(layer.order, {}) and row not in anchors.get(
                    upper_layer.order, {}
                )
                path = upper if keep_lower else lower
                path.provenance.setdefault("ordering_rejected_rows", []).append(int(row))
                path.samples[row] = -1
                path.visible[row] = False
                path.confidence[row] = 0
                for key in ("hybrid_accepted", "deep_identity_support", "seed_gap_support"):
                    if key in path.evidence:
                        path.evidence[key][row] = 0
