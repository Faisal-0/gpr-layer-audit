"""Exact history-aware inference on retained observations, never interpolated picks.

The state contains the preceding observation and an immutable template-bank mode.
Contracted chain interiors are expanded so their geometry contributes to the same
objective used by selection and competing-route max-marginals. This is an
experimental objective; its costs and margins are not calibrated probabilities.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .diverse_paths import matching_candidates


@dataclass(frozen=True)
class IdentityGeometry:
    """Physical scales and strengths, independent of sample/trace spacing.

    Curvature is a change in slope in ns/m per m. A Huber loss retains finite
    costs for abrupt geometry. Gaps reset geometry history: they are not evidence
    of a slope. Explicit structural breaks must be solved as separate intervals.
    """

    slope_weight: float = 0.0
    slope_scale_ns_per_m: float = 1.0
    curvature_weight: float = 0.0
    curvature_scale_ns_per_m2: float = 1.0
    curvature_quantization_guard: bool = False
    mode_switch_cost: float = 0.0
    maximum_states: int = 1_000_000

    def __post_init__(self):
        for name in ("slope_weight", "curvature_weight", "mode_switch_cost"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("slope_scale_ns_per_m", "curvature_scale_ns_per_m2"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_states < 1:
            raise ValueError("maximum_states must be positive")


@dataclass
class IdentityPathResult:
    samples: np.ndarray
    modes: np.ndarray
    path_margin: np.ndarray
    path_alternate: np.ndarray
    hypotheses: list[np.ndarray]
    objective: float
    diagnostics: dict


def _huber(value):
    value = abs(value)
    return 0.5 * value * value if value <= 1 else value - 0.5


def immutable_seed_modes(table, prototypes, correlations, anchors, measurement, valid):
    """Separate untouched seed-template evidence, with no pointwise identity max.

    Mode changes require an explicit regime/lobe change in supplied seed metadata.
    Ordinary additional observations do not silently create regime boundaries.
    Invalid seed contexts cannot furnish identity evidence.
    """
    if len(prototypes) != len(correlations):
        raise ValueError("Each immutable seed template requires correlation evidence")
    rr = np.arange(len(table.samples))[:, None]
    cc = np.clip(table.samples, 0, measurement.shape[1] - 1)
    observable = table.valid & (table.samples >= 0) & valid[rr, cc]
    modes, metadata = [], []
    for prototype, correlation in zip(prototypes, correlations, strict=True):
        row, sample, radius = (
            int(prototype.chainage_m),
            prototype.sample_index,
            prototype.radius_samples,
        )
        if (
            row != prototype.chainage_m
            or anchors.get(row) != sample
            or sample < radius
            or sample + radius >= measurement.shape[1]
            or not np.all(valid[row, sample - radius : sample + radius + 1])
        ):
            continue
        compatible = np.sign(measurement[rr, cc]) == prototype.polarity
        modes.append(np.where(observable & compatible, np.clip(correlation[rr, cc], 0, 1), 0))
        metadata.append(
            {
                "seed_row": row,
                "seed_sample": sample,
                "regime_id": prototype.regime_id,
                "polarity": prototype.polarity,
                "selected_lobe": prototype.selected_lobe,
            }
        )
    changes = set()
    ordered = sorted(metadata, key=lambda mode: mode["seed_row"])
    for before, after in zip(ordered[:-1], ordered[1:], strict=True):
        if before["regime_id"] != after["regime_id"]:
            changes.add(after["seed_row"])
    values = np.asarray(modes) if modes else np.zeros((1, *table.samples.shape))
    return values, changes, metadata


def _geometry_cost(before, current, after, samples, x, dt, config):
    # A nonadjacent edge carries no measurements in its gap. Do not smooth it.
    if after[0] != current[0] + 1:
        return 0.0
    distance = x[after[0]] - x[current[0]]
    slope = dt * (samples[after] - samples[current]) / distance
    cost = config.slope_weight * distance * _huber(slope / config.slope_scale_ns_per_m)
    if before is not None and current[0] == before[0] + 1:
        previous_distance = x[current[0]] - x[before[0]]
        previous_slope = dt * (samples[current] - samples[before]) / previous_distance
        support = (distance + previous_distance) / 2
        curvature = (slope - previous_slope) / support
        if config.curvature_quantization_guard:
            # Each picked sample represents a quantized time. The worst-case
            # three-point stencil error is dt * (1/dx_left + 1/dx_right).
            uncertainty = dt * (1 / distance + 1 / previous_distance) / support
            curvature = max(0.0, abs(curvature) - uncertainty)
        cost += (
            config.curvature_weight * support * _huber(curvature / config.curvature_scale_ns_per_m2)
        )
    return cost


def solve_identity_component(
    segments,
    edges,
    table,
    scores,
    anchors,
    pulse_width,
    *,
    x_m,
    dt_ns,
    config,
    geometry=None,
    mode_scores=None,
    mode_change_rows=(),
    breaks=(),
    cancel=None,
):
    """Select an exact seeded path and margins under one second-order objective.

    ``scores`` has the immutable candidate-table shape. Optional ``mode_scores``
    has shape (modes, rows, candidates), and supplies additive evidence from each
    immutable template/regime. A mode persists for a whole route, except at
    explicit ``mode_change_rows``. Callers must remove any pointwise maximum
    contribution from ``scores`` before supplying separate mode evidence.

    The solver restores all internal observations but no discarded graph edges.
    States can grow with incoming/outgoing degree; the explicit resource limit
    raises instead of silently pruning competitors and reporting false certainty.
    """
    geometry = geometry or IdentityGeometry()
    x = np.asarray(x_m, dtype=float)
    samples = np.asarray(table.samples)
    rows = len(samples)
    if x.shape != (rows,) or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("x_m must be a finite strictly increasing row coordinate")
    if not np.isfinite(dt_ns) or dt_ns <= 0:
        raise ValueError("dt_ns must be finite and positive")
    if np.shape(scores) != samples.shape or not np.all(np.isfinite(scores)):
        raise ValueError("Candidate scores must be finite and match the candidate table")
    modes = np.zeros((1, *samples.shape)) if mode_scores is None else np.asarray(mode_scores)
    if modes.ndim != 3 or modes.shape[1:] != samples.shape or not np.all(np.isfinite(modes)):
        raise ValueError("mode_scores must be finite (modes, rows, candidates) evidence")
    if not len(modes):
        raise ValueError("At least one identity mode is required")
    empty = np.full(rows, -1, np.int32)
    no_path = IdentityPathResult(
        empty.copy(),
        empty.copy(),
        np.zeros(rows),
        empty.copy(),
        [empty.copy()],
        float("-inf"),
        {"feasible": False, "states": 0, "exact_on_retained_graph": True},
    )
    if not anchors or not segments:
        return no_path
    if any(min(anchors) < row <= max(anchors) for row in breaks):
        raise ValueError("Solve structural-break regions separately")
    # Forward rectangle widths match the existing objective on uniform grids.
    widths = np.r_[np.diff(x), x[-1] - x[-2]] if rows > 1 else np.ones(1)
    valid_segments = {
        s.index: s
        for s in segments
        if not any(row in anchors and samples[row, col] != anchors[row] for row, col in s.nodes)
    }
    starts = {s.nodes[0] for s in valid_segments.values()}
    stops = {s.nodes[-1] for s in valid_segments.values()}
    node_edges = {}
    for segment in valid_segments.values():
        if len(segment.nodes) > 1:
            # Sum of internal correspondence costs is preserved exactly at uniform
            # sampling; geometry uses each actual interior observation separately.
            penalty = config.transition_cost_per_m * segment.transition_cost
            internal_distances = [
                x[b[0]] - x[a[0]]
                for a, b in zip(segment.nodes[:-1], segment.nodes[1:], strict=True)
            ]
            mean_step = sum(internal_distances) / len(internal_distances)
            for a, b in zip(segment.nodes[:-1], segment.nodes[1:], strict=True):
                node_edges[a, b] = penalty * mean_step / (len(segment.nodes) - 1)
    for (a, b), quality in edges.items():
        if a not in valid_segments or b not in valid_segments:
            continue
        source, target = valid_segments[a].nodes[-1], valid_segments[b].nodes[0]
        if source[0] >= target[0]:
            raise ValueError("Correspondence graph must be acyclic in physical row order")
        if any(source[0] < row < target[0] for row in anchors):
            continue
        if any(source[0] < row <= target[0] for row in breaks):
            continue
        distance = x[target[0]] - x[source[0]]
        gap = max(0, distance - widths[source[0]])
        node_edges[source, target] = (
            config.transition_cost_per_m * (1 - quality) * distance + config.gap_cost_per_m * gap
        )
    incoming = defaultdict(list)
    nodes = sorted({node for segment in valid_segments.values() for node in segment.nodes})
    for (a, b), penalty in node_edges.items():
        incoming[b].append((a, penalty))
    by_node = defaultdict(list)
    states, forward, backward, parent, successor, transitions = [], [], [], [], [], []
    changes = set(mode_change_rows)
    first_seed, last_seed = min(anchors), max(anchors)
    for position, node in enumerate(nodes):
        if cancel and position % 64 == 0 and cancel():
            raise InterruptedError("Analysis cancelled")
        state_ids = {}

        def add_state(
            previous, mode, total, predecessor, increment, state_ids=state_ids, node=node
        ):
            key = previous, mode
            if key not in state_ids:
                if len(states) >= geometry.maximum_states:
                    raise RuntimeError(
                        "Exact identity graph exceeds maximum_states; no pruning used"
                    )
                state_ids[key] = len(states)
                states.append((previous, node, mode))
                forward.append(float("-inf"))
                backward.append(float("-inf"))
                parent.append(None)
                successor.append(None)
                transitions.append([])
                by_node[node].append(state_ids[key])
            index = state_ids[key]
            if total > forward[index]:
                forward[index], parent[index] = total, predecessor
            if predecessor is not None:
                transitions[predecessor].append((index, increment))

        if node in starts and node[0] <= first_seed:
            for mode in range(len(modes)):
                value = widths[node[0]] * (scores[node] + modes[mode][node])
                add_state(None, mode, value, None, value)
        for previous, penalty in incoming[node]:
            for previous_id in by_node[previous]:
                before, _, previous_mode = states[previous_id]
                allowed = range(len(modes)) if node[0] in changes else (previous_mode,)
                for mode in allowed:
                    value = widths[node[0]] * (scores[node] + modes[mode][node]) - penalty
                    value -= _geometry_cost(before, previous, node, samples, x, dt_ns, geometry)
                    if mode != previous_mode:
                        value -= geometry.mode_switch_cost
                    add_state(previous, mode, forward[previous_id] + value, previous_id, value)
    ends = [i for i, (_, node, _) in enumerate(states) if node in stops and node[0] >= last_seed]
    if not ends:
        no_path.diagnostics["states"] = len(states)
        return no_path
    end_set = set(ends)
    winner = max(ends, key=lambda i: forward[i])
    optimum = forward[winner]
    for index in ends:
        backward[index] = 0.0
    for index in reversed(range(len(states))):
        for following, value in transitions[index]:
            total = value + backward[following]
            if total > backward[index]:
                backward[index], successor[index] = total, following

    def recover(left, right=None):
        route, route_modes = empty.copy(), empty.copy()
        chain = []
        while left is not None:
            chain.append(left)
            left = parent[left]
        while right is not None:
            chain.append(right)
            right = successor[right]
        for index in chain:
            _, node, mode = states[index]
            route[node[0]], route_modes[node[0]] = samples[node], mode
        return route, route_modes

    selected, selected_modes = recover(winner)
    selected_states, cursor = [], winner
    while cursor is not None:
        selected_states.append(cursor)
        cursor = parent[cursor]
    selected_states.reverse()
    decomposition = dict(
        candidate_evidence=0.0, correspondence_cost=0.0, geometry_cost=0.0, identity_switch_cost=0.0
    )
    for position, index in enumerate(selected_states):
        _, node, mode = states[index]
        decomposition["candidate_evidence"] += float(
            widths[node[0]] * (scores[node] + modes[mode][node])
        )
        if position:
            before, previous, previous_mode = states[selected_states[position - 1]]
            decomposition["correspondence_cost"] += node_edges[previous, node]
            decomposition["geometry_cost"] += _geometry_cost(
                before, previous, node, samples, x, dt_ns, geometry
            )
            if mode != previous_mode:
                decomposition["identity_switch_cost"] += geometry.mode_switch_cost
    same = matching_candidates(table, selected, max(2, pulse_width / 4))
    competitors = np.full(rows, -np.inf)
    competitor_routes = [None] * rows

    def compete(start, stop, value, route):
        if not np.isfinite(value):
            return
        improve = np.flatnonzero(value > competitors[start:stop] + 1e-12) + start
        competitors[improve] = value
        for row in improve:
            competitor_routes[row] = route

    for index, (previous, node, _) in enumerate(states):
        total = forward[index] + backward[index]
        if not same[node]:
            compete(node[0], node[0] + 1, total, (index, successor[index]))
        # An exact source-state prefix / feasible sink-state suffix is missing.
        if previous is None and parent[index] is None:
            compete(nodes[0][0], node[0], total, (index, successor[index]))
        if index in end_set:
            compete(node[0] + 1, nodes[-1][0] + 1, forward[index], (index, None))
        for following, value in transitions[index]:
            target = states[following][1]
            if target[0] > node[0] + 1:
                compete(
                    node[0] + 1,
                    target[0],
                    forward[index] + value + backward[following],
                    (index, following),
                )
    margin, alternate = np.zeros(rows), empty.copy()
    hypotheses, seen, rival_cache = [selected], set(), {}
    for row, description in enumerate(competitor_routes):
        if selected[row] < 0:
            continue
        if description is None:
            margin[row] = 1.0
            continue
        if description not in rival_cache:
            rival, _ = recover(*description)
            same_selected = np.any((samples == rival[:, None]) & same, axis=1)
            different = ((selected < 0) != (rival < 0)) | (
                (selected >= 0) & (rival >= 0) & ~same_selected
            )
            rival_cache[description] = rival, different
        rival, different = rival_cache[description]
        alternate[row] = rival[row]
        left, right = row, row
        while left > 0 and different[left - 1]:
            left -= 1
        while right + 1 < rows and different[right + 1]:
            right += 1
        denominator = max(widths[row], float(np.sum(widths[left : right + 1])))
        margin[row] = np.clip((optimum - competitors[row]) / denominator, 0, 1)
        key = rival.tobytes()
        if len(hypotheses) < 3 and key not in seen and not np.array_equal(rival, selected):
            seen.add(key)
            hypotheses.append(rival)
    return IdentityPathResult(
        selected,
        selected_modes,
        margin,
        alternate,
        hypotheses,
        optimum,
        {
            "feasible": True,
            "states": len(states),
            "observations": len(nodes),
            "retained_edges": len(node_edges),
            "exact_on_retained_graph": True,
            "geometry_reset_at_gaps": True,
            "persistent_modes": len(modes),
            "objective": "identity_geometry_v1",
            "calibrated_probability": False,
            "selected_objective": optimum,
            "selected_objective_terms": decomposition,
        },
    )


def solve_identity_intervals(
    segments,
    edges,
    support,
    table,
    scores,
    anchors,
    breaks,
    pulse_width,
    step,
    config,
    diagnostics,
    *,
    dt_ns,
    geometry=None,
    mode_scores=None,
    mode_change_rows=(),
    cancel=None,
):
    """Drop-in interval solver adapter; physical sample interval is mandatory."""
    rows = len(table.samples)
    result = [np.full(rows, -1, np.int32) for _ in range(3)]
    margins, alternate = np.zeros(rows), np.full(rows, -1, np.int32)
    unresolved, details = [], []
    bounds = sorted({0, rows, *breaks})
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        seeds = sorted(row for row in anchors if start <= row < stop)
        if not seeds:
            continue
        intervals = (
            [(start, stop - 1)]
            if mode_scores is not None
            else [
                (start, seeds[0]),
                *zip(seeds[:-1], seeds[1:], strict=True),
                (seeds[-1], stop - 1),
            ]
        )
        for left, right in intervals:
            required = {row: sample for row, sample in anchors.items() if left <= row <= right}
            subset = [s for s in segments if s.start >= left and s.stop <= right]
            solved = solve_identity_component(
                subset,
                edges,
                table,
                scores,
                required,
                pulse_width,
                x_m=np.arange(rows) * step,
                dt_ns=dt_ns,
                config=config,
                geometry=geometry,
                mode_scores=mode_scores,
                mode_change_rows=mode_change_rows,
                cancel=cancel,
            )
            details.append({"start_row": left, "stop_row": right, **solved.diagnostics})
            if not solved.diagnostics["feasible"]:
                unresolved.append(
                    {
                        "start_row": left,
                        "stop_row": right,
                        "reason": "no_route_honours_both_endpoint_observations",
                    }
                )
                continue
            for rank, destination in enumerate(result):
                path = solved.hypotheses[min(rank, len(solved.hypotheses) - 1)]
                destination[left : right + 1] = path[left : right + 1]
            margins[left : right + 1] = solved.path_margin[left : right + 1]
            alternate[left : right + 1] = solved.path_alternate[left : right + 1]
    for path in result:
        for row, sample in anchors.items():
            path[row] = sample
    diagnostics.update(
        path_margin=margins,
        path_alternate=alternate,
        unresolved_intervals=unresolved,
        identity_objective=details,
    )
    distinct = []
    for path in result:
        if not any(np.array_equal(path, old) for old in distinct):
            distinct.append(path)
    return distinct
