from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.processing.conventional_config import ConventionalConfig
from gpr_layer_audit.processing.hybrid import ReflectorSegment
from gpr_layer_audit.processing.identity_path import IdentityGeometry, solve_identity_component


def scene(samples, *, contracted=False):
    samples = np.asarray(samples, dtype=int)
    table = SimpleNamespace(
        samples=samples, polarities=np.ones_like(samples), valid=samples >= 0, component_maps={}
    )
    rows, columns = samples.shape
    if contracted:
        segments = [ReflectorSegment(0, [(0, 0)], {0})]
        segments += [
            ReflectorSegment(c + 1, [(r, c) for r in range(1, rows - 1)]) for c in range(columns)
        ]
        segments.append(ReflectorSegment(columns + 1, [(rows - 1, 0)], {rows - 1}))
        edges = {(0, c + 1): 1 for c in range(columns)}
        edges.update({(c + 1, columns + 1): 1 for c in range(columns)})
    else:
        segments = [
            ReflectorSegment(
                r * columns + c, [(r, c)], {r} if r in (0, rows - 1) and c == 0 else set()
            )
            for r in range(rows)
            for c in range(columns)
        ]
        edges = {
            (r * columns + a, (r + 1) * columns + b): 1
            for r in range(rows - 1)
            for a in range(columns)
            for b in range(columns)
        }
    anchors = {0: samples[0, 0], rows - 1: samples[-1, 0]}
    return table, segments, edges, anchors


def solve(case, scores, geometry=None, **kwargs):
    table, segments, edges, anchors = case
    return solve_identity_component(
        segments,
        edges,
        table,
        scores,
        anchors,
        4,
        x_m=kwargs.pop("x_m", np.arange(len(table.samples), dtype=float)),
        dt_ns=kwargs.pop("dt_ns", 1.0),
        config=ConventionalConfig(),
        geometry=geometry,
        **kwargs,
    )


def test_matches_exhaustive_second_order_reference():
    rng = np.random.default_rng(893)
    case = scene([[10, 20], [11, 18], [11, 17], [12, 19], [12, 20]])
    scores = rng.uniform(0, 1, (5, 2))
    options = IdentityGeometry(
        slope_weight=0.13, slope_scale_ns_per_m=2, curvature_weight=0.7, curvature_scale_ns_per_m2=3
    )

    def huber(v):
        return v * v / 2 if abs(v) <= 1 else abs(v) - 0.5

    routes = []
    for interior in product(range(2), repeat=3):
        cols = [0, *interior, 0]
        depths = case[0].samples[np.arange(5), cols]
        slopes = np.diff(depths)
        cost = sum(scores[r, c] for r, c in enumerate(cols))
        cost -= 0.13 * sum(huber(v / 2) for v in slopes)
        cost -= 0.7 * sum(huber(v / 3) for v in np.diff(slopes))
        routes.append((cost, depths))
    expected = max(routes, key=lambda pair: pair[0])
    actual = solve(case, scores, options)
    assert actual.objective == pytest.approx(expected[0])
    assert np.array_equal(actual.samples, expected[1])
    assert actual.diagnostics["exact_on_retained_graph"]
    for row in range(1, 4):
        competitors = [pair for pair in routes if abs(pair[1][row] - expected[1][row]) > 2]
        if not competitors:
            assert actual.path_margin[row] == 1
            continue
        rival_cost, rival = max(competitors, key=lambda pair: pair[0])
        different = abs(rival - expected[1]) > 2
        left, right = row, row
        while left > 0 and different[left - 1]:
            left -= 1
        while right < 4 and different[right + 1]:
            right += 1
        expected_margin = min(1, (expected[0] - rival_cost) / (right - left + 1))
        assert actual.path_margin[row] == pytest.approx(expected_margin)
        assert actual.path_alternate[row] == rival[row]


def test_contracted_geometry_and_competitor_use_same_history_objective():
    case = scene([[10, 20], [10, 11], [10, 18], [10, 11], [10, 20]], contracted=True)
    scores = np.tile([0.6, 0.9], (5, 1))
    plain = solve(case, scores)
    assert plain.samples[2] == 18
    curved = solve(case, scores, IdentityGeometry(curvature_weight=0.1))
    assert np.all(curved.samples == 10)
    assert curved.path_alternate[2] == 18
    assert curved.path_margin[2] > 0
    # Expanding the same allowed chains must not alter objective or margins.
    table, segments, edges, anchors = case
    expanded = [
        ReflectorSegment(i, [node], {node[0]} if node[0] in anchors else set())
        for i, node in enumerate(node for s in segments for node in s.nodes)
    ]
    membership = {s.nodes[0]: s.index for s in expanded}
    links = {
        (membership[a.nodes[-1]], membership[b.nodes[0]]): quality
        for (aa, bb), quality in edges.items()
        for a in [segments[aa]]
        for b in [segments[bb]]
    }
    for segment in segments:
        links.update(
            {
                (membership[a], membership[b]): 1
                for a, b in zip(segment.nodes[:-1], segment.nodes[1:], strict=True)
            }
        )
    other = solve((table, expanded, links, anchors), scores, IdentityGeometry(curvature_weight=0.1))
    assert other.objective == pytest.approx(curved.objective)
    assert np.allclose(other.path_margin, curved.path_margin)


def test_persistent_bank_does_not_take_pointwise_maximum():
    case = scene([[10, 20], [10, 20], [10, 20], [10, 20]], contracted=True)
    banks = np.full((2, 4, 2), 0.7)
    banks[0, 1, 1], banks[0, 2, 1] = 1.0, 0.0
    banks[1, 1, 1], banks[1, 2, 1] = 0.0, 1.0
    pointwise = solve(case, banks.max(axis=0))
    persistent = solve(case, np.zeros((4, 2)), mode_scores=banks)
    assert np.all(pointwise.samples[1:3] == 20)
    assert np.all(persistent.samples == 10)
    assert len(set(persistent.modes)) == 1


def test_explicit_regime_change_and_missing_observation_are_preserved():
    case = scene([[10, 20], [10, 20], [10, 20], [10, 20]])
    banks = np.zeros((2, 4, 2))
    banks[0, :2, 0], banks[1, 2:, 0] = 1, 1
    result = solve(case, np.zeros((4, 2)), mode_scores=banks, mode_change_rows=(2,))
    assert result.modes.tolist() == [0, 0, 1, 1]
    assert result.objective == 4
    table, segments, _, anchors = case
    # Reacquisition skips row 2, without inventing its sample or curvature.
    subset = [s for s in segments if s.nodes[0] in ((0, 0), (1, 0), (3, 0))]
    links = {(0, 2): 1, (2, 6): 1}
    gap = solve(
        (table, subset, links, anchors), np.ones((4, 2)), IdentityGeometry(curvature_weight=100)
    )
    assert gap.samples.tolist() == [10, 10, -1, 10]
    with pytest.raises(ValueError, match="structural-break"):
        solve(case, np.ones((4, 2)), breaks=(2,))


def test_sampling_invariant_slope_cost_and_no_free_mode_switch():
    coarse = scene([[10], [12], [14]])
    fine = scene([[10], [11], [12], [13], [14]])
    geometry = IdentityGeometry(slope_weight=0.2, curvature_weight=0.5)
    a = solve(coarse, np.zeros((3, 1)), geometry)
    b = solve(fine, np.zeros((5, 1)), geometry, x_m=np.arange(5) * 0.5)
    assert a.objective == pytest.approx(b.objective)
    # Changing the sampling interval with equivalent physical event times agrees.
    c = solve(scene([[20], [24], [28]]), np.zeros((3, 1)), geometry, dt_ns=0.5)
    assert a.objective == pytest.approx(c.objective)


def test_limits_and_cancellation_cannot_silently_drop_competitors():
    case = scene([[10, 20]] * 5)
    scores = np.ones((5, 2))
    with pytest.raises(RuntimeError, match="no pruning"):
        solve(case, scores, replace(IdentityGeometry(), maximum_states=2))
    with pytest.raises(InterruptedError):
        solve(case, scores, cancel=lambda: True)
    with pytest.raises(ValueError, match="strictly increasing"):
        solve(case, scores, x_m=np.zeros(5))


def test_quantization_guard_does_not_penalize_rounding_a_gently_sloped_event():
    case = scene([[10], [10], [11], [11], [11]])
    scores = np.zeros((5, 1))
    unguarded = solve(case, scores, IdentityGeometry(curvature_weight=1))
    guarded = solve(
        case, scores, IdentityGeometry(curvature_weight=1, curvature_quantization_guard=True)
    )
    assert unguarded.objective < 0
    assert guarded.objective == 0
    sharp = solve(
        scene([[10], [10], [16], [10], [10]]),
        scores,
        IdentityGeometry(curvature_weight=1, curvature_quantization_guard=True),
    )
    assert sharp.objective < 0


def test_ordinary_seed_does_not_reset_persistent_mode_between_intervals():
    from gpr_layer_audit.processing.identity_path import solve_identity_intervals

    table, segments, edges, anchors = scene([[10, 20]] * 5)
    anchors[2] = 10
    for segment in segments:
        row, col = segment.nodes[0]
        segment.seed_rows = {row} if anchors.get(row) == table.samples[row, col] else set()
    modes = np.zeros((2, 5, 2))
    modes[0, :3, 0], modes[1, 2:, 0] = 1, 1
    diagnostics = {}
    paths = solve_identity_intervals(
        segments,
        edges,
        np.ones(len(segments)),
        table,
        np.zeros((5, 2)),
        anchors,
        (),
        4,
        1,
        ConventionalConfig(),
        diagnostics,
        dt_ns=1,
        mode_scores=modes,
    )
    assert np.all(paths[0] == 10)
    assert len(diagnostics["identity_objective"]) == 1
    # A free reset at the middle ordinary seed would produce objective 5.
    assert diagnostics["identity_objective"][0]["selected_objective"] == 3
