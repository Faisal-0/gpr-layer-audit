from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.processing.conventional_config import ConventionalConfig
from gpr_layer_audit.processing.diverse_paths import distinct_complete_paths, matching_candidates
from gpr_layer_audit.processing.hybrid import ReflectorSegment, solve_complete_intervals
from gpr_layer_audit.processing.path_ambiguity import component_path_margins


def test_close_timing_variants_cannot_consume_distinct_reflector_slots():
    table = SimpleNamespace(samples=np.array([[60] * 6, [60, 61, 62, 63, 90, 110], [60] * 6]))
    segments = [ReflectorSegment(0, [(0, 0)], {0})]
    segments += [ReflectorSegment(c + 1, [(1, c)]) for c in range(6)]
    segments += [ReflectorSegment(7, [(2, 0)], {2})]
    edges = {(0, c + 1): 1.0 for c in range(6)} | {(c + 1, 7): 1.0 for c in range(6)}
    scores = np.ones((3, 6))
    scores[1] = [1, 0.99, 0.98, 0.97, 0.9, 0.8]
    config = ConventionalConfig(distinct_path_inference=True)
    diagnostics = {}
    paths = solve_complete_intervals(
        segments,
        edges,
        np.ones(8),
        table,
        scores,
        {0: 60, 2: 60},
        (),
        16,
        1,
        config,
        diagnostics,
    )
    assert [p.tolist() for p in paths] == [[60, 60, 60], [60, 90, 60], [60, 110, 60]]
    assert not diagnostics["unresolved_intervals"]


def test_adjacent_same_polarity_lobes_separated_by_zero_remain_competitors():
    lobes = np.ones((3, 128), int)
    lobes[:, 61:] = 2
    lobes[:, 62:] = 3
    table = SimpleNamespace(
        samples=np.array([[60, 60], [60, 62], [60, 60]]),
        polarities=np.ones((3, 2)),
        component_maps={"hybrid_lobe_id": lobes},
    )
    segments = [
        ReflectorSegment(0, [(0, 0)], {0}),
        ReflectorSegment(1, [(1, 0)]),
        ReflectorSegment(2, [(1, 1)]),
        ReflectorSegment(3, [(2, 0)], {2}),
    ]
    edges = {(0, 1): 1, (0, 2): 1, (1, 3): 1, (2, 3): 1}
    config = ConventionalConfig(distinct_path_inference=True)
    scores = np.ones((3, 2))
    paths = distinct_complete_paths(segments, edges, table, scores, {0: 60, 2: 60}, 8, 1, config)
    assert len(paths) == 2
    assert {int(p[1]) for p in paths} == {60, 62}
    margins, alternate = component_path_margins(
        segments,
        edges,
        table,
        scores,
        {0: 60, 2: 60},
        paths[0],
        8,
        step=1,
        config=config,
    )
    assert margins[1] == 0
    assert alternate[1] != paths[0][1]


def test_distinct_inference_matches_exhaustive_feasible_identity_alternatives():
    from gpr_layer_audit.processing.path_objective import local_score, transition_penalty

    rng = np.random.default_rng(282)
    for _ in range(10):
        table = SimpleNamespace(samples=np.tile([60, 61, 80, 100], (5, 1)))
        segments = [ReflectorSegment(0, [(0, 0)], {0})]
        for row in range(1, 4):
            segments.extend(ReflectorSegment(len(segments) + c, [(row, c)]) for c in range(4))
        segments.append(ReflectorSegment(len(segments), [(4, 0)], {4}))
        # Construct stable consecutive indices independently of extend's iterator.
        for index, segment in enumerate(segments):
            segment.index = index
        edges = {
            (a.index, b.index): float(rng.uniform(0.8, 1))
            for a in segments
            for b in segments
            if 0 < b.start - a.stop <= 2
        }
        scores = rng.uniform(0.1, 1, (5, 4))
        config = ConventionalConfig(distinct_path_inference=True)
        all_paths = []

        def visit(
            sequence,
            total,
            segments=segments,
            table=table,
            all_paths=all_paths,
            edges=edges,
            scores=scores,
        ):
            a = sequence[-1]
            if a == len(segments) - 1:
                path = np.full(5, -1)
                for index in sequence:
                    for row, col in segments[index].nodes:
                        path[row] = table.samples[row, col]
                all_paths.append((total, path))
                return
            for (left, right), q in edges.items():
                if left == a:
                    visit(
                        [*sequence, right],
                        total
                        + local_score(segments[right], scores, 0.4)
                        - transition_penalty(segments[a], segments[right], q, 0.4),
                    )

        visit([0], local_score(segments[0], scores, 0.4))
        expected = []
        for _total, path in sorted(all_paths, key=lambda p: -p[0]):
            if all(np.any((path >= 0) & (old >= 0) & (abs(path - old) > 2)) for old in expected):
                expected.append(path)
            if len(expected) == 3:
                break
        actual = distinct_complete_paths(
            segments,
            edges,
            table,
            scores,
            {0: 60, 4: 60},
            8,
            0.4,
            config,
        )
        assert len(actual) == len(expected)
        assert all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True))


def test_missing_routes_are_explicit_and_incompatible_endpoints_are_unresolved():
    table = SimpleNamespace(samples=np.full((5, 1), 60))
    segments = [ReflectorSegment(0, [(0, 0)], {0}), ReflectorSegment(1, [(4, 0)], {4})]
    config = ConventionalConfig(distinct_path_inference=True)
    diagnostic = {}
    paths = solve_complete_intervals(
        segments,
        {(0, 1): 1},
        np.ones(2),
        table,
        np.ones((5, 1)),
        {0: 60, 4: 60},
        (),
        7,
        0.4,
        config,
        diagnostic,
    )
    assert paths[0].tolist() == [60, -1, -1, -1, 60]
    assert not diagnostic["unresolved_intervals"]
    diagnostic = {}
    paths = solve_complete_intervals(
        segments,
        {},
        np.ones(2),
        table,
        np.ones((5, 1)),
        {0: 60, 4: 60},
        (),
        7,
        0.4,
        config,
        diagnostic,
    )
    assert paths[0].tolist() == [60, -1, -1, -1, 60]
    assert diagnostic["unresolved_intervals"]
    with pytest.raises(InterruptedError):
        distinct_complete_paths(
            segments,
            {},
            table,
            np.ones((5, 1)),
            {0: 60, 4: 60},
            7,
            0.4,
            config,
            cancel=lambda: True,
        )


def test_matching_candidates_never_identifies_missing_state_as_measurement():
    table = SimpleNamespace(
        samples=np.array([[-1, 60], [60, 62]]), polarities=np.array([[0, 1], [1, -1]])
    )
    assert matching_candidates(table, np.array([-1, 60]), 4).tolist() == [
        [False, False],
        [True, False],
    ]
