from types import SimpleNamespace

import numpy as np

from gpr_layer_audit.processing.hybrid import ReflectorSegment
from gpr_layer_audit.processing.path_ambiguity import component_path_margins


def fork_scene(strength=1.0):
    samples = np.tile([60, 75], (8, 1))
    table = SimpleNamespace(samples=samples)
    segments = [
        ReflectorSegment(0, [(0, 0)], {0}),
        ReflectorSegment(1, [(r, 0) for r in range(1, 7)]),
        ReflectorSegment(2, [(r, 1) for r in range(1, 7)]),
        ReflectorSegment(3, [(7, 0)], {7}),
    ]
    edges = {(0, 1): 1.0, (1, 3): 1.0, (0, 2): 1.0, (2, 3): 1.0}
    scores = np.tile([1.0, strength], (8, 1))
    return segments, edges, table, scores


def test_equal_complete_seeded_routes_remain_ambiguous():
    segments, edges, table, scores = fork_scene()
    margin, alternate = component_path_margins(
        segments, edges, table, scores, {0: 60, 7: 60}, np.full(8, 60), 7
    )
    assert np.all(margin[1:7] == 0)
    assert np.all(alternate[1:7] == 75)


def test_path_margin_compares_whole_route_instead_of_one_bright_pixel():
    segments, edges, table, scores = fork_scene(0.2)
    scores[4, 1] = 3.0
    margin, alternate = component_path_margins(
        segments, edges, table, scores, {0: 60, 7: 60}, np.full(8, 60), 7
    )
    assert np.all(margin[1:7] > 0.1)
    assert alternate[4] == 75


def test_missing_route_is_a_competing_hypothesis():
    segments, edges, table, scores = fork_scene(0.2)
    edges[0, 3] = 1.0
    scores[1:7] = -0.1  # Exactly offsets the six-row missing penalty.
    margin, alternate = component_path_margins(
        segments, edges, table, scores, {0: 60, 7: 60}, np.full(8, 60), 7
    )
    assert np.allclose(margin[1:7], 0)
    assert np.all(alternate[1:7] == -1)


def test_incompatible_seeds_cannot_get_positive_path_margin():
    segments, edges, table, scores = fork_scene(0.2)
    edges = {(0, 1): 1.0, (2, 3): 1.0}
    margin, _ = component_path_margins(
        segments, edges, table, scores, {0: 60, 7: 60}, np.full(8, 60), 7
    )
    assert not np.any(margin)


def test_suboptimal_selected_route_cannot_claim_margin():
    segments, edges, table, scores = fork_scene(2.0)
    margin, _ = component_path_margins(
        segments, edges, table, scores, {0: 60, 7: 60}, np.full(8, 60), 7
    )
    assert not np.any(margin[1:7])


def test_additional_click_prioritizes_long_observable_disagreement():
    from gpr_layer_audit.processing.path_ambiguity import suggest_path_observations

    first = np.full(240, 60)
    second = first.copy()
    second[50:160] = 75
    second[190:215] = 90
    proposals = suggest_path_observations(
        [first, second], np.ones((240, 128)), {3: 60, 235: 60}, 1.0, 7
    )
    assert 75 <= proposals[0]["row"] <= 135
    assert proposals[0]["candidate_samples"] == [60, 75]
    assert proposals[0]["contested_observations"] == 110


def test_additional_click_excludes_unobservable_intervals_and_existing_seeds():
    from gpr_layer_audit.processing.path_ambiguity import suggest_path_observations

    first, second = np.full(240, 60), np.full(240, 75)
    measurement = np.ones((240, 128))
    measurement[50:200] = 0
    proposals = suggest_path_observations([first, second], measurement, {20: 60, 220: 60}, 1.0, 7)
    assert proposals
    for proposal in proposals:
        row = proposal["row"]
        assert row < 50 or row >= 200
        assert row not in (20, 220)


def test_single_path_does_not_invent_a_competing_lobe_request():
    from gpr_layer_audit.processing.path_ambiguity import suggest_path_observations

    assert suggest_path_observations([np.full(240, 60)], np.ones((240, 128)), {}, 1.0, 7) == []


def test_close_opposite_lobe_still_competes_with_selected_path():
    segments, edges, table, scores = fork_scene()
    table.samples[:, 1] = 62
    table.polarities = np.tile([-1, 1], (8, 1))
    margin, alternate = component_path_margins(
        segments, edges, table, scores, {0: 60, 7: 60}, np.full(8, 60), 7
    )
    assert np.all(margin[1:7] == 0)
    assert np.all(alternate[1:7] == 62)


def test_dynamic_path_margins_match_exhaustive_small_graphs():
    rng = np.random.default_rng(23)
    for _ in range(20):
        table = SimpleNamespace(samples=np.tile([60, 75], (5, 1)))
        segments = [ReflectorSegment(0, [(0, 0)], {0})]
        for row in range(1, 4):
            for col in (0, 1):
                segments.append(ReflectorSegment(len(segments), [(row, col)]))
        segments.append(ReflectorSegment(7, [(4, 0)], {4}))
        edges = {
            (left.index, right.index): float(rng.uniform(0.8, 1))
            for left in segments
            for right in segments
            if 0 < right.start - left.stop <= 2
        }
        scores = rng.uniform(0.05, 1, (5, 2))
        all_paths = []

        def enumerate_paths(
            path,
            total,
            segments=segments,
            table=table,
            all_paths=all_paths,
            edges=edges,
            scores=scores,
        ):
            index = path[-1]
            if index == 7:
                samples = np.full(5, -1)
                for node in path:
                    row, col = segments[node].nodes[0]
                    samples[row] = table.samples[row, col]
                all_paths.append((total, samples))
                return
            for (left, right), quality in edges.items():
                if left != index:
                    continue
                row, col = segments[right].nodes[0]
                missing = segments[right].start - segments[left].stop - 1
                enumerate_paths(
                    [*path, right], total + scores[row, col] - (1 - quality) - 0.1 * missing
                )

        enumerate_paths([0], scores[0, 0])
        best_score, selected = max(all_paths, key=lambda item: item[0])
        margin, _ = component_path_margins(
            segments, edges, table, scores, {0: 60, 4: 60}, selected, 7
        )
        for row in range(1, 4):
            if selected[row] < 0:
                assert margin[row] == 0
                continue
            competing = max(total for total, path in all_paths if path[row] != selected[row])
            assert np.isclose(margin[row], min(1, (best_score - competing) / 5))
