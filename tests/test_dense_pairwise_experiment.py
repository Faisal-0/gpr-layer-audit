"""Direct packet arithmetic and exact oracle contracts for isolated experiments."""

from itertools import product
from pathlib import Path

import numpy as np


def test_pairwise_cost_matches_direct_packets_and_ignores_masked_values(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dense_pairwise_model import pairwise_cost

    data = np.random.default_rng(672).normal(size=(4, 30))
    valid = np.ones_like(data, bool)
    valid[1, 7] = False
    costs = pairwise_cost(data, valid, 3, 2, 0.1)
    changed = data.copy()
    changed[~valid] = 1e30
    np.testing.assert_array_equal(costs, pairwise_cost(changed, valid, 3, 2, 0.1))
    for row in range(3):
        for offset in range(-2, 3):
            for target in range(5, 25):
                source = target - offset
                if not (
                    valid[row, source - 3 : source + 4].all()
                    and valid[row + 1, target - 3 : target + 4].all()
                ):
                    assert costs[row, offset + 2, target] == 0.065
                    continue
                left = data[row, source - 3 : source + 4]
                right = data[row + 1, target - 3 : target + 4]
                correlation = np.corrcoef(left, right)[0, 1]
                np.testing.assert_allclose(costs[row, offset + 2, target], (1 - correlation) * 0.1)


def test_dense_coverage_oracle_matches_exhaustive_routes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from diagnose_patchnet_dense import maximum_correct_route

    reward = np.random.default_rng(33).integers(0, 2, size=(6, 4)).astype(bool)
    reward[2] = False  # An unknown row is never background or oracle reward.
    anchors = {0: 1, 3: 3, 5: 2}
    routes = [
        route
        for route in product(range(1, 4), repeat=6)
        if all(route[r] == c for r, c in anchors.items())
        and all(abs(b - a) <= 1 for a, b in zip(route, route[1:], strict=False))
    ]
    expected = max(sum(reward[r, c] for r, c in enumerate(route)) for route in routes)
    assert maximum_correct_route(reward, 0, anchors, 1) == expected
    assert maximum_correct_route(reward, 0, {0: 1, 1: 3}, 1) is None


def test_interval_normalized_margin_is_not_local_certainty():
    from gpr_layer_audit.processing.seeded_challenger import dense_interval_dp

    scores = []
    for rows in (101, 1001):
        _, costs = dense_interval_dp(
            np.tile([0.0, 0.065, 0.1], (rows, 1)), np.zeros((rows - 1, 3, 3)), 0, 0
        )
        difference = costs[rows // 2, 2] - costs[rows // 2, 0]
        np.testing.assert_allclose(difference, 0.23)
        scores.append(difference / (rows * 0.1))
    # Moving distant anchors changes this uncalibrated gate, despite an
    # identical local alternative and exact objective difference.
    assert scores[0] > 0.02 > scores[1]
