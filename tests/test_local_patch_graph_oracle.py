"""An oracle must obey the original graph and cannot credit unknown rows/seeds."""

from pathlib import Path

import numpy as np


def test_oracle_cannot_restore_discarded_observations_or_forbidden_displacements(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from diagnose_local_patch_edges import oracle_recoverable

    data = np.ones((5, 6), np.float32)
    candidate = np.full(data.shape, np.nan)
    candidate[1, 1] = candidate[2, 4] = 0
    seeds = {0: 2, 4: 2}
    # Both individual reviewed events are reachable, but not on the same route.
    points = {0: 2, 1: 1, 2: 4, 4: 2}
    result, route = oracle_recoverable(data, candidate, points, seeds, 0, 1, 0)
    assert result["candidate_retained"] == 2
    assert result["maximum_graph_correct"] == 1
    assert not result["retained_route_all_correct_feasible"]
    assert route[0] == route[-1] == 2
    assert np.all(abs(np.diff(route)) <= 1)
    candidate[2, 4] = np.nan
    result, _ = oracle_recoverable(data, candidate, points, seeds, 0, 1, 0)
    assert result["candidate_retained"] == result["maximum_graph_correct"] == 1
    assert result["retained_route_all_correct_feasible"]
