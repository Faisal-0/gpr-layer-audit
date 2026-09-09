"""Gap bypass and exact row-potential invariance counterexamples."""

from pathlib import Path

import numpy as np

from gpr_layer_audit.processing.seeded_challenger import dense_interval_dp


def test_absolute_gap_can_win_only_because_of_row_score_origin(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dense_relative_gap import relative_gap_unary

    observed = np.array([[True, False]] * 5)
    corr = np.array([[0.7, 0.0]] * 5)
    edges = np.zeros((4, 3, 2))
    legacy = np.where(observed, 1 - corr, 0.65)
    shifted_legacy = np.where(observed, 1 - (corr - 0.5), 0.65)
    assert np.all(dense_interval_dp(legacy, edges)[0] == 0)
    assert np.all(dense_interval_dp(shifted_legacy, edges)[0] == 1)
    new = relative_gap_unary(corr, observed, 0.65, -1)
    shifted = relative_gap_unary(corr - 0.5, observed, 0.65, -1)
    path, marginal = dense_interval_dp(new, edges)
    changed_path, changed_marginal = dense_interval_dp(shifted, edges)
    np.testing.assert_array_equal(path, changed_path)
    np.testing.assert_allclose(changed_marginal - marginal, 2.5)


def test_surface_and_missing_rows_do_not_create_observations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dense_relative_gap import relative_gap_unary

    corr = np.array([[1.0, 0.2, -0.2], [1.0, 1.0, 1.0]])
    visible = np.array([[True, True, False], [True, False, False]])
    cost = relative_gap_unary(corr, visible, 0.65, 0)
    np.testing.assert_allclose(cost[0], [1.45, 0.8, 1.45])
    np.testing.assert_allclose(cost[1], 0.65)
    np.testing.assert_array_equal(visible, [[True, True, False], [True, False, False]])
