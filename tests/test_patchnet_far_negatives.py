"""Training-negative scope and label exclusion for the fixed patch matcher."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


def test_far_negatives_include_strong_remote_matches_without_unknowns(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiment_patchnet_far_negatives import negative_indices

    candidates = np.array([10, 20, 30, 48, 49, 50, 51, 52, 70, 80, 90, 95])
    target = np.zeros((len(candidates), 2), np.float32)
    target[5] = 1
    target[-1] = -1
    correlations = np.array([0.1, 0.9, 0.3, 0.2, 0.1, 1, 0.7, 0.1, 0.8, 0.85, 0.6, 100])
    selected = negative_indices(candidates, target, 50, 2, correlations)
    assert len(selected) == 6
    assert set(candidates[selected[3:]]) == {20, 70, 80}
    assert 50 not in candidates[selected] and 95 not in candidates[selected]


def test_shortage_fills_without_duplicate_or_positive_targets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiment_patchnet_far_negatives import negative_indices

    candidates = np.arange(20)
    target = np.zeros((20, 2), np.float32)
    target[8:13] = 1
    selected = negative_indices(candidates, target, 10, 20, np.ones(20))
    assert len(selected) == len(set(selected)) == 6
    assert np.all(target[selected, 0] == 0)
