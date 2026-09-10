"""Training-only adjacent-pair eligibility and reachable hard-negative contracts."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


def test_pairing_requires_exact_prior_label_and_excludes_operating_targets(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiment_local_patch_pairs import eligible_pairs

    points = {0: 10, 4: 11, 8: 12, 9: 13, 16: 15, 20: 15}
    assert eligible_pairs(points, {8}, 4).tolist() == [4, 20]


def test_reachable_negatives_take_priority_and_unknowns_never_enter(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiment_local_patch_pairs import negative_indices

    candidates = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
    target = np.zeros((len(candidates), 2), np.float32)
    target[6] = 1
    target[1] = -1
    chosen = negative_indices(candidates, target, 14, 6, 10, 3)
    assert candidates[chosen[:2]].tolist() == [8, 6]
    assert 14 not in candidates[chosen] and 4 not in candidates[chosen]
    assert len(chosen) == len(set(chosen)) == 6
