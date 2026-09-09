from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.processing.adaptive_templates import adaptive_template_scores
from gpr_layer_audit.processing.conventional_config import ConventionalConfig


def template_scene():
    x = np.arange(-4, 5, dtype=np.float32)
    original = np.exp(-x * x / 3)
    changed = np.exp(-x * x / 7) + 0.15 * x * np.exp(-x * x / 4)
    waves = np.stack([original, changed])
    waves -= waves.mean(axis=1, keepdims=True)
    waves /= np.linalg.norm(waves, axis=1, keepdims=True)
    waveform = np.tile(waves[1], (16, 2, 1))
    waveform[:, 1] = -waves[1]
    waveform[0, 0] = waves[0]
    waveform[15, 0] = waves[0][::-1]
    samples = np.tile([20, 40], (16, 1))
    return SimpleNamespace(
        samples=samples,
        valid=np.ones((16, 2), bool),
        waveforms=waveform,
        polarities=np.tile([1, -1], (16, 1)),
    )


def compute(table, *, margin=None, breaks=(), anchors=None, allowed=None):
    return adaptive_template_scores(
        table,
        np.full(16, 20),
        np.full((16, 64), 0.95),
        np.ones(16) if margin is None else margin,
        np.ones((16, 64)),
        anchors or {0: 20, 15: 20},
        breaks,
        0.1,
        ConventionalConfig(adapt_seed_templates=True),
        allowed_rows=allowed,
    )


def test_adaptation_preserves_originals_and_excludes_ambiguous_observations():
    table = template_scene()
    original = table.waveforms.copy()
    margin = np.ones(16)
    margin[4] = 0
    bonus, records = compute(table, margin=margin)
    assert records and np.max(bonus[:, 0]) > 0
    assert np.all(bonus[:, 1] == 0)  # Opposite lobe never inherits this seed identity.
    assert np.array_equal(table.waveforms, original)
    assert 4 not in {r for record in records for r in record["source_rows"]}
    # Changing an ambiguous observation cannot modify the learned template bank.
    table.waveforms[4, 0] = np.roll(table.waveforms[4, 0], 3)
    _, changed_records = compute(table, margin=margin)
    assert changed_records == records


def test_templates_remain_in_their_seed_region_and_do_not_cross_breaks():
    table = template_scene()
    bonus, records = compute(table, breaks=(8,), anchors={0: 20})
    assert records
    assert all(max(r["source_rows"]) < 8 for r in records)
    assert np.all(bonus[8:] == 0)
    _, records = compute(table)
    assert {r["seed_row"] for r in records} == {0, 15}
    assert all(all(row < 8 for row in r["source_rows"]) for r in records if r["seed_row"] == 0)


def test_ordering_uncertainty_and_sparse_support_cannot_update_templates():
    table = template_scene()
    allowed = np.zeros(16, bool)
    allowed[[0, 1, 15]] = True
    bonus, records = compute(table, allowed=allowed)
    assert records == [] and not np.any(bonus)


def test_template_adaptation_cancels_and_has_no_feedback_pass():
    table = template_scene()
    with pytest.raises(InterruptedError):
        adaptive_template_scores(
            table,
            np.full(16, 20),
            np.ones((16, 64)),
            np.ones(16),
            np.ones((16, 64)),
            {0: 20},
            (),
            0.1,
            ConventionalConfig(adapt_seed_templates=True),
            cancel=lambda: True,
        )


def test_tracker_records_guarded_update_and_preserves_clicked_reflector():
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence
    from gpr_layer_audit.processing.tracker import pick_interfaces

    axis = np.arange(128)
    data = np.zeros((61, 128), np.float32)
    for row in range(61):
        z = (axis - 65) / (2.2 + row / 120)
        data[row] = -(1 - z * z) * np.exp(-z * z / 2)
    result = pick_interfaces(
        data,
        15,
        [LayerSpec(2, "Base", 20, 95, 4)],
        method="seed_hybrid",
        anchor_samples={2: {3: 65, 30: 65, 57: 65}},
        horizontal_step_m=0.1,
        ml_policy="off",
        hybrid_evidence=HybridEvidence(data, 0.05, 0.1),
        conventional_config={"adapt_seed_templates": True},
    )[2]
    assert all(result.samples[r] == 65 for r in (3, 30, 57))
    assert np.all(abs(result.samples[result.visible] - 65) <= 2)
    assert result.provenance["template_adaptation"]["original_seed_templates_preserved"]
    assert result.provenance["template_adaptation"]["status"] == "one_guarded_pass"


def test_tracker_discards_an_update_that_changes_its_own_support_observation(monkeypatch):
    import gpr_layer_audit.processing.adaptive_templates as adaptation
    import gpr_layer_audit.processing.hybrid as hybrid
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence
    from gpr_layer_audit.processing.tracker import pick_interfaces

    original = hybrid.solve_complete_intervals
    calls = 0

    def solve(*args, **kwargs):
        nonlocal calls
        calls += 1
        paths = original(*args, **kwargs)
        if calls == 2:
            paths[0][10] += 5
        return paths

    def propose(table, *args, **kwargs):
        return np.ones_like(table.samples, dtype=float), [{"source_rows": [10]}]

    monkeypatch.setattr(hybrid, "solve_complete_intervals", solve)
    monkeypatch.setattr(adaptation, "adaptive_template_scores", propose)
    axis = np.arange(128)
    z = (axis - 65) / 2.2
    data = np.tile(-(1 - z * z) * np.exp(-z * z / 2), (31, 1)).astype(np.float32)
    result = pick_interfaces(
        data,
        15,
        [LayerSpec(2, "Base", 20, 95, 4)],
        method="seed_hybrid",
        anchor_samples={2: {3: 65, 15: 65, 27: 65}},
        horizontal_step_m=0.1,
        ml_policy="off",
        hybrid_evidence=HybridEvidence(data, 0.05, 0.1),
        conventional_config={"adapt_seed_templates": True},
    )[2]
    assert calls == 2
    assert result.samples[10] == 65
    assert result.provenance["template_adaptation"]["status"] == "discarded_source_path_changed"
    assert result.provenance["template_adaptation"]["updates"] == []
