from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gpr_layer_audit.processing.seed_graph import (
    _candidate_table,
    _component_maps,
    _event_emissions,
    _propagated_positives,
    _template_bank,
    _template_feature_maps,
)


def _training_table(branch_agreement):
    names = (
        "signed_seed_correlation",
        "phase_cycle_agreement",
        "oriented_coherence",
        "preprocessing_agreement",
        "polarity_agreement",
        "design_tiebreak",
    )
    features = np.ones((9, 3, len(names)))
    features[:, 0, names.index("preprocessing_agreement")] = branch_agreement
    features[:, 1, :3] = 0.1
    return SimpleNamespace(
        samples=np.tile([100, 104, -1], (9, 1)),
        valid=np.tile([True, True, False], (9, 1)),
        features=features,
        feature_names=names,
        polarities=np.tile([-1, -1, 0], (9, 1)),
        waveforms=np.tile([[0, 1, 0, -1, 0, 1, 0], [0, -1, 0, 1, 0, -1, 0], [0] * 7], (9, 1, 1)),
        tracklet_support=np.ones((9, 3)),
    )


def test_tracklet_support_alone_cannot_create_training_labels():
    table = _training_table(0.0)
    assert _propagated_positives(table, {4: 100}, 7, 1.0, 2) == {4: 100}
    table.features[:, 0, table.feature_names.index("preprocessing_agreement")] = 1
    positives = _propagated_positives(table, {4: 100}, 7, 1.0, 2)
    assert len(positives) > 1 and all(sample == 100 for sample in positives.values())


def test_conflicting_polarity_stays_in_table_but_cannot_silently_replace_seeded_lobe():
    table = _training_table(1.0)
    table.polarities[:, 1] = 1
    before = table.valid.copy()
    emissions = _event_emissions(
        table,
        np.ones((9, 3)),
        {0: 100, 8: 100},
        {},
        set(),
        np.zeros(9, dtype=bool),
        0.10,
    )
    assert np.all(np.isneginf(emissions[:, 1]))
    assert np.array_equal(table.valid, before)
    # Different confirmed endpoint lobes allow an explicit regime decision.
    different_regimes = _event_emissions(
        table,
        np.ones((9, 3)),
        {0: 100, 8: 104},
        {},
        set(),
        np.zeros(9, dtype=bool),
        0.10,
    )
    assert np.all(np.isfinite(different_regimes[1:8, :2]))


def test_both_confirmed_lobe_modes_survive_before_graph_selection():
    rows, samples = 80, 160
    x = (np.arange(samples) - 100) / 3.0
    wave = (1 - x * x) * np.exp(-x * x / 2)
    data = np.tile(wave, (rows, 1)).astype(np.float32)
    anchors = {5: 105, 75: 100}
    metadata = {
        5: {"polarity": -1, "regime_id": "first", "canonical_sample_index": 100},
        75: {"polarity": 1, "regime_id": "second", "canonical_sample_index": 100},
    }
    bank = _template_bank(data, anchors, 11, metadata)
    maps, _ = _component_maps(data, data, None, bank, 7)
    table = _candidate_table(
        data,
        data,
        maps,
        np.full(rows, 80),
        np.full(rows, 100),
        np.full(rows, 120),
        anchors,
        np.zeros(rows, dtype=bool),
        7,
    )
    for row in range(rows):
        candidates = table.samples[row, table.valid[row] & (table.samples[row] >= 0)]
        assert np.any(abs(candidates - 100) <= 1), (row, candidates)
        assert np.any(abs(candidates - 105) <= 1), (row, candidates)


def test_template_winner_cannot_borrow_identity_from_opposite_polarity_regime():
    data = np.zeros((3, 50), dtype=np.float32)
    shape = np.asarray([0, -0.3, -0.6, 1, 0.01, 1, -0.6, -0.3, 0])
    data[:, 16:25] = shape
    data[1, 20] = -0.1
    data[1, 18] = -0.1
    data[2, 20] = -0.01
    bank = _template_bank(data, {0: 20, 1: 20}, 6)
    maps, correlations = _template_feature_maps(data, bank)
    assert correlations[0][2, 20] > correlations[1][2, 20]
    assert maps["prototype_index"][2, 20] == 1
    assert maps["polarity_agreement"][2, 20] == 1


def test_joint_pass_reconciliation_cannot_cross_interfaces():
    from gpr_layer_audit.processing.pass_selection import select_joint_passes

    samples = np.tile(np.asarray([[80, 40], [100, 60]]), (12, 1, 1))
    support = np.tile(np.asarray([[0.9, 0.8], [0.8, 0.9]]), (12, 1, 1))
    preferred = np.tile([0, 1], (12, 1))  # Independent choice would select 80/60.
    choices = select_joint_passes(
        samples, samples, np.ones_like(samples), support, preferred, [5, 5], 7, 0.4, set()
    )
    selected = np.take_along_axis(
        np.concatenate((samples, np.full((12, 2, 1), -1)), axis=2),
        choices[:, :, None],
        axis=2,
    )[:, :, 0]
    assert np.all(selected[:, 0] >= 0) and np.all(selected[:, 1] >= selected[:, 0] + 5)


def test_joint_pass_switch_requires_gap_or_structural_break():
    from gpr_layer_audit.processing.pass_selection import select_joint_passes

    samples = np.tile([[[100, 130]]], (20, 1, 1))
    support = np.tile([[[0.9, 0.1]]], (20, 1, 1))
    support[10:] = [0.1, 0.9]
    preferred = np.zeros((20, 1), dtype=int)
    preferred[10:] = 1
    choices = select_joint_passes(
        samples, samples, np.ones_like(samples), support, preferred, [5], 7, 0.4, set()
    )[:, 0]
    assert not np.any((choices[:-1] == 0) & (choices[1:] == 1))
    with_break = select_joint_passes(
        samples, samples, np.ones_like(samples), support, preferred, [5], 7, 0.4, {10}
    )[:, 0]
    assert np.all(with_break[:10] == 0) and np.all(with_break[10:] == 1)


def test_joint_pass_reconciliation_cannot_drop_confirmed_seed_to_improve_score():
    from gpr_layer_audit.processing.pass_selection import select_joint_passes

    samples = np.tile([[[100, 130]]], (20, 1, 1))
    support = np.tile([[[0.1, 0.9]]], (20, 1, 1))
    preferred = np.ones((20, 1), dtype=int)
    choices = select_joint_passes(
        samples,
        samples,
        np.ones_like(samples),
        support,
        preferred,
        [5],
        7,
        0.4,
        set(),
        required_samples={0: {10: 100}},
    )[:, 0]
    assert choices[10] == 0


def test_final_visibility_gate_cannot_change_selected_event_family(monkeypatch):
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing import seed_graph

    def path(sample, score):
        samples = np.full(12, sample, dtype=np.int32)
        samples[0] = 80
        evidence = {
            "graph_selected_sample": samples.astype(float),
            "canonical_event_sample": samples.astype(float),
            "selected_lobe_code": np.ones(12),
            "pre_gate_confidence": np.full(12, score),
        }
        return seed_graph.SeedConditionedPath(
            samples,
            np.full(12, score),
            np.zeros((12, 140)),
            samples.copy(),
            np.ones(12, dtype=bool),
            np.zeros(12, dtype=bool),
            evidence,
            samples.copy(),
            samples.copy(),
            np.zeros(12, dtype=bool),
            True,
            {"audit_graph_selected": np.zeros((12, 140))},
        )

    signal, guided = path(80, 0.9), path(100, 0.6)
    monkeypatch.setattr(
        seed_graph,
        "_pick_seed_conditioned_pass",
        lambda *args, **kwargs: {1: guided if kwargs["search_corridors"] else signal},
    )

    def select():
        return seed_graph.pick_seed_conditioned_interfaces(
            np.zeros((12, 140)),
            20,
            [LayerSpec(1, "Asphalt", 5, 120, 5)],
            anchor_samples={1: {0: 80}},
            feature_branches=None,
            search_corridors={1: object()},
            design_weight=0.10,
            pulse_width_samples=7,
            break_rows=set(),
            anomaly_mask=np.zeros(12, dtype=bool),
            max_interpolation_rows=0,
        )[1]

    original = select()
    signal.confidence[1:] = 0
    signal.samples[1:] = -1
    signal.visible[1:] = False
    more_conservative = select()
    assert np.array_equal(
        original.evidence["graph_selected_sample"],
        more_conservative.evidence["graph_selected_sample"],
    )
    assert np.all(more_conservative.samples[1:] == -1)  # Gap, not an alternative family.
