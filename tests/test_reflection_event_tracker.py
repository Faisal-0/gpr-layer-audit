from __future__ import annotations

import numpy as np

from gpr_layer_audit.models import LayerSpec, SearchCorridor
from gpr_layer_audit.processing.tracker import pick_interfaces


def _ricker(axis: np.ndarray, centre: float, width: float, amplitude: float = 1.0):
    z = (axis - centre) / width
    return amplitude * (1.0 - z**2) * np.exp(-0.5 * z**2)


def test_seeded_negative_lobe_does_not_slip_to_positive_half_cycle():
    rows, sample_count = 120, 180
    axis = np.arange(sample_count, dtype=float)
    centres = np.asarray(
        [105.0 + 3.0 * np.sin(row / 17.0) for row in range(rows)], dtype=float
    )
    negative_lobes = np.rint(centres + np.sqrt(3.0) * 2.6).astype(int)
    data = np.asarray(
        [
            _ricker(axis, 40.0, 2.4, 1.8)
            + _ricker(axis, centre, 2.6, 0.9)
            for centre in centres
        ],
        dtype=float,
    )
    data += np.random.default_rng(21).normal(0.0, 0.012, data.shape)
    layer = LayerSpec(1, "Interface", 45, 100, 4)
    seed_rows = (8, 60, 112)
    anchors = {1: {row: int(negative_lobes[row]) for row in seed_rows}}

    path = pick_interfaces(data, 40, [layer], anchor_samples=anchors)[1]

    visible = path.samples >= 0
    assert np.mean(visible) > 0.90
    selected = path.samples[visible]
    trough_error = np.abs(selected - negative_lobes[visible])
    positive_error = np.abs(selected - np.rint(centres[visible]).astype(int))
    assert np.mean(trough_error) < 1.5
    assert np.mean(trough_error < positive_error) > 0.99
    assert np.median(path.evidence["tracklet_support"][visible]) > 0.70
    assert np.median(path.evidence["cycle_slip_risk"][visible]) < 0.25
    assert np.median(path.evidence["drop_seed_stability"][visible]) > 0.60


def test_wavelet_lobes_are_grouped_into_reflection_packets():
    rows, sample_count = 40, 150
    axis = np.arange(sample_count, dtype=float)
    data = np.asarray(
        [_ricker(axis, 40.0, 2.4, 1.5) + _ricker(axis, 95.0, 2.8) for _ in range(rows)]
    )
    layer = LayerSpec(1, "Interface", 40, 75, 4)
    anchors = {1: {5: 100, 20: 100, 35: 100}}

    path = pick_interfaces(data, 40, [layer], anchor_samples=anchors)[1]

    packet_map = path.candidate_components["event_packet_centre"]
    # All observable lobes remain available; they are members of one packet,
    # not three independently identified physical interfaces. Suppressing the
    # alternatives before selection caused real-road continuation losses.
    for row in range(rows):
        centres = packet_map[row, 88:103]
        assert len(np.unique(centres[np.isfinite(centres)])) == 1
        assert np.all(np.isfinite(packet_map[row, [90, 95, 100]]))
    assert np.all(path.samples == 100)  # Retention must not change the selected lobe.


def test_joint_solver_follows_varying_base_instead_of_seed_interpolation():
    rows, sample_count = 100, 210
    axis = np.arange(sample_count, dtype=float)
    asphalt = np.full(rows, 78, dtype=int)
    base = np.rint(130.0 + 9.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))).astype(int)
    data = np.asarray(
        [
            _ricker(axis, 40.0, 2.3, 1.6)
            + _ricker(axis, asphalt[row], 2.5, 1.0)
            - _ricker(axis, base[row], 3.0, 0.75)
            for row in range(rows)
        ]
    )
    data += np.random.default_rng(8).normal(0.0, 0.015, data.shape)
    layers = [
        LayerSpec(1, "Asphalt", 25, 60, 5),
        LayerSpec(2, "Base", 70, 120, 12),
    ]
    anchors = {
        1: {5: int(asphalt[5]), 50: int(asphalt[50]), 94: int(asphalt[94])},
        2: {5: int(base[5]), 50: int(base[50]), 94: int(base[94])},
    }

    paths = pick_interfaces(data, 40, layers, anchor_samples=anchors)

    visible = paths[2].samples >= 0
    assert np.mean(visible) > 0.85
    assert np.mean(np.abs(paths[2].samples[visible] - base[visible])) < 2.0
    straight = np.interp(np.arange(rows), [5, 50, 94], [base[5], base[50], base[94]])
    assert np.mean(np.abs(paths[2].samples[visible] - straight[visible])) > 2.5
    assert np.median(paths[2].evidence["joint_hypothesis_support"][visible]) > 0.70
    assert np.median(paths[2].evidence["forward_backward_agreement"][visible]) > 0.70


def test_design_pass_is_independent_and_reports_event_family_conflict():
    rows, sample_count = 70, 170
    axis = np.arange(sample_count, dtype=float)
    data = np.asarray(
        [
            _ricker(axis, 40.0, 2.4, 1.4)
            + _ricker(axis, 78.0, 2.6, 0.75)
            + _ricker(axis, 126.0, 2.8, 1.5)
            for _ in range(rows)
        ]
    )
    layer = LayerSpec(1, "Interface", 25, 110, 4)
    lower = np.full(rows, 68.0)
    centre = np.full(rows, 78.0)
    upper = np.full(rows, 88.0)
    corridor = SearchCorridor(
        layer_order=1,
        chainage_m=np.arange(rows, dtype=float) * 0.4,
        lower_sample=lower,
        centre_sample=centre,
        upper_sample=upper,
        gap_lower_samples=lower - 40.0,
        gap_centre_samples=centre - 40.0,
        gap_upper_samples=upper - 40.0,
        source="test-design",
    )

    signal = pick_interfaces(data, 40, [layer])[1]
    guided = pick_interfaces(data, 40, [layer], search_corridors={1: corridor})[1]

    assert np.array_equal(guided.signal_only_samples, signal.samples)
    assert np.median(guided.signal_only_samples[guided.signal_only_samples >= 0]) > 115
    assert np.median(guided.design_guided_samples[guided.design_guided_samples >= 0]) < 90
    assert np.mean(guided.design_conflict) > 0.90
    assert np.mean(guided.evidence["branch_multimodality"]) > 0.90


def test_drop_one_seed_keeps_phase_family_or_moves_rows_to_no_pick():
    rows, sample_count = 90, 175
    axis = np.arange(sample_count, dtype=float)
    centres = 103.0 + 5.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    troughs = np.rint(centres + np.sqrt(3.0) * 2.7).astype(int)
    data = np.asarray(
        [
            _ricker(axis, 40.0, 2.3, 1.5)
            + _ricker(axis, centre, 2.7, 0.85)
            for centre in centres
        ]
    )
    data += np.random.default_rng(31).normal(0.0, 0.015, data.shape)
    layer = LayerSpec(1, "Interface", 45, 105, 4)
    seed_rows = (8, 44, 82)
    all_anchors = {row: int(troughs[row]) for row in seed_rows}
    reference = pick_interfaces(
        data, 40, [layer], anchor_samples={1: all_anchors}
    )[1].samples

    for removed in seed_rows:
        anchors = {row: sample for row, sample in all_anchors.items() if row != removed}
        dropout = pick_interfaces(data, 40, [layer], anchor_samples={1: anchors})[1].samples
        comparable = (reference >= 0) & (dropout >= 0)
        assert np.mean(comparable) > 0.80
        assert np.all(np.abs(reference[comparable] - dropout[comparable]) <= 7)


def test_single_tracklet_preserves_signed_slope_through_turning_points():
    rows, sample_count = 150, 190
    axis = np.arange(sample_count, dtype=float)
    centres = 112.0 + 10.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, rows))
    troughs = np.rint(centres + np.sqrt(3.0) * 2.8).astype(int)
    data = np.asarray(
        [_ricker(axis, 40.0, 2.3, 1.5) + _ricker(axis, centre, 2.8) for centre in centres]
    )
    data += np.random.default_rng(52).normal(0.0, 0.01, data.shape)
    layer = LayerSpec(1, "Turning interface", 48, 105, 4)
    seed_row = rows // 2

    path = pick_interfaces(
        data,
        40,
        [layer],
        anchor_samples={1: {seed_row: int(troughs[seed_row])}},
    )[1]

    visible = path.samples >= 0
    assert np.mean(visible) > 0.88
    assert np.mean(np.abs(path.samples[visible] - troughs[visible])) < 1.8
    assert np.mean(path.evidence["tracklet_support"][visible] >= 0.55) > 0.70


def test_compatible_seed_tracklets_share_one_event_family():
    rows, sample_count = 100, 175
    axis = np.arange(sample_count, dtype=float)
    centres = 104.0 + 5.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    troughs = np.rint(centres + np.sqrt(3.0) * 2.7).astype(int)
    data = np.asarray(
        [_ricker(axis, 40.0, 2.3, 1.5) + _ricker(axis, centre, 2.7) for centre in centres]
    )
    seeds = (8, 49, 91)
    layer = LayerSpec(1, "Interface", 45, 105, 4)

    path = pick_interfaces(
        data,
        40,
        [layer],
        anchor_samples={1: {row: int(troughs[row]) for row in seeds}},
    )[1]

    family_map = path.candidate_components["event_family_index"]
    families = [int(family_map[row, troughs[row]]) for row in seeds]
    assert min(families) >= 0
    assert len(set(families)) == 1


def test_base_identity_survives_imperfect_asphalt_stripping():
    rows, sample_count = 120, 190
    axis = np.arange(sample_count, dtype=float)
    asphalt_centres = 84.0 + 2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    base_centres = 118.0 + 6.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    asphalt_wrong_lobe = np.rint(asphalt_centres + np.sqrt(3.0) * 2.7).astype(int)
    base_lobes = np.rint(base_centres + np.sqrt(3.0) * 3.0).astype(int)
    data = np.asarray(
        [
            _ricker(axis, 40.0, 2.3, 1.6)
            + _ricker(axis, asphalt_centres[row], 2.7, 1.1)
            - _ricker(axis, base_centres[row], 3.0, 0.72)
            for row in range(rows)
        ]
    )
    data += np.random.default_rng(91).normal(0.0, 0.015, data.shape)
    seed_rows = (8, 60, 111)
    layers = [
        LayerSpec(1, "Asphalt", 25, 70, 5),
        LayerSpec(2, "Base", 60, 120, 12),
    ]
    anchors = {
        1: {row: int(asphalt_wrong_lobe[row]) for row in seed_rows},
        2: {row: int(base_lobes[row]) for row in seed_rows},
    }

    paths = pick_interfaces(data, 40, layers, anchor_samples=anchors)

    visible = paths[2].samples >= 0
    assert np.mean(visible) > 0.82
    assert np.mean(np.abs(paths[2].samples[visible] - base_lobes[visible])) < 2.5


def test_weak_seeded_base_packet_is_retained_beside_stronger_ringing():
    rows, sample_count = 110, 220
    axis = np.arange(sample_count, dtype=float)
    true_centres = 151.0 + 5.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    true_lobes = np.rint(true_centres + np.sqrt(3.0) * 3.0).astype(int)
    data = np.asarray(
        [
            _ricker(axis, 40.0, 2.3, 1.7)
            + _ricker(axis, 82.0, 2.7, 1.2)
            + _ricker(axis, 111.0, 2.8, 1.65)
            - _ricker(axis, true_centres[row], 3.0, 0.38)
            for row in range(rows)
        ]
    )
    data += np.random.default_rng(221).normal(0.0, 0.012, data.shape)
    seeds = (8, 55, 101)
    layers = [
        LayerSpec(1, "Asphalt", 25, 70, 5),
        LayerSpec(2, "Base", 60, 160, 12),
    ]
    anchors = {
        1: {row: 82 for row in seeds},
        2: {row: int(true_lobes[row]) for row in seeds},
    }

    path = pick_interfaces(data, 40, layers, anchor_samples=anchors)[2]

    packet_map = path.candidate_components["event_canonical_sample"]
    retained = np.asarray(
        [
            np.any(
                np.isfinite(packet_map[row])
                & (np.abs(packet_map[row] - true_lobes[row]) <= 7.0)
            )
            for row in range(rows)
        ]
    )
    assert np.mean(retained) > 0.95
    visible = path.samples >= 0
    assert np.mean(visible) > 0.80
    assert np.mean(np.abs(path.samples[visible] - true_lobes[visible])) < 3.0
