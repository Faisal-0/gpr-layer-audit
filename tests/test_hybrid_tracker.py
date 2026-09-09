from __future__ import annotations

import numpy as np
import pytest

from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.hybrid import constrained_dtw
from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence, LearnedEvidence
from gpr_layer_audit.processing.tracker import pick_interfaces


def scene(rows=55, samples=128, *, gap=False, ringing=False, drift=False):
    x = np.arange(samples)
    data = np.zeros((rows, samples), np.float32)
    targets = []
    for row in range(rows):
        centre = 65 + (round(3 * np.sin(row / 20)) if drift else 0)
        z = (x - centre) / 2.2
        data[row] = -(1 - z * z) * np.exp(-z * z / 2)
        if ringing:
            z = (x - 90) / 2.2
            data[row] -= 4 * (1 - z * z) * np.exp(-z * z / 2)
        targets.append(centre)
    if gap:
        data[23:28] = 0
    return data, targets


def run(data, targets, *, rows=(3, 18, 45), **kwargs):
    return pick_interfaces(
        data,
        15,
        [LayerSpec(2, "Base", 20, 95, 4)],
        anchor_samples={2: {r: targets[r] for r in rows}},
        method="seed_hybrid",
        horizontal_step_m=0.4,
        **kwargs,
    )[2]


def test_reciprocal_dtw_detects_opposite_waveform():
    wave = np.array([0, -0.2, -1, -0.2, 0.0])
    assert constrained_dtw(wave, wave, 1)[0] == 1
    assert constrained_dtw(wave, -wave, 1)[0] < 0.75


@pytest.mark.parametrize("ringing,drift", [(False, False), (True, False), (True, True)])
def test_seed_identity_survives_stronger_competing_reflector(ringing, drift):
    data, expected = scene(ringing=ringing, drift=drift)
    result = run(data, expected)
    assert all(result.samples[row] == expected[row] for row in (3, 18, 45))
    accepted = result.visible
    assert np.count_nonzero(accepted) >= len(data) * 0.4
    assert np.all(abs(result.samples[accepted] - np.asarray(expected)[accepted]) <= 2)


def test_gap_reacquires_but_does_not_export_measurement():
    data, expected = scene(gap=True)
    result = run(data, expected, rows=(3, 18))
    assert np.all(result.samples[23:28] == -1)
    assert not np.any(result.interpolated)
    assert np.count_nonzero(result.visible[28:]) >= 10


def test_break_prevents_seed_identity_transfer():
    data, expected = scene()
    result = run(data, expected, rows=(3, 18), break_rows={28})
    assert not np.any(result.visible[28:])


def test_ml_cannot_create_measurements_in_absent_signal():
    data = np.zeros((30, 128), np.float32)
    learned = LearnedEvidence(np.ones_like(data), np.ones(30), np.ones_like(data, bool), 2, 0.35)
    evidence = HybridEvidence(data, 0.05, 0.4, {2: learned})
    result = run(data, [65] * 30, rows=(3, 18), hybrid_evidence=evidence)
    assert set(np.flatnonzero(result.visible)).issubset({3, 18})


def test_model_policy_failures_and_recorded_fallback(tmp_path):
    data, expected = scene()
    result = run(data, expected, model_path=str(tmp_path / "missing"))
    assert result.provenance["ml_status"] == "unavailable"
    with pytest.raises(ValueError, match="requires a model"):
        run(data, expected, ml_policy="require")


def test_cancel_interrupts_before_dense_work():
    data, expected = scene()
    with pytest.raises(InterruptedError):
        run(data, expected, cancel=lambda: True)


def test_evidence_grid_mismatch_rejected():
    data, expected = scene()
    with pytest.raises(ValueError, match="match"):
        run(data, expected, hybrid_evidence=HybridEvidence(data[:2], 0.05, 0.4))


def test_no_crossing_with_seeded_upper_interface():
    data, expected = scene()
    x = np.arange(128)
    z = (x - 45) / 2
    data += ((1 - z * z) * np.exp(-z * z / 2))[None, :]
    layers = [LayerSpec(2, "Base", 15, 70, 4), LayerSpec(3, "Subbase", 25, 95, 4)]
    paths = pick_interfaces(
        data,
        15,
        layers,
        method="seed_hybrid",
        anchor_samples={2: {3: 45, 45: 45}, 3: {3: 65, 45: 65}},
    )
    both = paths[2].visible & paths[3].visible
    assert np.all(paths[3].samples[both] >= paths[2].samples[both] + 4)


def test_global_pipeline_preserves_hybrid_contract(synthetic_acquisition):
    from gpr_layer_audit.models import AcquisitionFileSet
    from gpr_layer_audit.processing.pipeline import AnalysisOptions, analyze_acquisition

    road, plate, _ = synthetic_acquisition
    options = AnalysisOptions(
        tracker_method="seed_hybrid",
        stack_size=12,
        validate_seed_dropout=False,
        auto_fine_retrack=False,
        layer_specs=LayerSpec.defaults()[:2],
        anchors={1: [(1.0, 97.0), (20.0, 97.0)], 2: [(1.0, 134.0), (20.0, 134.0)]},
    )
    result = analyze_acquisition(AcquisitionFileSet(road), AcquisitionFileSet(plate), options)
    assert result.parameters["tracker_method"] == "seed_hybrid"
    assert "2" in result.parameters["hybrid_provenance"]
    for pick in result.picks:
        if pick.layer_order == 2 and pick.selected_lobe_sample is None:
            assert not np.isfinite(pick.twtt_ns)
    assert any("correspondence" in name for name in result.display_radargrams)


def test_independent_seeded_regions_continue_on_both_sides_of_break():
    data, expected = scene()
    result = run(data, expected, rows=(3, 45), break_rows={28})
    assert np.count_nonzero(result.visible[:28]) >= 15
    assert np.count_nonzero(result.visible[28:]) >= 15


def test_contradictory_identity_seeds_do_not_force_interpolated_switch():
    data, expected = scene(ringing=True)
    expected[45] = 90
    result = run(data, expected, rows=(3, 45))
    assert result.samples[3] == 65 and result.samples[45] == 90
    assert not np.any(result.interpolated)
    assert not np.any((result.samples > 67) & (result.samples < 88))


@pytest.mark.parametrize("offset", [12, 18, 25])
def test_weak_seeded_reflector_with_gain_noise_and_width_drift(offset):
    rng = np.random.default_rng(42)
    axis = np.arange(128)
    expected = np.rint(62 + 2 * np.sin(np.arange(90) / 24)).astype(int)
    data = []
    for row, sample in enumerate(expected):
        width = 2.0 + 0.35 * np.sin(row / 35)
        target = (axis - sample) / width
        ringing = (axis - sample - offset) / 2.2
        trace = -0.30 * (1 - target**2) * np.exp(-(target**2) / 2)
        trace -= (1 - ringing**2) * np.exp(-(ringing**2) / 2)
        trace += rng.normal(0, 0.008, len(axis))
        data.append(trace * (1 + 0.5 * np.sin(row / 20)))
    result = run(np.asarray(data, np.float32), expected, rows=(3, 42, 85))
    assert np.mean(result.visible) >= 0.5
    assert np.all(abs(result.samples[result.visible] - expected[result.visible]) <= 2)


def test_abrupt_seeded_regime_change_does_not_bridge_structural_break():
    data, expected = scene()
    x = np.arange(data.shape[1])
    for row in range(28, len(data)):
        z = (x - 82) / 2.2
        data[row] = (1 - z * z) * np.exp(-z * z / 2)
        expected[row] = 82
    result = run(data, expected, rows=(3, 45), break_rows={28})
    assert np.mean(result.visible) >= 0.7
    assert np.all(abs(result.samples[result.visible] - np.asarray(expected)[result.visible]) <= 2)
    assert not np.any(result.interpolated)


def test_segment_contraction_preserves_overlapping_nonlocal_route():
    from types import SimpleNamespace

    from gpr_layer_audit.processing.hybrid import _contract_links, solve_segments

    # A valid route leaves A before its endpoint and enters B after its start.
    # Contracting the full chains would incorrectly discard A -> B as overlapping.
    samples = np.tile([60, 64], (9, 1))
    table = SimpleNamespace(samples=samples, valid=np.ones_like(samples, bool))
    links = [((r, c), (r + 1, c), 0.98, 1) for r in range(8) for c in (0, 1)]
    links.append(((3, 0), (5, 1), 0.95, 2))
    following = {a: b for a, b, _, gap in links if gap == 1}
    preceding = {b: a for a, b, _, gap in links if gap == 1}
    anchors = {1: 60, 7: 64}
    segments, membership = _contract_links(table, anchors, links, following, preceding)
    edges = {}
    for a, b, quality, _ in links:
        left, right = membership[a], membership[b]
        if left != right:
            assert segments[left].stop < segments[right].start
            edges[left, right] = quality
    paths = solve_segments(
        segments, edges, np.ones(len(segments)), table, np.ones(samples.shape), anchors, set(), 7
    )
    assert np.all(paths[0][:4] == 60)
    assert paths[0][4] == -1
    assert np.all(paths[0][5:] == 64)


def dropout_scene(gap, *, target_returns=True, offset=35):
    axis = np.arange(180)
    truth = np.rint(80 + 5 * np.sin(np.arange(180) / 50)).astype(int)
    data = []
    for row, centre in enumerate(truth):
        target = (axis - centre) / 2.2
        ringing = (axis - 80 - offset) / 2.2
        trace = -2 * (1 - ringing**2) * np.exp(-(ringing**2) / 2)
        if target_returns or row < 65:
            trace -= 0.3 * (1 - target**2) * np.exp(-(target**2) / 2)
        data.append(trace)
    data = np.asarray(data, np.float32)
    data[65 : 65 + gap] = 0
    return data, truth


@pytest.mark.parametrize("gap", [5, 15, 35])
def test_nonlocal_reacquisition_improves_correct_coverage_over_local_tracker(gap):
    data, truth = dropout_scene(gap)
    outputs = {}
    for method in ("joint_seed_adaptive", "seed_hybrid"):
        outputs[method] = pick_interfaces(
            data,
            15,
            [LayerSpec(2, "Base", 20, 150, 4)],
            anchor_samples={2: {r: int(truth[r]) for r in (3, 25, 50)}},
            method=method,
            horizontal_step_m=0.4,
        )[2]
    combined, established = outputs["seed_hybrid"], outputs["joint_seed_adaptive"]
    assert not np.any(combined.visible[65 : 65 + gap])
    assert np.all(abs(combined.samples[combined.visible] - truth[combined.visible]) <= 2)
    assert np.sum(combined.visible[65 + gap :]) >= np.sum(established.visible[65 + gap :]) + 30


@pytest.mark.parametrize("offset", [10, 14, 20])
def test_nonlocal_reacquisition_cannot_replace_disappeared_target_with_ringing(offset):
    data, truth = dropout_scene(15, target_returns=False, offset=offset)
    path = pick_interfaces(
        data,
        15,
        [LayerSpec(2, "Base", 20, 150, 4)],
        anchor_samples={2: {r: int(truth[r]) for r in (3, 25, 50)}},
        method="seed_hybrid",
        horizontal_step_m=0.4,
    )[2]
    assert not np.any(path.visible[65:])
    assert path.samples[50] == truth[50]
    if offset == 10:
        assert any(w["row"] == 50 for w in path.provenance["seed_identity_warnings"])


def test_filtered_sign_reversal_does_not_change_measurement_lobe_identity():
    measurement, expected = scene()
    filtered = measurement.copy()
    filtered[20:35] *= -1
    result = run(
        filtered,
        expected,
        rows=(3, 18, 45),
        hybrid_evidence=HybridEvidence(measurement, 0.05, 0.4),
        seed_metadata={
            2: {
                row: {"polarity": 1, "phase_class": 0, "analytic_phase_rad": 0}
                for row in (3, 18, 45)
            }
        },
    )
    assert np.mean(result.visible[20:35]) >= 0.8
    assert np.all(result.samples[result.visible] == np.asarray(expected)[result.visible])
    assert np.all(result.evidence["selected_lobe_code"][result.visible] == 1)


def test_path_score_does_not_reward_fragmenting_a_weaker_reflector():
    from types import SimpleNamespace

    from gpr_layer_audit.processing.hybrid import ReflectorSegment, solve_segments

    samples = np.tile([60, 75], (6, 1))
    table = SimpleNamespace(samples=samples, valid=np.ones_like(samples, bool))
    segments = [
        ReflectorSegment(0, [(0, 0)], {0}),
        ReflectorSegment(1, [(r, 0) for r in range(1, 6)]),
    ]
    segments += [ReflectorSegment(r + 1, [(r, 1)]) for r in range(1, 6)]
    edges = {(0, 1): 1.0, (0, 2): 1.0, **{(r, r + 1): 1.0 for r in range(2, 6)}}
    scores = np.tile([1.2, 1.0], (6, 1))
    path = solve_segments(
        segments, edges, np.ones(len(segments)), table, scores, {0: 60}, set(), 7
    )[0]
    assert np.all(path == 60)


@pytest.mark.parametrize("contracted", [True, False])
def test_path_scoring_preserves_transition_cost_under_segment_contraction(contracted):
    from types import SimpleNamespace

    from gpr_layer_audit.processing.hybrid import ReflectorSegment, solve_segments

    table = SimpleNamespace(samples=np.tile([60, 75], (6, 1)), valid=np.ones((6, 2), bool))
    segments = [ReflectorSegment(0, [(0, 0)], {0})]
    edges = {}
    for candidate, quality in ((0, 0.8), (1, 1.0)):
        if contracted:
            index = len(segments)
            segments.append(
                ReflectorSegment(
                    index, [(r, candidate) for r in range(1, 6)], transition_cost=4 * (1 - quality)
                )
            )
            edges[0, index] = quality
        else:
            previous = 0
            for row in range(1, 6):
                index = len(segments)
                segments.append(ReflectorSegment(index, [(row, candidate)]))
                edges[previous, index] = quality
                previous = index
    scores = np.tile([1.1, 1.0], (6, 1))
    path = solve_segments(
        segments, edges, np.ones(len(segments)), table, scores, {0: 60}, set(), 7
    )[0]
    assert np.all(path[1:] == 75)


def test_correspondence_support_cannot_bypass_a_contradictory_manual_seed():
    from types import SimpleNamespace

    from gpr_layer_audit.processing.hybrid import correspondence_graph

    samples = np.tile([60, 75], (8, 1))
    wave = np.array([0.0, -0.2, -1.0, -0.2, 0.0], np.float32)
    wave /= np.linalg.norm(wave)
    table = SimpleNamespace(
        samples=samples,
        valid=np.ones_like(samples, bool),
        waveforms=np.tile(wave, (8, 2, 1)),
        polarities=-np.ones_like(samples),
        phase_classes=np.full_like(samples, 4),
        component_maps={},
        dense_radar_score=np.zeros((8, 100), np.float32),
    )
    _, _, _, support = correspondence_graph(table, {0: 75, 3: 60}, set(), 7, 0.4)
    assert support[1, 75] > 0.8
    assert support[7, 60] > 0.8
    assert support[7, 75] == 0


@pytest.mark.parametrize("separate_lobe", [False, True])
def test_correspondence_margin_distinguishes_duplicate_timing_from_another_lobe(separate_lobe):
    from types import SimpleNamespace

    from gpr_layer_audit.processing.hybrid import correspondence_graph

    # Two feature branches locate the same broad negative lobe two samples
    # apart. Their near-identical waveform scores used to suppress both links.
    samples = np.tile([40, 42], (2, 1))
    wave = np.array([0, -0.2, -1, -0.2, 0], np.float32)
    wave /= np.linalg.norm(wave)
    table = SimpleNamespace(
        samples=samples,
        valid=np.ones_like(samples, bool),
        waveforms=np.tile(wave, (2, 2, 1)),
        polarities=-np.ones_like(samples),
        phase_classes=np.full_like(samples, 4),
        component_maps={},
        dense_radar_score=np.zeros((2, 80), np.float32),
    )
    measurement = np.zeros((2, 80), np.float32)
    measurement[:, 38:45] = -1
    if separate_lobe:
        # Same-polarity endpoints are insufficient: a positive ringing cycle
        # between them makes the second candidate a real competing reflector.
        measurement[:, 41] = 1
    _, _, _, support = correspondence_graph(
        table, {0: 40}, set(), 7, 0.4, measurement=measurement
    )
    assert support[0, 40] == 1
    assert support[1, 40] > 0.8
    # A genuine second lobe remains a route hypothesis, with an ambiguity
    # penalty; duplicate timings on one lobe do not consume another route.
    assert (support[1, 42] > 0) == separate_lobe
    if separate_lobe:
        assert support[1, 40] < 1


def test_repeated_small_correspondence_jumps_cannot_escape_nearby_seed_neighbourhood():
    from types import SimpleNamespace

    from gpr_layer_audit.processing.hybrid import correspondence_graph

    # Each 4-sample hop passes the pairwise displacement test; seven hops
    # would nevertheless transfer seed identity to an unrelated shallow band.
    samples = (80 - 4 * np.arange(8))[:, None]
    wave = np.array([0, -0.2, -1, -0.2, 0], np.float32)
    wave /= np.linalg.norm(wave)
    table = SimpleNamespace(
        samples=samples,
        valid=np.ones_like(samples, bool),
        waveforms=np.tile(wave, (8, 1, 1)),
        polarities=-np.ones_like(samples),
        phase_classes=np.full_like(samples, 4),
        component_maps={},
        dense_radar_score=np.zeros((8, 100), np.float32),
    )
    _, _, _, support = correspondence_graph(table, {0: 80}, set(), 7, 0.4)
    assert support[2, 72] > 0.8
    assert not np.any(support[3:])


def test_ranked_correspondences_reserve_three_distinct_lobes():
    from gpr_layer_audit.processing.hybrid import _ranked_lobe_matches

    scores = np.array([[1, .99, .98, .95, .92, .7], [-np.inf] * 6])
    same_lobe = np.eye(6, dtype=bool)
    same_lobe[:3, :3] = True
    selected = _ranked_lobe_matches(scores, same_lobe)
    assert np.array_equal(np.flatnonzero(selected[0]), [0, 3, 4])
    assert not np.any(selected[1])


def test_locally_second_best_connection_can_satisfy_later_seed():
    from types import SimpleNamespace

    from gpr_layer_audit.processing.hybrid import correspondence_graph, solve_segments

    samples = np.tile([60, 66], (6, 1))
    wave = np.array([0, -0.2, -1, -0.2, 0], np.float32)
    wave /= np.linalg.norm(wave)
    table = SimpleNamespace(
        samples=samples, valid=np.ones_like(samples, bool),
        waveforms=np.tile(wave, (6, 2, 1)), polarities=-np.ones_like(samples),
        phase_classes=np.full_like(samples, 4), component_maps={},
        dense_radar_score=np.zeros((6, 100), np.float32),
    )
    anchors = {0: 60, 5: 66}
    segments, edges, support, _ = correspondence_graph(table, anchors, set(), 7, .4)
    scores = np.tile([1.0, .99], (6, 1))
    diagnostics = {}
    paths = solve_segments(segments, edges, support, table, scores, anchors, set(), 7,
                           diagnostics=diagnostics)
    assert paths[0][0] == 60 and paths[0][-1] == 66
    assert np.all(paths[0] >= 0)
    assert any(np.any((p >= 0) & (p != paths[0])) for p in paths[1:])


def test_unproven_whole_trace_registration_is_inactive_by_default(monkeypatch):
    from gpr_layer_audit.processing import trace_registration

    def unexpected(*args, **kwargs):
        raise AssertionError("Unvalidated registration should be explicitly requested")

    monkeypatch.setattr(trace_registration, "sparse_registration", unexpected)
    data, expected = scene()
    result = run(data, expected)
    assert not result.provenance["whole_trace_registration"]


def test_opt_in_registration_preserves_gap_and_exact_seeds():
    data, expected = scene(gap=True, ringing=True)
    evidence = HybridEvidence(data, .05, .4, provenance={"whole_trace_registration": True})
    result = run(data, expected, rows=(3, 18), hybrid_evidence=evidence)
    assert result.provenance["whole_trace_registration"]
    assert result.samples[3] == expected[3] and result.samples[18] == expected[18]
    assert np.all(result.samples[23:28] == -1)
    assert not np.any(result.interpolated)
    assert np.all(abs(result.samples[result.visible] - np.asarray(expected)[result.visible]) <= 2)


def test_exact_route_ambiguity_withholds_nonseed_measurements(monkeypatch):
    import gpr_layer_audit.processing.hybrid as hybrid

    original = hybrid.solve_complete_intervals

    def tied_routes(*args, **kwargs):
        paths = original(*args, **kwargs)
        kwargs["diagnostics"]["path_margin"][:] = 0
        return paths

    monkeypatch.setattr(hybrid, "solve_complete_intervals", tied_routes)
    data, expected = scene()
    result = run(data, expected)
    assert set(np.flatnonzero(result.visible)) == {3, 18, 45}
    assert np.all(result.provisional_samples >= 0)
    assert not np.any(result.interpolated)
