from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.checkpoints import evaluate_retention_audit
from gpr_layer_audit.models import (
    CandidateEvent,
    InterfacePick,
    LayerSpec,
    PickStatus,
    SeedStation,
    TrackingEvidence,
    ValidationCheckpoint,
    VisibilityState,
)
from gpr_layer_audit.processing.joint_graph import joint_family_beam
from gpr_layer_audit.processing.pipeline import (
    _anchor_rows,
    _boundary_conditions,
    _candidate_events,
    _seed_metadata_rows,
)
from gpr_layer_audit.processing.seed_graph import _joint_states_at_row
from gpr_layer_audit.processing.seed_guidance import SeedGuide, derive_seed_guide


def _workspace(order, samples, emissions, anchors=None):
    samples = np.asarray(samples, dtype=np.int32)
    shape = samples.shape
    table = SimpleNamespace(
        samples=samples,
        canonical_samples=samples.astype(float),
        valid=samples >= 0,
        phase_classes=np.zeros(shape, dtype=np.int8),
        polarities=np.ones(shape, dtype=np.int8),
        family_indices=np.tile(np.arange(shape[1]), (shape[0], 1)),
        regime_indices=np.zeros(shape, dtype=np.int16),
        waveforms=np.ones((*shape, 7), dtype=np.float32),
    )
    return SimpleNamespace(
        table=table,
        emissions=np.asarray(emissions, dtype=float),
        anchors=anchors or {},
        layer=LayerSpec(order, str(order), 1, 100, 5),
    )


def _one_row_gap_guide(value):
    guide = derive_seed_guide(2, {0: value, 1: value}, mode="gap")
    assert guide is not None
    return SeedGuide(
        guide.values[:1],
        guide.uncertainty[:1],
        guide.extrapolated[:1],
        np.asarray([], dtype=int),
        np.asarray([], dtype=float),
        "gap",
    )


def test_joint_search_selects_combinations_not_independent_complete_paths():
    # The independently best upper layer crosses the strongest lower layer.
    # The joint search must choose a different upper event, keeping the base.
    top = _workspace(1, [[40, 80, -1]] * 12, [[0.9, 1.0, -3]] * 12)
    base = _workspace(2, [[60, 95, -1]] * 12, [[1.0, 0.1, -3]] * 12)
    paths, scores = joint_family_beam([top, base], set(), 0.4)
    assert len(paths) > 1 and np.all(np.diff(scores) <= 0)
    assert np.all(paths[0][1] == 40)
    assert np.all(paths[0][2] == 60)
    for hypothesis in paths:
        visible = (hypothesis[1] >= 0) & (hypothesis[2] >= 0)
        assert np.all(hypothesis[2][visible] >= hypothesis[1][visible] + 5)


def test_joint_seed_is_hard_and_canonical_crossing_is_infeasible():
    top = _workspace(1, [[40, 80, -1]] * 4, [[0.9, 1, -np.inf]] * 4, {2: 40})
    base = _workspace(2, [[60, 95, -1]] * 4, [[1, 0.1, -np.inf]] * 4)
    # Display lobes are ordered but the second base candidate's event is not.
    base.table.canonical_samples[:, 1] = 35
    paths, _ = joint_family_beam([top, base], set(), 0.4)
    assert all(path[1][2] == 40 for path in paths)
    assert all(np.all(path[2] == 60) for path in paths)


def test_joint_hypotheses_keep_distinct_prefixes_after_confirmed_anchor():
    # Two equally supported ridges converge on one final confirmed event.
    # Keeping only the best predecessor per destination fabricates certainty
    # about the ambiguous prefix even though the anchor resolves only its row.
    samples = [[100 + row * 2, 120 - row * 2, -1] for row in range(5)]
    samples.extend([[110, 110, -1]] * 3)
    workspace = _workspace(2, samples, [[1, 1, -np.inf]] * 8, {5: 110, 7: 110})
    workspace.table.family_indices[:, :2] = 0
    # The converged rows contain one physical candidate, not duplicate lobes.
    workspace.emissions[5:, 1] = -np.inf
    paths, scores = joint_family_beam([workspace], set(), 1.0, beam_size=32, top_n=32)
    prefixes = {tuple(path[2][:5]) for path in paths}
    assert len(prefixes) > 2
    assert all(np.all(path[2][5:] == 110) for path in paths)
    assert {paths[0][2][0], paths[1][2][0]} == {100, 120}
    assert scores[0] == pytest.approx(scores[1])


def test_joint_missing_state_is_independent_for_each_layer():
    top = _workspace(1, [[40, -1]] * 8, [[1, -np.inf]] * 8)
    base = _workspace(2, [[80, -1]] * 8, [[1, -2]] * 8)
    base.emissions[3:5] = [-np.inf, 0]
    base.table.valid[3:5, 0] = False
    paths, _ = joint_family_beam([top, base], {3, 5}, 0.4)
    assert np.all(paths[0][1] == 40)
    assert np.all(paths[0][2][3:5] == -1)


def test_three_layer_beam_retains_weak_guide_compatible_lookahead_states():
    rows = 8
    top_samples = [*range(40, 50), 80, -1]
    base_samples = [*range(60, 70), 120, -1]
    subbase_samples = [*range(80, 90), 170, -1]
    emissions = [[0.50] * 10 + [0.40, -3.0]] * rows
    top = _workspace(1, [top_samples] * rows, emissions)
    base = _workspace(2, [base_samples] * rows, emissions)
    subbase = _workspace(3, [subbase_samples] * rows, emissions)
    base.gap_seed_guide = derive_seed_guide(rows, {0: 40, rows - 1: 40}, mode="gap")
    subbase.gap_seed_guide = derive_seed_guide(
        rows, {0: 50, rows - 1: 50}, mode="gap"
    )
    for workspace in (top, base, subbase):
        workspace.radar_emissions = workspace.emissions.copy()
        workspace.absolute_seed_guide = None

    states, _ = _joint_states_at_row([top, base, subbase], 3, maximum_states=128)
    assert len(states) <= 128
    assert np.any(np.all(states == np.asarray([10, 10, 10]), axis=1))

    paths, _ = joint_family_beam(
        [top, base, subbase], set(), 0.4, beam_size=64, top_n=8
    )
    assert np.all(paths[0][1] == 80)
    assert np.all(paths[0][2] == 120)
    assert np.all(paths[0][3] == 170)


def test_joint_geometry_has_one_total_budget_and_noncontiguous_layers_fail():
    top = _workspace(1, [[40]], [[0.0]])
    base = _workspace(2, [[100]], [[0.0]])
    subbase = _workspace(3, [[180]], [[0.0]])
    base.gap_seed_guide = _one_row_gap_guide(20)
    subbase.gap_seed_guide = _one_row_gap_guide(20)
    states, scores = _joint_states_at_row([top, base, subbase], 0, maximum_states=8)
    assert states.shape == (1, 3)
    assert scores[0] == pytest.approx(-0.35)

    with pytest.raises(ValueError, match="contiguous"):
        joint_family_beam([top, subbase], set(), 0.4)


def test_gap_scoring_uses_canonical_coordinates_and_supersedes_absolute_penalty():
    top = _workspace(1, [[40, 80]], [[0.0, 0.0]])
    base = _workspace(2, [[100, 120]], [[-0.35, -0.35]])
    top.table.canonical_samples[:] = [[40, 70]]
    base.table.canonical_samples[:] = [[110, 110]]
    top.radar_emissions = np.zeros((1, 2))
    base.radar_emissions = np.zeros((1, 2))
    base.gap_seed_guide = _one_row_gap_guide(40)

    states, scores = _joint_states_at_row([top, base], 0, maximum_states=16)
    score_by_state = {tuple(state): score for state, score in zip(states, scores, strict=True)}

    # Display gaps favour (40, 100), but canonical gaps favour (80, 120).
    assert score_by_state[(1, 1)] == pytest.approx(0.0)
    assert score_by_state[(0, 0)] == pytest.approx(-0.35)


def test_local_window_does_not_move_distant_seeds_to_endpoints():
    chainage = np.arange(100.0, 110.1, 0.1)
    anchors = {2: [(0, 200), (105, 250), (999, 300)]}
    stations = [
        SeedStation(str(x), x, {2: y}, {2: VisibilityState.VISIBLE}, user_confirmed={2: True})
        for x, y in anchors[2]
    ]
    assert _anchor_rows(anchors, chainage) == {2: {50: 250}}
    assert list(_seed_metadata_rows(stations, chainage)[2]) == [50]


def _pick(sample=150, status=PickStatus.HIGH_CONFIDENCE):
    return InterfacePick(
        2,
        "Base",
        0,
        10.0,
        sample,
        2.0,
        -1.0,
        0.8,
        status,
        selected_lobe_sample=155 if sample >= 0 else None,
        evidence=TrackingEvidence(graph_selected_sample=155, pre_gate_confidence=0.8),
    )


def test_refinement_boundary_keeps_clicked_lobe_and_canonical_offset():
    result = SimpleNamespace(picks=[_pick()], chainage_m=np.array([9.6, 10, 10.4]))
    anchors, metadata = {}, {}
    _boundary_conditions(result, np.array([10.0, 20.0]), anchors, metadata)
    assert anchors == {2: {0: 155}}
    assert metadata[2][0]["canonical_sample_index"] == 150
    # An unresolved/missing boundary must not be replaced by a faraway pick.
    assert 1 not in anchors[2]


def test_retention_audit_distinguishes_selection_from_final_visibility_gate():
    pick = _pick(-1, PickStatus.UNRESOLVED)
    result = SimpleNamespace(
        picks=[pick],
        chainage_m=np.array([10.0]),
        search_corridors={},
        candidate_events=[CandidateEvent(2, 10, 155, 2, 0.7, 0, -1)],
    )
    checkpoint = ValidationCheckpoint(
        "held-out",
        2,
        10.0,
        155,
        VisibilityState.VISIBLE,
        canonical_sample_index=150,
        user_confirmed=True,
    )
    record = evaluate_retention_audit(result, [checkpoint])[0]
    assert record.candidate_generated and record.graph_selected
    assert not record.visible and not record.accepted
    assert record.loss_stage == "confidence_or_visibility_gate"
    pick.evidence.graph_selected_sample = 200
    rejected = evaluate_retention_audit(result, [checkpoint])[0]
    assert rejected.loss_stage == "graph_or_ranker_selection"


def test_candidate_export_uses_solver_rank_and_does_not_reprune_lobes():
    components = {
        "event_canonical_sample": np.full((1, 80), np.nan),
        "audit_candidate_rank": np.full((1, 80), np.nan),
        "audit_candidate_score": np.full((1, 80), np.nan),
        "audit_graph_selected": np.zeros((1, 80)),
    }
    components["event_canonical_sample"][0, [40, 42]] = [35, 37]
    components["audit_candidate_rank"][0, [40, 42]] = [2, 1]
    components["audit_candidate_score"][0, [40, 42]] = [0.3, 0.7]
    components["audit_graph_selected"][0, 42] = 1
    feature = np.zeros((1, 80))
    feature[0, 40] = 1  # Generic intensity disagrees with the learned ranker.
    path = SimpleNamespace(candidate_components=components, feature=feature, evidence={})
    events = _candidate_events({2: path}, np.ones((1, 80)), np.array([10]), {})
    assert [(event.sample_index, event.rank) for event in events] == [(42, 1), (40, 2)]
    assert events[0].graph_selected
    assert [event.radar_score for event in events] == [0.7, 0.3]


def test_joint_rejects_inconsistent_hard_seeds_instead_of_silent_null():
    top = _workspace(1, [[80, -1]], [[1, -np.inf]], {0: 80})
    base = _workspace(2, [[60, -1]], [[1, -np.inf]], {0: 60})
    with pytest.raises(ValueError, match="No ordered joint candidates"):
        joint_family_beam([top, base], set(), 0.4)


def test_actual_seed_dropout_demotes_without_changing_selected_event():
    from gpr_layer_audit.processing.reliability import apply_seed_dropout_check

    pick = _pick()
    alternative_pick = _pick(190)
    alternative_pick.selected_lobe_sample = 195
    result = SimpleNamespace(picks=[pick], parameters={})
    alternative = SimpleNamespace(picks=[alternative_pick])
    before = (pick.sample_index, pick.selected_lobe_sample, pick.twtt_ns)
    audit = apply_seed_dropout_check(result, alternative, "withheld")
    assert audit[0]["demoted_picks"] == 1
    assert pick.status == PickStatus.REVIEW
    assert pick.evidence.drop_seed_stability == 0
    assert pick.competing_family_sample == 195
    assert pick.competing_family_id.startswith("dropout:withheld:")
    assert before == (pick.sample_index, pick.selected_lobe_sample, pick.twtt_ns)
    # A later agreeing run cannot erase a previously observed dependency.
    apply_seed_dropout_check(result, SimpleNamespace(picks=[_pick()]), "another")
    assert pick.evidence.drop_seed_stability == 0


def test_seed_dropout_identifies_the_withheld_manual_station_for_reconfirmation():
    from gpr_layer_audit.processing.reliability import apply_seed_dropout_check

    station = SeedStation(
        "suspect",
        10.0,
        {2: 155.0},
        {2: VisibilityState.VISIBLE},
        user_confirmed={2: True},
    )
    result = SimpleNamespace(picks=[_pick()], parameters={})
    independent = _pick(190)
    independent.selected_lobe_sample = 195
    alternative = SimpleNamespace(picks=[independent])

    audit = apply_seed_dropout_check(result, alternative, station)

    assert audit[0]["station_inconsistent"]
    assert audit[0]["station_chainage_m"] == 10.0
    assert audit[0]["withheld_manual_sample"] == 155
    assert audit[0]["independent_sample"] == 195


def test_radargram_export_cannot_join_across_missing_or_review_rows():
    from gpr_layer_audit.export.audit import _overlay_series

    picks = [_pick(), _pick(-1, PickStatus.UNRESOLVED), _pick(), _pick(status=PickStatus.REVIEW)]
    values = _overlay_series(picks, {PickStatus.HIGH_CONFIDENCE})
    assert values[[0, 2]].tolist() == [155, 155]
    assert np.all(np.isnan(values[[1, 3]]))


def test_accuracy_default_grid_does_not_coarsen_long_roads():
    from gpr_layer_audit.processing.pipeline import _effective_stack

    assert _effective_stack(20_000, 0, 0.025) == 16
    assert _effective_stack(2_000_000, 0, 0.025) == 16


def test_explicit_exhaustive_fine_mode_refines_entire_uncertain_span():
    from gpr_layer_audit.processing.pipeline import _automatic_fine_windows

    result = SimpleNamespace(
        chainage_m=np.arange(1001),
        review_issues=[SimpleNamespace(start_chainage_m=0, end_chainage_m=1000)],
    )
    windows = _automatic_fine_windows(result, None)
    assert len(windows) > 3
    assert windows[0][0] == 0 and windows[-1][1] == 1000
    assert all(right[0] <= left[1] for left, right in zip(windows, windows[1:], strict=False))


def test_interactive_default_refines_one_local_high_information_window():
    from gpr_layer_audit.processing.pipeline import (
        AnalysisOptions,
        _automatic_fine_windows,
    )

    result = SimpleNamespace(
        chainage_m=np.arange(1001),
        review_issues=[
            SimpleNamespace(
                start_chainage_m=0,
                end_chainage_m=1000,
                suggested_chainage_m=640,
            )
        ],
    )
    maximum = AnalysisOptions().max_auto_fine_regions
    assert maximum == 1
    assert _automatic_fine_windows(result, maximum) == [(635.0, 645.0)]
    result.review_issues[0].suggested_chainage_m = 0
    assert _automatic_fine_windows(result, maximum) == [(5.0, 15.0)]


def test_joint_graph_stable_top_k_matches_full_stable_sort():
    from gpr_layer_audit.processing.joint_graph import _stable_top_indices

    rng = np.random.default_rng(8128)
    values = rng.integers(-5, 6, size=(257, 67)).astype(float)
    values.ravel()[::101] = -np.inf
    values.ravel()[::307] = np.nan
    for count in (0, 1, 7, 128, values.size):
        flat = values.ravel()
        expected = np.argsort(-flat, kind="stable")
        expected = expected[np.isfinite(flat[expected])][:count]
        assert np.array_equal(_stable_top_indices(values, count), expected)


def test_seed_dropout_refits_the_same_fine_workflow_without_duplicate_anchor_leakage():
    from gpr_layer_audit.processing.pipeline import AnalysisOptions, _seed_dropout_options

    station = SeedStation("withheld", 10, {2: 150}, {2: VisibilityState.VISIBLE},
                          user_confirmed={2: True})
    options = AnalysisOptions(
        seed_stations=[station],
        anchors={2: [(10, 150), (20, 160)]},
        auto_fine_retrack=True,
        max_auto_fine_regions=None,
    )
    child = _seed_dropout_options(options, station, [(4.0, 24.0)])
    assert child.auto_fine_retrack and child.max_auto_fine_regions is None
    assert child.fine_retrack_windows_m == [(4.0, 24.0)]
    assert not child.validate_seed_dropout
    assert child.seed_stations == [] and child.anchors == {2: [(20, 160)]}
    assert options.seed_stations == [station] and len(options.anchors[2]) == 2


def test_seed_on_packet_edge_is_retained_once_even_far_from_canonical_peak():
    from gpr_layer_audit.processing.seed_graph import _packet_representatives

    generic, identity, reflectivity = (np.zeros(160) for _ in range(3))
    generic[110] = identity[110] = 1
    reflectivity[119] = 1
    packets = _packet_representatives(
        np.asarray([100, 110, 119, 125]),
        generic,
        reflectivity,
        identity,
        100,
        7,
    )
    assert sum(packet[0] == 100 for packet in packets) == 1
    seeded = next(packet for packet in packets if packet[0] == 100)
    assert seeded[1] == 119 and 100 in seeded[2]


def test_benchmark_requires_checkpoints_for_every_target_road_and_layer():
    from gpr_layer_audit.benchmark import _missing_checkpoint_pairs

    cases = [{"case_id": "road-a"}, {"case_id": "road-b"}]
    assert len(_missing_checkpoint_pairs(cases, [])) == 4
    checkpoints = [
        {"case_id": "road-a", "layer_order": 1, "checkpoints": 30},
        {"case_id": "road-a", "layer_order": 2, "checkpoints": 29},
    ]
    missing = _missing_checkpoint_pairs(cases, checkpoints)
    assert len(missing) == 3
    assert {"case_id": "road-a", "layer_order": 2} in missing
