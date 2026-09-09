from __future__ import annotations

import numpy as np
import pytest

from gpr_layer_audit.checkpoints import evaluate_retention_audit
from gpr_layer_audit.io import DZTFile
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    DielectricSource,
    LayerDesign,
    LayerSpec,
    PickStatus,
    ReviewIssue,
    SeedStation,
    ValidationCheckpoint,
    VisibilityState,
)
from gpr_layer_audit.processing import (
    AnalysisOptions,
    PreprocessingOptions,
    analyze_acquisition,
    retrack_segment,
)
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.dielectric import (
    surface_reflection_dielectric,
    thickness_from_twtt_mm,
)
from gpr_layer_audit.processing.pipeline import (
    _additional_seed_requests,
    _design_calibrated_dielectric,
    _dropout_seed_requests,
    _initial_seed_requests,
    _invalidate_design_calibration_after_dropout,
    _unknown_layer_seed_target,
    _validate_enabled_layer_sequence,
)
from gpr_layer_audit.processing.preprocessing import subtract_tracked_reflection


def test_surface_reflection_dielectric_recovers_known_value():
    epsilon = 7.0
    ratio = (np.sqrt(epsilon) - 1) / (np.sqrt(epsilon) + 1)
    output = surface_reflection_dielectric(np.asarray([-ratio * 1000] * 9), -1000)
    assert np.nanmedian(output) == pytest.approx(epsilon, rel=0.02)


def test_thickness_conversion():
    assert thickness_from_twtt_mm(2.0, 7.0) == pytest.approx(113.3, rel=0.01)


def test_enabled_interfaces_must_be_contiguous_from_asphalt_downward():
    layers = LayerSpec.defaults()
    layers[1].analysis_enabled = False

    with pytest.raises(ValueError, match="contiguous from asphalt"):
        _validate_enabled_layer_sequence(layers)

    layers[2].analysis_enabled = False
    _validate_enabled_layer_sequence(layers)


def test_design_calibrated_dielectric_requires_consistent_seed_gaps():
    stations = [
        SeedStation(
            "a",
            10.0,
            {1: 130.0, 2: 190.0},
            user_confirmed={1: True, 2: True},
        ),
        SeedStation(
            "b",
            90.0,
            {1: 131.0, 2: 192.0},
            user_confirmed={1: True, 2: True},
        ),
        SeedStation(
            "c",
            170.0,
            {1: 130.0, 2: 190.0},
            user_confirmed={1: True, 2: True},
        ),
    ]
    designs = [
        LayerDesign(1, "Asphalt", 50.0),
        LayerDesign(2, "Base course", 100.0),
    ]
    values, audit = _design_calibrated_dielectric(
        100,
        0.029296875,
        LayerSpec.defaults()[:2],
        designs,
        [],
        stations,
    )

    assert values[1] == pytest.approx(7.2, rel=0.08)
    assert values[2] == pytest.approx(7.2, rel=0.08)
    assert audit["1"]["status"] == "accepted"
    assert audit["2"]["status"] == "accepted"

    stations[1].samples[2] = 240.0
    values, audit = _design_calibrated_dielectric(
        100,
        0.029296875,
        LayerSpec.defaults()[:2],
        designs,
        [],
        stations,
    )

    assert 2 not in values
    assert audit["2"]["status"] == "rejected"


def test_dropout_contradiction_invalidates_design_calibrated_dielectric():
    dielectric = {
        1: (6.0, DielectricSource.DESIGN_CALIBRATED),
        2: (8.0, DielectricSource.DESIGN_CALIBRATED),
        3: (9.0, DielectricSource.ANALYST),
    }
    calibration = {
        "1": {"status": "accepted", "inferred_dielectric": 6.0},
        "2": {"status": "accepted", "inferred_dielectric": 8.0},
    }

    invalidated = _invalidate_design_calibration_after_dropout(
        dielectric,
        [
            {"layer_order": 1, "station_inconsistent": True},
            {"layer_order": 2, "station_inconsistent": True},
        ],
        calibration,
    )

    assert invalidated == {1, 2}
    assert dielectric[1] == (None, DielectricSource.UNRESOLVED)
    assert dielectric[2] == (None, DielectricSource.UNRESOLVED)
    assert dielectric[3] == (9.0, DielectricSource.ANALYST)
    assert calibration["1"]["reason"] == "seed_identity_dropout_failed"


def test_adaptive_subtraction_tolerates_small_path_timing_errors():
    rows, samples = 35, 180
    axis = np.arange(samples, dtype=float)
    true_centre = 82
    wave = (1.0 - ((axis - true_centre) / 3.2) ** 2) * np.exp(
        -0.5 * ((axis - true_centre) / 3.2) ** 2
    )
    data = np.tile(wave, (rows, 1)).astype(np.float32)
    path = true_centre + np.resize(np.asarray([-2, -1, 0, 1, 2]), rows)
    fixed, _, _ = subtract_tracked_reflection(
        data, path, maximum_shift_samples=0
    )
    adaptive, improvement, _ = subtract_tracked_reflection(
        data, path, maximum_shift_samples=2
    )
    window = slice(true_centre - 8, true_centre + 9)
    assert np.sum(adaptive[:, window] ** 2) <= np.sum(fixed[:, window] ** 2)
    assert np.median(np.max(improvement[:, window], axis=1)) > 0.70


def test_pipeline_tracks_interfaces_and_attaches_gps(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, report_interval_m=5, accept_scan_dielectric=True),
    )
    assert result.calibrated_radargram.shape == (60, 256)
    assert {item.layer_order for item in result.picks} == {1, 2, 3}
    visible = [item for item in result.picks if item.sample_index >= 0]
    assert visible and all(np.isnan(item.twtt_ns) for item in visible)
    assert result.parameters["required_seed_orders"]
    assert any(item.latitude is not None for item in result.picks)
    sources = {(item.layer_order, item.dielectric_source) for item in result.thickness}
    assert (1, DielectricSource.REFLECTION) in sources
    assert (2, DielectricSource.ASSUMED_SCAN) in sources
    assert (3, DielectricSource.ASSUMED_SCAN) in sources
    assert all(
        item.thickness_mm is None
        for item in result.thickness
        if item.status == PickStatus.UNRESOLVED
    )
    assert all(
        item.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.REVIEW, PickStatus.UNRESOLVED}
        for item in result.picks
    )


def test_checkpoint_retention_audit_never_changes_tracker_output(
    synthetic_acquisition,
):
    road_path, plate_path, _ = synthetic_acquisition
    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, accept_scan_dielectric=True),
    )
    event = next(item for item in result.candidate_events if item.layer_order == 2)
    before = np.asarray(
        [item.sample_index for item in result.picks if item.layer_order == 2]
    )
    checkpoint = ValidationCheckpoint(
        checkpoint_id="base-audit-001",
        layer_order=2,
        chainage_m=event.chainage_m,
        sample_index=float(event.sample_index),
        canonical_sample_index=event.canonical_sample_index,
        visibility=VisibilityState.VISIBLE,
        user_confirmed=True,
        selected_lobe=event.selected_lobe,
        pulse_width_samples=max(1.0, event.pulse_width_samples),
    )

    audit = evaluate_retention_audit(result, [checkpoint])
    after = np.asarray(
        [item.sample_index for item in result.picks if item.layer_order == 2]
    )

    assert audit[0].candidate_generated
    assert audit[0].loss_stage in {
        "retained",
        "graph_or_ranker_selection",
        "confidence_or_visibility_gate",
    }
    assert np.array_equal(before, after, equal_nan=True)


def test_unresolved_interface_does_not_manufacture_dependent_thickness(
    synthetic_acquisition,
):
    road_path, plate_path, _ = synthetic_acquisition
    options = AnalysisOptions(stack_size=4, accept_scan_dielectric=True)
    options.layer_specs[2].min_offset_samples = 180
    options.layer_specs[2].max_offset_samples = 180
    result = analyze_acquisition(
        AcquisitionFileSet(road_path), AcquisitionFileSet(plate_path), options
    )
    third_picks = [item for item in result.picks if item.layer_order == 3]
    third_results = [item for item in result.thickness if item.layer_order == 3]
    assert all(item.status == PickStatus.UNRESOLVED for item in third_picks)
    assert all(item.thickness_mm is None for item in third_results)


def test_gain_mismatch_fails_closed(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    raw = bytearray(plate_path.read_bytes())
    raw[512] = 6
    plate_path.write_bytes(raw)
    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, accept_scan_dielectric=False),
    )
    assert not result.diagnostics.valid_for_dielectric
    assert all(
        item.dielectric_source == DielectricSource.UNRESOLVED
        for item in result.thickness
    )
    assert all(item.thickness_mm is None for item in result.thickness)


def test_consistent_design_and_manual_seeds_calibrate_base_dielectric(
    synthetic_acquisition,
):
    road_path, plate_path, _ = synthetic_acquisition
    stations = [
        SeedStation(
            "a",
            5.0,
            {1: 97.0, 2: 134.0},
            user_confirmed={1: True, 2: True},
        ),
        SeedStation(
            "b",
            12.0,
            {1: 97.0, 2: 134.0},
            user_confirmed={1: True, 2: True},
        ),
        SeedStation(
            "c",
            18.0,
            {1: 97.0, 2: 134.0},
            user_confirmed={1: True, 2: True},
        ),
    ]
    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(
            stack_size=4,
            accept_scan_dielectric=False,
            layer_specs=LayerSpec.defaults()[:2],
            seed_stations=stations,
            layer_designs=[
                LayerDesign(1, "Asphalt", 50.8),
                LayerDesign(2, "Base course", 101.6),
            ],
            validate_seed_dropout=False,
        ),
    )

    base = [item for item in result.thickness if item.layer_order == 2]
    assert base
    assert all(
        item.dielectric_source == DielectricSource.DESIGN_CALIBRATED
        for item in base
    )
    assert result.parameters["design_dielectric_calibration"]["2"]["status"] == "accepted"


def test_interpretation_preprocessing_cannot_change_reflection_dielectric(
    synthetic_acquisition,
):
    road_path, plate_path, _ = synthetic_acquisition
    enabled = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, preprocessing=PreprocessingOptions(enabled=True)),
    )
    disabled = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, preprocessing=PreprocessingOptions(enabled=False)),
    )
    enabled_epsilon = next(item.dielectric for item in enabled.thickness if item.layer_order == 1)
    disabled_epsilon = next(item.dielectric for item in disabled.thickness if item.layer_order == 1)
    assert enabled_epsilon == pytest.approx(disabled_epsilon, rel=0, abs=1e-12)
    assert "zero-phase automatic spectral band-pass" in enabled.diagnostics.preprocessing_steps
    assert "zero-phase automatic spectral band-pass" not in disabled.diagnostics.preprocessing_steps


def test_anchor_retracking_is_bounded(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    options = AnalysisOptions(stack_size=4, accept_scan_dielectric=True)
    result = analyze_acquisition(
        AcquisitionFileSet(road_path), AcquisitionFileSet(plate_path), options
    )
    centre = float(result.chainage_m[len(result.chainage_m) // 2])
    before = {(item.layer_order, item.chainage_m): item.sample_index for item in result.picks}
    current = next(
        item.sample_index
        for item in result.picks
        if item.layer_order == 1 and item.chainage_m == centre
    )
    options.anchors = {1: [(centre, current + 4)]}
    retrack_segment(result, options, centre - 2.0, centre + 2.0, context_m=2.0)
    outside = [
        item
        for item in result.picks
        if item.chainage_m < centre - 2.0 or item.chainage_m > centre + 2.0
    ]
    assert all(before[(item.layer_order, item.chainage_m)] == item.sample_index for item in outside)
    anchored = next(
        item for item in result.picks if item.layer_order == 1 and item.chainage_m == centre
    )
    assert anchored.selected_lobe_sample == current + 4
    assert anchored.canonical_event_sample == anchored.sample_index
    assert any(
        event.layer_order == 1 and event.chainage_m == centre
        and event.sample_index == anchored.selected_lobe_sample
        for event in result.candidate_events
    )
    segment = result.parameters["fine_retracked_segments"][-1]
    assert segment["fine_stack_size"] == 1
    assert segment["coarse_stack_size"] == 4


@pytest.mark.parametrize("method", ["joint_seed_adaptive", "seed_hybrid"])
def test_saved_correction_replays_locally_without_becoming_a_model_seed(
    synthetic_acquisition, method,
):
    road_path, plate_path, _ = synthetic_acquisition
    dzx_path = road_path.with_suffix(".DZX")
    dzx_path.write_text(
        dzx_path.read_text(encoding="utf-8").replace(
            "<unitsPerScan>0.1</unitsPerScan>",
            "<unitsPerScan>1.0</unitsPerScan>",
        ),
        encoding="utf-8",
    )
    common = dict(
        tracker_method=method,
        stack_size=4,
        auto_fine_retrack=False,
        validate_seed_dropout=False,
    )
    baseline = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(**common),
    )
    centre = float(baseline.chainage_m[len(baseline.chainage_m) // 2])
    event = min(
        (item for item in baseline.candidate_events if item.layer_order == 2),
        key=lambda item: (
            abs(item.chainage_m - centre),
            item.rank,
        ),
    )
    correction = SeedStation(
        "local-base-correction",
        event.chainage_m,
        samples={2: event.sample_index},
        visibility={2: VisibilityState.VISIBLE},
        role="correction",
        user_confirmed={2: True},
        phase_class={2: event.phase_class},
        analytic_phase_rad={2: event.analytic_phase_rad},
        polarity={2: event.polarity},
        selected_lobe={2: event.selected_lobe},
        canonical_samples={
            2: event.canonical_sample_index or event.sample_index
        },
        pulse_width_samples={2: event.pulse_width_samples},
        event_ids={2: event.event_id},
        family_ids={2: event.event_family_id or ""},
    )

    corrected = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(seed_stations=[correction], **common),
    )

    assert corrected.parameters["required_seed_orders"] == baseline.parameters[
        "required_seed_orders"
    ]
    assert corrected.parameters["required_seed_count"] == baseline.parameters[
        "required_seed_count"
    ]
    assert corrected.parameters["provisional_independent_preview"] is True
    assert corrected.parameters["local_correction_replay"] == {
        "count": 1,
        "windows_m": [[event.chainage_m - 25.0, event.chainage_m + 25.0]],
        "policy": "bounded_plus_or_minus_25_m_after_global_fit",
        "satisfies_model_seed_requirements": False,
    }
    assert corrected.seed_stations == [correction]
    before = {
        (item.layer_order, item.chainage_m): (
            item.sample_index,
            item.status,
            item.visibility,
        )
        for item in baseline.picks
    }
    outside = [
        item
        for item in corrected.picks
        if abs(item.chainage_m - event.chainage_m) > 25.0
    ]
    assert outside
    assert all(
        before[(item.layer_order, item.chainage_m)]
        == (item.sample_index, item.status, item.visibility)
        for item in outside
    )
    corrected_pick = min(
        (item for item in corrected.picks if item.layer_order == 2),
        key=lambda item: abs(item.chainage_m - event.chainage_m),
    )
    assert corrected_pick.status == PickStatus.ACCEPTED
    assert corrected_pick.is_accepted_measurement


def test_coarse_retrack_fallback_uses_saved_tracking_input(
    synthetic_acquisition, tmp_path
):
    road_path, plate_path, _ = synthetic_acquisition
    options = AnalysisOptions(stack_size=4, auto_fine_retrack=False)
    result = analyze_acquisition(
        AcquisitionFileSet(road_path), AcquisitionFileSet(plate_path), options
    )
    centre = float(result.chainage_m[len(result.chainage_m) // 2])
    current = next(
        item.sample_index
        for item in result.picks
        if item.layer_order == 1 and item.chainage_m == centre
    )
    result.source.dzt_path = tmp_path / "missing-road.DZT"
    options.anchors = {1: [(centre, current + 3)]}

    retrack_segment(result, options, centre - 1.0, centre + 1.0, context_m=2.0)

    anchored = next(
        item for item in result.picks if item.layer_order == 1 and item.chainage_m == centre
    )
    assert anchored.selected_lobe_sample == current + 3


def test_partial_calibration_preserves_absolute_trace_centres(synthetic_acquisition):
    road_path, _, _ = synthetic_acquisition
    calibrated = calibrate(
        DZTFile(road_path),
        None,
        stack_size=4,
        start_trace=40,
        stop_trace=60,
    )

    assert calibrated.trace_centres.tolist() == [41.5, 45.5, 49.5, 53.5, 57.5]


def test_requested_seed_count_is_adaptive_and_never_exceeds_five(synthetic_acquisition):
    road_path, plate_path, expected = synthetic_acquisition
    preview = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, auto_fine_retrack=False),
    )
    stations: list[SeedStation] = []
    for index, chainage in enumerate(preview.proposed_seed_chainages, 1):
        nearest = min(
            (item for item in preview.picks if item.layer_order == 1),
            key=lambda item: abs(item.chainage_m - chainage),
        )
        trace = int(nearest.trace_index)
        samples = {order: float(expected[order][trace]) for order in (1, 2, 3)}
        stations.append(
            SeedStation(
                station_id=f"seed-{index}",
                chainage_m=chainage,
                samples=samples,
                visibility={order: VisibilityState.VISIBLE for order in samples},
                user_confirmed={order: True for order in samples},
            )
        )

    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(
            stack_size=4,
            seed_stations=stations,
            validate_seed_dropout=False,
        ),
    )

    # Base/subbase require three distributed observations so the subsequent
    # leave-one-out identity audit retains two independent training stations.
    assert len(stations) == 3
    assert all(
        request.layer_orders == [1, 2, 3]
        for request in preview.proposed_seed_requests
    )
    assert len(result.seed_stations) <= 5
    assert preview.parameters["required_seed_orders"] == [1, 2, 3]
    assert result.parameters["required_seed_orders"] == []
    assert preview.parameters["provisional_independent_preview"] is True
    assert result.parameters["provisional_independent_preview"] is False
    fine_plan = result.parameters["automatic_fine_retrack_plan"]
    assert fine_plan["policy"] == "bounded_high_information"
    assert len(fine_plan["selected_windows_m"]) <= 1
    assert fine_plan["selected_core_span_m"] <= 10.0 + 1e-9


def test_ambiguity_seed_requests_preserve_layer_and_reason():
    chainage = np.arange(0.0, 501.0, 1.0)
    issues = [
        ReviewIssue(
            "base-disconnected",
            2,
            "Base course",
            250.0,
            350.0,
            ["Candidate reflector is not connected to a confirmed seed"],
            "Inspect",
            priority=0.95,
            suggested_chainage_m=302.0,
        ),
        ReviewIssue(
            "asphalt-low",
            1,
            "Asphalt",
            70.0,
            130.0,
            ["Low survey-normalized evidence support"],
            "Inspect",
            priority=0.60,
            suggested_chainage_m=101.0,
        ),
    ]

    requests = _additional_seed_requests(
        chainage, issues, [SeedStation("existing", 0.0)], limit=2
    )

    assert [item.chainage_m for item in requests] == [302.0, 101.0]
    assert [item.layer_orders for item in requests] == [[2], [1]]
    assert "not connected" in requests[0].reason


def test_inconsistent_existing_seed_is_requested_even_at_five_station_limit():
    stations = [SeedStation(f"seed-{index}", float(index * 100)) for index in range(5)]
    result = type("Result", (), {"parameters": {"seed_dropout_audit": [{
        "station_id": "seed-2",
        "layer_order": 2,
        "station_inconsistent": True,
        "withheld_manual_sample": 243.0,
        "independent_sample": 256.0,
    }]}})()

    requests = _dropout_seed_requests(result, stations)

    assert len(requests) == 1
    assert requests[0].chainage_m == 200.0
    assert requests[0].layer_orders == [2]
    assert "256.0 instead of 243.0" in requests[0].reason
    assert AnalysisOptions().validate_seed_dropout


def test_parallel_seed_dropout_matches_serial(synthetic_acquisition):
    road_path, plate_path, expected = synthetic_acquisition
    layer_specs = LayerSpec.defaults()[:1]
    preview = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(
            stack_size=4,
            layer_specs=layer_specs,
            auto_fine_retrack=False,
            validate_seed_dropout=False,
        ),
    )
    stations = []
    for index, chainage in enumerate((5.0, 15.0, 25.0), 1):
        nearest = min(
            preview.picks,
            key=lambda item: abs(item.chainage_m - chainage),
        )
        sample = float(expected[1][nearest.trace_index])
        stations.append(
            SeedStation(
                f"seed-{index}",
                nearest.chainage_m,
                {1: sample},
                {1: VisibilityState.VISIBLE},
                user_confirmed={1: True},
            )
        )

    def run(workers: int):
        return analyze_acquisition(
            AcquisitionFileSet(road_path),
            AcquisitionFileSet(plate_path),
            AnalysisOptions(
                stack_size=4,
                layer_specs=layer_specs,
                seed_stations=stations,
                auto_fine_retrack=False,
                seed_dropout_workers=workers,
            ),
        )

    serial, parallel = run(1), run(3)

    assert serial.parameters["seed_dropout_audit"] == parallel.parameters[
        "seed_dropout_audit"
    ]
    assert [
        (item.sample_index, item.status, item.evidence.drop_seed_stability)
        for item in serial.picks
    ] == [
        (item.sample_index, item.status, item.evidence.drop_seed_stability)
        for item in parallel.picks
    ]
    assert [item.issue_id for item in serial.review_issues] == [
        item.issue_id for item in parallel.review_issues
    ]
    assert [item.source_issue_id for item in serial.proposed_seed_requests] == [
        item.source_issue_id for item in parallel.proposed_seed_requests
    ]


def test_seed_dropout_worker_count_is_bounded(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    with pytest.raises(ValueError, match="between zero and five"):
        analyze_acquisition(
            AcquisitionFileSet(road_path),
            AcquisitionFileSet(plate_path),
            AnalysisOptions(seed_dropout_workers=6),
        )
    with pytest.raises(ValueError, match="max_auto_fine_regions"):
        analyze_acquisition(
            AcquisitionFileSet(road_path),
            AcquisitionFileSet(plate_path),
            AnalysisOptions(max_auto_fine_regions=-1),
        )
    with pytest.raises(ValueError, match="fine_retrack_windows_m"):
        analyze_acquisition(
            AcquisitionFileSet(road_path),
            AcquisitionFileSet(plate_path),
            AnalysisOptions(fine_retrack_windows_m=[(20.0, 10.0)]),
        )


def test_seed_dropout_default_is_memory_safe_serial(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    stations = [
        SeedStation(
            f"seed-{index}",
            chainage,
            {1: 95.0},
            user_confirmed={1: True},
        )
        for index, chainage in enumerate((5.0, 15.0, 25.0), 1)
    ]
    layers = LayerSpec.defaults()
    layers[1].analysis_enabled = False
    layers[1].audit_enabled = False
    layers[2].analysis_enabled = False
    layers[2].audit_enabled = False

    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(
            stack_size=4,
            layer_specs=layers,
            seed_stations=stations,
            auto_fine_retrack=False,
        ),
    )

    assert result.parameters["seed_dropout_workers_used"] == 1


def test_inconsistent_seed_requests_nearby_regime_companion_when_capacity_remains():
    stations = [
        SeedStation("seed-a", 25.0),
        SeedStation("suspect", 100.0),
        SeedStation("seed-c", 175.0),
    ]
    result = type(
        "Result",
        (),
        {
            "chainage_m": np.arange(0.0, 201.0, 1.0),
            "parameters": {
                "seed_dropout_audit": [
                    {
                        "station_id": "suspect",
                        "layer_order": 2,
                        "station_inconsistent": True,
                        "withheld_manual_sample": 258.0,
                        "independent_sample": 241.0,
                    }
                ]
            },
        },
    )()

    requests = _dropout_seed_requests(result, stations)

    assert len(requests) == 2
    assert requests[0].chainage_m == 100.0
    assert requests[1].chainage_m in {75.0, 125.0}
    assert requests[1].layer_orders == [2]
    assert requests[1].source_issue_id == "dropout-companion:suspect:L2"
    assert "local event/velocity regime" in requests[1].reason


def test_deeper_unknown_layers_require_three_observations_for_identity_audit():
    assert _unknown_layer_seed_target(1, []) == 2
    assert _unknown_layer_seed_target(1, [180.0, 181.0]) == 2
    assert _unknown_layer_seed_target(1, [180.0, 225.0]) == 3
    assert _unknown_layer_seed_target(2, []) == 3
    assert _unknown_layer_seed_target(3, [280.0, 281.0]) == 3


def test_design_thickness_does_not_replace_manual_interface_identity(
    synthetic_acquisition,
):
    road_path, plate_path, _ = synthetic_acquisition
    designs = [
        LayerDesign(1, "Asphalt", 50.8, 7.0),
        LayerDesign(2, "Base course", 101.6, 7.0),
        LayerDesign(3, "Sub-base course", 152.4, 7.0),
    ]

    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(
            stack_size=4,
            layer_designs=designs,
            auto_fine_retrack=False,
            validate_seed_dropout=False,
        ),
    )

    assert result.parameters["required_seed_orders"] == [1, 2, 3]
    assert result.parameters["required_seed_count"] == 3
    assert len(result.proposed_seed_requests) == 3
    assert result.parameters["provisional_independent_preview"] is True
    assert all(
        request.layer_orders == [1, 2, 3]
        for request in result.proposed_seed_requests
    )
    assert all(
        item.status == PickStatus.REVIEW
        for item in result.picks
        if item.sample_index >= 0
    )
    assert all(
        np.isnan(item.twtt_ns)
        for item in result.picks
        if item.source.value == "auto" and item.sample_index >= 0
    )
    assert all(item.status == PickStatus.UNRESOLVED for item in result.thickness)
    assert all(item.thickness_mm is None for item in result.thickness)
    assert all(np.isnan(item.twtt_ns) for item in result.thickness)
    assert all(
        item.individual_thickness_mm is None
        and item.cumulative_depth_mm is None
        for item in result.profile
    )


def test_followup_initial_seed_requests_cover_unobserved_road_spans():
    chainage = np.arange(0.0, 101.0, 1.0)
    candidate_feature = np.zeros((len(chainage), 32), dtype=float)
    candidate_feature[50, -1] = 10.0
    stations = [
        SeedStation(
            "known-base",
            50.0,
            {2: 240.0},
            {2: VisibilityState.VISIBLE},
            user_confirmed={2: True},
        )
    ]

    requests = _initial_seed_requests(
        chainage,
        candidate_feature,
        {2},
        count=2,
        stations=stations,
    )

    assert len(requests) == 2
    assert all(abs(item.chainage_m - 50.0) >= 25.0 for item in requests)
    assert all(item.layer_orders == [2] for item in requests)


def test_unknown_subbase_requests_third_station_with_two_observations(
    synthetic_acquisition,
):
    road_path, plate_path, _ = synthetic_acquisition
    designs = [
        LayerDesign(1, "Asphalt", 50.8, 7.0),
        LayerDesign(2, "Base course", 101.6, 7.0),
        LayerDesign(3, "Sub-base course", None, 7.0),
    ]
    stations = [
        SeedStation(
            "subbase-a",
            5.0,
            {3: 150.0},
            {3: VisibilityState.VISIBLE},
            user_confirmed={3: True},
        ),
        SeedStation(
            "subbase-b",
            20.0,
            {3: 205.0},
            {3: VisibilityState.VISIBLE},
            user_confirmed={3: True},
        ),
    ]

    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(
            stack_size=4,
            layer_designs=designs,
            seed_stations=stations,
            auto_fine_retrack=False,
        ),
    )

    # Any two deeper-layer observations require a third station for the
    # independence audit. These are deliberately off-reflector, so this fixture
    # does not establish that the unseeded base is unambiguous; a joint solver
    # may also request its identity.
    assert 3 in result.parameters["required_seed_orders"]
    assert result.parameters["required_seed_count"] == 3
    assert 1 <= len(result.proposed_seed_chainages) <= 5 - len(stations)


def test_acceptance_threshold_cannot_change_raw_graph_family(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    results = [
        analyze_acquisition(
            AcquisitionFileSet(road_path), AcquisitionFileSet(plate_path),
            AnalysisOptions(
                stack_size=4, confidence_threshold=threshold,
                layer_confidence_thresholds={}, auto_fine_retrack=False,
                validate_seed_dropout=False,
            ),
        )
        for threshold in (0.0, 1.0)
    ]
    observed = [
        [(p.evidence.graph_selected_sample, p.evidence.event_family_index) for p in result.picks]
        for result in results
    ]
    assert observed[0] == observed[1]
