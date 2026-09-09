"""Product invariants discovered while exercising native processed correction replay."""

import json
import os
from copy import deepcopy
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from conftest import pulse, write_dzt
from PySide6.QtWidgets import QApplication

from gpr_layer_audit import cli
from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    LayerSpec,
    PickStatus,
    ReviewIssue,
    SeedStation,
    VisibilityState,
)
from gpr_layer_audit.native_seed_io import load_native_observations
from gpr_layer_audit.processing import processed_tracking as processed
from gpr_layer_audit.processing.pipeline import (
    AnalysisOptions,
    analyze_acquisition,
    resolve_review_issue,
    retrack_segment,
)
from gpr_layer_audit.processing.seed_graph import SeedConditionedPath
from gpr_layer_audit.project import ProjectStore
from gpr_layer_audit.ui import main_window as ui


@pytest.fixture(scope="module")
def qt_application():
    return QApplication.instance() or QApplication([])


def _path(sample, rows=41):
    samples = np.full(rows, sample, np.int32)
    return SeedConditionedPath(
        samples=samples,
        confidence=np.ones(rows),
        feature=np.ones((rows, 128), np.float32),
        alternate_samples=samples + 8,
        visible=np.ones(rows, bool),
        interpolated=np.zeros(rows, bool),
        evidence={"hybrid_backend": np.ones(rows), "hybrid_accepted": np.ones(rows)},
        signal_only_samples=samples.copy(),
        design_guided_samples=samples.copy(),
        design_conflict=np.zeros(rows, bool),
        design_constrained=False,
        provisional_samples=samples.copy(),
        provenance={"hypothesis_samples": [samples.tolist(), (samples + 8).tolist()]},
    )


@pytest.fixture
def dispatch_case(tmp_path, monkeypatch):
    """A deterministic solver makes scope/replay defects visible without score noise."""
    path = tmp_path / "processed.DZT"
    data = np.tile(100000 * pulse(128, 50) + 70000 * pulse(128, 90), (41, 1))
    write_dzt(path, data)
    seed = SeedStation("initial", 0.5, samples={2: 50, 3: 90}, user_confirmed={2: True, 3: True})
    options = AnalysisOptions(
        input_mode="processed",
        tracker_method="seed_hybrid",
        seed_stations=[seed],
        layer_specs=[LayerSpec(2, "Base", 1, 127, 1), LayerSpec(3, "Subbase", 1, 127, 1)],
    )
    calls = []

    def solve(result, options, stations, *, cancel=None):
        calls.append([s.station_id for s in stations])
        if cancel and cancel():
            raise InterruptedError("cancelled in solver")
        anchors = processed.native_anchors(stations, result.chainage_m)
        paths = {2: _path(50), 3: _path(90)}
        for station in stations:
            if station.role == "correction":
                # Depend on every supplied correction, exposing future-input leakage.
                for order, sample in station.samples.items():
                    paths[order] = _path(int(sample))
        for order, points in anchors.items():
            for row, sample in points.items():
                paths[order].samples[row] = sample
                paths[order].provisional_samples[row] = sample
        return paths, anchors

    monkeypatch.setattr(processed, "_run", solve)
    return AcquisitionFileSet(path), options, calls


def test_native_import_is_exact_hash_checked_and_does_not_need_reference(dispatch_case, tmp_path):
    source, _, _ = dispatch_case
    seed_path = tmp_path / "seeds.json"
    document = {
        "schema": "conventional-native-seeds-v1",
        "mode": "processed",
        "dzt_sha256": fingerprint_file(source.dzt_path),
        "observations": {"3": [{"trace": 13, "sample": 90, "channel": 0}]},
    }
    seed_path.write_text(json.dumps(document))
    station = load_native_observations(seed_path, source.dzt_path)[0]
    assert station.chainage_m == 1.3
    assert station.visible_sample(3) == 90
    assert station.canonical_samples[3] == 90
    document["observations"]["3"][0]["trace"] = 13.0
    seed_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="must be integers"):
        load_native_observations(seed_path, source.dzt_path)
    document["dzt_sha256"] = "wrong-source"
    seed_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="different processed DZT"):
        load_native_observations(seed_path, source.dzt_path)


def test_cli_native_subbase_observation_enables_its_layer(dispatch_case, monkeypatch, tmp_path):
    source, options, _ = dispatch_case
    monkeypatch.setattr(
        "gpr_layer_audit.native_seed_io.load_native_observations",
        lambda *args: options.seed_stations,
    )
    captured = []
    monkeypatch.setattr(
        cli, "analyze_acquisition", lambda road, plate, options, **kw: captured.append(options)
    )
    monkeypatch.setattr(cli, "export_audit_package", lambda *args: tmp_path)
    args = cli.build_parser().parse_args(
        [
            "analyze",
            str(source.dzt_path),
            "--input-mode",
            "processed",
            "--native-seeds",
            str(tmp_path / "native.json"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    cli._run_analysis(args, source.dzt_path)
    assert next(layer for layer in captured[0].layer_specs if layer.order == 3).analysis_enabled


def test_gui_native_import_enables_subbase_without_an_extra_unrelated_click(
    dispatch_case,
    monkeypatch,
    tmp_path,
    qt_application,
):
    source, options, _ = dispatch_case
    window = ui.MainWindow()
    window.road = source
    window.project_store = ProjectStore.create(tmp_path / "native.gprproj", "native")
    monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *args: ("native.json", "JSON"))
    monkeypatch.setattr(
        "gpr_layer_audit.native_seed_io.load_native_observations",
        lambda *args: options.seed_stations,
    )
    captured = []
    monkeypatch.setattr(
        window, "run_analysis", lambda: captured.append(window._layers_from_controls())
    )
    try:
        window.load_native_seed_observations()
        assert next(layer for layer in captured[0] if layer.order == 3).analysis_enabled
    finally:
        window.close()


def test_local_merge_preserves_other_layers_and_outside_competing_hypotheses(dispatch_case):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    previous = deepcopy(result.processed_paths)
    before_picks = list(result.picks)
    options.seed_stations.append(
        SeedStation(
            "correction",
            2.0,
            samples={2: 60},
            user_confirmed={2: True},
            role="correction",
        )
    )
    retrack_segment(result, options, 1.8, 2.2)
    outside = (result.chainage_m < 1.8) | (result.chainage_m > 2.2)
    np.testing.assert_array_equal(
        result.processed_paths[2].samples[outside], previous[2].samples[outside]
    )
    np.testing.assert_array_equal(result.processed_paths[3].samples, previous[3].samples)
    for old in previous[2].provenance["hypothesis_samples"]:
        assert any(
            np.array_equal(np.asarray(old)[outside], np.asarray(new)[outside])
            for new in result.processed_paths[2].provenance["hypothesis_samples"]
        )
    for old, new in zip(before_picks, result.picks, strict=True):
        if new.layer_order == 3 or not 1.8 <= new.chainage_m <= 2.2:
            assert old is new
    assert result.processed_paths[2].samples[20] == 60


def test_preserved_accepted_picks_and_paths_cannot_diverge(dispatch_case):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    options.seed_stations.append(
        SeedStation(
            "correction",
            2.0,
            samples={2: 60},
            user_confirmed={2: True},
            role="correction",
        )
    )
    # Both representations drive subsequent requests, exported values and replay.
    retrack_segment(result, options, 1.8, 2.2, preserve_accepted=True)
    for pick in result.picks:
        if pick.status in {PickStatus.ACCEPTED, PickStatus.HIGH_CONFIDENCE}:
            assert (
                result.processed_paths[pick.layer_order].samples[pick.trace_index]
                == pick.selected_lobe_sample
            )


def test_saved_corrections_replay_in_saved_order_with_only_answer_prefix(dispatch_case, tmp_path):
    source, options, calls = dispatch_case
    initial = list(options.seed_stations)
    first = SeedStation(
        "first-at-right", 3.0, samples={2: 60}, user_confirmed={2: True}, role="correction"
    )
    second = SeedStation(
        "second-at-left", 1.0, samples={2: 70}, user_confirmed={2: True}, role="correction"
    )
    store = ProjectStore.create(tmp_path / "replay.gprproj", "replay")
    for station in [*initial, first, second]:
        store.save_seed_station(station)
    options.seed_stations = store.seed_stations()  # SQL returns chainage order, not action order.
    options.local_correction_order = [first.station_id, second.station_id]
    result = analyze_acquisition(source, options=options)
    assert calls == [
        ["initial"],
        ["initial", first.station_id],
        ["initial", first.station_id, second.station_id],
    ]
    assert result.parameters["local_correction_order"] == [first.station_id, second.station_id]
    assert result.parameters["local_correction_replay"]["count"] == 2


def test_guard_withholds_automation_and_rejects_conflicting_exact_correction():
    previous = {2: _path(50), 3: _path(90)}
    layers = [LayerSpec(2, "Base", 1, 127, 1), LayerSpec(3, "Subbase", 1, 127, 1)]
    changed = {2: _path(95), 3: _path(90)}
    processed.guard_local_order(changed, previous, layers, {}, {2}, 18, 22)
    assert np.all(changed[2].samples[18:23] == -1)
    assert not np.any(changed[2].visible[18:23])
    assert np.all(changed[2].evidence["hybrid_accepted"][18:23] == 0)
    with pytest.raises(ValueError, match="preserved neighboring interface"):
        processed.guard_local_order(
            {2: _path(95), 3: _path(90)}, previous, layers, {2: {20: 95}}, {2}, 18, 22
        )


def test_worker_cancellation_keeps_previous_result_and_emits_only_cancelled(
    dispatch_case, qt_application
):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    previous = deepcopy(result.processed_paths[2].samples)
    worker = ui.ProcessedCorrectionWorker(result, options, 2.0)
    events = []
    worker.signals.cancelled.connect(lambda: events.append("cancelled"))
    worker.signals.result.connect(lambda value: events.append("result"))
    worker.signals.error.connect(lambda value: events.append(value))
    worker.cancel()
    worker.run()
    assert events == ["cancelled"]
    np.testing.assert_array_equal(result.processed_paths[2].samples, previous)


def test_public_retrack_checks_late_cancellation_before_publishing(dispatch_case, monkeypatch):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    before = list(result.picks)
    old_paths = result.processed_paths
    solve = processed._run
    cancelled = False

    def finishes_as_cancel_arrives(*args, **kwargs):
        nonlocal cancelled
        output = solve(*args, **kwargs)
        cancelled = True
        return output

    monkeypatch.setattr(processed, "_run", finishes_as_cancel_arrives)
    with pytest.raises(InterruptedError):
        retrack_segment(result, options, 1.8, 2.2, cancel=lambda: cancelled)
    assert result.processed_paths is old_paths
    assert all(old is current for old, current in zip(before, result.picks, strict=True))


def test_unconfirmed_layer_does_not_expand_local_correction_scope(dispatch_case):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    previous = list(result.picks)
    options.seed_stations.append(
        SeedStation(
            "partly-confirmed",
            2.0,
            samples={2: 60, 3: 95},
            user_confirmed={2: True, 3: False},
            role="correction",
        )
    )
    retrack_segment(result, options, 1.8, 2.2)
    assert result.parameters["processed_local_retracks"][-1]["layer_orders"] == [2]
    for old, current in zip(previous, result.picks, strict=True):
        if old.layer_order == 3:
            assert current is old


def test_overlapping_corrections_scope_each_answer_and_saved_replay(
    dispatch_case,
    monkeypatch,
    tmp_path,
):
    source, options, _ = dispatch_case
    solve = processed._run

    def coupled_solver(result, options, stations, **kwargs):
        paths, anchors = solve(result, options, stations, **kwargs)
        if any(s.station_id == "later-base" for s in stations):
            # A changed upper interpretation changes a regenerated lower proposal.
            # That new lower proposal must remain outside the current action's scope.
            paths[3].samples[25] = 97
            paths[3].provisional_samples[25] = 97
        return paths, anchors

    monkeypatch.setattr(processed, "_run", coupled_solver)
    result = analyze_acquisition(source, options=options)
    first = SeedStation(
        "earlier-subbase", 2.0, samples={3: 95}, user_confirmed={3: True}, role="correction"
    )
    second = SeedStation(
        "later-base", 2.1, samples={2: 60}, user_confirmed={2: True}, role="correction"
    )
    options.seed_stations.append(first)
    retrack_segment(result, options, 0, 27, layer_orders={3})
    lower = deepcopy(result.processed_paths[3])
    previous_picks = list(result.picks)
    options.seed_stations.append(second)
    retrack_segment(result, options, 0, 27.1, layer_orders={2})
    np.testing.assert_array_equal(result.processed_paths[3].samples, lower.samples)
    np.testing.assert_array_equal(
        result.processed_paths[3].provisional_samples, lower.provisional_samples
    )
    for old, current in zip(previous_picks, result.picks, strict=True):
        if old.layer_order == 3:
            assert current is old
    assert result.parameters["processed_local_retracks"][-1]["layer_orders"] == [2]
    store = ProjectStore.create(tmp_path / "overlap.gprproj", "overlap")
    for station in options.seed_stations:
        store.save_seed_station(station)
    store.save_analysis(result)
    options.seed_stations = store.seed_stations()
    options.local_correction_order = store.latest_parameters()["local_correction_order"]
    replayed = analyze_acquisition(source, options=options)
    assert [item["layer_orders"] for item in replayed.parameters["processed_local_retracks"]] == [
        [3],
        [2],
    ]
    np.testing.assert_array_equal(replayed.processed_paths[3].samples, lower.samples)


def test_not_visible_answer_is_retained_and_not_requested_again(dispatch_case):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    first = result.proposed_seed_requests[0]
    order = first.layer_orders[0]
    options.seed_stations.append(
        SeedStation(
            "not-visible-answer",
            first.chainage_m,
            role="correction",
            visibility={order: VisibilityState.NOT_VISIBLE},
            user_confirmed={order: True},
        )
    )
    retrack_segment(result, options, max(0, first.chainage_m - 0.1), first.chainage_m + 0.1)
    row = round(first.chainage_m / result.header.distance_per_trace_m)
    pick = next(p for p in result.picks if p.layer_order == order and p.trace_index == row)
    assert not np.isfinite(pick.twtt_ns)
    assert pick.visibility == VisibilityState.NOT_VISIBLE
    assert result.processed_paths[order].samples[row] < 0
    assert all(
        q.chainage_m != first.chainage_m or q.layer_orders != [order]
        for q in result.proposed_seed_requests
    )


def test_processed_requests_default_to_deep_layers_and_honor_explicit_scope(dispatch_case):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    assert options.query_layer_orders == [2, 3]
    result.processed_paths[1] = _path(30)
    result.processed_paths[1].provenance["suggested_observations"] = [
        {"row": 20, "priority": 1e9, "candidate_samples": [30, 38]}
    ]
    processed._refresh_paths_and_requests(result, options)
    assert result.proposed_seed_requests[0].layer_orders[0] in (2, 3)
    options.query_layer_orders = [1]
    processed._refresh_paths_and_requests(result, options)
    assert result.proposed_seed_requests[0].layer_orders == [1]


def test_gui_requested_answer_is_a_local_correction(
    dispatch_case,
    monkeypatch,
    tmp_path,
    qt_application,
):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    window = ui.MainWindow()
    window.road, window.result, window.options = source, result, options
    window.project_store = ProjectStore.create(tmp_path / "gui-answer.gprproj", "gui-answer")
    local_calls = []
    errors = []

    def run_worker(worker):
        local_calls.append(worker.chainage)
        worker.run()

    monkeypatch.setattr(window, "thread_pool", SimpleNamespace(start=run_worker))
    monkeypatch.setattr(window, "_analysis_error", errors.append)
    monkeypatch.setattr(
        ui.QMessageBox, "question", lambda *args: ui.QMessageBox.StandardButton.Save
    )
    try:
        window._populate_seed_controls()
        request = result.proposed_seed_requests[0]
        window.seed_combo.setCurrentIndex(window.seed_combo.findData(request.chainage_m))
        window.add_seed_pick(request.layer_orders[0], request.chainage_m, 60)
        answer = next(s for s in options.seed_stations if s.chainage_m == request.chainage_m)
        assert answer.role == "correction"
        assert local_calls == [request.chainage_m]
        assert not errors
        assert window.result is not result
        assert (
            window.result.parameters["processed_local_retracks"][-1]["operation"]
            == "local_correction"
        )
        assert options.local_correction_order == [answer.station_id]
        assert (
            window.project_store.review_events()[-1]["details"]["operation"] == "local_correction"
        )
        assert window.project_store.seed_stations()[0].role == "correction"
    finally:
        window.close()


@pytest.mark.parametrize("action", ["not_visible", "absent", "accept"])
def test_review_actions_sync_only_the_affected_processed_path(dispatch_case, action):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    previous = deepcopy(result.processed_paths)
    issue = ReviewIssue("review", 2, "Base", 1.8, 2.2, [], "Inspect")
    result.review_issues = [issue]
    resolve_review_issue(result, options, issue.issue_id, action)
    use = (result.chainage_m >= 1.8) & (result.chainage_m <= 2.2)
    np.testing.assert_array_equal(
        result.processed_paths[2].samples[~use], previous[2].samples[~use]
    )
    np.testing.assert_array_equal(result.processed_paths[3].samples, previous[3].samples)
    for pick in result.picks:
        if pick.layer_order != 2 or not 1.8 <= pick.chainage_m <= 2.2:
            continue
        assert result.processed_paths[2].visible[pick.trace_index] == pick.is_accepted_measurement
        if action == "accept":
            assert result.processed_paths[2].samples[pick.trace_index] == pick.selected_lobe_sample
        else:
            assert result.processed_paths[2].samples[pick.trace_index] == -1
            assert not np.isfinite(pick.twtt_ns)
    assert all(
        not (q.layer_orders == [2] and 1.8 <= q.chainage_m <= 2.2)
        for q in result.proposed_seed_requests
    )


def test_explicit_processed_dielectric_is_identified_and_reopened(
    dispatch_case,
    monkeypatch,
    tmp_path,
    qt_application,
):
    source, options, _ = dispatch_case
    options.analyst_dielectric = {2: 8.0}
    result = analyze_acquisition(source, options=options)
    assert result.parameters["dielectric_by_layer"]["2"] == {
        "value": 8.0,
        "source": "analyst",
    }
    assert not result.diagnostics.valid_for_dielectric
    store = ProjectStore.create(tmp_path / "dielectric.gprproj", "dielectric")
    store.set_layers(options.layer_specs)
    for station in options.seed_stations:
        store.save_seed_station(station)
    store.save_analysis(result)
    window = ui.MainWindow()
    monkeypatch.setattr(
        ui.QFileDialog, "getOpenFileName", lambda *args: (str(store.path), "Project")
    )
    try:
        window.open_project()
        assert window.options.analyst_dielectric == {2: 8.0}
        reopened = analyze_acquisition(source, options=window.options)
        assert (
            reopened.parameters["dielectric_by_layer"] == result.parameters["dielectric_by_layer"]
        )
    finally:
        window.close()


def test_confirmation_identity_does_not_transfer_to_regenerated_family(dispatch_case):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    issue = SimpleNamespace(layer_order=2, start_chainage_m=1.8, end_chainage_m=2.2)
    details = ui._review_proposal_snapshot(result, issue)
    assert ui._review_proposal_matches(result, issue, details)
    selected = next(p for p in result.picks if p.layer_order == 2 and p.trace_index == 20)
    selected.event_family_id = "different-family-at-same-sample"
    assert not ui._review_proposal_matches(result, issue, details)


def test_fit_processed_passes_immutable_seed_metadata_and_pulse_roles(monkeypatch):
    captured = {}
    monkeypatch.setattr(processed, "pick_interfaces", lambda *args, **kw: captured.update(kw))
    metadata = {2: {3: {"family_id": "seed-family"}, 9: {"regime_id": "reviewed-change"}}}
    original = deepcopy(metadata)
    processed.fit_processed(
        np.ones((12, 128)),
        np.ones((12, 128), bool),
        0,
        0.1,
        0.2,
        [LayerSpec(2, "Base", 1, 127, 1)],
        {2: {3: 50, 9: 60}},
        seed_metadata=metadata,
        pulse_anchors={2: {3: 50}},
    )
    assert metadata == original
    received = captured["seed_metadata"][2]
    assert received[3] == {"family_id": "seed-family", "pulse_estimation_use": True}
    assert received[9] == {"regime_id": "reviewed-change", "pulse_estimation_use": False}


def test_unresolved_proposals_have_no_accepted_twtt(dispatch_case, monkeypatch):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    unresolved = deepcopy(result.processed_paths)
    unresolved[2].samples[20] = -1
    unresolved[2].visible[20] = False
    unresolved[2].evidence["hybrid_accepted"][20] = 0
    picks = processed._paths_to_picks(result, options, unresolved, {})
    pick = next(p for p in picks if p.layer_order == 2 and p.trace_index == 20)
    assert not np.isfinite(pick.twtt_ns)
    assert not pick.is_accepted_measurement
