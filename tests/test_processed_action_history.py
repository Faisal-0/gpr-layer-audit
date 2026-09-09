"""Scoped correction history must reproduce observations, not final station state."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_processed_workflow_verification import dispatch_case as dispatch_case
from test_processed_workflow_verification import qt_application as qt_application

from gpr_layer_audit.models import SeedStation, VisibilityState
from gpr_layer_audit.processing import processed_tracking as processed
from gpr_layer_audit.processing.pipeline import analyze_acquisition, retrack_segment
from gpr_layer_audit.project import ProjectStore
from gpr_layer_audit.ui import main_window as ui


def _reopen(source, options, result, tmp_path):
    store = ProjectStore.create(tmp_path / "history.gprproj", "history")
    for station in options.seed_stations:
        store.save_seed_station(station)
    store.save_analysis(result)
    reopened_options = deepcopy(options)
    reopened_options.seed_stations = store.seed_stations()
    reopened_options.local_correction_actions = store.latest_parameters()[
        "local_correction_actions"
    ]
    reopened = analyze_acquisition(source, options=reopened_options)
    for order, path in result.processed_paths.items():
        np.testing.assert_array_equal(reopened.processed_paths[order].samples, path.samples)
        np.testing.assert_array_equal(
            reopened.processed_paths[order].provisional_samples, path.provisional_samples
        )
    return reopened


def test_interleaved_layers_at_one_station_keep_each_action_scope(
    dispatch_case, monkeypatch, tmp_path
):
    source, options, _ = dispatch_case
    solve = processed._run

    def coupled(result, options, stations, **kwargs):
        paths, anchors = solve(result, options, stations, **kwargs)
        if any(s.station_id == "later-base" for s in stations):
            paths[3].samples[25] = 97
            paths[3].provisional_samples[25] = 97
        return paths, anchors

    monkeypatch.setattr(processed, "_run", coupled)
    result = analyze_acquisition(source, options=options)
    shared = SeedStation(
        "shared", 2.0, samples={3: 95}, user_confirmed={3: True}, role="correction"
    )
    later = SeedStation(
        "later-base", 2.1, samples={2: 60}, user_confirmed={2: True}, role="correction"
    )
    options.seed_stations.append(shared)
    retrack_segment(result, options, 0, 27, layer_orders={3})
    options.seed_stations.append(later)
    retrack_segment(result, options, 0, 27.1, layer_orders={2})
    shared.samples[2], shared.user_confirmed[2] = 65, True
    retrack_segment(result, options, 0, 27, layer_orders={2})
    assert result.processed_paths[3].samples[25] == 95
    log = result.parameters["local_correction_actions"]
    assert [action["layer_orders"] for action in log] == [[3], [2], [2]]
    assert log[0]["stations"][-1]["samples"] == {"3": 95}
    assert all(s["station_id"] != "later-base" for s in log[0]["stations"])
    reopened = _reopen(source, options, result, tmp_path)
    assert reopened.processed_paths[3].samples[25] == 95
    assert reopened.parameters["local_correction_actions"] == log


@pytest.mark.parametrize("negative", [False, True])
def test_repeated_layer_edits_preserve_old_samples_visibility_and_exact_windows(
    dispatch_case, tmp_path, negative
):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    station = SeedStation(
        "repeat", 2.0, samples={2: 60}, user_confirmed={2: True}, role="correction"
    )
    options.seed_stations.append(station)
    retrack_segment(result, options, 1.8, 2.2, layer_orders={2})
    if negative:
        station.samples.pop(2)
        station.visibility[2] = VisibilityState.NOT_VISIBLE
    else:
        station.samples[2] = 70
    retrack_segment(result, options, 1.9, 2.1, layer_orders={2})
    assert result.processed_paths[2].samples[18] == 60
    assert result.processed_paths[2].samples[20] == (-1 if negative else 70)
    log = result.parameters["local_correction_actions"]
    assert log[0]["stations"][-1]["samples"] == {"2": 60}
    assert [(a["start_chainage_m"], a["end_chainage_m"]) for a in log] == [
        (1.8, 2.2), (1.9, 2.1)
    ]
    _reopen(source, options, result, tmp_path)


def test_cancelled_gui_correction_publishes_neither_station_nor_action(
    dispatch_case, monkeypatch, tmp_path, qt_application
):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    before = deepcopy(result.processed_paths[2].samples)
    window = ui.MainWindow()
    window.road, window.result, window.options = source, result, options
    window.project_store = ProjectStore.create(tmp_path / "cancel.gprproj", "cancel")
    for station in options.seed_stations:
        window.project_store.save_seed_station(station)

    def cancel_worker(worker):
        worker.cancel()
        worker.run()

    monkeypatch.setattr(window, "thread_pool", SimpleNamespace(start=cancel_worker))
    monkeypatch.setattr(ui.QMessageBox, "question", lambda *a: ui.QMessageBox.StandardButton.Save)
    try:
        window._populate_seed_controls()
        request = result.proposed_seed_requests[0]
        window.seed_combo.setCurrentIndex(window.seed_combo.findData(request.chainage_m))
        window.add_seed_pick(request.layer_orders[0], request.chainage_m, 60)
        assert [s.station_id for s in window.options.seed_stations] == ["initial"]
        assert [s.station_id for s in window.project_store.seed_stations()] == ["initial"]
        assert result.parameters["local_correction_actions"] == []
        assert window.options.local_correction_actions == []
        np.testing.assert_array_equal(result.processed_paths[2].samples, before)
    finally:
        window.close()


def test_cancelled_public_fit_does_not_append_history(dispatch_case):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    options.seed_stations.append(
        SeedStation("cancel", 2.0, samples={2: 60}, user_confirmed={2: True}, role="correction")
    )
    with pytest.raises(InterruptedError):
        retrack_segment(result, options, 1.8, 2.2, layer_orders={2}, cancel=lambda: True)
    assert options.local_correction_actions == result.parameters["local_correction_actions"] == []


def test_gui_commits_each_layer_action_and_restores_history_on_open(
    dispatch_case, monkeypatch, tmp_path, qt_application
):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    window = ui.MainWindow()
    window.road, window.result, window.options = source, result, options
    window.project_store = ProjectStore.create(tmp_path / "gui-history.gprproj", "history")
    window.project_store.set_layers(options.layer_specs)
    for station in options.seed_stations:
        window.project_store.save_seed_station(station)
    monkeypatch.setattr(window, "thread_pool", SimpleNamespace(start=lambda worker: worker.run()))
    monkeypatch.setattr(ui.QMessageBox, "question", lambda *a: ui.QMessageBox.StandardButton.Save)
    try:
        window._populate_seed_controls()
        correction = window.seed_combo.findData("correction", ui.Qt.ItemDataRole.UserRole + 1)
        window.seed_combo.setCurrentIndex(correction)
        window.add_seed_pick(3, 2.0, 95)
        window.add_seed_pick(2, 2.1, 60)
        window.add_seed_pick(2, 2.0, 65)
        log = deepcopy(window.result.parameters["local_correction_actions"])
        assert [action["layer_orders"] for action in log] == [[3], [2], [2]]
        assert log[0]["stations"][-1]["samples"] == {"3": 95.0}
        assert window.options.local_correction_actions == log
        saved_path = str(window.project_store.path)
        monkeypatch.setattr(ui.QFileDialog, "getOpenFileName", lambda *a: (saved_path, "Project"))
        window.open_project()
        assert window.options.local_correction_actions == log
    finally:
        window.close()


@pytest.mark.parametrize("cancel_undo", [False, True])
def test_gui_correction_undo_records_removal_or_restores_cancelled_state(
    dispatch_case, monkeypatch, tmp_path, qt_application, cancel_undo
):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    original = deepcopy(result.processed_paths[2].samples)
    window = ui.MainWindow()
    window.road, window.result, window.options = source, result, options
    window.project_store = ProjectStore.create(tmp_path / "undo.gprproj", "undo")
    for station in options.seed_stations:
        window.project_store.save_seed_station(station)
    monkeypatch.setattr(window, "thread_pool", SimpleNamespace(start=lambda worker: worker.run()))
    monkeypatch.setattr(ui.QMessageBox, "question", lambda *a: ui.QMessageBox.StandardButton.Save)
    try:
        window._populate_seed_controls()
        index = window.seed_combo.findData("correction", ui.Qt.ItemDataRole.UserRole + 1)
        window.seed_combo.setCurrentIndex(index)
        window.add_seed_pick(2, 2.0, 60)
        corrected = deepcopy(window.result.processed_paths[2].samples)
        if cancel_undo:
            def cancel_worker(worker):
                worker.cancel()
                worker.run()
            monkeypatch.setattr(window, "thread_pool", SimpleNamespace(start=cancel_worker))
        window.undo_seed_station()
        expected = corrected if cancel_undo else original
        np.testing.assert_array_equal(window.result.processed_paths[2].samples, expected)
        assert len(window.project_store.seed_stations()) == (2 if cancel_undo else 1)
        assert len(window.options.local_correction_actions) == (1 if cancel_undo else 2)
        _reopen(source, window.options, window.result, tmp_path)
    finally:
        window.close()


def test_legacy_replay_is_explicit_and_freezes_inferred_actions(dispatch_case):
    source, options, _ = dispatch_case
    options.seed_stations.append(
        SeedStation("legacy", 2.0, samples={2: 60}, user_confirmed={2: True}, role="correction")
    )
    result = analyze_acquisition(source, options=options)
    assert result.parameters["local_correction_replay"]["mode"] == "legacy final-station fallback"
    assert len(result.parameters["local_correction_actions"]) == 1


def test_gui_undo_uses_station_creation_order_not_road_distance(
    dispatch_case, monkeypatch, tmp_path, qt_application
):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    window = ui.MainWindow()
    window.road, window.result, window.options = source, result, options
    window.project_store = ProjectStore.create(tmp_path / "reverse-order.gprproj", "undo")
    for station in options.seed_stations:
        window.project_store.save_seed_station(station)
    monkeypatch.setattr(window, "thread_pool", SimpleNamespace(start=lambda worker: worker.run()))
    monkeypatch.setattr(ui.QMessageBox, "question", lambda *a: ui.QMessageBox.StandardButton.Save)
    try:
        window._populate_seed_controls()
        index = window.seed_combo.findData("correction", ui.Qt.ItemDataRole.UserRole + 1)
        window.seed_combo.setCurrentIndex(index)
        window.add_seed_pick(2, 3.0, 60)
        window.add_seed_pick(2, 2.0, 70)
        window.undo_seed_station()
        corrections = [s for s in window.options.seed_stations if s.role == "correction"]
        assert [s.chainage_m for s in corrections] == [3.0]
        assert window.options.local_correction_actions[-1]["end_chainage_m"] == 27.0
        _reopen(source, window.options, window.result, tmp_path)
    finally:
        window.close()


@pytest.mark.parametrize("field,value", [("source_sha256", "other"), ("processed_stride", 4)])
def test_saved_actions_fail_closed_on_source_or_grid_change(dispatch_case, field, value):
    source, options, _ = dispatch_case
    result = analyze_acquisition(source, options=options)
    retrack_segment(result, options, 1.8, 2.2, layer_orders={2})
    options.local_correction_actions[0][field] = value
    with pytest.raises(ValueError, match="different source or native grid"):
        analyze_acquisition(source, options=options)
