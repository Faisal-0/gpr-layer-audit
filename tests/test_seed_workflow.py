"""Focused operator-loop checks; no full-road processing or new test machinery."""

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from gpr_layer_audit.models import (
    DielectricSource,
    InterfacePick,
    LayerSpec,
    PickStatus,
    SeedRequest,
    SeedStation,
    VisibilityState,
)
from gpr_layer_audit.processing import AnalysisOptions
from gpr_layer_audit.processing.pipeline import _aggregate_results, _apply_seed_visibility
from gpr_layer_audit.project import ProjectStore
from gpr_layer_audit.ui.main_window import AnalysisWorker, MainWindow


@pytest.fixture
def seed_window(monkeypatch):
    app = QApplication.instance() or QApplication(["seed-workflow", "-platform", "offscreen"])
    window = MainWindow()
    window.result = SimpleNamespace(
        chainage_m=np.arange(0.2, 100.0, 0.4),
        stack_size=16,
        seed_stations=[],
        proposed_seed_chainages=[40.2],
        parameters={"required_seed_orders": [], "required_seed_count": 0},
        review_issues=[SimpleNamespace(layer_order=2, suggested_chainage_m=40.2)],
    )
    monkeypatch.setattr(window.radar, "refresh_guides", lambda: None)
    monkeypatch.setattr(window.radar, "focus_chainage", lambda *args: None)
    monkeypatch.setattr(window.radar, "set_active_layer", lambda *args: None)
    window._populate_seed_controls()
    yield window
    window.close()
    app.processEvents()


def test_suggested_model_seed_uses_clicked_trace_not_navigation_target(seed_window):
    station = seed_window._selected_or_clicked_station(44.25)
    assert station.role == "initial"
    assert station.chainage_m == pytest.approx(44.2)
    assert seed_window.active_layer_combo.currentData() == 2
    # An adjacent trace is not silently folded into the previous station.
    next_station = seed_window._selected_or_clicked_station(44.6)
    assert next_station.station_id != station.station_id


def test_seed_request_label_uses_backend_layer_and_reason(seed_window):
    seed_window.result.proposed_seed_requests = [
        SeedRequest(
            40.2,
            [2],
            "Candidate reflector is not connected to a confirmed seed",
            0.9,
        )
    ]
    seed_window._populate_seed_controls()

    assert "Base course" in seed_window.seed_combo.itemText(0)
    assert "Resolve ambiguity" in seed_window.seed_combo.itemText(0)
    assert "not connected" in seed_window.seed_combo.itemData(
        0, Qt.ItemDataRole.ToolTipRole
    )


def test_arbitrary_model_seed_enables_rerun_and_preserves_mode(seed_window):
    index = seed_window.seed_combo.findText("Model seed at clicked chainage")
    seed_window.seed_combo.setCurrentIndex(index)
    station = seed_window._selected_or_clicked_station(70.2)
    station.samples[2] = 280.0
    station.visibility[2] = VisibilityState.VISIBLE
    station.user_confirmed[2] = True
    seed_window._populate_seed_controls()
    assert station.role == "initial"
    assert seed_window.track_button.isEnabled()
    assert seed_window.seed_combo.currentData(Qt.ItemDataRole.UserRole + 1) == "model"
    assert seed_window._station_complete(station)  # no irrelevant subbase pick needed


def test_explicit_correction_stays_local_and_preserves_mode(seed_window):
    seed_window.seed_combo.setCurrentIndex(
        seed_window.seed_combo.findText("Correction at clicked chainage")
    )
    station = seed_window._selected_or_clicked_station(20.2)
    station.samples[2] = 280.0
    station.user_confirmed[2] = True
    seed_window._populate_seed_controls()
    assert station.role == "correction"
    assert not seed_window.track_button.isEnabled()
    assert seed_window.seed_combo.currentData(Qt.ItemDataRole.UserRole + 1) == "correction"


def test_base_only_model_seed_can_rerun_while_subbase_remains_unknown(seed_window):
    seed_window.result.parameters.update(required_seed_orders=[3], required_seed_count=2)
    seed_window.seed_combo.setCurrentIndex(
        seed_window.seed_combo.findText("Model seed at clicked chainage")
    )
    seed_window.active_layer_combo.setCurrentIndex(seed_window.active_layer_combo.findData(2))
    station = seed_window._selected_or_clicked_station(70.2)
    station.samples[2] = 280.0
    station.user_confirmed[2] = True
    seed_window._populate_seed_controls()
    assert seed_window.track_button.isEnabled()
    assert seed_window.active_layer_combo.currentData() == 2


def test_training_limit_never_silently_converts_model_seed_to_correction(seed_window, monkeypatch):
    seed_window.options.seed_stations = [SeedStation(str(i), i * 2.0) for i in range(5)]
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    assert seed_window._selected_or_clicked_station(70.2) is None
    assert len(seed_window.options.seed_stations) == 5
    assert warnings


def test_running_worker_owns_snapshot_of_seeds():
    options = AnalysisOptions(seed_stations=[SeedStation("first", 10.0, {2: 280.0})])
    worker = AnalysisWorker(None, None, options)
    options.seed_stations[0].samples[2] = 290.0
    options.seed_stations.append(SeedStation("second", 80.0))
    assert len(worker.options.seed_stations) == 1
    assert worker.options.seed_stations[0].samples[2] == 280.0


def test_radar_click_persists_matching_trace_metadata_and_enables_updated_seed(
    seed_window, tmp_path, monkeypatch,
):
    seed_window.project_store = ProjectStore.create(tmp_path / "seed.gprproj", "synthetic")
    seed_window.result.calibrated_radargram = np.zeros((250, 340))
    row = int(np.argmin(abs(seed_window.result.chainage_m - 44.2)))
    chainage = float(seed_window.result.chainage_m[row])
    seed_window.result.candidate_events = [SimpleNamespace(
        layer_order=2, chainage_m=chainage, sample_index=280,
        waveform_correlation=0.9, phase_class=5, analytic_phase_rad=1.2,
        polarity=1, selected_lobe="positive_peak", canonical_sample_index=280,
        pulse_width_samples=7, event_id="synthetic-event", event_family_id="base",
        competing_family_id=None, competing_event_ids=[],
    )]
    local_calls = []
    monkeypatch.setattr(seed_window, "_local_retrack", lambda value: local_calls.append(value))
    seed_window.add_seed_pick(2, 44.25, 280.0)
    station = seed_window.options.seed_stations[0]
    saved = seed_window.project_store.seed_stations()[0]
    assert saved.chainage_m == chainage
    assert saved.event_ids[2] == "synthetic-event"
    assert saved.role == "initial"
    assert seed_window.track_button.isEnabled()
    assert not local_calls
    # Editing an already-processed station must enable retraining too.
    seed_window._analyzed_training_station_ids.add(station.station_id)
    seed_window._populate_seed_controls()
    assert not seed_window.track_button.isEnabled()
    seed_window.add_seed_pick(2, 44.25, 281.0)
    assert seed_window.track_button.isEnabled()
    assert seed_window.project_store.seed_stations()[0].samples[2] == 281.0


@pytest.mark.parametrize("visibility", [VisibilityState.NOT_VISIBLE, VisibilityState.ABSENT])
def test_negative_seed_survives_tracking_and_prevents_thickness_across_its_gap(visibility):
    chainage = np.asarray([0.2, 0.6, 1.0])
    picks = [
        InterfacePick(
            order, f"Layer {order}", row, float(distance), float(sample), 1.0,
            1.0, 0.9, PickStatus.HIGH_CONFIDENCE, selected_lobe_sample=float(sample),
        )
        for order, sample in [(1, 190), (2, 280)]
        for row, distance in enumerate(chainage)
    ]
    station = SeedStation("no-base", 0.6, visibility={2: visibility}, user_confirmed={2: True})
    _apply_seed_visibility(picks, [station], chainage)
    base = [pick for pick in picks if pick.layer_order == 2]
    assert [pick.sample_index for pick in base] == [280, -1, 280]
    assert base[1].visibility == visibility
    assert base[1].selected_lobe_sample is None
    assert all(pick.sample_index == 190 for pick in picks if pick.layer_order == 1)
    thickness = _aggregate_results(
        picks, LayerSpec.defaults(), SimpleNamespace(sample_interval_ns=0.03), 1.2,
        {1: (7.0, DielectricSource.ASSUMED_SCAN), 2: (7.0, DielectricSource.ASSUMED_SCAN)},
        156,
    )
    assert next(item for item in thickness if item.layer_order == 2).thickness_mm is None
