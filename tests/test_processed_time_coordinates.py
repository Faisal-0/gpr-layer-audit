"""Time-origin regressions; these do not rerun or change tracking inference."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.models import (
    AcquisitionFileSet,
    AnalysisResult,
    CalibrationDiagnostics,
    CandidateEvent,
    InterfacePick,
    LayerSpec,
    PickSource,
    PickStatus,
    ReviewIssue,
    SearchCorridor,
    VisibilityState,
)
from gpr_layer_audit.processing.pipeline import AnalysisOptions
from gpr_layer_audit.time_coordinates import (
    image_time_bounds_ns,
    measurement_zero_sample,
    sample_from_time_ns,
    sample_time_ns,
    sample_twtt_ns,
)


def _pick(row=0, *, order=2, canonical=261.25, display=264):
    return InterfacePick(
        order,
        f"Layer {order}",
        row,
        row * 0.025,
        canonical,
        (canonical - 102) * 0.029296875,
        10.0,
        1.0,
        PickStatus.ACCEPTED,
        source=PickSource.SEED,
        selected_lobe_sample=float(display),
        canonical_event_sample=canonical,
    )


@pytest.fixture
def coordinate_result(tmp_path):
    radar = np.arange(3 * 512, dtype=np.float32).reshape(3, 512)
    chainage = np.arange(3) * 0.025
    unresolved = _pick(1)
    unresolved.sample_index = -1
    unresolved.selected_lobe_sample = None
    unresolved.status = PickStatus.UNRESOLVED
    unresolved.visibility = VisibilityState.NOT_VISIBLE
    unresolved.twtt_ns = float("nan")
    result = AnalysisResult(
        source=AcquisitionFileSet(tmp_path / "processed.DZT"),
        header=SimpleNamespace(
            position_ns=-3.0,
            sample_interval_ns=0.029296875,
            range_ns=15.0,
            samples_per_trace=512,
            distance_per_trace_m=0.025,
        ),
        stack_size=1,
        chainage_m=chainage,
        calibrated_radargram=radar,
        surface_samples_raw=np.full(3, 102),
        reference_surface_sample=102,
        picks=[_pick(), unresolved, _pick(2, canonical=263.25, display=266)],
        thickness=[],
        review_issues=[],
        diagnostics=CalibrationDiagnostics(False),
        interpretation_input_radargram=radar,
        sample_validity=np.ones_like(radar, dtype=bool),
        parameters={"input_mode": "processed", "processed_stride": 1},
        provisional_paths={2: np.array([264, 265, 266])},
        signal_only_paths={2: np.array([264, -1, 266])},
        design_guided_paths={2: np.array([270, 271, 272])},
        candidate_events=[CandidateEvent(2, 0.0, 264, 1, 1.0, 0.0, 1)],
        search_corridors={
            2: SearchCorridor(
                2,
                chainage,
                np.full(3, 250),
                np.full(3, 260),
                np.full(3, 270),
                np.ones(3),
                np.ones(3),
                np.ones(3),
                "test radar corridor",
            )
        },
    )
    return result


@pytest.fixture
def qapp(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize("mode", ["processed", "raw"])
def test_gui_image_lines_candidates_and_clicks_share_time_coordinates(
    coordinate_result, mode, qapp, monkeypatch
):
    from PySide6.QtCore import QPointF, Qt

    from gpr_layer_audit.ui.radar_view import RadarView

    result = coordinate_result
    result.parameters["input_mode"] = mode
    origin = -3.0 if mode == "processed" else 0.0
    dt = result.header.sample_interval_ns
    view = RadarView()
    view.set_active_layer(2)
    view.set_result(result)
    expected = origin + np.array([264, np.nan, 266]) * dt
    np.testing.assert_allclose(view._pick_curves[0].yData, expected, equal_nan=True)
    np.testing.assert_allclose(
        view._pick_curves[1].yData,
        origin + np.array([np.nan, 265, np.nan]) * dt,
        equal_nan=True,
    )
    np.testing.assert_allclose(view._pick_curves[2].yData, origin + np.arange(270, 273) * dt)
    for index, sample in ((0, 250), (1, 270), (3, 260)):
        np.testing.assert_allclose(view._corridor_items[index].yData, origin + sample * dt)
    np.testing.assert_allclose(view._a_curve.yData, origin + np.arange(512) * dt)
    assert view._candidate_ridges.points()[0].pos().y() == origin + 264 * dt
    assert view._candidate_markers.points()[0].pos().y() == origin + 264 * dt
    assert view._family_preview_curves[0].yData[0] == origin + 264 * dt
    if mode == "processed":
        assert view.image.mapToParent(QPointF(0.5, 264.5)).y() == origin + 264 * dt
        for row, chainage in enumerate(result.chainage_m):
            assert view.image.mapToParent(QPointF(row + 0.5, 264.5)).x() == pytest.approx(chainage)
    else:
        assert image_time_bounds_ns(result) == (0.0, 15.0)
        assert view.image.mapToParent(QPointF(0, 0)).y() == 0
    requested, moved = [], []
    view.anchorRequested.connect(lambda *values: requested.append(values))
    view.locationChanged.connect(lambda *values: moved.append(values))
    monkeypatch.setattr(view, "_position", lambda _: (0.025, origin + 261 * dt))
    event = SimpleNamespace(
        button=lambda: Qt.MouseButton.LeftButton,
        modifiers=lambda: Qt.KeyboardModifier.ControlModifier,
        scenePos=lambda: QPointF(),
    )
    view._mouse_clicked(event)
    view._mouse_moved(QPointF())
    assert requested == [(2, 0.025, 261.0)]
    assert moved == [(0.025, origin + 261 * dt, 261)]
    monkeypatch.setattr(view, "_position", lambda _: (0.025, origin + 260.75 * dt))
    view._mouse_moved(QPointF())
    assert moved[-1][2] == (261 if mode == "processed" else 260)
    assert result.picks[0].sample_index == 261.25
    assert result.picks[0].selected_lobe_sample == 264
    view.close()


@pytest.mark.parametrize("mode", ["processed", "raw"])
def test_export_raster_and_display_lobe_use_the_same_origin(
    coordinate_result, mode, monkeypatch, tmp_path
):
    from gpr_layer_audit.export import audit

    result = coordinate_result
    result.parameters["input_mode"] = mode
    figure, axis = audit.plt.subplots()
    monkeypatch.setattr(audit.plt, "subplots", lambda **_: (figure, axis))
    audit._radargram(result, tmp_path / f"{mode}.png")
    top, bottom = axis.images[0].get_extent()[3], axis.images[0].get_extent()[2]
    origin = -3.0 if mode == "processed" else 0.0
    dt = result.header.sample_interval_ns
    if mode == "processed":
        assert top + 0.5 * dt == origin
        assert bottom - 0.5 * dt == origin + 511 * dt
        left, right = axis.images[0].get_extent()[:2]
        step = result.header.distance_per_trace_m * result.parameters["processed_stride"]
        assert left + step / 2 == result.chainage_m[0]
        assert right - step / 2 == result.chainage_m[-1]
    else:
        assert (top, bottom) == (0, 15)
    np.testing.assert_allclose(
        axis.lines[0].get_ydata(),
        origin + np.array([264, np.nan, 266]) * dt,
        equal_nan=True,
    )


def test_processed_picks_use_canonical_time_without_changing_graph_surface(
    coordinate_result, monkeypatch
):
    from gpr_layer_audit.processing import pipeline, processed_tracking

    result = coordinate_result
    recorded = []
    originals = deepcopy(result.picks)

    def interface_picks(*args):
        recorded.append(args[3])
        return deepcopy(originals)

    monkeypatch.setattr(pipeline, "_interface_picks", interface_picks)
    picks = processed_tracking._paths_to_picks(result, AnalysisOptions(), {}, {})
    assert recorded == [102]
    assert result.reference_surface_sample == 102
    assert measurement_zero_sample(result) == 102.4
    assert picks[0].twtt_ns == -3 + 261.25 * 0.029296875
    assert picks[0].twtt_ns == originals[0].twtt_ns - 0.01171875
    assert picks[0].sample_index == originals[0].sample_index
    assert picks[0].selected_lobe_sample == originals[0].selected_lobe_sample
    assert np.isnan(picks[1].twtt_ns)


def test_processed_thickness_uses_fractional_zero_only_for_surface_interval(coordinate_result):
    from gpr_layer_audit.processing.processed_tracking import _refresh_result

    result = coordinate_result
    result.picks = [
        _pick(order=1, canonical=132, display=135),
        _pick(order=2, canonical=261, display=264),
        _pick(order=3, canonical=327, display=330),
    ]
    result.parameters["dielectric_by_layer"] = {
        str(order): {"value": 4.0, "source": "analyst"} for order in (1, 2, 3)
    }
    options = AnalysisOptions(layer_specs=[LayerSpec(o, f"Layer {o}", 1, 511) for o in (1, 2, 3)])
    _refresh_result(result, options)
    first, base, subbase = result.thickness
    assert first.top_sample == 102.4
    assert first.twtt_ns == pytest.approx(-3 + 132 * 0.029296875)
    assert base.twtt_ns == (261 - 132) * 0.029296875
    assert subbase.twtt_ns == (327 - 261) * 0.029296875
    assert all(item.dielectric_source.value == "analyst" for item in result.thickness)
    assert all(item.thickness_mm is not None for item in result.thickness)
    assert result.reference_surface_sample == 102


@pytest.mark.parametrize("mode", ["processed", "raw"])
def test_confirmation_and_review_aggregation_preserve_canonical_display_roles(
    coordinate_result, mode, monkeypatch
):
    from gpr_layer_audit.processing import processed_tracking
    from gpr_layer_audit.processing.pipeline import resolve_review_issue

    result = coordinate_result
    result.parameters["input_mode"] = mode
    result.picks = [_pick(order=1, canonical=132.25, display=135)]
    result.picks[0].status = PickStatus.REVIEW
    result.picks[0].twtt_ns = float("nan")
    result.candidate_events = []
    result.review_issues = [ReviewIssue("time-review", 1, "Asphalt", 0, 0, [], "accept")]
    result.processed_paths = {
        1: SimpleNamespace(
            samples=np.array([135]),
            visible=np.array([False]),
            confidence=np.array([0.0]),
            evidence={},
        )
    }
    monkeypatch.setattr(processed_tracking, "_refresh_paths_and_requests", lambda *_: None)
    resolve_review_issue(result, AnalysisOptions(), "time-review", "accept")
    expected_zero = 102.4 if mode == "processed" else 102
    assert result.picks[0].sample_index == 132.25
    assert result.picks[0].selected_lobe_sample == 135
    assert result.picks[0].twtt_ns == pytest.approx((132.25 - expected_zero) * 0.029296875)
    assert result.thickness[0].top_sample == expected_zero
    assert result.reference_surface_sample == 102


@pytest.mark.data
def test_actual_mandiali_header_and_native_seed_times_agree_without_inference(coordinate_result):
    from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file

    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / "benchmarks/seeded-evaluation-inputs.json").read_text())["cases"][
        "mandiali-short"
    ]
    dzt_path = root / case["dzt"]
    if not dzt_path.exists():
        pytest.skip("The reviewed local Mandiali DZT is not installed")
    road = DZTFile(dzt_path)
    assert fingerprint_file(dzt_path) == case["dzt_sha256"]
    assert road.header.position_ns == -3.0
    assert road.header.sample_interval_ns == 0.029296875
    result = coordinate_result
    result.header = road.header
    assert measurement_zero_sample(result) == 102.4
    seeds = json.loads((root / case["seed_source"]).read_text())["observations"]
    for observations in seeds.values():
        for seed in observations:
            time = sample_time_ns(result, seed["sample"])
            assert time == pytest.approx(seed["time_ns"], abs=5.1e-8)
            assert sample_from_time_ns(result, time) == seed["sample"]
            assert sample_twtt_ns(result, seed["sample"]) == time
            assert road.channel()[seed["trace"], seed["sample"]] == seed["recorded_amplitude"]
