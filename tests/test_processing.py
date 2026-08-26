from __future__ import annotations

import numpy as np
import pytest

from gpr_layer_audit.models import AcquisitionFileSet, DielectricSource, PickStatus
from gpr_layer_audit.processing import (
    AnalysisOptions,
    PreprocessingOptions,
    analyze_acquisition,
    retrack_segment,
)
from gpr_layer_audit.processing.dielectric import (
    surface_reflection_dielectric,
    thickness_from_twtt_mm,
)


def test_surface_reflection_dielectric_recovers_known_value():
    epsilon = 7.0
    ratio = (np.sqrt(epsilon) - 1) / (np.sqrt(epsilon) + 1)
    output = surface_reflection_dielectric(np.asarray([-ratio * 1000] * 9), -1000)
    assert np.nanmedian(output) == pytest.approx(epsilon, rel=0.02)


def test_thickness_conversion():
    assert thickness_from_twtt_mm(2.0, 7.0) == pytest.approx(113.3, rel=0.01)


def test_pipeline_tracks_interfaces_and_attaches_gps(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, report_interval_m=5, accept_scan_dielectric=True),
    )
    assert result.calibrated_radargram.shape == (60, 256)
    assert {item.layer_order for item in result.picks} == {1, 2, 3}
    assert all(item.twtt_ns > 0 for item in result.picks)
    assert any(item.latitude is not None for item in result.picks)
    sources = {(item.layer_order, item.dielectric_source) for item in result.thickness}
    assert (1, DielectricSource.REFLECTION) in sources
    assert (2, DielectricSource.ASSUMED_SCAN) in sources
    assert (3, DielectricSource.ASSUMED_SCAN) in sources
    assert all(
        (item.thickness_mm is None) == (item.status == PickStatus.UNRESOLVED)
        for item in result.thickness
    )
    assert all(
        item.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.REVIEW, PickStatus.UNRESOLVED}
        for item in result.picks
    )


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
    assert all(item.dielectric_source == DielectricSource.UNRESOLVED for item in result.thickness)
    assert all(item.thickness_mm is None for item in result.thickness)


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
    assert anchored.sample_index == current + 4
