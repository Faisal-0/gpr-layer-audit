from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from conftest import pulse, write_dzt

from gpr_layer_audit.models import AcquisitionFileSet, LayerSpec, SeedStation
from gpr_layer_audit.processing.active_queries import request_observation
from gpr_layer_audit.processing.pipeline import (
    AnalysisOptions,
    analyze_acquisition,
    retrack_segment,
)
from gpr_layer_audit.processing.processed_tracking import fit_processed, native_anchors


def station(row, sample, *, role="initial"):
    return SeedStation(
        str(row), row * 0.1, samples={2: sample}, user_confirmed={2: True}, role=role
    )


@pytest.fixture
def processed_case(tmp_path):
    measurement = np.tile(10000 * pulse(128, 70), (61, 1))
    measurement += np.tile(20000 * pulse(128, 91), (61, 1))
    measurement[29:32] = 0
    path = tmp_path / "processed.DZT"
    write_dzt(path, measurement, scans_per_meter=10)
    options = AnalysisOptions(
        input_mode="processed",
        tracker_method="seed_hybrid",
        layer_specs=[LayerSpec(2, "Base", 1, 127, 1)],
        seed_stations=[station(r, 70) for r in (5, 25, 55)],
    )
    return AcquisitionFileSet(path), options


def test_processed_application_and_public_tracker_share_native_coordinates(processed_case):
    source, options = processed_case
    result = analyze_acquisition(source, options=options)
    anchors = native_anchors(options.seed_stations, result.chainage_m)
    paths = fit_processed(
        result.interpretation_input_radargram,
        result.sample_validity,
        0,
        result.header.sample_interval_ns,
        0.1,
        options.layer_specs,
        anchors,
    )
    assert result.parameters["coordinate_provenance"]["trace_stride"] == 1
    assert result.reference_surface_sample == 0
    for pick, sample in zip(result.picks, paths[2].samples, strict=True):
        assert pick.selected_lobe_sample == (sample if sample >= 0 else None)
    for row in (5, 25, 55):
        assert result.picks[row].selected_lobe_sample == 70
    assert all(not np.isfinite(result.picks[row].twtt_ns) for row in (29, 30, 31))


def test_local_correction_keeps_outside_objects_and_cancellation_transactional(processed_case):
    source, options = processed_case
    result = analyze_acquisition(source, options=options)
    before = list(result.picks)
    options.seed_stations.append(station(40, 91, role="correction"))
    with pytest.raises(InterruptedError):
        retrack_segment(result, options, 3.8, 4.2, cancel=lambda: True)
    assert all(a is b for a, b in zip(before, result.picks, strict=True))
    retrack_segment(result, options, 3.8, 4.2)
    for old, current in zip(before, result.picks, strict=True):
        if not 3.8 <= current.chainage_m <= 4.2:
            assert current is old
    assert result.picks[40].selected_lobe_sample == 91
    assert result.parameters["processed_local_retracks"][-1]["layer_orders"] == [2]


def test_incompatible_processed_grid_fails_without_snapping():
    with pytest.raises(ValueError, match="exact native trace"):
        native_anchors([station(5, 70)], np.arange(16) * 0.4)


def test_requests_are_reference_blind_skip_gaps_and_visited(processed_case):
    source, options = processed_case
    result = analyze_acquisition(source, options=options)
    anchors = native_anchors(options.seed_stations, result.chainage_m)
    paths = fit_processed(
        result.interpretation_input_radargram,
        result.sample_validity,
        0,
        result.header.sample_interval_ns,
        0.1,
        options.layer_specs,
        anchors,
    )
    before = deepcopy(anchors)
    first = request_observation(
        paths, result.interpretation_input_radargram, result.sample_validity, anchors, 0.1
    )
    second = request_observation(
        paths,
        result.interpretation_input_radargram,
        result.sample_validity,
        anchors,
        0.1,
        visited={(first["layer_order"], first["row"])},
    )
    assert first["row"] not in (5, 25, 29, 30, 31, 55)
    assert second["row"] != first["row"]
    assert anchors == before
    assert "references" not in asdict(options)


def test_new_local_observation_does_not_recut_original_seed_packets():
    from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse

    radar = -np.ones((4, 96))
    for row, width in enumerate((7, 9, 11, 15)):
        radar[row, 48 - width // 2 : 49 + width // 2] = 1
    valid = np.ones_like(radar, dtype=bool)
    original = {0: 48, 1: 48, 2: 48}
    corrected = {**original, 3: 48}
    config = ConventionalConfig()
    initial = resolve_pulse(radar, valid, original, {}, 0.03, config)
    refitted = resolve_pulse(radar, valid, corrected, {}, 0.03, config)
    frozen = resolve_pulse(
        radar, valid, corrected, {3: {"pulse_estimation_use": False}}, 0.03, config
    )
    assert refitted.lobe_samples != initial.lobe_samples
    assert frozen == initial
