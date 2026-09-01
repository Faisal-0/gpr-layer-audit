from __future__ import annotations

import numpy as np

from gpr_layer_audit.io import DZTFile
from gpr_layer_audit.models import AcquisitionFileSet
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition
from gpr_layer_audit.processing.calibration import calibrate
from gpr_layer_audit.processing.preprocessing import measurement_packet_support


def test_measurement_support_prefers_a_laterally_coherent_packet():
    rng = np.random.default_rng(12)
    rows, samples = 41, 180
    data = rng.normal(0.0, 0.10, (rows, samples)).astype(np.float32)
    axis = np.arange(samples, dtype=float)
    packet = (1.0 - ((axis - 105.0) / 3.0) ** 2) * np.exp(
        -0.5 * ((axis - 105.0) / 3.0) ** 2
    )
    data += packet[None, :]

    support = measurement_packet_support(data)

    assert float(np.median(support[:, 105])) > 0.60
    assert float(np.median(support[:, 105])) > 2.0 * float(
        np.median(support[:, 35])
    )


def test_calibration_preserves_pre_subtraction_measurement(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    calibrated = calibrate(DZTFile(road_path), DZTFile(plate_path), stack_size=4)

    assert calibrated.measurement_radargram.shape == calibrated.radargram.shape
    assert not np.array_equal(calibrated.measurement_radargram, calibrated.radargram)


def test_gain_mismatched_plate_cannot_rewrite_road_measurement(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    with plate_path.open("r+b") as stream:
        stream.seek(512)
        stream.write(b"\x06\x00\x00\x00\x00\x04")

    calibrated = calibrate(DZTFile(road_path), DZTFile(plate_path), stack_size=4)

    assert not calibrated.diagnostics.gain_compatible
    assert not calibrated.diagnostics.valid_for_dielectric
    assert not calibrated.diagnostics.plate_subtraction_applied
    assert np.array_equal(calibrated.measurement_radargram, calibrated.radargram)
    assert np.any(calibrated.plate_template)
    assert any(
        "waveform subtraction skipped" in message
        for message in calibrated.diagnostics.messages
    )


def test_analysis_exposes_true_pre_subtraction_measurement(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    expected = calibrate(DZTFile(road_path), DZTFile(plate_path), stack_size=4)
    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, auto_fine_retrack=False),
    )

    assert np.array_equal(
        result.interpretation_input_radargram,
        expected.measurement_radargram,
    )
    assert np.array_equal(
        result.display_radargrams["Pre-subtraction measurement"],
        expected.measurement_radargram,
    )
    assert any(item.evidence.measurement_support > 0 for item in result.picks)
    assert all(
        item.evidence.measurement_support_gate > 0.5
        for item in result.picks
        if item.sample_index >= 0
    )
