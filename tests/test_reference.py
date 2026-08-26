from __future__ import annotations

from pathlib import Path

import pytest

from gpr_layer_audit.models import AcquisitionFileSet
from gpr_layer_audit.processing import AnalysisOptions, analyze_acquisition
from gpr_layer_audit.reference import (
    ManualReferencePoint,
    evaluate_manual_reference,
    read_manual_reference,
)


def test_reference_diagnostic_is_comparison_only(synthetic_acquisition):
    road_path, plate_path, _ = synthetic_acquisition
    result = analyze_acquisition(
        AcquisitionFileSet(road_path),
        AcquisitionFileSet(plate_path),
        AnalysisOptions(stack_size=4, accept_scan_dielectric=True),
    )
    samples_before = [item.sample_index for item in result.picks]
    diagnostics = evaluate_manual_reference(
        result,
        [ManualReferencePoint(1, 5.0, 55.0), ManualReferencePoint(2, 10.0, 140.0)],
    )
    assert len(diagnostics) == 2
    assert all(item.measured_interface_depth_mm is not None for item in diagnostics)
    assert [item.sample_index for item in result.picks] == samples_before


@pytest.mark.data
def test_talagang_reference_flags_interpolated_cells():
    path = Path("GPR Data/talagang/Talagang_layer_thickness_.xlsx")
    if not path.exists():
        pytest.skip("Talagang manual reference not available")
    points = read_manual_reference(path)
    assert len(points) == 756
    assert sum(item.interpolated for item in points if item.layer_order == 1) == 11
    assert sum(item.interpolated for item in points if item.layer_order == 2) == 94
