from __future__ import annotations

import numpy as np

from gpr_layer_audit.models import LayerSpec, SearchCorridor
from gpr_layer_audit.processing.tracker import pick_interfaces


def _ricker(axis: np.ndarray, centre: float, width: float, amplitude: float) -> np.ndarray:
    z = (axis - centre) / width
    return amplitude * (1.0 - z**2) * np.exp(-0.5 * z**2)


def test_narrow_design_corridor_cannot_accept_or_inflate_a_deep_alternate():
    rows, samples = 140, 190
    axis = np.arange(samples, dtype=float)
    intended = np.rint(108.0 + 3.0 * np.sin(np.arange(rows) / 12.0)).astype(int)
    radargram = np.asarray(
        [
            _ricker(axis, 70.0, 2.3, 1.2)
            + _ricker(axis, float(intended[row]), 2.7, 0.78)
            + _ricker(axis, 148.0, 2.8, 1.25)
            for row in range(rows)
        ],
        dtype=np.float32,
    )
    layers = [
        LayerSpec(1, "Asphalt", 20, 55, 4),
        LayerSpec(2, "Base", 50, 125, 7),
    ]
    seed_rows = (5, 134)
    anchors = {
        1: {row: 70 for row in seed_rows},
        2: {row: int(intended[row]) for row in seed_rows},
    }
    lower = np.full(rows, 140.0)
    centre = np.full(rows, 148.0)
    upper = np.full(rows, 156.0)
    corridor = SearchCorridor(
        layer_order=2,
        chainage_m=np.arange(rows, dtype=float) * 0.4,
        lower_sample=lower,
        centre_sample=centre,
        upper_sample=upper,
        gap_lower_samples=lower - 70.0,
        gap_centre_samples=centre - 70.0,
        gap_upper_samples=upper - 70.0,
        source="test-narrow-wrong-design",
    )

    radar = pick_interfaces(radargram, 40, layers, anchor_samples=anchors)[2]
    designed = pick_interfaces(
        radargram,
        40,
        layers,
        anchor_samples=anchors,
        search_corridors={2: corridor},
    )[2]

    design_graph = designed.evidence["design_guided_graph_selected_sample"]
    assert np.mean(design_graph < 0) > 0.50
    np.testing.assert_array_equal(designed.samples, radar.samples)
    np.testing.assert_array_equal(designed.confidence, radar.confidence)
    assert np.all(designed.evidence["selected_design_path"] == 0.0)
    for row in seed_rows:
        assert designed.samples[row] == intended[row]
        assert designed.confidence[row] == 1.0
