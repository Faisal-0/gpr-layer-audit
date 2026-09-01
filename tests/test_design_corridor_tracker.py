from __future__ import annotations

import numpy as np
import pytest

from gpr_layer_audit.models import (
    DielectricSource,
    LayerDesign,
    LayerSpec,
    SeedStation,
    VisibilityState,
)
from gpr_layer_audit.processing.corridor import build_search_corridors
from gpr_layer_audit.processing.tracker import pick_interfaces


def _pulse(axis: np.ndarray, centre: float, width: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((axis - centre) / width) ** 2)


def _corridors(
    rows: int,
    layers: list[LayerSpec],
    designs: list[LayerDesign],
    seeds: list[SeedStation] | None = None,
):
    return build_search_corridors(
        np.arange(rows, dtype=float) * 0.4,
        40,
        0.029296875,
        layers,
        designs,
        [],
        {order: (7.0, DielectricSource.ASSUMED_SCAN) for order in range(1, 4)},
        seeds or [],
    )


def test_talagang_design_maps_to_expected_interface_samples():
    layers = LayerSpec.defaults()
    corridors = build_search_corridors(
        np.arange(4, dtype=float),
        156,
        0.029296875,
        layers,
        [
            LayerDesign(1, "Asphalt bottom", 50.8, dielectric=7.0),
            LayerDesign(2, "Base bottom", 101.6, dielectric=7.0),
            LayerDesign(3, "Subbase bottom", None, dielectric=7.0),
        ],
        [],
        {order: (7.0, DielectricSource.ASSUMED_SCAN) for order in range(1, 4)},
        [],
    )

    np.testing.assert_allclose(corridors[1].centre_sample[0], 187, atol=0.6)
    np.testing.assert_allclose(corridors[2].centre_sample[0], 248, atol=0.8)


def test_design_corridor_is_an_audit_alternate_not_a_measurement_override():
    rows, samples = 100, 170
    axis = np.arange(samples, dtype=float)
    data = np.asarray(
        [
            _pulse(axis, 71 + 1.5 * np.sin(row / 20), 2.2, 0.75)
            + _pulse(axis, 128, 2.2, 1.8)
            for row in range(rows)
        ]
    )
    layer = LayerSpec(1, "Asphalt bottom", 10, 110, 3)
    corridors = _corridors(
        rows, [layer], [LayerDesign(1, layer.name, 50.8, dielectric=7.0)]
    )

    radar = pick_interfaces(data, 40, [layer])[1]
    path = pick_interfaces(data, 40, [layer], search_corridors=corridors)[1]

    np.testing.assert_array_equal(path.samples, radar.samples)
    np.testing.assert_array_equal(path.confidence, radar.confidence)
    np.testing.assert_allclose(
        np.nanmedian(path.design_guided_samples[path.design_guided_samples >= 0]),
        71,
        atol=3,
    )
    assert np.mean(path.design_conflict) > 0.9


def test_design_alone_cannot_manufacture_a_visible_layer_from_noise():
    rows, samples = 90, 150
    data = np.random.default_rng(12).normal(0.0, 0.002, (rows, samples))
    layer = LayerSpec(1, "Asphalt bottom", 10, 100, 3)
    corridors = _corridors(
        rows, [layer], [LayerDesign(1, layer.name, 50.8, dielectric=7.0)]
    )

    path = pick_interfaces(data, 40, [layer], search_corridors=corridors)[1]

    assert np.mean(path.samples >= 0) < 0.05
    assert np.max(path.confidence) < 0.45


def test_two_seeds_recenter_an_incorrect_tentative_design():
    rows, samples = 100, 180
    axis = np.arange(samples, dtype=float)
    data = np.asarray([_pulse(axis, 102, 2.4) for _ in range(rows)])
    layer = LayerSpec(1, "Asphalt bottom", 10, 130, 3)
    chainage = np.arange(rows, dtype=float) * 0.4
    seeds = [
        SeedStation(
            "a",
            float(chainage[10]),
            {1: 102.0},
            {1: VisibilityState.VISIBLE},
            user_confirmed={1: True},
        ),
        SeedStation(
            "b",
            float(chainage[90]),
            {1: 102.0},
            {1: VisibilityState.VISIBLE},
            user_confirmed={1: True},
        ),
    ]
    corridors = build_search_corridors(
        chainage,
        40,
        0.029296875,
        [layer],
        [LayerDesign(1, layer.name, 20.0, dielectric=7.0)],
        [],
        {1: (7.0, DielectricSource.ASSUMED_SCAN)},
        seeds,
    )

    path = pick_interfaces(
        data,
        40,
        [layer],
        anchor_samples={1: {10: 102, 90: 102}},
        search_corridors=corridors,
    )[1]

    assert np.mean(np.abs(path.samples[path.samples >= 0] - 102)) < 2.0


def test_one_disagreeing_seed_does_not_widen_the_global_corridor():
    rows = 120
    chainage = np.arange(rows, dtype=float) * 0.4
    layer = LayerSpec(1, "Asphalt bottom", 10, 140, 3)
    seeds = [
        SeedStation(
            f"seed-{index}",
            float(chainage[row]),
            {1: float(sample)},
            {1: VisibilityState.VISIBLE},
            user_confirmed={1: True},
        )
        for index, (row, sample) in enumerate(((10, 100), (60, 102), (110, 150)))
    ]

    corridor = build_search_corridors(
        chainage,
        40,
        0.029296875,
        [layer],
        [LayerDesign(1, layer.name, 100.0, dielectric=7.0)],
        [],
        {1: (7.0, DielectricSource.ASSUMED_SCAN)},
        seeds,
    )[1]

    assert "local_seed_outlier" in corridor.source
    assert corridor.seed_outlier_chainages_m == [pytest.approx(chainage[110])]
    assert np.median(corridor.gap_centre_samples) == pytest.approx(61.0, abs=2.0)
    assert np.median(corridor.gap_upper_samples - corridor.gap_centre_samples) <= 15.0


def test_anomaly_mask_breaks_all_interface_paths_without_crossing_order():
    rows, samples = 80, 180
    axis = np.arange(samples, dtype=float)
    data = np.asarray(
        [
            _pulse(axis, 70, 2.2)
            + _pulse(axis, 105, 2.5, 0.8)
            + _pulse(axis, 145, 3.0, 0.65)
            for _ in range(rows)
        ]
    )
    layers = [
        LayerSpec(1, "Asphalt bottom", 10, 60, 3),
        LayerSpec(2, "Base bottom", 35, 105, 5),
        LayerSpec(3, "Subbase bottom", 70, 140, 5),
    ]
    designs = [
        LayerDesign(1, layers[0].name, 50.8, dielectric=7.0),
        LayerDesign(2, layers[1].name, 58.0, dielectric=7.0),
        LayerDesign(3, layers[2].name, 66.0, dielectric=7.0),
    ]
    anomaly = np.zeros(rows, dtype=bool)
    anomaly[34:45] = True

    paths = pick_interfaces(
        data,
        40,
        layers,
        search_corridors=_corridors(rows, layers, designs),
        anomaly_mask=anomaly,
    )

    for path in paths.values():
        assert np.all(path.samples[34:45] == -1)
    valid = np.all(np.column_stack([paths[order].samples for order in (1, 2, 3)]) >= 0, axis=1)
    picked = np.column_stack([paths[order].samples for order in (1, 2, 3)])[valid]
    assert np.all(np.diff(picked, axis=1) > 0)
