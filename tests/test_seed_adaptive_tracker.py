from __future__ import annotations

import numpy as np
import pytest

from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.tracker import TRACKER_METHODS, pick_interfaces


def _pulse(axis: np.ndarray, centre: float, width: float) -> np.ndarray:
    z = (axis - centre) / width
    return (1.0 - z**2) * np.exp(-0.5 * z**2)


def _three_layer_radargram():
    rows, samples = 120, 220
    axis = np.arange(samples, dtype=float)
    data = np.random.default_rng(4).normal(0, 0.03, (rows, samples))
    truth = {1: [], 2: [], 3: []}
    for row in range(rows):
        change = 5 if row >= 65 else 0
        samples_at_row = {
            1: 70 + round(3 * np.sin(row / 17)) + change,
            2: 111 + round(4 * np.sin(row / 21)) + change,
            3: 158 + round(5 * np.sin(row / 25)) + change,
        }
        for order, sample in samples_at_row.items():
            truth[order].append(sample)
        data[row] += _pulse(axis, samples_at_row[1], 2.2)
        # Exercise a genuine polarity reversal halfway through the line.
        data[row] += (-0.8 if row < 60 else 0.8) * _pulse(
            axis, samples_at_row[2], 2.8
        )
        data[row] += 0.65 * _pulse(axis, samples_at_row[3], 3.2)
    layers = [
        LayerSpec(1, "Asphalt", 20, 60, 4),
        LayerSpec(2, "Base", 50, 100, 8),
        LayerSpec(3, "Subbase", 90, 150, 8),
    ]
    anchor_rows = (5, 60, 115)
    anchors = {
        order: {row: truth[order][row] for row in anchor_rows}
        for order in truth
    }
    return data, layers, truth, anchors


def test_joint_tracker_handles_three_layers_change_point_and_polarity_reversal():
    data, layers, truth, anchors = _three_layer_radargram()
    paths = pick_interfaces(data, 40, layers, anchor_samples=anchors)

    for order in (1, 2, 3):
        expected = np.asarray(truth[order])
        assert np.all(paths[order].samples >= 0)
        assert np.mean(np.abs(paths[order].samples - expected)) < 1.5
        for row, sample in anchors[order].items():
            assert paths[order].samples[row] == sample
            assert paths[order].confidence[row] == 1.0
    assert np.all(paths[2].samples - paths[1].samples >= layers[1].min_gap_samples)
    assert np.all(paths[3].samples - paths[2].samples >= layers[2].min_gap_samples)


def test_thin_overlapping_reflections_remain_ordered():
    rows, samples = 80, 150
    axis = np.arange(samples, dtype=float)
    data = np.random.default_rng(2).normal(0, 0.015, (rows, samples))
    first: list[int] = []
    second: list[int] = []
    for row in range(rows):
        top = 70 + round(2 * np.sin(row / 12))
        bottom = top + 12 + round(np.sin(row / 14))
        first.append(top)
        second.append(bottom)
        data[row] += _pulse(axis, top, 3.5)
        data[row] -= 0.8 * _pulse(axis, bottom, 3.5)
    layers = [
        LayerSpec(1, "Thin 1", 20, 45, 4),
        LayerSpec(2, "Thin 2", 31, 65, 6),
    ]
    seed_rows = (5, 40, 74)
    anchors = {
        1: {row: first[row] for row in seed_rows},
        2: {row: second[row] for row in seed_rows},
    }

    paths = pick_interfaces(data, 40, layers, anchor_samples=anchors)

    assert np.mean(np.abs(paths[1].samples - first)) < 1.5
    assert np.mean(np.abs(paths[2].samples - second)) < 1.5
    assert np.all(paths[2].samples - paths[1].samples >= 6)


def test_missing_evidence_stays_no_pick_even_with_wrong_design_prior():
    rows, samples = 100, 160
    axis = np.arange(samples, dtype=float)
    data = np.zeros((rows, samples), dtype=float)
    for row in [*range(35), *range(65, rows)]:
        data[row] = _pulse(axis, 90, 2.5)
    data += np.random.default_rng(1).normal(0, 0.005, data.shape)
    layer = LayerSpec(1, "Interface", 35, 80, 4)

    path = pick_interfaces(
        data,
        40,
        [layer],
        anchor_samples={1: {5: 90, 94: 90}},
        design_prior_samples={1: np.full(rows, 70.0)},
        design_prior_widths={1: np.full(rows, 7.0)},
        design_weight=0.20,
    )[1]

    assert np.all(path.samples[40:60] == -1)
    assert np.all(path.confidence[40:60] == 0)
    assert path.design_guided_samples is not None
    assert path.design_conflict is not None


def test_design_pass_never_changes_preserved_signal_only_path():
    data, layers, _, anchors = _three_layer_radargram()
    signal = pick_interfaces(data, 40, layers, anchor_samples=anchors)
    wrong_priors = {order: np.full(len(data), 45.0) for order in (1, 2, 3)}
    widths = {order: np.full(len(data), 7.0) for order in (1, 2, 3)}
    guided = pick_interfaces(
        data,
        40,
        layers,
        anchor_samples=anchors,
        design_prior_samples=wrong_priors,
        design_prior_widths=widths,
        design_weight=0.20,
    )

    for order in signal:
        assert np.array_equal(
            guided[order].signal_only_samples, signal[order].signal_only_samples
        )


def test_incorrect_crossing_seeds_are_rejected():
    data, layers, _, anchors = _three_layer_radargram()
    anchors[2][60] = anchors[1][60] + 2
    with pytest.raises(ValueError, match="out of order"):
        pick_interfaces(data, 40, layers, anchor_samples=anchors)


@pytest.mark.parametrize("method", TRACKER_METHODS)
def test_current_tracker_is_reproducible(method):
    data, layers, _, anchors = _three_layer_radargram()
    first = pick_interfaces(data, 40, layers, anchor_samples=anchors, method=method)
    second = pick_interfaces(data, 40, layers, anchor_samples=anchors, method=method)

    assert set(first) == {1, 2, 3}
    for order in first:
        assert np.array_equal(first[order].samples, second[order].samples)
