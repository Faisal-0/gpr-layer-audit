"""Explicit waveform modes, invalid boundaries and exact competing-route costs."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from gpr_layer_audit.processing.seeded_challenger import dense_interval_dp, pick_seeded_packet


@pytest.fixture
def mechanism(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import dense_basin_emissions

    return dense_basin_emissions


def test_split_same_sign_peaks_and_ambiguous_valley(mechanism):
    data = np.array([[0, 1, 3, 2, 1, 2, 4, 2, 0]], float)
    mapping, peaks = mechanism.emission_map(data, np.ones_like(data, bool), data != 0)
    np.testing.assert_array_equal(mapping, [[-1, 2, 2, 2, -1, 6, 6, 6, -1]])
    assert set(np.flatnonzero(peaks[0])) == {2, 6}
    other = mechanism.other_events(np.arange(9, dtype=float), mapping[0], 2)
    assert np.isinf(other[1:4]).all()
    assert np.isfinite(other[6])  # The second peak must survive the same sign.


def test_invalid_neighbors_cannot_change_measured_modes(mechanism):
    data = np.array([[0, 1, 3, 2, 5, 6, 2, 4, 2, 0]], float)
    valid = np.ones_like(data, bool)
    valid[:, 4:6] = False
    first = mechanism.emission_map(data, valid, valid)
    changed = data.copy()
    changed[~valid] = np.nan
    second = mechanism.emission_map(changed, valid, valid)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
    assert not first[1][0, 3] and not first[1][0, 6]


def test_emission_min_marginal_matches_exhaustive_routes(mechanism):
    rng = np.random.default_rng(410)
    unary, edges = rng.random((4, 5)), rng.random((3, 3, 5))
    selected, costs = dense_interval_dp(unary, edges)
    identities = np.tile([1, 1, -1, 4, 4], (4, 1))
    routes = []
    for route in product(range(5), repeat=4):
        if any(abs(b - a) > 1 for a, b in zip(route, route[1:], strict=False)):
            continue
        total = sum(unary[r, s] for r, s in enumerate(route))
        total += sum(
            edges[r, b - a + 1, b] for r, (a, b) in enumerate(zip(route, route[1:], strict=False))
        )
        routes.append((route, total))
    for row, sample in enumerate(selected):
        candidates = mechanism.other_events(costs[row], identities[row], sample)
        expected = min(
            cost
            for route, cost in routes
            if (
                identities[row, route[row]] != identities[row, sample]
                if identities[row, sample] >= 0
                else route[row] != sample
            )
        )
        assert min(candidates) == pytest.approx(expected)


def test_exact_peak_seeds_and_no_measurements_in_dropout(mechanism):
    columns = np.arange(50)
    trace = np.exp(-(((columns - 15) / 3) ** 2)) - 0.6 * np.exp(-(((columns - 25) / 3) ** 2))
    data = np.tile(trace, (30, 1)).astype(np.float32)
    valid = np.ones_like(data, bool)
    valid[12:15] = False
    picker, _ = mechanism.basin_picker(pick_seeded_packet)
    result = picker(
        data,
        {2: 15, 27: 15},
        valid=valid,
        dt_ns=0.03,
        dx_m=0.1,
        pulse_width_samples=5,
        reference_surface=0,
    )
    assert result.samples[2] == result.samples[27] == 15
    assert np.all(result.samples[12:15] == -1)
    assert np.all(result.provisional_samples[12:15] == -1)
    with pytest.raises(ValueError, match="exact supported measured extrema"):
        picker(
            data,
            {2: 14, 27: 15},
            valid=valid,
            dt_ns=0.03,
            dx_m=0.1,
            pulse_width_samples=5,
            reference_surface=0,
        )
