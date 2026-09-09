from __future__ import annotations

import numpy as np
import pytest

from gpr_layer_audit.processing.trace_registration import balance_waveforms, register_pairs


def wavelets(positions, amplitudes, samples=128):
    axis = np.arange(samples)
    return sum(a * (1 - ((axis - p) / 2.2) ** 2) * np.exp(-.5 * ((axis - p) / 2.2) ** 2)
               for p, a in zip(positions, amplitudes, strict=True)).astype(np.float32)


def test_whole_trace_warp_preserves_reflector_order_and_weak_interface():
    left = wavelets([30, 60, 83, 108], [1, -.2, -.2, .7])[None]
    right = wavelets([32, 65, 87, 109], [1.5, -.25, -.25, .9])[None]
    forward, inverse, quality, _ = register_pairs(
        balance_waveforms(left, 7), balance_waveforms(right, 7), 12, 7
    )
    for first, second in ((30, 32), (60, 65), (83, 87), (108, 109)):
        assert abs(forward[0, first] - second) <= 1
        assert abs(inverse[0, second] - first) <= 1
        assert quality[0, first] > .8
    assert np.all(np.diff(forward) >= 0)
    assert np.max(abs(forward - np.arange(128))) <= 12


def test_gain_balancing_preserves_polarity_and_does_not_fill_zero_signal():
    signal = wavelets([30, 65, 95], [1, -.25, .5])
    data = np.stack([signal, signal * 100, -signal, np.zeros_like(signal)])
    balanced = balance_waveforms(data, 7)
    assert np.all(np.isfinite(balanced))
    np.testing.assert_allclose(balanced[0], balanced[1], atol=1e-6)
    np.testing.assert_allclose(balanced[0], -balanced[2], atol=1e-6)
    assert not np.any(balanced[3])


def test_identical_and_empty_traces_keep_identity_coordinates():
    data = np.stack([wavelets([30, 65, 95], [1, -.2, .4]), np.zeros(128)])
    forward, inverse, quality, reverse_quality = register_pairs(data, data, 8, 7)
    np.testing.assert_array_equal(forward, np.tile(np.arange(128), (2, 1)))
    np.testing.assert_array_equal(inverse, forward)
    assert np.all(quality == 1) and np.all(reverse_quality == 1)
    # Agreement is registration fit, not an interface visibility declaration.


def test_registration_batch_matches_individual_results_and_cancels():
    left = np.stack([wavelets([40, 80], [1, -.3]), wavelets([45, 90], [.8, -.2])])
    right = np.stack([wavelets([43, 82], [1, -.3]), wavelets([44, 94], [.8, -.2])])
    batch = register_pairs(left, right, 8, 7)
    for row in range(2):
        single = register_pairs(left[row:row + 1], right[row:row + 1], 8, 7)
        for together, alone in zip(batch, single, strict=True):
            np.testing.assert_array_equal(together[row], alone[0])
    with pytest.raises(InterruptedError):
        register_pairs(left, right, 8, 7, cancel=lambda: True)


def test_invalid_registration_inputs_fail_clearly():
    with pytest.raises(ValueError):
        register_pairs(np.zeros((2, 8)), np.zeros((2, 9)), 3, 7)
    with pytest.raises(ValueError):
        register_pairs(np.full((1, 8), np.nan), np.zeros((1, 8)), 3, 7)
    with pytest.raises(ValueError):
        register_pairs(np.zeros((1, 8)), np.zeros((1, 8)), 0, 7)


def test_sparse_registration_does_not_cross_structural_regions():
    from gpr_layer_audit.processing.trace_registration import sparse_registration

    measurement = np.tile(wavelets([40, 80], [1, -.3]), (6, 1))
    candidates = np.full((6, 1), 80)
    regions = np.array([0, 0, 0, 1, 1, 1])
    result = sparse_registration(measurement, candidates, [1, 2], regions, 7, .4)
    low, forward, inverse, quality, reverse_quality = result[1]
    assert quality[1, 80 - low] > .99
    assert not np.any(quality[2])
    assert not np.any(reverse_quality[3])
    assert forward[1, 80 - low] == inverse[2, 80 - low] == 80


def test_registration_matches_exhaustive_tiny_alignment_cost():
    # Independent enumeration of every allowed path verifies the batched
    # band-index recurrences, including both stretching directions.
    from functools import lru_cache

    left = np.array([[.2, -.8, .1, .7, -.4]], np.float32)
    right = np.array([[.2, -.7, -.5, .6, -.3]], np.float32)
    error = (left[0, :, None] - right[0, None, :]) ** 2

    @lru_cache(None)
    def paths(i, j):
        if i == j == 0:
            return [(float(error[0, 0]), [(0, 0)])]
        if i < 0 or j < 0 or abs(i - j) > 2:
            return []
        result = []
        for di, dj in ((1, 1), (1, 2), (2, 1)):
            extra = [(i, j)]
            penalty = 0
            if dj == 2:
                extra.insert(0, (i, j - 1))
                penalty = .02
            if di == 2:
                extra.insert(0, (i - 1, j))
                penalty = .02
            for cost, path in paths(i - di, j - dj):
                result.append((cost + sum(float(error[a, b]) for a, b in extra) + penalty,
                               path + extra))
        return result

    _, expected = min(paths(4, 4), key=lambda item: item[0])
    forward, inverse, _, _ = register_pairs(left, right, 2, 2)
    for row in range(5):
        assert forward[0, row] == np.mean([j for i, j in expected if i == row])
        assert inverse[0, row] == np.mean([i for i, j in expected if j == row])


def test_same_centre_waveform_does_not_allow_a_neighbouring_cycle_to_swap_order():
    # The two identical negative packets are ambiguous in isolation. The
    # unequal neighbours provide an ordered context for their identities.
    left = wavelets([24, 50, 70, 100], [.8, -.2, -.2, .5])[None]
    right = wavelets([27, 53, 72, 104], [.8, -.2, -.2, .5])[None]
    forward, inverse, quality, _ = register_pairs(
        balance_waveforms(left, 7), balance_waveforms(right, 7), 24, 7
    )
    assert abs(forward[0, 50] - 53) <= 1
    assert abs(forward[0, 70] - 72) <= 1
    assert abs(inverse[0, 53] - 50) <= 1
    assert quality[0, 50] > .8


def test_sparse_reverse_evidence_comes_from_an_independent_alignment(monkeypatch):
    import gpr_layer_audit.processing.trace_registration as registration

    calls = []

    def align(left, right, *args):
        calls.append((left.copy(), right.copy()))
        marker = 3 if len(calls) == 1 else 7
        return (np.full_like(left, marker), np.full_like(left, -999),
                np.full_like(left, .9 if marker == 3 else .8), np.zeros_like(left))

    monkeypatch.setattr(registration, "register_pairs", align)
    data = np.stack([wavelets([40, 80], [1, -.3]), wavelets([42, 83], [1, -.3])])
    result = registration.sparse_registration(data, np.full((2, 1), 80), [1], np.zeros(2), 7, .4)
    origin, forward, backward, quality, reverse_quality = result[1]
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], calls[1][1])
    np.testing.assert_array_equal(calls[0][1], calls[1][0])
    assert np.all(forward[0] == origin + 3)
    assert np.all(backward[1] == origin + 7)
    assert np.allclose(quality[0], .9) and np.allclose(reverse_quality[1], .8)


def test_global_registration_masks_invalid_amplitudes_and_fit_evidence():
    from gpr_layer_audit.processing.trace_registration import sparse_registration

    data = np.tile(wavelets([40, 80], [1, -.3]), (3, 1))
    valid = np.ones_like(data, bool)
    valid[:, :2] = False
    damaged = data.copy()
    damaged[:, :2] = 1e25
    np.testing.assert_allclose(balance_waveforms(data, 7, valid),
                               balance_waveforms(damaged, 7, valid))
    valid[1, 80] = False
    origin, _, _, quality, reverse = sparse_registration(
        damaged, np.full((3, 1), 80), [1], np.zeros(3), 7, .4, valid=valid
    )[1]
    assert quality[1, 80 - origin] == reverse[1, 80 - origin] == 0
