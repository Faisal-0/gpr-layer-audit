"""Differential tests against the existing independent NumPy DTW implementation."""

import numpy as np
import pytest

pytest.importorskip("numba")

from gpr_layer_audit.processing.waveform_matching import batch_dtw
from gpr_layer_audit.processing.waveform_matching_fast import (
    batch_dtw_fast,
    reciprocal_dtw_fast,
)


def reference_reciprocal(left, right, band):
    forward, shift = batch_dtw(left, right, band)
    backward, reverse_shift = batch_dtw(right, left, band)
    return np.minimum(forward, backward), np.maximum(abs(shift), abs(reverse_shift))


def assert_identical(left, right, band):
    before_left, before_right = np.array(left, copy=True), np.array(right, copy=True)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        for reference, compiled in (
            (batch_dtw, batch_dtw_fast),
            (reference_reciprocal, reciprocal_dtw_fast),
        ):
            expected, observed = reference(left, right, band), compiled(left, right, band)
            for a, b in zip(expected, observed, strict=True):
                np.testing.assert_array_equal(a, b, strict=True)
    np.testing.assert_array_equal(before_left, left)
    np.testing.assert_array_equal(before_right, right)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 32, 89])
@pytest.mark.parametrize("band", [0, 1, 2, 5, 100])
def test_random_exact(n, band):
    rng = np.random.default_rng(917 + n + band)
    left, right = rng.normal(size=(2, 9, n))
    assert_identical(left, right, band)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_ties_zero_negative_and_noncontiguous(dtype):
    rng = np.random.default_rng(310)
    left = rng.integers(-2, 3, (12, 178)).astype(dtype)[::-2, ::2]
    right = rng.integers(-2, 3, (12, 178)).astype(dtype)[::-2, ::2]
    left[0], right[0] = 0, 0
    left[1], right[1] = -1, -1
    for band in (0, 1, 5, 89):
        assert_identical(left, right, band)


@pytest.mark.parametrize("shape", [(0, 0), (0, 32), (0, 89), (3, 0)])
def test_empty(shape):
    assert_identical(np.empty(shape), np.empty(shape), 3)


@pytest.mark.parametrize("band", [-1, 0.5, 2.8, np.nan, np.inf])
def test_unusual_band_reference_semantics(band):
    assert_identical(np.arange(12).reshape(2, 6), np.ones((2, 6)), band)


def test_nonfinite_values_reference_semantics():
    left = np.array([[0, np.nan, 1, 1], [np.inf, 1, -np.inf, 3], [0, 1e300, 2, 3]])
    right = np.array([[0, 1, np.nan, 2], [np.inf, 1, 2, -np.inf], [0, 0, 1e300, 3]])
    for band in (0, 1, 4):
        assert_identical(left, right, band)


@pytest.mark.parametrize(
    ("left", "right"),
    [([1, 2], [1, 2]), (np.zeros((2, 3)), np.zeros((2, 4))),
     (np.zeros((2, 3)), np.zeros((3, 3))), (0, 0)],
)
def test_invalid_shape_contract(left, right):
    for function in (batch_dtw, batch_dtw_fast):
        with pytest.raises(ValueError, match="matching shapes"):
            function(left, right, 2)
