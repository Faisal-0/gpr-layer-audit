"""No fictitious horizontal measurement from an unsupported slope field."""

from pathlib import Path

import numpy as np


def test_zero_confidence_has_no_motion_preference(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dense_geometry_confidence import geometry_cost

    slope = np.array([[0, 1, -1], [0, 1, -1]], float)
    offsets = np.arange(-5, 6)
    actual = geometry_cost(slope, np.zeros_like(slope), offsets, 0.03, 0.1, 0.2, 0.15)
    np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_partial_confidence_preserves_direction_and_scales_penalty(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dense_geometry_confidence import geometry_cost

    slope = np.full((2, 1), 0.6)
    offsets = np.arange(-4, 5)
    full = geometry_cost(slope, np.ones_like(slope), offsets, 0.03, 0.1, 0.2, 0.15)
    half = geometry_cost(slope, np.full_like(slope, 0.5), offsets, 0.03, 0.1, 0.2, 0.15)
    np.testing.assert_allclose(half, full / 2)
    assert offsets[np.argmin(full[0, :, 0])] == offsets[np.argmin(half[0, :, 0])] == 2
    converted = geometry_cost(slope * 1000, np.ones_like(slope), offsets, 30, 0.1, 200, 0.15)
    np.testing.assert_allclose(converted, full)
