"""Independent arithmetic and exact-route checks for frozen local patch edges."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from gpr_layer_audit.processing import seeded_challenger as dense


@pytest.fixture
def mechanism(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import local_patch_edge_model

    return local_patch_edge_model


def _patches(values, support=None):
    values = np.asarray(values, np.float32)
    if support is None:
        support = np.ones(values.shape, bool)
    return np.stack((values, support), axis=1)


def test_ncc_matches_centered_common_pixel_arithmetic(mechanism):
    rng = np.random.default_rng(472)
    a, b = rng.normal(size=(2, 3, 5, 7)).astype(np.float32)
    # Match the production temporal L2 normalization, independently for each A-scan.
    a /= np.linalg.norm(a, axis=2, keepdims=True)
    b /= np.linalg.norm(b, axis=2, keepdims=True)
    left, right = _patches(a), _patches(b)
    left[:, 1, 0, :2] = 0
    right[:, 1, 4, -3:] = 0
    correlation, supported = mechanism.common_patch_ncc(left, right)
    assert supported.all()
    for row in range(len(a)):
        common = (left[row, 1] > 0) & (right[row, 1] > 0)
        expected = np.corrcoef(a[row][common].astype(float), b[row][common].astype(float))[0, 1]
        assert correlation[row] == pytest.approx(expected, abs=1e-12)

    # Values outside either member's mask cannot enter centering or variance.
    left[:, 0][left[:, 1] == 0] = np.nan
    right[:, 0][right[:, 1] == 0] = 1e30
    changed = mechanism.common_patch_ncc(left, right)
    np.testing.assert_array_equal(changed[0], correlation)
    np.testing.assert_array_equal(changed[1], supported)


def test_common_support_threshold_and_variance_apply_to_both_arms(mechanism):
    values = np.tile(np.arange(1365, dtype=np.float32).reshape(21, 65), (6, 1, 1))
    left, right = _patches(values), _patches(values * 2 + 3)
    # A 21x65 patch requires ceil(0.75*1365)=1024 common pixels.
    left[0, 1].flat[1024:] = 0
    left[1, 1].flat[1023:] = 0
    # Both individual masks exceed 75%, but their intersection does not.
    left[2, 1].flat[:200] = 0
    right[2, 1].flat[-200:] = 0
    # Whole-patch variance cannot substitute for variance of common observations.
    left[3, 0] = 5
    left[3, 0].flat[-100:] = 7
    right[3, 1].flat[-100:] = 0
    source_usable = np.array([True, True, True, True, False, True])
    target_usable = np.array([True, True, True, True, True, False])
    correlation, supported = mechanism.common_patch_ncc(
        left, right, source_usable, target_usable
    )
    np.testing.assert_array_equal(supported, [True, False, False, False, False, False])
    assert correlation[0] == pytest.approx(1)
    classical, learned = mechanism.edge_costs(correlation, np.ones(6), supported)
    np.testing.assert_array_equal(classical[1:], np.full(5, 0.65))
    np.testing.assert_array_equal(learned[1:], np.full(5, 0.65))


def test_signed_ncc_costs_and_forward_model_order(mechanism):
    values = np.arange(35, dtype=np.float32).reshape(1, 5, 7)
    correlation, support = mechanism.common_patch_ncc(_patches(values), _patches(-values))
    assert support[0] and correlation[0] == pytest.approx(-1)
    classical, learned = mechanism.edge_costs(
        np.array([-1, 0, 1, np.nan]), np.array([0, 0.5, 1, np.nan]),
        np.array([True, True, True, False]),
    )
    np.testing.assert_array_equal(classical, [2, 1, 0, 0.65])
    np.testing.assert_array_equal(learned, classical)

    class DirectionSensitiveMatcher:
        def compare(self, candidate, seed):
            return 2 * candidate - seed

    source = np.array([[1, 4], [3, 8]], np.float32)
    target = np.array([[7, 2], [9, 6]], np.float32)
    np.testing.assert_array_equal(
        mechanism.compare_forward(DirectionSensitiveMatcher(), source, target),
        2 * target - source,
    )


def _enumerate(unary, transitions, left, right):
    rows, columns = unary.shape
    radius = transitions.shape[1] // 2
    reference = np.full_like(unary, np.inf)
    routes = []
    for route in product(range(columns), repeat=rows):
        if route[0] != left or route[-1] != right:
            continue
        if any(abs(b - a) > radius for a, b in zip(route, route[1:], strict=False)):
            continue
        total = sum(unary[r, s] for r, s in enumerate(route))
        total += sum(
            transitions[r, b - a + radius, b]
            for r, (a, b) in enumerate(zip(route, route[1:], strict=False))
        )
        routes.append((total, route))
        for row, sample in enumerate(route):
            reference[row, sample] = min(reference[row, sample], total)
    return min(routes), reference


def test_injected_edge_axes_scaling_and_all_min_marginals_are_exact(mechanism, monkeypatch):
    rng = np.random.default_rng(914)
    radar = rng.normal(size=(5, 5)).astype(np.float32)
    per_metre = rng.uniform(0.1, 2, size=(4, 3, 5))
    original_edges = per_metre.copy()
    capture = []
    real_dp = dense.dense_interval_dp

    def record(unary, transitions, left, right, cancel=None):
        path, marginal = real_dp(unary, transitions, left, right, cancel)
        capture.append((unary.copy(), transitions.copy(), left, right, path, marginal))
        return path, marginal

    monkeypatch.setattr(dense, "dense_interval_dp", record)
    picker = mechanism.instrument(dense.pick_seeded_packet, per_metre)
    for dx in (0.1, 0.2):
        result = picker(
            radar, {0: 1, 4: 3}, valid=np.ones_like(radar, bool), dt_ns=0.25,
            dx_m=dx, pulse_width_samples=1, context_radius=1, reference_surface=-1,
            config=dense.PacketConfig(
                geometry="none", geometry_weight=0, max_slope_ns_per_m=1
            ),
        )
        assert result.samples[0] == 1 and result.samples[4] == 3
        unary, transition, left, right, path, marginal = capture[-1]
        np.testing.assert_array_equal(transition, per_metre * dx)
        best, expected = _enumerate(unary, transition, left, right)
        assert tuple(path) == best[1]
        np.testing.assert_allclose(marginal, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(capture[0][4], capture[1][4])
    np.testing.assert_allclose(capture[1][5], 2 * capture[0][5], rtol=1e-12)
    np.testing.assert_array_equal(per_metre, original_edges)


def test_zero_edges_reproduce_every_output_and_candidate(mechanism):
    rng = np.random.default_rng(116)
    data = rng.normal(size=(9, 25)).astype(np.float32)
    valid = np.ones_like(data, bool)
    valid[4] = False
    config = dense.PacketConfig(geometry="none", max_slope_ns_per_m=1, guide_weight=1)
    kwargs = dict(
        valid=valid, dt_ns=0.2, dx_m=0.1, pulse_width_samples=2, context_radius=2,
        reference_surface=0, config=config,
    )
    baseline = dense.pick_seeded_packet(data, {1: 8, 7: 10}, **kwargs)
    changed = mechanism.instrument(dense.pick_seeded_packet, np.zeros((8, 3, 25)))(
        data, {1: 8, 7: 10}, **kwargs
    )
    for field in (
        "samples", "confidence", "alternate_samples", "visible", "interpolated",
        "provisional_samples", "signal_only_samples", "design_guided_samples",
    ):
        np.testing.assert_array_equal(getattr(changed, field), getattr(baseline, field))
    for key, value in baseline.candidate_components.items():
        np.testing.assert_array_equal(changed.candidate_components[key], value)
    for key, value in baseline.evidence.items():
        np.testing.assert_array_equal(changed.evidence[key], value)
    assert changed.provisional_samples[4] == changed.samples[4] == -1
