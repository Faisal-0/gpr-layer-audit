"""Exact inference and upstream numerical contracts, independent of road labels."""

from itertools import product

import numpy as np
import pytest

from gpr_layer_audit.processing.seeded_challenger import (
    PacketConfig,
    dense_interval_dp,
    pick_seeded_packet,
    pyseistr_geometry,
    tensor_geometry,
)


@pytest.mark.parametrize("left,right", [(None, None), (0, 2), (None, 1), (2, None)])
def test_exact_dp_min_marginals_match_enumeration(left, right):
    rng = np.random.default_rng(35)
    unary = rng.uniform(size=(4, 3))
    transition = rng.uniform(size=(3, 3, 3))
    path, marginals = dense_interval_dp(unary, transition, left, right)
    reference = np.full_like(unary, np.inf)
    all_paths = []
    for route in product(range(3), repeat=4):
        if (left is not None and route[0] != left) or (right is not None and route[-1] != right):
            continue
        if any(abs(b - a) > 1 for a, b in zip(route, route[1:], strict=False)):
            continue
        cost = sum(unary[r, c] for r, c in enumerate(route)) + sum(
            transition[r, b - a + 1, b]
            for r, (a, b) in enumerate(zip(route, route[1:], strict=False))
        )
        all_paths.append((cost, route))
        for r, c in enumerate(route):
            reference[r, c] = min(reference[r, c], cost)
    np.testing.assert_allclose(marginals, reference)
    assert tuple(path) == min(all_paths)[1]


def test_no_route_returns_explicit_gap():
    path, margin = dense_interval_dp(np.zeros((2, 4)), np.zeros((1, 3, 4)), 0, 3)
    assert np.all(path == -1)
    assert np.all(np.isinf(margin))


def _radar(rows=60, samples=128, slope=0.2):
    x = np.arange(rows)[:, None]
    t = np.arange(samples)[None, :]
    u = (t - 45 - slope * x) / 2.5
    return ((1 - u * u) * np.exp(-u * u / 2)).astype(np.float32)


@pytest.mark.parametrize("guide_weight", [0, 1])
def test_exact_clicks_and_masks_survive_packet_tracking(guide_weight):
    data = _radar(slope=0)
    valid = np.ones(data.shape, bool)
    valid[28:32] = False
    data[28:32] = 0
    anchors = {3: 46, 23: 46, 55: 46}
    path = pick_seeded_packet(
        data,
        anchors,
        valid=valid,
        dt_ns=0.03,
        dx_m=0.1,
        pulse_width_samples=6,
        config=PacketConfig(minimum_margin=0, guide_weight=guide_weight),
    )
    assert all(path.samples[r] == s for r, s in anchors.items())
    assert np.all(path.samples[28:32] == -1)
    assert np.all(path.provisional_samples[28:32] == -1)
    assert not np.any(path.interpolated)


def test_no_seed_means_no_physical_identity():
    data = _radar()
    path = pick_seeded_packet(
        data, {}, valid=np.ones_like(data, bool), dt_ns=0.03, dx_m=0.1, pulse_width_samples=6
    )
    assert not np.any(path.visible)
    assert np.all(path.samples == -1)


def test_structural_break_does_not_transfer_seed_identity():
    data = _radar(slope=0)
    path = pick_seeded_packet(
        data,
        {3: 45, 20: 45},
        valid=np.ones_like(data, bool),
        dt_ns=0.03,
        dx_m=0.1,
        pulse_width_samples=6,
        breaks=(30,),
    )
    assert np.all(path.provisional_samples[30:] == -1)


def test_tensor_axes_and_units_and_invalid_neighborhood():
    data = _radar()
    valid = np.ones_like(data, bool)
    slope, confidence = tensor_geometry(data, valid, 0.03, 0.1)
    assert np.median(slope[10:45, 43:60]) == pytest.approx(0.06, abs=0.015)
    valid[30] = False
    _, confidence = tensor_geometry(data, valid, 0.03, 0.1)
    assert not np.any(confidence[27:34])


def test_upstream_adapter_axes_units_and_python_mask_guard():
    pytest.importorskip("pyseistr")
    data = _radar(rows=40)
    valid = np.ones_like(data, bool)
    slope, support = pyseistr_geometry(data, valid, 0.03, 0.1)
    assert slope.shape == data.shape
    assert np.median(slope[10:30, 43:55]) == pytest.approx(0.06, abs=0.025)
    valid[20] = False
    masked, support = pyseistr_geometry(data, valid, 0.03, 0.1)
    assert not np.any(masked[15:26])
    assert not np.any(support[15:26])
    with pytest.raises(ValueError, match="does not implement masks"):
        pyseistr_geometry(data, valid, 0.03, 0.1, implementation="python")


def test_tracking_cancellation_is_observed():
    data = _radar(rows=100)
    with pytest.raises(InterruptedError):
        pick_seeded_packet(
            data,
            {0: 45, 99: 65},
            valid=np.ones_like(data, bool),
            dt_ns=0.03,
            dx_m=0.1,
            pulse_width_samples=6,
            cancel=lambda: True,
        )


def test_top_three_lobe_retention_cannot_establish_global_identity():
    from gpr_layer_audit.processing.hybrid import _ranked_lobe_matches

    # The endpoint-constrained globally correct fourth lobe is only 0.06 below
    # the local best, but disappears before any exact path inference can see it.
    scores = np.array([[0.95, 0.93, 0.91, 0.89]])
    same_lobe = np.eye(4, dtype=bool)
    production = _ranked_lobe_matches(scores, same_lobe)
    generous = _ranked_lobe_matches(scores, same_lobe, count=4)
    assert not production[0, 3]
    assert generous[0, 3]
