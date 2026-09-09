from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.processing.continuation import _advect
from gpr_layer_audit.processing.motion_prediction import compose_motion, contextual_waveforms


def motion_maps():
    shape = (15, 64)
    forward = np.tile(np.array([1, 1, -1, -1, 0] * 3)[:, None], (1, 64)).astype(np.float32)
    # Independently supplied reverse motion; deliberately not the inverse field.
    backward = -np.roll(forward, 2, axis=0)
    return {
        "motion_forward_shift": forward,
        "motion_backward_shift": backward,
        "motion_forward_support": np.full(shape, 0.95, np.float32),
        "motion_backward_support": np.full(shape, 0.9, np.float32),
    }


def test_composed_motion_matches_stepwise_advection_in_both_directions():
    maps = motion_maps()
    result = compose_motion(maps, [1, 2, 5, 10])
    for gap, fields in result.items():
        for direction, start, stop in (("forward", 0, gap), ("backward", 14, 14 - gap)):
            expected, support = _advect(np.array([20, 30]), start, stop, maps, direction)
            assert np.allclose(
                fields[f"motion_{direction}_shift"][start, [20, 30]], expected - [20, 30]
            )
            assert np.allclose(fields[f"motion_{direction}_support"][start, [20, 30]], support)
    # A local one-sample slope does not imply ten samples of displacement.
    assert result[10]["motion_forward_shift"][0, 30] == 0


def test_invalid_intermediate_signal_and_boundaries_invalidate_prediction():
    maps = motion_maps()
    valid = np.ones((15, 64), bool)
    valid[3] = False
    result = compose_motion(maps, [5], valid=valid)[5]
    assert result["motion_forward_support"][0, 30] == 0
    assert result["motion_backward_support"][7, 30] == 0
    assert result["motion_forward_support"][8, 30] > 0
    assert result["motion_forward_support"][14, 30] == 0


def test_composed_motion_cancels():
    with pytest.raises(InterruptedError):
        compose_motion(motion_maps(), [10], cancel=lambda: True)


def test_context_averages_aligned_noise_without_mutating_reference_waveforms():
    rng = np.random.default_rng(1)
    clean = np.exp(-(((np.arange(64) - 32) / 2) ** 2)).astype(np.float32)
    data = np.tile(clean, (9, 1)) + rng.normal(0, 0.15, (9, 64)).astype(np.float32)
    raw = data[:, None, 28:37].copy()
    raw -= raw.mean(axis=2, keepdims=True)
    raw /= np.linalg.norm(raw, axis=2, keepdims=True)
    table = SimpleNamespace(samples=np.full((9, 1), 32), valid=np.ones((9, 1), bool), waveforms=raw)
    maps = {
        f"motion_{d}_{k}": np.full_like(data, v)
        for d in ("forward", "backward")
        for k, v in (("shift", 0), ("support", 1))
    }
    before = raw.copy()
    averaged = contextual_waveforms(table, data, maps, 4, 0.1, 0.3)
    expected = clean[28:37] - clean[28:37].mean()
    expected /= np.linalg.norm(expected)
    assert np.mean(averaged[:, 0] @ expected) > np.mean(raw[:, 0] @ expected)
    assert np.array_equal(raw, before)
    # An explicit break plus unobservable neighbours prevents importing context.
    valid = np.ones_like(data, bool)
    valid[4:] = False
    isolated = contextual_waveforms(table, data, maps, 4, 0.1, 0.3, valid=valid, breaks=(4,))
    assert np.allclose(isolated[4:], raw[4:])
    # Zero-valued samples remain without a waveform even beside visible reflectors.
    data[4] = 0
    raw[4] = 0
    missing = contextual_waveforms(table, data, maps, 4, 0.1, 0.3)
    assert np.all(missing[4] == 0)


def test_tracker_with_motion_context_preserves_clicked_lobe_and_missing_signal():
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence
    from gpr_layer_audit.processing.tracker import pick_interfaces

    axis = np.arange(128)
    centres = 65 + np.rint(2 * np.sin(np.arange(55) / 15)).astype(int)
    data = np.zeros((55, 128), np.float32)
    for row, centre in enumerate(centres):
        z, rival = (axis - centre) / 2.2, (axis - 90) / 2.2
        data[row] = -(1 - z * z) * np.exp(-z * z / 2) - 4 * (1 - rival * rival) * np.exp(
            -rival * rival / 2
        )
    data[25:28] = 0
    anchors = {r: int(centres[r]) for r in (3, 18, 45)}
    result = pick_interfaces(
        data,
        15,
        [LayerSpec(2, "Base", 20, 95, 4)],
        method="seed_hybrid",
        anchor_samples={2: anchors},
        horizontal_step_m=0.4,
        ml_policy="off",
        hybrid_evidence=HybridEvidence(data, 0.05, 0.4),
        conventional_config={"integrated_motion": True, "waveform_context_radius_m": 0.4},
    )[2]
    assert all(result.samples[r] == s for r, s in anchors.items())
    assert np.count_nonzero(result.visible) >= 15
    assert np.all(abs(result.samples[result.visible] - centres[result.visible]) <= 2)
    assert np.all(result.samples[25:28] == -1)
    assert not np.any(result.interpolated)


def test_bright_shallow_reflector_does_not_erase_wide_base_observability():
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence
    from gpr_layer_audit.processing.tracker import pick_interfaces

    axis = np.arange(512)
    z, upper = (axis - 260) / 15, (axis - 100) / 4
    trace = -(1 - z * z) * np.exp(-z * z / 2) + 500 * (1 - upper * upper) * np.exp(
        -upper * upper / 2
    )
    data = np.tile(trace, (21, 1)).astype(np.float32)
    result = pick_interfaces(
        data,
        110,
        [LayerSpec(2, "Base", 1, 360, 4)],
        method="seed_hybrid",
        anchor_samples={2: {2: 260, 10: 260, 18: 260}},
        horizontal_step_m=0.4,
        ml_policy="off",
        hybrid_evidence=HybridEvidence(data, 0.03, 0.4),
    )[2]
    assert np.count_nonzero(result.visible) >= 15
    assert np.all(abs(result.samples[result.visible] - 260) <= 2)
    assert np.median(result.evidence["measurement_support"][result.visible]) > 0.015


@pytest.mark.parametrize("integrated_motion", [False, True])
@pytest.mark.parametrize("context_radius", [0.0, 0.4])
def test_symmetric_split_waveform_cannot_gain_identity_from_motion_tie_order(
    integrated_motion, context_radius
):
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence
    from gpr_layer_audit.processing.tracker import pick_interfaces

    axis = np.arange(180)

    def wave(centre):
        z = (axis - centre) / 5
        return (1 - z * z) * np.exp(-z * z / 2)

    data = np.tile(wave(90), (55, 1)).astype(np.float32)
    data[25] = wave(84) + wave(96)
    result = pick_interfaces(
        data,
        20,
        [LayerSpec(3, "Subbase", 30, 140, 4)],
        method="seed_hybrid",
        anchor_samples={3: {3: 90, 18: 90, 45: 90}},
        horizontal_step_m=0.4,
        ml_policy="off",
        hybrid_evidence=HybridEvidence(data, 0.05, 0.4),
        conventional_config={
            "integrated_motion": integrated_motion,
            "waveform_context_radius_m": context_radius,
            "minimum_motion_margin": 0.02,
        },
    )[3]
    assert result.samples[25] == -1
    assert result.evidence["hybrid_path_margin"][25] < 0.02
    assert np.count_nonzero(result.visible) >= 45
    assert all(result.samples[row] == 90 for row in (3, 18, 45))


def test_ambiguous_intermediate_motion_cannot_become_reliable_by_composition():
    maps = motion_maps()
    for direction in ("forward", "backward"):
        maps[f"motion_{direction}_margin"] = np.ones((15, 64), np.float32)
    maps["motion_forward_margin"][3] = 0
    composed = compose_motion(maps, [5], minimum_margin=0.02)[5]
    assert composed["motion_forward_support"][0, 30] == 0
    assert composed["motion_forward_support"][5, 30] > 0


@pytest.mark.parametrize("length,pairs,band", [(17, 24, 1), (31, 128, 2), (43, 32, 3), (87, 8, 7)])
def test_short_and_long_reciprocal_batches_match_independent_scalar_alignment(length, pairs, band):
    from gpr_layer_audit.processing.hybrid import constrained_dtw
    from gpr_layer_audit.processing.waveform_matching import reciprocal_dtw

    rng = np.random.default_rng(34)
    a, b = rng.normal(size=(2, pairs, length)).astype(np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    # Include exact ties and an opposite lobe, not only random floating-point cases.
    b[0] = a[0]
    b[1] = -a[1]
    expected = []
    for left, right in zip(a, b, strict=True):
        q, shift = constrained_dtw(left, right, band)
        reverse, reverse_shift = constrained_dtw(right, left, band)
        expected.append((min(q, reverse), max(abs(shift), abs(reverse_shift))))
    q, shift = reciprocal_dtw(a, b, band)
    assert np.allclose(np.column_stack((q, shift)), expected, atol=1e-12)
