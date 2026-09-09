"""Keep dense timing ambiguity and real local correction scope independently checkable."""

from pathlib import Path

import numpy as np
import pytest


def test_timing_competitors_change_margins_without_changing_selection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dense_replay_adapter import timing_aware_picker

    from gpr_layer_audit.processing.seeded_challenger import PacketConfig, pick_seeded_packet

    x = np.arange(121)[:, None]
    t = np.arange(128)[None, :]
    u = (t - 48 - 0.04 * x) / 4
    data = ((1 - u * u) * np.exp(-u * u / 2)).astype(np.float32)
    kwargs = dict(
        valid=np.ones_like(data, bool),
        dt_ns=0.03,
        dx_m=0.025,
        pulse_width_samples=10,
        config=PacketConfig(guide_weight=1),
    )
    anchors = {8: 48, 60: 50, 112: 52}
    before = pick_seeded_packet(data, anchors, **kwargs)
    after = timing_aware_picker()(data, anchors, **kwargs)
    np.testing.assert_array_equal(before.provisional_samples, after.provisional_samples)
    has_competitor = before.alternate_samples >= 0
    assert np.all(after.confidence[has_competitor] <= before.confidence[has_competitor] + 1e-12)
    assert np.any(after.confidence[has_competitor] < before.confidence[has_competitor] - 1e-6)
    assert all(after.samples[r] == s for r, s in anchors.items())


def test_dense_adapter_preserves_frozen_pulse_and_local_observations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from dense_replay_adapter import fit_dense

    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.processed_tracking import merge_local_paths

    t = np.arange(128)
    u = (t - 50) / 4
    data = np.broadcast_to((1 - u * u) * np.exp(-u * u / 2), (150, 128)).astype(np.float32).copy()
    anchors = {2: {10: 50, 75: 50, 135: 50}}
    kwargs = dict(
        method="dense_packet_research",
        config={"guide_weight": 1, "timing_ambiguity": True},
        pulse_anchors=anchors,
    )
    layers = [LayerSpec(2, "Base", 1, 127, 1)]
    original = fit_dense(data, np.ones_like(data, bool), 0, 0.03, 0.025, layers, anchors, **kwargs)
    corrected = fit_dense(
        data,
        np.ones_like(data, bool),
        0,
        0.03,
        0.025,
        layers,
        {2: anchors[2] | {105: 52}},
        **kwargs,
    )
    assert original[2].provenance["resolved_pulse"] == corrected[2].provenance["resolved_pulse"]
    merged = merge_local_paths(original, corrected, orders={2}, start_row=95, stop_row=115)[2]
    assert merged.samples[105] == 52
    outside = np.r_[np.arange(95), np.arange(116, len(data))]
    for field in ("samples", "visible", "provisional_samples", "alternate_samples", "confidence"):
        np.testing.assert_array_equal(
            getattr(merged, field)[outside], getattr(original[2], field)[outside]
        )
    with pytest.raises(ValueError, match="not application dispatch"):
        fit_dense(
            data,
            np.ones_like(data, bool),
            0,
            0.03,
            0.025,
            layers,
            anchors,
            method="seed_hybrid",
            config=kwargs["config"],
        )
