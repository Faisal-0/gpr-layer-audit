"""The tail change must leave bracketed objectives exactly unchanged."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from gpr_layer_audit.processing import seeded_challenger as dense


def test_only_tail_guide_is_removed_from_full_objective(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from experiment_local_patch_tail_guide import instrument

    radar = np.random.default_rng(482).normal(size=(11, 25)).astype(np.float32)
    config = dense.PacketConfig(geometry="none", max_slope_ns_per_m=1, guide_weight=1)
    kwargs = dict(
        valid=np.ones_like(radar, bool),
        dt_ns=0.2,
        dx_m=0.1,
        pulse_width_samples=2,
        context_radius=2,
        reference_surface=0,
    )
    seeds = {3: 8, 7: 10}
    captured = []
    real_dp = dense.dense_interval_dp

    def record(unary, transition, left, right, cancel=None):
        captured.append((unary.copy(), transition.copy(), left, right))
        return real_dp(unary, transition, left, right, cancel)

    monkeypatch.setattr(dense, "dense_interval_dp", record)
    full = dense.pick_seeded_packet(radar, seeds, config=config, **kwargs)
    guided = captured[:]
    captured.clear()
    dense.pick_seeded_packet(radar, seeds, config=replace(config, guide_weight=0), **kwargs)
    unguided = captured[:]
    captured.clear()
    result = instrument(dense.pick_seeded_packet, np.zeros((10, 3, 25)))(
        radar, seeds, config=config, **kwargs
    )
    assert len(captured) == 3
    for i, span in enumerate(captured):
        control = guided[i] if i == 1 else unguided[i]
        for actual, expected in zip(span[:2], control[:2], strict=True):
            np.testing.assert_array_equal(actual, expected)
        assert span[2:] == control[2:]
    assert not np.array_equal(guided[0][0], captured[0][0])
    np.testing.assert_array_equal(result.provisional_samples[4:7], full.provisional_samples[4:7])
    for row, sample in seeds.items():
        assert result.samples[row] == sample
