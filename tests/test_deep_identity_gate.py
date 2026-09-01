"""Focused regressions for conservative deep-interface identity handling."""

from __future__ import annotations

import numpy as np

from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.tracker import pick_interfaces


def _ricker(axis: np.ndarray, centre: int, width: float, amplitude: float) -> np.ndarray:
    coordinate = (axis - centre) / width
    return amplitude * (1.0 - coordinate**2) * np.exp(-0.5 * coordinate**2)


def test_strong_smooth_deep_competitor_is_withheld_when_seeded_family_fades():
    """A persistent stronger reflector must not replace a seeded deep family.

    The intended base packet moves by roughly 16 samples over the line and is
    removed for a short interval.  A stronger, perfectly smooth packet remains
    in that interval, making it a deliberately attractive but semantically
    wrong dropout recovery candidate.  The conservative identity gate should
    preserve a review gap instead of accepting that candidate.
    """

    rows, sample_count = 120, 210
    axis = np.arange(sample_count, dtype=float)
    asphalt = np.rint(
        76.0 + 2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    ).astype(int)
    intended_base = np.rint(
        124.0 + 8.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    ).astype(int)
    wrong_base = 153
    data = np.asarray(
        [
            _ricker(axis, 40, 2.3, 1.5)
            + _ricker(axis, int(asphalt[row]), 2.5, 0.9)
            + _ricker(axis, int(intended_base[row]), 3.0, 0.34)
            + _ricker(axis, wrong_base, 3.0, 0.95)
            for row in range(rows)
        ]
    )
    dropout = slice(45, 63)
    data[dropout] = np.asarray(
        [
            _ricker(axis, 40, 2.3, 1.5)
            + _ricker(axis, int(asphalt[row]), 2.5, 0.9)
            + _ricker(axis, wrong_base, 3.0, 0.95)
            for row in range(dropout.start, dropout.stop)
        ]
    )
    data += np.random.default_rng(901).normal(0.0, 0.018, data.shape)

    layers = [
        LayerSpec(1, "Asphalt", 25, 60, 5),
        LayerSpec(2, "Base", 65, 130, 12),
    ]
    seed_rows = (8, 38, 72, 102)
    anchors = {
        1: {row: int(asphalt[row]) for row in seed_rows},
        2: {row: int(intended_base[row]) for row in seed_rows},
    }
    seed_metadata = {
        1: {
            row: {"canonical_sample_index": int(asphalt[row] - 3)}
            for row in seed_rows
        },
        2: {
            row: {"canonical_sample_index": int(intended_base[row] + 4)}
            for row in seed_rows
        },
    }

    path = pick_interfaces(
        data,
        40,
        layers,
        anchor_samples=anchors,
        seed_metadata=seed_metadata,
        max_interpolation_rows=0,
    )[2]

    ordinary = np.r_[0 : dropout.start, dropout.stop : rows]
    assert np.mean(np.abs(path.samples[ordinary] - intended_base[ordinary]) <= 2) > 0.95
    assert np.array_equal(path.samples[list(seed_rows)], intended_base[list(seed_rows)])

    # The strong smooth packet is still present in every dropout trace, but it
    # is a high-scoring candidate.  Identity exclusion, rather than candidate
    # generation accidentally missing it, is what forces the review gap.
    assert path.candidate_components["audit_candidate_rank"][50, wrong_base] == 1
    assert path.candidate_components["audit_candidate_score"][50, wrong_base] > 0.75
    assert (
        path.candidate_components["seed_mismatched_persistent_packet"][50, wrong_base]
        == 1
    )
    assert np.all(path.samples[dropout] == -1)
    assert not np.any(np.abs(path.samples - wrong_base) <= 3)
    assert np.all(path.confidence[dropout] == 0.0)
    assert np.all(~path.visible[dropout])
