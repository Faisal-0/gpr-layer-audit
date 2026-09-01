"""Regressions for direct seed-lineage deep-interface acceptance."""

from __future__ import annotations

import numpy as np

from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.tracker import pick_interfaces


def _ricker(axis: np.ndarray, centre: int, width: float, amplitude: float) -> np.ndarray:
    coordinate = (axis - centre) / width
    return amplitude * (1.0 - coordinate**2) * np.exp(-0.5 * coordinate**2)


def test_generic_coherence_cannot_replace_direct_seed_family_through_dropout():
    """A strong persistent packet is detection evidence, not seeded identity."""

    rows, sample_count = 120, 210
    axis = np.arange(sample_count, dtype=float)
    asphalt = np.rint(
        76.0 + 2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    ).astype(int)
    intended = np.rint(
        124.0 + 8.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))
    ).astype(int)
    wrong = 153
    data = np.asarray(
        [
            _ricker(axis, 40, 2.3, 1.5)
            + _ricker(axis, int(asphalt[row]), 2.5, 0.9)
            + _ricker(axis, int(intended[row]), 3.0, 0.34)
            + _ricker(axis, wrong, 3.0, 0.95)
            for row in range(rows)
        ]
    )
    dropout = slice(45, 63)
    data[dropout] = np.asarray(
        [
            _ricker(axis, 40, 2.3, 1.5)
            + _ricker(axis, int(asphalt[row]), 2.5, 0.9)
            + _ricker(axis, wrong, 3.0, 0.95)
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
        2: {row: int(intended[row]) for row in seed_rows},
    }
    metadata = {
        1: {
            row: {"canonical_sample_index": int(asphalt[row] - 3)}
            for row in seed_rows
        },
        2: {
            row: {"canonical_sample_index": int(intended[row] + 4)}
            for row in seed_rows
        },
    }

    path = pick_interfaces(
        data,
        40,
        layers,
        anchor_samples=anchors,
        seed_metadata=metadata,
        max_interpolation_rows=0,
    )[2]

    assert path.provisional_samples is not None
    assert np.array_equal(
        path.provisional_samples, path.evidence["guided_graph_selected_sample"]
    )

    ordinary = np.r_[0 : dropout.start, dropout.stop : rows]
    ordinary_visible = ordinary[path.samples[ordinary] >= 0]
    assert len(ordinary_visible) / len(ordinary) > 0.95
    assert np.mean(np.abs(path.samples[ordinary_visible] - intended[ordinary_visible]) <= 2) > 0.99
    assert np.all(path.evidence["direct_seed_family_support"][ordinary_visible] == 1.0)
    assert np.all(path.evidence["deep_identity_support"][ordinary_visible] == 1.0)
    assert np.all(
        path.evidence["independent_radar_identity_agreement"][ordinary_visible] == 1.0
    )

    # The wrong packet is deliberately the strongest candidate and has high
    # generic/local coherence.  It still has neither direct seed reachability
    # nor a phase-locked seed tracklet, so the dropout must remain unresolved.
    candidate = path.candidate_components
    assert candidate["audit_candidate_rank"][50, wrong] == 1
    assert candidate["audit_candidate_score"][50, wrong] > 0.75
    assert candidate["generic_radar_score"][50, wrong] > 0.75
    assert candidate["oriented_coherence"][50, wrong] > 0.75
    assert candidate["seed_reachable"][50, wrong] == 0
    assert candidate["tracklet_support"][50, wrong] == 0
    assert np.all(path.evidence["direct_seed_family_support"][dropout] == 0.0)
    assert np.all(path.evidence["deep_identity_support"][dropout] == 0.0)
    assert np.all(path.samples[dropout] == -1)
    assert np.all(path.confidence[dropout] == 0.0)
    assert np.all(~path.visible[dropout])
