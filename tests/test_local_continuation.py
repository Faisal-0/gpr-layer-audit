from types import SimpleNamespace

import numpy as np

from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.continuation import (
    attach_spatial_lineage,
    candidate_links,
    local_phase_motion,
)
from gpr_layer_audit.processing.joint_graph import joint_family_beam
from gpr_layer_audit.processing.seed_graph import _event_emissions


def _wave(axis, centre):
    z = (axis - centre) / 3
    return -(1 - z * z) * np.exp(-z * z / 2)


def _case(rows=120):
    target = np.rint(140 + 18 * np.sin(np.linspace(0, 2 * np.pi, rows))).astype(int)
    axis = np.arange(210)
    data = np.array([_wave(axis, t) + 1.8 * _wave(axis, 100) for t in target])
    samples = np.column_stack((target, np.full(rows, 100), np.full(rows, -1)))
    waveforms = np.zeros((rows, 3, 23))
    for row in range(rows):
        for index in (0, 1):
            sample = samples[row, index]
            waveform = data[row, sample - 11 : sample + 12].copy()
            waveform -= waveform.mean()
            waveforms[row, index] = waveform / np.linalg.norm(waveform)
    shape = samples.shape
    table = SimpleNamespace(
        samples=samples,
        canonical_samples=samples.astype(float),
        valid=samples >= 0,
        features=np.zeros((*shape, 2)),
        tracklet_support=np.zeros(shape),
        polarities=np.tile([-1, -1, 0], (rows, 1)),
        phase_classes=np.zeros(shape, dtype=int),
        family_indices=np.zeros(shape, dtype=int),
        regime_indices=np.zeros(shape, dtype=int),
        waveforms=waveforms,
        dense_radar_score=np.zeros_like(data),
        component_maps=local_phase_motion(data, 7),
    )
    anchors = {0: int(target[0]), rows - 1: int(target[-1])}
    score = np.tile([0.55, 1.0, 0], (rows, 1))
    workspace = SimpleNamespace(
        table=table, anchors=anchors, layer=LayerSpec(2, "Base", 60, 180, 5)
    )
    return data, target, workspace, score


def test_endpoint_seeds_keep_unreachable_ringing_admissible_but_penalized():
    data, target, workspace, score = _case()
    table = workspace.table
    workspace.emissions = _event_emissions(
        table,
        score,
        workspace.anchors,
        {},
        set(),
        np.zeros(len(data), bool),
        0,
    )
    unconstrained, _ = joint_family_beam([workspace], set(), 0.4)
    assert np.mean(unconstrained[0][2] == 100) > 0.90  # Reproduces the actual defect.

    attach_spatial_lineage(table, workspace.anchors, set(), 7, 0.4)
    assert np.all(table.seed_reachable[:, 0])
    assert not np.any(table.seed_reachable[:, 1])
    workspace.emissions = _event_emissions(
        table,
        score,
        workspace.anchors,
        {},
        set(),
        np.zeros(len(data), bool),
        0,
    )
    # Only the confirmed seed rows remain hard constraints. Elsewhere a broken
    # local lineage is advisory and both physical alternatives stay searchable.
    assert np.all(np.isfinite(workspace.emissions[1:-1, :2]))
    paths, _ = joint_family_beam([workspace], set(), 0.4)
    assert np.all(paths[0][2] >= 0)


def test_local_motion_preserves_signed_displacement_and_bidirectional_alignment():
    axis = np.arange(180)
    data = np.array([_wave(axis, 90), _wave(axis, 94)])
    maps = local_phase_motion(data, 7)
    assert maps["motion_forward_shift"][0, 90] == 4
    assert maps["motion_backward_shift"][1, 94] == -4
    assert maps["motion_forward_support"][0, 90] > 0.99
    assert maps["motion_backward_support"][1, 94] > 0.99


def test_opposite_polarity_and_structural_break_cannot_merge_lineages():
    data, _, workspace, _ = _case(rows=12)
    table = workspace.table
    table.polarities[6:, 0] = 1
    assert not candidate_links(table, 5, 6, 7)[0, 0]
    attach_spatial_lineage(table, {0: table.samples[0, 0]}, {6}, 7, 0.4)
    assert not np.any(table.seed_reachable[6:])
    assert not np.any(table.continuation_links[6])


def test_zero_signal_has_no_local_motion_support():
    maps = local_phase_motion(np.zeros((4, 100)), 7)
    assert np.all(maps["motion_forward_support"] == 0)
    assert np.all(maps["motion_backward_support"] == 0)
    assert np.all(maps["motion_forward_shift"] == 0)
