import numpy as np
import pytest

from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.seed_guidance import (
    derive_seed_guide,
    guide_conflict_mask,
    score_candidates_against_guide,
)
from gpr_layer_audit.processing.tracker import pick_interfaces


def test_smooth_absolute_guide_tracks_seeds_and_softly_ranks_candidates():
    guide = derive_seed_guide(21, {0: 100, 10: 105, 20: 110})

    assert guide is not None
    assert guide.mode == "absolute"
    assert np.allclose(guide.values, 100 + 0.5 * np.arange(21))
    assert np.all(guide.uncertainty[[0, 10, 20]] == 2.0)
    assert guide.uncertainty[5] > guide.uncertainty[0]

    candidates = np.column_stack((guide.values, guide.values + 8, np.full(21, -1)))
    adjustment = score_candidates_against_guide(candidates, guide)
    assert np.allclose(adjustment[:, 0], 0.0)
    assert np.all(adjustment[:, 1] < 0.0)
    assert np.all(adjustment[:, 2] == 0.0)  # no-pick is never suppressed
    assert np.all(adjustment <= 0.0)
    assert np.all(adjustment >= -0.35)


def test_seed_observed_regime_change_is_preserved_and_locally_allowed():
    guide = derive_seed_guide(31, [0, 10, 11, 30], [120, 120, 160, 160])

    assert guide is not None
    assert guide.values[10] == 120
    assert guide.values[11] == 160
    assert np.all(guide.values[:11] == 120)
    assert np.all(guide.values[11:] == 160)
    # The abrupt seed is exact rather than pulled toward the remote majority.
    adjustment = score_candidates_against_guide(
        np.asarray([120.0, guide.values[11]]),
        derive_seed_guide(2, {0: 120, 1: 160}),
    )
    assert adjustment[1] == 0.0


def test_abrupt_alternating_seed_regimes_preserve_an_ambiguous_envelope():
    # This mimics alternating Daska gap modes.  Sparse endpoints prove that
    # both modes occur, but cannot locate the transition between traces.
    guide = derive_seed_guide(31, {0: 55, 10: 79, 20: 60, 30: 84}, mode="gap")

    assert guide is not None
    assert np.all(guide.ambiguous[1:10])
    assert np.all(guide.ambiguous[11:20])
    assert np.all(guide.ambiguous[21:30])
    assert not np.any(guide.ambiguous[[0, 10, 20, 30]])
    assert np.isnan(guide.values[5])  # no invented midpoint/transition
    assert guide.lower_bounds[5] == 55
    assert guide.upper_bounds[5] == 79

    # Every physically plausible intermediate or endpoint mode remains
    # admissible.  Only values outside the observed endpoint envelope score.
    candidates = np.full((31, 7), -1.0)
    candidates[5] = [55.0, 60.0, 67.0, 79.0, 49.0, 85.0, -1.0]
    adjustment = score_candidates_against_guide(candidates, guide, upper_samples=np.zeros(31))
    assert np.all(adjustment[5, :4] == 0.0)
    assert np.all(adjustment[5, 4:6] < 0.0)
    assert adjustment[5, 6] == 0.0
    assert np.all(np.isfinite(adjustment))
    assert np.all((adjustment >= -0.35) & (adjustment <= 0.0))

    conflict = guide_conflict_mask(
        np.asarray([55.0, 67.0, 49.0, 79.0]),
        derive_seed_guide(4, {0: 55, 3: 79}),
        pulse_width_samples=2.0,
        uncertainty_multiple=1.0,
    )
    assert np.array_equal(conflict, np.asarray([False, False, True, False]))


def test_abrupt_regime_envelope_is_reversal_invariant():
    forward = derive_seed_guide(31, {0: 55, 10: 79, 20: 60, 30: 84})
    reverse = derive_seed_guide(31, {0: 84, 10: 60, 20: 79, 30: 55})

    assert forward is not None and reverse is not None
    assert np.array_equal(forward.ambiguous[::-1], reverse.ambiguous)
    assert np.allclose(forward.lower_bounds[::-1], reverse.lower_bounds)
    assert np.allclose(forward.upper_bounds[::-1], reverse.upper_bounds)
    assert np.allclose(forward.uncertainty[::-1], reverse.uncertainty)
    assert np.allclose(forward.values[::-1], reverse.values, equal_nan=True)


def test_insufficient_seed_evidence_produces_no_guide_or_adjustment():
    assert derive_seed_guide(12, {}) is None
    assert derive_seed_guide(12, {6: 140}) is None
    candidates = np.asarray([[130, 150, -1], [131, 151, -1]])
    assert np.all(score_candidates_against_guide(candidates, None) == 0.0)


@pytest.mark.parametrize(
    ("seeds", "mode", "message"),
    [
        ({12: 140}, "absolute", "row range"),
        ({-1: 140}, "absolute", "row range"),
        ({"bad": 140}, "absolute", "numeric row"),
        ({3: np.nan}, "absolute", "finite"),
        ({3: -1}, "gap", "non-negative"),
    ],
)
def test_invalid_single_seed_is_rejected_before_insufficient_evidence(seeds, mode, message):
    with pytest.raises(ValueError, match=message):
        derive_seed_guide(12, seeds, mode=mode)


def test_safe_extrapolation_holds_endpoint_and_weakens_with_distance():
    guide = derive_seed_guide(31, {10: 100, 20: 110})

    assert guide is not None
    assert np.all(guide.values[:11] == 100)
    assert np.all(guide.values[20:] == 110)
    assert guide.uncertainty[0] > guide.uncertainty[9] > guide.uncertainty[10]
    assert guide.uncertainty[30] > guide.uncertainty[21] > guide.uncertainty[20]
    assert np.all(guide.extrapolated[:10])
    assert not np.any(guide.extrapolated[10:21])
    assert np.all(guide.extrapolated[21:])

    candidates = guide.values + 6.0
    adjustment = score_candidates_against_guide(candidates, guide)
    # The same mismatch is penalized less far outside observed support.
    assert adjustment[0] > adjustment[10]


def test_gap_guide_scores_relative_layer_spacing_without_changing_candidates():
    guide = derive_seed_guide(5, {0: 40, 4: 44}, mode="gap")
    upper = np.arange(100, 105)
    matching = upper + guide.values
    lower = np.column_stack((matching, matching + 20, np.full(5, -1)))
    before = lower.copy()

    adjustment = score_candidates_against_guide(lower, guide, upper_samples=upper)

    assert np.array_equal(lower, before)
    assert np.allclose(adjustment[:, 0], 0.0)
    assert np.all(adjustment[:, 1] < 0.0)
    assert np.all(adjustment[:, 2] == 0.0)
    with pytest.raises(ValueError, match="upper_samples"):
        score_candidates_against_guide(lower, guide)


def test_guide_conflict_is_a_review_signal_and_never_overrides_a_seed():
    guide = derive_seed_guide(9, {0: 100, 8: 100}, base_uncertainty=2.0)
    selected = np.asarray([100, 101, 102, 130, 131, 132, 102, 101, 140])

    conflict = guide_conflict_mask(
        selected,
        guide,
        pulse_width_samples=5.0,
        uncertainty_multiple=2.0,
    )

    assert np.array_equal(
        conflict,
        np.asarray([False, False, False, True, True, True, False, False, False]),
    )


def _ricker(axis, centre, width, amplitude=1.0):
    z = (axis - centre) / width
    return amplitude * (1.0 - z**2) * np.exp(-0.5 * z**2)


def test_seed_guided_deep_tracker_rejects_stronger_parallel_event_and_keeps_dropout_gap():
    rows, sample_count = 100, 210
    axis = np.arange(sample_count, dtype=float)
    asphalt = np.rint(76.0 + 2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))).astype(int)
    base = np.rint(126.0 + 3.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, rows))).astype(int)
    competitor = np.full(rows, 153, dtype=int)
    data = np.asarray(
        [
            _ricker(axis, 40.0, 2.3, 1.5)
            + _ricker(axis, asphalt[row], 2.5, 0.9)
            - _ricker(axis, base[row], 3.0, 0.34)
            - _ricker(axis, competitor[row], 3.0, 0.80)
            for row in range(rows)
        ]
    )
    # Remove the weak seeded family locally while leaving the stronger, wrong
    # reflector. Geometry must not turn that wrong radar event into confidence.
    dropout = slice(44, 59)
    data[dropout] = np.asarray(
        [
            _ricker(axis, 40.0, 2.3, 1.5)
            + _ricker(axis, asphalt[row], 2.5, 0.9)
            - _ricker(axis, competitor[row], 3.0, 0.80)
            for row in range(dropout.start, dropout.stop)
        ]
    )
    data += np.random.default_rng(901).normal(0.0, 0.018, data.shape)
    layers = [
        LayerSpec(1, "Asphalt", 25, 60, 5),
        LayerSpec(2, "Base", 65, 130, 12),
    ]
    seed_rows = (8, 38, 72, 92)
    anchors = {
        1: {row: int(asphalt[row]) for row in seed_rows},
        2: {row: int(base[row]) for row in seed_rows},
    }
    metadata = {
        1: {
            row: {"canonical_sample_index": int(asphalt[row] - 3)}
            for row in seed_rows
        },
        2: {
            row: {"canonical_sample_index": int(base[row] + 4)}
            for row in seed_rows
        },
    }

    path = pick_interfaces(
        data, 40, layers, anchor_samples=anchors, seed_metadata=metadata
    )[2]

    ordinary = np.r_[0 : dropout.start, dropout.stop : rows]
    assert np.mean(np.abs(path.samples[ordinary] - base[ordinary]) <= 2) > 0.98
    assert not np.any(np.abs(path.samples - competitor) <= 3)
    assert np.all(path.samples[dropout] == -1)
    assert np.all(path.confidence[dropout] == 0.0)
    assert path.evidence["absolute_seed_guide_extrapolated"][0] == 1.0
    assert path.evidence["absolute_seed_guide_extrapolated"][50] == 0.0
    assert path.evidence["absolute_seed_guide_deviation"][seed_rows[1]] == 0.0
    assert path.evidence["seed_gap_guide_deviation"][seed_rows[2]] == 0.0
    assert np.array_equal(
        path.evidence["radar_only_confidence"], path.evidence["pre_gate_confidence"]
    )
    assert np.any(
        path.evidence["guided_graph_selected_sample"]
        != path.evidence["unguided_graph_selected_sample"]
    )
    guided = path.evidence["guided_graph_selected_sample"]
    unguided = path.evidence["unguided_graph_selected_sample"]
    independent_disagreement = (
        (guided >= 0)
        & (unguided >= 0)
        & (np.abs(guided - unguided) > 6.0)
    )
    independent_disagreement[list(seed_rows)] = False
    assert np.any(independent_disagreement)
    assert np.all(path.samples[independent_disagreement] == -1)
    assert np.all(path.confidence[independent_disagreement] == 0.0)
