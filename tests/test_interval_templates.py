from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.processing.interval_templates import interval_seed_scores


def scene(step=1):
    rows = round(20 / step) + 1
    seeds = {round(r / step): 20 for r in (0, 10, 20)}
    table = SimpleNamespace(samples=np.tile([20, 30], (rows, 1)), valid=np.ones((rows, 2), bool))
    prototypes = [
        SimpleNamespace(
            chainage_m=float(row),
            sample_index=20,
            radius_samples=2,
            polarity=1,
            regime_id="default",
            selected_lobe="positive",
        )
        for row in seeds
    ]
    correlations = []
    for index in range(3):
        values = np.full((rows, 64), 0.9, np.float32)
        values[:, 30] = 1 if index == 0 else 0.1
        correlations.append(values)
    return table, prototypes, correlations, seeds, np.ones((rows, 64)), np.ones((rows, 64), bool)


def compute(inputs, breaks=(), step=1, **kwargs):
    table, prototypes, correlations, anchors, measurement, valid = inputs
    return interval_seed_scores(
        table, prototypes, correlations, anchors, breaks, measurement, valid, step, **kwargs
    )


def test_distant_seed_cannot_outvote_both_local_endpoint_observations():
    inputs = scene()
    scores, diagnostics = compute(inputs)
    assert scores[15, 0] == pytest.approx(0.9)
    assert scores[15, 1] == pytest.approx(0.1)
    assert scores[5, 1] == pytest.approx(0.55)
    assert scores[20, 1] == pytest.approx(0.1)
    assert diagnostics["original_seed_templates_preserved"]
    original = [c.copy() for c in inputs[2]]
    compute(inputs)
    assert all(np.array_equal(a, b) for a, b in zip(original, inputs[2], strict=True))


def test_interval_scores_have_same_physical_weights_across_sampling_resolutions():
    coarse, _ = compute(scene(), step=1)
    fine, _ = compute(scene(0.25), step=0.25)
    assert np.allclose(coarse, fine[::4])


def test_explicit_regimes_remain_alternatives_and_breaks_isolate_templates():
    inputs = scene()
    inputs[1][1].regime_id = "new-regime"
    scores, diagnostics = compute(inputs)
    assert scores[5, 1] == 1
    assert any(r["policy"] == "competing_explicit_regimes" for r in diagnostics["regions"])
    inputs[3].pop(0)
    scores, _ = compute(inputs, breaks=(8,))
    assert not np.any(scores[:8])
    assert scores[9, 1] == pytest.approx(0.1)


def test_invalid_seed_context_invalid_candidates_and_opposite_lobes_supply_no_score():
    inputs = scene()
    inputs[5][0, 18] = False  # Invalid context makes the first seed template unusable.
    inputs[5][5, 20] = False
    inputs[4][6, 20] = -1
    scores, _ = compute(inputs)
    assert scores[1, 1] == pytest.approx(0.1)
    assert scores[5, 0] == 0
    assert scores[6, 0] == 0
    with pytest.raises(InterruptedError):
        compute(inputs, cancel=lambda: True)


def test_seed_region_objective_preserves_complete_endpoints_and_repairs_distant_template_switch():
    from gpr_layer_audit.processing.conventional_config import ConventionalConfig
    from gpr_layer_audit.processing.hybrid import ReflectorSegment, solve_complete_intervals

    inputs = scene()
    table = inputs[0]
    segments = [
        ReflectorSegment(0, [(0, 0)], {0}),
        ReflectorSegment(1, [(r, 0) for r in range(1, 10)]),
        ReflectorSegment(2, [(10, 0)], {10}),
        ReflectorSegment(3, [(r, 0) for r in range(11, 20)]),
        ReflectorSegment(4, [(r, 1) for r in range(11, 20)]),
        ReflectorSegment(5, [(20, 0)], {20}),
    ]
    edges = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (2, 4): 1, (3, 5): 1, (4, 5): 1}
    global_scores = np.tile([0.9, 1], (21, 1))
    regional, _ = compute(inputs)
    paths = []
    for scores in (global_scores, regional):
        diagnostic = {}
        paths.append(
            solve_complete_intervals(
                segments,
                edges,
                np.ones(6),
                table,
                scores,
                inputs[3],
                (),
                7,
                1,
                ConventionalConfig(),
                diagnostic,
            )[0]
        )
        assert diagnostic["unresolved_intervals"] == []
    assert np.all(paths[0][11:20] == 30)
    assert np.all(paths[1] == 20)
    assert all(paths[1][r] == s for r, s in inputs[3].items())


@pytest.mark.parametrize(
    "options",
    [
        {"interval_seed_scoring": True},
        {"distinct_path_inference": True},
        {"interval_seed_scoring": True, "distinct_path_inference": True},
    ],
)
def test_tracker_scoring_keeps_clicks_lobe_identity_and_missing_signal(options):
    from gpr_layer_audit.models import LayerSpec
    from gpr_layer_audit.processing.hybrid_evidence import HybridEvidence
    from gpr_layer_audit.processing.tracker import pick_interfaces

    axis = np.arange(128)
    z = (axis - 65) / 2.2
    waveform = -(1 - z * z) * np.exp(-z * z / 2)
    data = np.tile(waveform, (61, 1)).astype(np.float32)
    data[17:22] = 0
    result = pick_interfaces(
        data,
        15,
        [LayerSpec(2, "Base", 20, 95, 4)],
        method="seed_hybrid",
        anchor_samples={2: {3: 65, 30: 65, 57: 65}},
        horizontal_step_m=0.1,
        ml_policy="off",
        hybrid_evidence=HybridEvidence(data, 0.05, 0.1),
        conventional_config=options,
    )[2]
    assert all(result.samples[r] == 65 for r in (3, 30, 57))
    assert not np.any(result.visible[17:22])
    assert np.all(abs(result.samples[result.visible] - 65) <= 2)
    assert np.count_nonzero(result.visible) >= 35
    assert result.provenance["seed_identity_scoring"]["policy"] == (
        "interval_seed_scores_v1"
        if options.get("interval_seed_scoring")
        else "strongest_seed_correlation"
    )
