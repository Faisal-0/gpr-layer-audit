from __future__ import annotations

import itertools

import numpy as np
import pytest

from gpr_layer_audit.ml.processed_dense_decoder import (
    DenseDecoderConfig,
    decode_dense_path,
    direct_predictions,
    seed_interpolation,
)
from gpr_layer_audit.ml.processed_dense_eval import (
    aggregate_reports,
    canonical_sha256,
    evaluate_evidence_stages,
    evaluate_native_predictions,
    freeze_failure_atlas,
    scorer_provenance,
    select_training_threshold,
    validate_evidence_provenance,
)


def score(radar, refs, proposal, *, accepted=None, **kwargs):
    return evaluate_native_predictions(
        radar,
        refs,
        proposal,
        np.ones(len(radar), bool) if accepted is None else accepted,
        initial_seeds=kwargs.pop("initial_seeds", {}),
        native_dx_m=0.1,
        dt_ns=0.03,
        tolerance_samples=2,
        road_group=kwargs.pop("road_group", "train-a"),
        layer=3,
        **kwargs,
    )


def test_native_signed_lobe_contract_and_fixed_interaction_denominator():
    radar = np.tile([1, 2, -1, -2, 1, 2], (7, 1))
    refs = np.array([1, 1, np.nan, 1, 1, 1, 1])
    proposal = np.array([1, 1, 1, 1, 2, 2, 2])
    report = score(radar, refs, proposal, initial_seeds={0: 1}, additional_seed_rows=[3])
    assert report["fixed_initial_nonseed_N"] == 5
    assert report["correct_accepted"] == 1
    assert report["wrong_accepted"] == 3  # within time tolerance but wrong signed lobe
    assert report["unresolved"] == 1
    assert report["manual_answers_in_initial_cohort"] == 1
    assert report["correct_proposals_before_gating"] == 1
    assert report["accepted_agreement"] == 0.25
    assert report["correct_coverage"] == 0.2
    original = score(radar, refs, proposal, initial_seeds={0: 1})
    assert original["cohort_sha256"] == report["cohort_sha256"]
    assert report["longest_contiguous_wrong_accepted_observed_span_m"] == pytest.approx(0.3)


def test_zero_acceptance_undefined_and_unknowns_break_wrong_spans():
    radar = np.tile([1, 2, -1, -2], (5, 1))
    refs = [1, 1, np.nan, 1, 1]
    report = score(radar, refs, np.full(5, 2))
    assert report["wrong_accepted_observed_runs"] == 2
    assert report["longest_contiguous_wrong_accepted_observed_span_m"] == pytest.approx(0.2)
    none = score(radar, refs, np.full(5, 2), accepted=np.zeros(5, bool))
    assert none["accepted_agreement"] is None
    assert none["unresolved"] == 4


def test_native_stratification_and_coordinate_grid_rejection():
    radar = np.ones((5, 6))
    report = score(
        radar,
        np.ones(5),
        np.ones(5),
        initial_seeds={1: 1, 3: 1},
        native_trace_indices=np.arange(5) * 4,
    )
    assert report["segments"]["bracketed"]["fixed_initial_nonseed_N"] == 1
    assert report["segments"]["left_tail"]["fixed_initial_nonseed_N"] == 1
    assert report["segments"]["right_tail"]["fixed_initial_nonseed_N"] == 1
    with pytest.raises(ValueError, match="Nonuniform"):
        score(radar, np.ones(5), np.ones(5), native_trace_indices=[0, 1, 2, 6, 7])
    irregular = score(
        radar, np.zeros(5), np.full(5, 5), native_trace_indices=[0, 1, 2, 6, 7], trace_step=1
    )
    assert irregular["longest_contiguous_wrong_accepted_observed_span_m"] == pytest.approx(0.3)


def test_dense_decoder_global_optimum_against_exhaustive_paths():
    p = np.array([[0.1, 0.6, 0.2, 0.1], [0.55, 0.1, 0.1, 0.25], [0.1, 0.2, 0.6, 0.1]])
    seed = {1: 2.4}
    config = DenseDecoderConfig(
        continuity_cost_per_ns=1.3, gap_probability=0.3, gap_transition_cost=0.2
    )
    actual = decode_dense_path(p, seed, dx_m=0.4, dt_ns=0.2, config=config)

    def objective(path):
        value = 0.0
        for row, sample in enumerate(path):
            value += (
                0
                if row in seed
                else np.log(config.gap_probability if sample < 0 else p[row, int(sample)])
            )
            if row:
                previous = path[row - 1]
                if sample < 0 and previous < 0:
                    pass
                elif sample < 0 or previous < 0:
                    value -= config.gap_transition_cost
                else:
                    value -= config.continuity_cost_per_ns * 0.2 * abs(sample - previous)
        return value

    candidates = [(a, 2.4, b) for a, b in itertools.product(range(-1, 4), repeat=2)]
    best = max(objective(path) for path in candidates)
    assert objective(actual.proposed_samples) == pytest.approx(best)
    assert actual.samples[1] == 2.4
    assert actual.manual[1]


def test_dense_decoder_exact_fractional_seeds_gaps_and_conflicts():
    p = np.zeros((6, 10))
    p[:, 7] = 1
    valid = np.ones_like(p, bool)
    valid[3] = False
    result = decode_dense_path(p, {1: 2.25, 5: 7.75}, sample_valid=valid, dx_m=0.4, dt_ns=0.03)
    assert result.samples[1] == 2.25
    assert result.samples[5] == 7.75
    assert result.gap_mask[3]
    assert result.samples[3] == -1
    assert result.diagnostics["all_native_depths_eligible"]
    with pytest.raises(ValueError, match="Contradictory"):
        decode_dense_path(p, [(1, 2), (1, 3)], dx_m=0.4, dt_ns=0.03)
    with pytest.raises(ValueError, match="break"):
        decode_dense_path(p, {1: 2.25}, dx_m=0.4, dt_ns=0.03, break_rows=[1])
    with pytest.raises(ValueError, match="exact"):
        decode_dense_path(p, {1.5: 2}, dx_m=0.4, dt_ns=0.03)


def test_direct_stage_has_no_seed_override_or_candidate_pruning():
    p = np.zeros((5, 200))
    p[:, 150] = 1
    direct = direct_predictions(p)
    assert np.all(direct.proposed_samples == 150)
    decoded = decode_dense_path(
        p, {0: 1, 4: 1}, dx_m=0.4, dt_ns=0.03, config=DenseDecoderConfig(continuity_cost_per_ns=0)
    )
    assert np.all(decoded.proposed_samples[1:4] == 150)
    assert np.all(decoded.proposed_samples[[0, 4]] == 1)


def test_uniform_evidence_can_abstain_and_breaks_are_explicit():
    p = np.full((8, 512), 1 / 512)
    result = decode_dense_path(p, {0: 7, 7: 9}, dx_m=0.4, dt_ns=0.03)
    assert np.all(result.gap_mask[1:7])
    assert np.array_equal(result.samples[[0, 7]], [7, 9])
    p[:, :] = 0
    p[:, 8] = 1
    broken = decode_dense_path(p, {}, dx_m=0.4, dt_ns=0.03, break_rows=[4])
    assert broken.gap_mask[4]
    assert np.all(broken.samples[[0, 1, 2, 3, 5, 6, 7]] == 8)


def test_interpolation_is_radar_free_and_preserves_exact_clicks():
    result = seed_interpolation(7, {1: 2.25, 5: 6.75})
    assert result[1] == 2.25 and result[5] == 6.75
    assert result[0] == 2.25 and result[6] == 6.75
    assert result[3] == 4.5
    interior = seed_interpolation(7, {1: 2.25, 5: 6.75}, extrapolate_tails=False)
    assert np.all(interior[[0, 6]] == -1)


def test_threshold_selection_rejects_outer_labels_and_unbuffered_inner_validation():
    radar = np.tile([1, 2, -1, -2], (40, 1))
    report = score(
        radar,
        np.ones(40),
        np.r_[np.ones(30), np.full(10, 2)],
        confidence=np.r_[np.full(30, 0.9), np.full(10, 0.2)],
    )
    with pytest.raises(ValueError, match="buffered"):
        select_training_threshold([report], training_groups=["train-a"], excluded_groups=["outer"])
    report.update(split_role="inner_validation", spatial_buffer_verified=True)
    calibrated = select_training_threshold(
        [report],
        training_groups=["train-a"],
        excluded_groups=["outer"],
        thresholds=[0, 0.5, 1],
    )
    assert calibrated["selected_threshold"] == 0.5
    with pytest.raises(ValueError, match="outer"):
        select_training_threshold([report], training_groups=["other"], excluded_groups=["train-a"])
    report["evaluation_observations"] = report["evaluation_observations"][-10:]
    no_point = select_training_threshold(
        [report],
        training_groups=["train-a"],
        excluded_groups=["outer"],
        thresholds=[0, 0.5, 1],
    )
    assert no_point["selected_threshold"] is None


def test_macro_weighting_uses_roads_and_undefined_precision_is_not_perfect():
    first = score(np.ones((100, 6)), np.zeros(100), np.zeros(100), road_group="long")
    second = score(np.ones((2, 6)), np.zeros(2), np.full(2, 5), road_group="short")
    third = score(
        np.ones((2, 6)), np.zeros(2), np.zeros(2), road_group="none", accepted=np.zeros(2, bool)
    )
    result = aggregate_reports([first, second, third])["layers"]["3"]
    assert result["macro_road"]["correct_coverage"] == pytest.approx(1 / 3)
    assert result["macro_road"]["accepted_agreement"] == 0.5
    assert result["roads_with_defined_agreement"] == 2


def test_stage_comparison_is_common_cohort_and_original_scorer_frozen():
    radar = np.ones((6, 8))
    p = np.zeros_like(radar)
    p[:, 4] = 1
    report = evaluate_evidence_stages(
        radar,
        np.full(6, 4),
        p,
        initial_seeds={0: 4, 5: 4},
        sample_valid=radar.astype(bool),
        native_dx_m=0.1,
        dt_ns=0.03,
        road_group="train-a",
        layer=3,
        tolerance_samples=2,
        operating_threshold=1.000001,
    )
    assert len({s["cohort_sha256"] for s in report["stages"].values()}) == 1
    assert report["stages"]["direct_evidence"]["correct_accepted"] == 4
    assert report["stages"]["direct_evidence_gated"]["accepted_agreement"] is None
    assert scorer_provenance()["normalized_lf_sha256"].startswith("fc5b2093")


def test_failure_atlas_immutable(tmp_path):
    entry = dict(
        road_group="gujrat",
        layer=3,
        start_native_trace=0,
        stop_native_trace=100,
        reason="fixed initial tail context; semantics pending visual review",
    )
    destination = tmp_path / "atlas.json"
    freeze_failure_atlas(destination, [entry], primary_artifact_hashes={"baseline": "a" * 64})
    with pytest.raises(FileExistsError):
        freeze_failure_atlas(destination, [entry], primary_artifact_hashes={})


def test_saved_evidence_requires_exact_source_seeds_grid_and_inductive_model():
    metadata = {
        "schema": "processed-native-probability-evidence-v1",
        "source_sha256": "a" * 64,
        "seed_sha256": "b" * 64,
        "probabilities_sha256": "c" * 64,
        "native_trace_indices_sha256": canonical_sha256([0, 4, 8]),
        "native_sample_indices_sha256": canonical_sha256([0, 1, 2]),
        "preprocessing_version": "native-v1",
        "layer": 3,
        "held_out_group": "outer",
        "model_sha256": "d" * 64,
        "weights_sha256": "e" * 64,
        "training_groups": ["train"],
    }
    kwargs = dict(
        source_sha256="a" * 64,
        seed_sha256="b" * 64,
        probabilities_sha256="c" * 64,
        native_trace_indices=[0, 4, 8],
        native_sample_indices=[0, 1, 2],
        preprocessing_version="native-v1",
        layer=3,
        road_group="outer",
    )
    validate_evidence_provenance(metadata, **kwargs)
    with pytest.raises(ValueError, match="source_sha256"):
        validate_evidence_provenance({**metadata, "source_sha256": "f" * 64}, **kwargs)
    with pytest.raises(ValueError, match="native_trace"):
        validate_evidence_provenance(metadata, **{**kwargs, "native_trace_indices": [1, 5, 9]})
    with pytest.raises(ValueError, match="out-of-training"):
        validate_evidence_provenance({**metadata, "training_groups": ["outer"]}, **kwargs)
