import numpy as np
import pytest

from gpr_layer_audit.ml.processed_correspondence import (
    PairLogistic,
    PatchContract,
    extract_patches,
    fit_pair_logistic,
    interval_weights,
    pair_features,
    reviewed_candidate_targets,
    score_candidates,
)


def test_unknown_and_support_rows_never_become_training_background():
    trace = np.array([0, 1, 2, 1, -1, -2, -1, 1, 2, 1, 0], float)
    candidates = np.array([2, 5, 8])
    np.testing.assert_array_equal(reviewed_candidate_targets(trace, candidates, None, 2), -1)
    np.testing.assert_array_equal(
        reviewed_candidate_targets(trace, candidates, 2, 2, support_row=True), -1
    )
    np.testing.assert_array_equal(reviewed_candidate_targets(trace, candidates, 2, 2), [1, 0, 0])
    # A timing miss on the same signed lobe is not a reflector-family negative.
    np.testing.assert_array_equal(reviewed_candidate_targets(trace, [1, 3], 3, 0.5), [-1, 1])


def test_native_sampling_and_invalid_gaps_remain_explicit():
    radar = np.ones((7, 51), np.float32)
    valid = np.ones_like(radar, bool)
    valid[3, 25] = False
    contract = PatchContract(temporal_radius_ns=0.2, temporal_points=5, lateral_points=3)
    values, support, usable = extract_patches(radar, valid, [3], [25], 0.1, 0.1, contract)
    assert not usable[0] and not support[0, 1, 2] and values[0, 1, 2] == 0
    with pytest.raises(ValueError, match="snapping"):
        extract_patches(radar, valid, [3.5], [25], 0.1, 0.1, contract)
    with pytest.raises(ValueError, match="Positive physical"):
        contract.offsets(0, 0.1)


def test_interval_weights_do_not_take_pointwise_best_seed():
    weights = interval_weights([-1, 0, 2, 5, 7, 10, 11], [0, 5, 10])
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    np.testing.assert_allclose(weights[2], [0.6, 0.4, 0])
    np.testing.assert_allclose(weights[-1], [0, 0, 1])
    with pytest.raises(ValueError, match="Unique sorted"):
        interval_weights([0], [0, 0])


def test_gain_invariant_features_preserve_polarity_competitors():
    wave = np.sin(np.linspace(-3, 3, 65)).astype(np.float32)
    radar = np.tile(wave, (9, 1))
    mask = np.ones_like(radar, bool)
    a, m, _ = extract_patches(radar, mask, [4], [32], 0.03, 0.05)
    b, _, _ = extract_patches(radar * 15, mask, [4], [32], 0.03, 0.05)
    np.testing.assert_allclose(
        pair_features(a, m, a[0], m[0]), pair_features(b, m, a[0], m[0]), atol=1e-6
    )
    assert not np.allclose(pair_features(a, m, a[0], m[0]), pair_features(-a, m, a[0], m[0]))


def test_regularized_fit_save_reopen_and_fixed_seed_inference(tmp_path):
    random = np.random.default_rng(3)
    features = random.normal(size=(200, 4))
    target = (features[:, 0] + features[:, 1] > 0).astype(int)
    model, diagnostics = fit_pair_logistic(features, target)
    assert diagnostics["success"]
    assert np.mean((model.evidence(features) >= 0.5) == target) > 0.9
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = PairLogistic.load(path)
    np.testing.assert_allclose(loaded.evidence(features), model.evidence(features), atol=0)
    with pytest.raises(ValueError, match="Both positive"):
        fit_pair_logistic(features, np.ones(200))
    radar = random.normal(size=(9, 65)).astype(np.float32)
    valid = np.ones_like(radar, bool)
    patches, support, _ = extract_patches(radar, valid, [2, 4, 6], [32, 32, 32], 0.03, 0.05)
    size = pair_features(patches, support, patches[0], support[0]).shape[1]
    constant = PairLogistic(PatchContract(), np.zeros(size), np.ones(size), np.zeros(size), 0)
    seeds = {2: 32, 6: 32}
    before = seeds.copy()
    np.testing.assert_allclose(
        score_candidates(constant, radar, valid, [4], [32], seeds, 0.03, 0.05), [0.5]
    )
    assert seeds == before
    valid[4, 32] = False
    assert np.isnan(score_candidates(constant, radar, valid, [4], [32], seeds, 0.03, 0.05)[0])
