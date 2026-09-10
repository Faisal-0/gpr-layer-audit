"""Contract tests for the native processed-coordinate predictor."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _radar_only_data(n_traces: int = 25) -> dict:
    rng = np.random.default_rng(17)
    amplitudes = rng.normal(0.0, 1.0, (n_traces, 512)).astype(np.float32)
    validity = np.ones_like(amplitudes, dtype=bool)
    record = {
        "input_mode": "processed",
        "physical_road_group": "held",
        "record_id": "held-id",
        "shape": [n_traces, 512],
        "horizontal_step_m": 0.025,
        "sample_interval_ns": 0.03,
        "source_sha256": "arbitrary-source-fingerprint",
        "time_origin_ns": 0.0,
    }
    return {
        "record": record,
        "amplitudes": amplitudes,
        "sample_validity": validity,
        "native_trace_indices": np.arange(n_traces, dtype=np.int64),
        "native_sample_indices": np.arange(512, dtype=np.int64),
        "distances_m": np.arange(n_traces, dtype=np.float64) * 0.025,
    }


@pytest.fixture(scope="module")
def radar_only_data() -> dict:
    return _radar_only_data()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the smallest genuine checkpoint accepted by the bundle contract."""
    torch = pytest.importorskip("torch")

    from gpr_layer_audit.ml.processed_training import TrainingConfig
    from gpr_layer_audit.ml.seed_context_model import SeedContextUNet

    output = tmp_path_factory.mktemp("processed-inference-model")
    config = TrainingConfig(
        held_out_group="held",
        layers=(1, 2, 3),
        width=4,
        fine_width=33,
        coarse_span_ratio=4,
        patch_depth=9,
        patch_width=5,
        normalization="none",
    )
    model = SeedContextUNet(width=config.width, conditioned=True, use_context=True)
    weights = output / "weights.pt"
    torch.save(model.state_dict(), weights)
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    metadata = {
        "schema": "processed-seed-context-training-v1",
        "config": asdict(config),
        "architecture": "SeedContextUNet",
        "held_out_group": "held",
        "training_road_groups": ["train-a", "train-b"],
        "training_record_ids": ["train-a-0", "train-b-0"],
        "training_source_sha256": ["train-a-source", "train-b-source"],
        "dataset_sha256": "d" * 64,
        "weights": "weights.pt",
        "weights_sha256": digest,
    }
    (output / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
    return output


@pytest.fixture
def predictor(checkpoint: Path, radar_only_data: dict):
    from gpr_layer_audit.ml.processed_inference import ProcessedPredictor

    return ProcessedPredictor.from_checkpoint(
        checkpoint,
        radar_only_data,
        device="cpu",
        trace_stride=4,
        acceptance_threshold=float("inf"),
    )


def _seeds(*, count: int = 1) -> dict[int, dict[int, float]]:
    return {
        1: {row: 41.25 + row * 0.5 for row in range(count)},
    }


def test_processed_inference_import_is_lazy_about_torch():
    code = """
import builtins

real_import = builtins.__import__
def import_without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('processed_inference imported torch eagerly')
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
import gpr_layer_audit.ml.processed_inference
print('ok')
"""
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert completed.stdout.strip() == "ok"


def test_checkpoint_rejects_raw_input_and_training_group_overlap(checkpoint, radar_only_data):
    from gpr_layer_audit.ml.processed_inference import ProcessedPredictor

    raw = dict(radar_only_data)
    raw["record"] = {**radar_only_data["record"], "input_mode": "raw"}
    with pytest.raises(ValueError, match="processed|raw"):
        ProcessedPredictor.from_checkpoint(
            checkpoint, raw, device="cpu", trace_stride=4, acceptance_threshold=float("inf")
        )

    overlap = dict(radar_only_data)
    overlap["record"] = {
        **radar_only_data["record"],
        "physical_road_group": "train-a",
    }
    with pytest.raises(ValueError, match="training|overlap|held"):
        ProcessedPredictor.from_checkpoint(
            checkpoint,
            overlap,
            device="cpu",
            trace_stride=4,
            acceptance_threshold=float("inf"),
        )


def test_predict_evidence_is_finite_float32_and_uses_native_stride(predictor):
    evidence = predictor.predict_evidence(_seeds())
    assert evidence
    assert 1 in evidence
    expected_shape = (int(np.ceil(25 / 4)), 512)
    for layer, probabilities in evidence.items():
        assert isinstance(layer, int)
        assert probabilities.dtype == np.float32
        assert probabilities.shape == expected_shape
        assert np.isfinite(probabilities).all()
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_predict_accepts_explicit_previous_probability_guidance(predictor):
    previous = {1: np.ones((25, 512), dtype=np.float32)}
    evidence = predictor.predict_evidence(_seeds(), previous_probability=previous)
    assert evidence[1].shape == (int(np.ceil(25 / 4)), 512)
    assert np.isfinite(evidence[1]).all()
    path = predictor.predict(_seeds(), previous_probability=previous)[1]
    assert path.provenance["previous_probability_guidance"] is True


def test_predict_supports_more_than_five_seeds_and_preserves_fractional_seeds(predictor):
    from gpr_layer_audit.processing.seed_graph import SeedConditionedPath

    seeds = _seeds(count=6)
    paths = predictor.predict(seeds)
    assert set(paths) == {1}
    path = paths[1]
    assert isinstance(path, SeedConditionedPath)
    assert path.samples.shape == (int(np.ceil(25 / 4)),)
    assert path.provisional_samples.shape == path.samples.shape
    for row, sample in seeds[1].items():
        assert path.samples[row] == pytest.approx(sample)
        assert path.provisional_samples[row] == pytest.approx(sample)
        assert path.visible[row]
    unseeded = np.ones(path.samples.shape, dtype=bool)
    unseeded[list(seeds[1])] = False
    np.testing.assert_array_equal(path.samples[unseeded], -1)
    np.testing.assert_array_equal(path.visible[unseeded], False)


def test_predict_rejects_nonretained_or_invalid_seed_rows(predictor):
    with pytest.raises(ValueError, match="seed|row|coordinate"):
        predictor.predict({1: {1.5: 41.25}})
    with pytest.raises(ValueError, match="seed|row|coordinate"):
        predictor.predict({1: {7: 41.25}})
    with pytest.raises(ValueError, match="seed|sample|coordinate"):
        predictor.predict({1: {1: 512.0}})


def test_predict_and_evidence_honor_cancellation(predictor):
    calls = []

    def cancel():
        calls.append(True)
        return True

    with pytest.raises(InterruptedError, match="cancel"):
        predictor.predict_evidence(_seeds(), cancel=cancel)
    assert calls


def _correction_predictor(predictor):
    predictor.config = replace(predictor.config, correction_aware=True)
    return predictor


def test_batch_size_one_and_four_match_on_seeded_and_empty_native_evidence(
    checkpoint, radar_only_data
):
    from gpr_layer_audit.ml.processed_inference import ProcessedPredictor

    common = dict(
        checkpoint=checkpoint,
        loaded_data_dict=radar_only_data,
        device="cpu",
        trace_stride=4,
        acceptance_threshold=float("inf"),
    )
    batch_one = ProcessedPredictor.from_checkpoint(**common, batch_size=1)
    batch_four = ProcessedPredictor.from_checkpoint(**common, batch_size=4)
    for seeds in (_seeds(), {1: {}}):
        first = batch_one.predict_evidence(seeds)[1]
        fourth = batch_four.predict_evidence(seeds)[1]
        np.testing.assert_allclose(first, fourth, rtol=2e-5, atol=2e-6)
        assert first.shape == (int(np.ceil(25 / 4)), 512)


def test_correction_history_scope_updates_only_inclusive_native_interval(predictor):
    correction = _correction_predictor(predictor)
    native_shape = (25, 512)
    old = {
        1: np.full(native_shape, 1.0, dtype=np.float32),
        2: np.full(native_shape, 2.0, dtype=np.float32),
    }
    correction._previous_probability = old
    previous_state = correction.export_replay_state()
    current = {
        1: np.full(native_shape, 10.0, dtype=np.float32),
        2: np.full(native_shape, 20.0, dtype=np.float32),
    }
    correction._previous_probability = current
    correction.commit_replay_scope(previous_state, 1, 1, 3)
    state = correction.export_replay_state()
    layer_one = state["previous_probability"][1]
    layer_two = state["previous_probability"][2]
    expected = np.full(native_shape, 1.0, dtype=np.float32)
    expected[1 * 4 : 3 * 4 + 1] = 10.0
    np.testing.assert_array_equal(layer_one, expected)
    np.testing.assert_array_equal(layer_two, 2.0)


def test_restore_replay_state_rejects_mismatched_model_and_stride(
    checkpoint, radar_only_data, predictor
):
    from gpr_layer_audit.ml.processed_inference import ProcessedPredictor

    correction = _correction_predictor(predictor)
    correction._previous_probability = {1: np.ones((25, 512), dtype=np.float32)}
    state = correction.export_replay_state()
    wrong_model = dict(state, model_sha256="0" * 64)
    with pytest.raises(ValueError, match="model_sha256"):
        correction.restore_replay_state(wrong_model)

    different_stride = ProcessedPredictor.from_checkpoint(
        checkpoint,
        radar_only_data,
        device="cpu",
        trace_stride=2,
        acceptance_threshold=float("inf"),
    )
    with pytest.raises(ValueError, match="trace_stride"):
        different_stride.restore_replay_state(state)


def test_explicit_correction_prior_is_used_after_successful_predict(predictor):
    correction = _correction_predictor(predictor)
    seeds = _seeds()
    correction.predict(seeds)
    prior = {1: np.ones((25, 512), dtype=np.float32)}
    path = correction.predict(seeds, previous_probability=prior)[1]
    assert path.provenance["previous_probability_guidance"] is True
    restored = correction.export_replay_state()
    assert restored["previous_probability"][1].dtype == np.float32
    assert restored["previous_probability"][1].shape == (25, 512)


def test_cancelled_correction_does_not_replace_previous_history(predictor):
    correction = _correction_predictor(predictor)
    seeds = _seeds()
    correction.predict(seeds)
    before = correction.export_replay_state()
    with pytest.raises(InterruptedError, match="cancel"):
        correction.predict(seeds, cancel=lambda: True)
    after = correction.export_replay_state()
    np.testing.assert_array_equal(
        after["previous_probability"][1], before["previous_probability"][1]
    )
