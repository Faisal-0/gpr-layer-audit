"""Numerical and label-boundary checks for the isolated patch matcher."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


model_module = load_module("patchnet_model")


def test_timing_misses_are_separate_from_wrong_lobe_and_unknown_targets():
    trace = np.array([0, 1, 2, 3, 2, 1, 0, -1, -2, -1, 0.0])
    candidates = np.array([1, 2, 3, 7, 8])
    np.testing.assert_array_equal(
        model_module.targets(trace, candidates, 3, 1),
        [[0, 1], [1, 1], [1, 1], [0, 0], [0, 0]],
    )
    assert np.all(model_module.targets(trace, candidates, None, 1) == -1)


def test_masked_amplitudes_cannot_change_features_or_create_observations():
    data = np.random.default_rng(42).normal(size=(65, 150)).astype(np.float32)
    valid = np.ones(data.shape, bool)
    valid[32, 75] = False
    original, usable = model_module.patches(
        data, valid, np.array([32]), np.array([75]), 0.029296875, 0.025
    )
    data[~valid] = 1e30
    changed, changed_usable = model_module.patches(
        data, valid, np.array([32]), np.array([75]), 0.029296875, 0.025
    )
    np.testing.assert_array_equal(original, changed)
    assert not usable[0] and not changed_usable[0]
    assert not model_module.native_candidates(data, valid, 0)[32, 75]


def test_masked_neighbors_cannot_change_measured_candidate_retention():
    data = np.array([[0, 1, 3, 2, 1, 2, 5, 2, 1, 0.0]])
    valid = np.ones_like(data, bool)
    valid[:, 4] = False
    expected = model_module.native_candidates(data, valid, 0)
    assert expected[0, 2] and expected[0, 6]
    for value in (-1e30, 1e30, np.nan):
        changed = data.copy()
        changed[~valid] = value
        np.testing.assert_array_equal(model_module.native_candidates(changed, valid, 0), expected)
    assert not expected[0, 4]


def test_prediction_preserves_native_seeds_without_opening_references(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    experiment = load_module("experiment_patchnet_dense")
    monkeypatch.setattr(experiment, "OUT", tmp_path)
    sample = np.arange(512)
    trace = np.sin(sample / 3) * np.exp(-(((sample - 180) / 60) ** 2))
    native = np.broadcast_to(trace.astype(np.float32), (84, 512)).copy()
    dzt = tmp_path / "operating-radar.DZT"
    dzt.write_bytes(b"reader substituted by native measurement fixture")
    entry = {
        "road": "jamshoro",
        "dzt": str(dzt),
        "dzt_sha256": experiment.sha(dzt),
        "dt_ns": 0.029296875,
        "dx_m": 0.025,
        "origin_ns": -3.0,
        "seeds": {"2": {"24": 175, "40": 175, "56": 175}, "3": {"24": 232, "40": 232, "56": 232}},
    }
    (tmp_path / "inputs.json").write_text(json.dumps([entry]))
    (tmp_path / "contract.json").write_text(json.dumps(experiment.CONTRACT))
    torch.save(model_module.PatchMatcher().state_dict(), tmp_path / "model.pt")
    (tmp_path / "training-result.json").write_text(
        json.dumps({"model_sha256": experiment.sha(tmp_path / "model.pt")})
    )
    monkeypatch.setattr(
        experiment, "DZTFile", lambda _: type("Radar", (), {"channel": lambda _: native})()
    )
    monkeypatch.setattr(
        experiment, "processed_boundary_mask", lambda data: np.ones(data.shape, bool)
    )

    def forbidden_reference(*args, **kwargs):
        raise AssertionError("Prediction opened a withheld reference")

    monkeypatch.setattr(experiment, "read_dzx", forbidden_reference)
    experiment.predict()
    manifest = experiment.read(tmp_path / "jamshoro-predictions/prediction-manifest.json")
    assert manifest["reference_opened_for_prediction"] is False
    for order in (2, 3):
        for arm in ("classical", "learned"):
            output = np.load(tmp_path / f"jamshoro-predictions/{arm}-layer{order}.npz")
            for row, observed in entry["seeds"][str(order)].items():
                assert output["samples"][int(row) // 4] == observed
