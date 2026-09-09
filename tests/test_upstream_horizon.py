"""Observation/coordinate contracts for the optional upstream research adapter."""

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.processing.upstream_horizon import (
    profile_paths,
    valid_runs,
    warped_context_cosine,
)


def test_valid_runs_do_not_connect_missing_observations():
    assert valid_runs([0, 1, 1, 0, 1, 1, 1, 0]) == [(1, 3), (4, 7)]


def test_mapping_preserves_all_centre_alternatives():
    path = np.array([(0, 0), (1, 1), (2, 2), (2, 3), (3, 4)])
    left = np.array([0, -2, 1, -1])
    right = np.array([0, -2, 1, 1, -1])
    assert warped_context_cosine(left, right, path, 2, 3, 4, 0) == pytest.approx(1)
    assert warped_context_cosine(left, right, path, 2, 4, 4, 0) == -1


def test_exact_dtw_never_maps_across_explicit_mask_gap():
    pytest.importorskip("tslearn")
    trace = np.array([2, 3, 4, 0, 5, 7, 9], float)
    valid = np.array([1, 1, 1, 0, 1, 1, 1], bool)
    first, second = profile_paths(trace, trace, valid, valid)
    assert np.array_equal(first, second)
    assert np.array_equal(first[:, 0], [0, 1, 2, 4, 5, 6])
    assert np.array_equal(first[:, 0], first[:, 1])


def test_upstream_pin_rejects_staged_source_change(tmp_path, monkeypatch):
    runner_path = Path(__file__).resolve().parents[1] / "scripts/experiment_upstream_horizon.py"
    spec = importlib.util.spec_from_file_location("horizon_experiment_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    source = checkout / "HorizonTracker_functions.py"
    source.write_text("# pinned source\n")
    subprocess.run(["git", "-C", str(checkout), "add", source.name], check=True)
    subprocess.run([
        "git", "-C", str(checkout), "-c", "user.name=contract-test", "-c",
        "user.email=contract-test@example.invalid", "commit", "-m", "pin",
    ], check=True, capture_output=True)
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(runner, "PIN", revision)
    source.write_text("raise RuntimeError('modified upstream must not execute')\n")
    subprocess.run(["git", "-C", str(checkout), "add", source.name], check=True)
    with pytest.raises(subprocess.CalledProcessError):
        runner.load_upstream(checkout)


def test_config_hash_identifies_parsed_bytes_when_file_changes_during_run(tmp_path, monkeypatch):
    from gpr_layer_audit import conventional

    runner_path = Path(__file__).resolve().parents[1] / "scripts/experiment_upstream_horizon.py"
    spec = importlib.util.spec_from_file_location("horizon_config_race_test", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config_path = tmp_path / "benchmarks/conventional-motion-calibrated-development.json"
    config_path.parent.mkdir()
    original_bytes = b'{"minimum_correspondence": 0.8}\n'
    config_path.write_bytes(original_bytes)
    source = tmp_path / "src"
    adapter = source / "gpr_layer_audit/processing/upstream_horizon.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text("# frozen adapter placeholder\n")
    output = {}

    def fake_evaluate_reference(*args, config, **kwargs):
        assert config == json.loads(original_bytes)
        config_path.write_text('{"minimum_correspondence": 0.2}\n')
        return {"config": config, "methods": {"seed_hybrid": {"layers": {}}}}

    monkeypatch.setattr(conventional, "evaluate_reference", fake_evaluate_reference)
    monkeypatch.setattr(conventional, "write_json", lambda path, result: output.update(result))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "test-packages")
    runner.run(
        SimpleNamespace(control=True, source=source, output=tmp_path / "result.json"),
        {"dzx": "unopened-reference.DZX", "seed_source": "unopened-seeds.json", "stride": 1},
    )
    expected_hash = hashlib.sha256(original_bytes).hexdigest()
    assert output["upstream_experiment"]["config_sha256"] == expected_hash
    assert output["upstream_experiment"]["config_sha256"] != hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
