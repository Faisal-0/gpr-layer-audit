"""Observation/coordinate contracts for the optional upstream research adapter."""

import importlib.util
import subprocess
from pathlib import Path

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
