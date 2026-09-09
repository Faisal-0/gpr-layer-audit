"""Interruption recovery must retain the hypotheses used to choose observations."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.io.dzx import DZXPick


@pytest.fixture
def replay_module():
    path = Path(__file__).parents[1] / "scripts/replay_seeded_interaction.py"
    spec = importlib.util.spec_from_file_location("replay_checkpoint_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_keeps_alternatives_and_refuses_changed_contract(replay_module, tmp_path):
    module = replay_module
    path = tmp_path / "replay.checkpoint"
    state = {
        "paths": {
            2: SimpleNamespace(
                samples=np.array([12, -1]),
                alternate_samples=np.array([[11, 13], [11, 14]]),
                provenance={"routes": [[12, 12], [13, 14]]},
            )
        },
        "visited": {(2, 7)},
        "anchors": {2: {0: 12, 10: 12}},
    }
    contract = {"source_sha256": "original", "policy": "active"}
    module.save_checkpoint(path, contract, state)
    result = module.load_checkpoint(path, contract)
    np.testing.assert_array_equal(
        result["paths"][2].alternate_samples, state["paths"][2].alternate_samples
    )
    assert result["visited"] == state["visited"]
    with pytest.raises(ValueError, match="source_sha256"):
        module.load_checkpoint(path, {**contract, "source_sha256": "changed"})
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="payload hash"):
        module.load_checkpoint(path, contract)


def test_atomic_interruption_preserves_previous_checkpoint(replay_module, tmp_path, monkeypatch):
    path = tmp_path / "replay.checkpoint"
    replay_module.save_checkpoint(path, {}, {"completed_iteration": 2})
    previous = path.read_bytes()

    def interrupted(*args):
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(replay_module.os, "replace", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        replay_module.save_checkpoint(path, {}, {"completed_iteration": 3})
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_repeats_only_unfinished_query_without_duplicate_scores(
    replay_module, tmp_path, monkeypatch
):
    module = replay_module
    case = {
        "dzt": "radar",
        "dzx": "reference",
        "seed_source": "seeds",
        "dzt_sha256": "hash",
        "dzx_sha256": "hash",
        "seed_sha256": "hash",
        "stride": 1,
        "physical_road_group": "test",
        "reference_label_mapping": {"1": 2},
    }
    seeds = {"observations": {"2": [{"trace": 0, "sample": 4}, {"trace": 7, "sample": 4}]}}
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setattr(
        module,
        "read",
        lambda path: (
            seeds
            if Path(path).name == "seeds"
            else {"cases": {"test": case}}
            if Path(path).name.endswith("inputs.json")
            else {}
        ),
    )
    monkeypatch.setattr(module, "fingerprint_file", lambda path: "hash")
    monkeypatch.setattr(module, "source_fingerprint", lambda path: "source-hash")
    monkeypatch.setattr(
        module,
        "DZTFile",
        lambda path: SimpleNamespace(
            channel=lambda: np.ones((8, 12)),
            header=SimpleNamespace(
                sample_interval_ns=0.1, distance_per_trace_m=1.0, position_ns=0.0
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "read_dzx",
        lambda path: SimpleNamespace(
            layers=[
                SimpleNamespace(
                    number=1, picks=[DZXPick(r, 4, 0, 0, 0.4, 1.0, 0.0, 0.0) for r in range(8)]
                )
            ]
        ),
    )
    monkeypatch.setattr(module, "resolve_pulse", lambda *args: SimpleNamespace(lobe_samples=2))
    monkeypatch.setattr(
        module,
        "layer_metrics",
        lambda *args: {
            "observations_excluding_seeds": 6,
            "accepted": 0,
            "accepted_agree": 0,
            "correct_coverage": 0.0,
            "evaluation_observations": [],
        },
    )
    fitted = []

    def fit(*args, **kwargs):
        fitted.append(kwargs["pulse_anchors"])
        return {
            2: SimpleNamespace(
                samples=np.full(8, -1),
                provisional_samples=np.full(8, 4),
                visible=np.ones(8, bool),
                provenance={},
                alternate_samples=np.tile([4, 7], (8, 1)),
            )
        }

    monkeypatch.setattr(module, "fit_processed", fit)
    monkeypatch.setattr(module, "merge_local_paths", lambda old, new, **kwargs: new)
    monkeypatch.setattr(module, "guard_local_order", lambda *args: None)
    requested = []
    interrupt = [True]

    def request(paths, *args, visited, **kwargs):
        row = 2 if (2, 2) not in visited else 4
        requested.append(row)
        assert paths[2].alternate_samples[0, 1] == 7
        if row == 4 and interrupt[0]:
            raise InterruptedError("after completed step one")
        return {"row": row, "layer_order": 2}

    monkeypatch.setattr(module, "request_observation", request)
    args = Namespace(
        output=tmp_path / "result.json",
        case="test",
        config=config_path,
        stride=None,
        seeding="three",
        method="seed_hybrid",
        policy="active",
        scope="local",
        layers=[2],
        actions=2,
        radius_m=2.0,
        freeze_pulse=True,
        resume=False,
    )
    with pytest.raises(InterruptedError):
        module.replay(args)
    assert len(fitted) == 2
    assert len(json.loads(args.output.read_text())["steps"]) == 2
    interrupt[0] = False
    args.resume = True
    completed = module.replay(args)
    assert requested == [2, 4, 4]
    assert [s["step"] for s in completed["steps"]] == [0, 1, 2]
    assert len(fitted) == 3  # Neither initial fit nor the first correction is rerun.
    assert all(value == {2: {0: 4, 7: 4}} for value in fitted)
    assert len(completed["actions"]) == 2
    module.replay(args)
    assert len(fitted) == 3  # Completed checkpoints are idempotent.
