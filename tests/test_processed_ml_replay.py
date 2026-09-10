"""Native action boundaries, frozen denominator, and real replay persistence."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.ml.processed_ml_replay import (
    ReplayConfig,
    checkpoint_helpers,
    exact_answers,
    replay_frozen_model,
    request_model_observation,
    score_fixed_cohort,
)
from gpr_layer_audit.models import LayerSpec
from gpr_layer_audit.processing.seed_graph import SeedConditionedPath


def path_for(samples, *, accepted=True):
    samples = np.asarray(samples, np.int32)
    count = len(samples)
    return SeedConditionedPath(
        samples=samples.copy() if accepted else np.full(count, -1, np.int32),
        confidence=np.full(count, 0.5),
        feature=np.zeros((count, 14), np.float32),
        alternate_samples=np.full(count, 12, np.int32),
        visible=np.full(count, accepted),
        interpolated=np.zeros(count, bool),
        evidence={"model_entropy": np.linspace(0, 1, count)},
        signal_only_samples=samples.copy(),
        design_guided_samples=samples.copy(),
        design_conflict=np.zeros(count, bool),
        design_constrained=False,
        provisional_samples=samples.copy(),
        provenance={"model_sha256": "b" * 64},
    )


@pytest.fixture
def setup():
    initial = {2: {0: 4, 50: 4, 100: 4}}
    references = {
        2: [SimpleNamespace(trace=r, sample=4 if r in initial[2] else 9) for r in range(101)]
    }
    calls = []

    def predict(anchors):
        calls.append({o: dict(a) for o, a in anchors.items()})
        samples = np.full(101, 9 if len(anchors[2]) > 3 else 4)
        for row, sample in anchors[2].items():
            samples[row] = sample
        return {2: path_for(samples)}

    kwargs = dict(
        predict=predict,
        measurement=np.ones((101, 14), np.float32),
        valid=np.ones((101, 14), bool),
        initial_anchors=initial,
        references=references,
        layers=[LayerSpec(2, "Base", 1, 13, 1)],
        dt_ns=0.05,
        dx_m=1.0,
        pulse_samples={2: 4.0},
        provenance={
            "coordinate_mode": "processed",
            "input_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "preprocessing_sha256": "c" * 64,
            "source_sha256": "d" * 64,
            "trace_stride": 4,
        },
    )
    return kwargs, calls


def test_real_local_merge_retains_outside_and_exact_answer_has_no_credit(setup, tmp_path):
    kwargs, calls = setup
    output = tmp_path / "local.json"
    result = replay_frozen_model(
        **kwargs, output=output, config=ReplayConfig(policy="fixed_spacing", budgets=(0, 1))
    )
    assert len(calls) == 2
    action = result["actions"][0]
    assert action["row"] == 20 and action["native_trace"] == 80
    assert action["operation"] == "local_correction" and action["affected_rows"] == [0, 45]
    metric = result["steps"][1]["layers"]["2"]
    assert metric["fixed_initial_observations"] == 98
    assert metric["analyst_supplied_rows_no_automatic_credit"] == [20]
    assert metric["correct_automatically_accepted"] == 44
    assert metric["correct_automatic_coverage"] == 44 / 98
    assert result["steps"][1]["actions_per_km"] == 10
    saved = np.load(tmp_path / "local-step1.npz")
    original = np.load(tmp_path / "local-step0.npz")
    for name in saved.files:
        np.testing.assert_array_equal(saved[name][46:], original[name][46:])
    assert saved["layer2_samples"][20] == 9
    assert result["steps"][1]["changes_after_action"]["2"][
        "newly_answered_rows_no_automatic_credit"
    ] == [20]


def test_explicit_global_model_seed_equal_cost_and_all_budget_prefixes(setup, tmp_path):
    kwargs, calls = setup
    result = replay_frozen_model(
        **kwargs,
        output=tmp_path / "global.json",
        config=ReplayConfig(policy="fixed_spacing", operation="global_model_seed"),
    )
    assert len(calls) == 5
    assert [a["row"] for a in result["actions"]] == [20, 40, 60, 80]
    assert [s["additional_requests"] for s in result["steps"] if s["reported_budget"]] == [
        0,
        1,
        2,
        4,
    ]
    assert result["actions"][0]["affected_rows"] == [0, 100]
    assert result["actions"][0]["model_weights_fitted"] is False
    metric = result["steps"][1]["layers"]["2"]
    assert metric["correct_automatic_coverage"] == 97 / 98
    assert metric["accepted_pick_agreement"] == 1
    assert all(s["layers"]["2"]["fixed_initial_observations"] == 98 for s in result["steps"])


def test_missing_exact_answer_is_charged_without_relocation_or_predictor_access(setup, tmp_path):
    kwargs, calls = setup
    kwargs["references"][2] = [p for p in kwargs["references"][2] if p.trace != 20]
    result = replay_frozen_model(
        **kwargs,
        output=tmp_path / "missing.json",
        config=ReplayConfig(policy="fixed_spacing", budgets=(0, 1)),
    )
    assert len(calls) == 1
    assert result["actions"][0]["row"] == 20
    assert result["actions"][0]["answer_status"] == "unavailable_reviewed_answer"
    assert "answer_sample" not in result["actions"][0]
    assert result["steps"][1]["additional_requests"] == 1
    assert result["steps"][1]["layers"]["2"]["analyst_supplied_in_initial_cohort"] == 0


def test_policies_see_no_references_and_choose_model_or_geometry(setup):
    kwargs, _ = setup
    paths = kwargs["predict"](kwargs["initial_anchors"])
    args = (paths, kwargs["measurement"], kwargs["valid"], kwargs["initial_anchors"], 1.0)
    assert request_model_observation(*args, policy="fixed_spacing")["row"] == 20
    assert request_model_observation(*args, policy="largest_interval_midpoint")["row"] == 25
    assert request_model_observation(*args, policy="uncertainty")["row"] == 99
    assert (
        request_model_observation(*args, policy="largest_interval_midpoint", visited={(2, 25)})[
            "row"
        ]
        == 75
    )
    # Radar observability is allowed; selecting another requested row uses no answers.
    kwargs["measurement"][99] = 0
    assert request_model_observation(*args, policy="uncertainty")["row"] == 98


def test_checkpoint_resume_preserves_actual_path_arrays_and_does_not_repeat_actions(
    setup, tmp_path
):
    kwargs, calls = setup
    real_predict = kwargs["predict"]
    fail = [True]

    def interrupted(anchors):
        if len(anchors[2]) == 5 and fail[0]:
            raise InterruptedError("bounded interruption")
        return real_predict(anchors)

    kwargs["predict"] = interrupted
    output = tmp_path / "resume.json"
    config = ReplayConfig(policy="fixed_spacing", operation="global_model_seed", budgets=(0, 1, 2))
    with pytest.raises(InterruptedError):
        replay_frozen_model(**kwargs, output=output, config=config)
    assert len(calls) == 2
    fail[0] = False
    result = replay_frozen_model(**kwargs, output=output, config=config, resume=True)
    assert len(calls) == 3
    assert [a["row"] for a in result["actions"]] == [20, 40]
    again = replay_frozen_model(**kwargs, output=output, config=config, resume=True)
    assert again == result and len(calls) == 3
    io = checkpoint_helpers()
    header = io.checkpoint_header(io.checkpoint_path(output))
    restored = io.load_checkpoint(io.checkpoint_path(output), header["contract"])
    assert restored["paths"][2].alternate_samples[4] == 12
    kwargs["provenance"]["model_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="contract changed"):
        replay_frozen_model(**kwargs, output=output, config=config, resume=True)


def test_fixed_denominator_counts_only_reviewed_initial_nonseeds_and_zero_is_undefined(setup):
    kwargs, _ = setup
    points = [p for p in kwargs["references"][2] if p.trace != 0]
    path = path_for(np.full(101, 9), accepted=False)
    result = score_fixed_cohort(
        {2: path},
        {2: points},
        kwargs["initial_anchors"],
        {2: {**kwargs["initial_anchors"][2], 20: 9}},
        kwargs["measurement"],
        {2: 4},
        1,
        0.05,
    )["2"]
    assert result["fixed_initial_observations"] == 98
    assert result["accepted_pick_agreement"] is None
    assert result["automatic_unresolved"] == 98
    assert result["unresolved_after_analyst_answers"] == 97


def test_contracts_reject_raw_offgrid_duplicate_and_changed_radius(setup, tmp_path):
    kwargs, _ = setup
    kwargs["provenance"]["coordinate_mode"] = "raw"
    with pytest.raises(ValueError, match="raw"):
        replay_frozen_model(**kwargs, output=tmp_path / "invalid.json")
    with pytest.raises(ValueError, match="exactly"):
        ReplayConfig(radius_m=50).validate()
    with pytest.raises(ValueError, match="exact native"):
        exact_answers({2: [SimpleNamespace(trace=0, sample=4.2)]}, (2, 10))
    with pytest.raises(ValueError, match="Duplicate"):
        exact_answers({2: [SimpleNamespace(trace=0, sample=4)] * 2}, (2, 10))


def test_unknown_reviewed_rows_break_wrong_observed_spans():
    paths = {2: path_for([4] * 7)}
    refs = {2: [SimpleNamespace(trace=r, sample=9) for r in [0, 1, 2, 4, 5, 6]]}
    metric = score_fixed_cohort(
        paths, refs, {2: {0: 4}}, {2: {0: 4}}, np.ones((7, 14)), {2: 4}, 2.0, 0.05
    )["2"]
    assert metric["longest_contiguous_wrong_accepted_observed_span_m"] == 6.0
    assert metric["longest_wrong_observed_endpoint_distance_m"] == 4.0


def test_checkpoint_helpers_are_original_repository_implementation():
    module = checkpoint_helpers()
    assert Path(module.__file__).name == "replay_seeded_interaction.py"


def test_cli_seed_artifact_keeps_exact_native_samples_and_refuses_grid_snapping(tmp_path):
    script = Path(__file__).parents[1] / "scripts/replay_processed_ml.py"
    spec = importlib.util.spec_from_file_location("_processed_replay_cli_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seeds = tmp_path / "seeds.json"
    seeds.write_text(
        json.dumps(
            {
                "mode": "processed",
                "dzt_sha256": "source",
                "dzx_sha256": "reference",
                "observations": {"2": [{"trace": 12, "sample": 7, "channel": 0}]},
            }
        )
    )
    args = SimpleNamespace(case=None, seeds=seeds, stride=4, layer=2)
    anchors, seed_hash = module.load_anchors(
        args, {"source_sha256": "source", "label_sha256": "reference", "shape": [100, 14]}
    )
    assert anchors == {2: {3: 7}} and len(seed_hash) == 64
    args.stride = 5
    with pytest.raises(ValueError, match="snapping"):
        module.load_anchors(
            args, {"source_sha256": "source", "label_sha256": "reference", "shape": [100, 14]}
        )


def test_native_fractional_proposal_errors_are_not_truncated():
    path = path_for([4, 4])
    path.provisional_samples = np.array([4.0, 4.75])
    metric = score_fixed_cohort(
        {2: path},
        {2: [SimpleNamespace(trace=1, sample=4)]},
        {2: {0: 4}},
        {2: {0: 4}},
        np.ones((2, 14)),
        {2: 4},
        1.0,
        0.1,
    )["2"]
    assert metric["evaluation_observations"][0]["proposal_native_sample"] == 4.75
    assert metric["evaluation_observations"][0]["proposal_error_ns"] == pytest.approx(0.075)


def test_replay_refuses_predictor_that_moves_authorized_sample(setup, tmp_path):
    kwargs, _ = setup
    kwargs["predict"] = lambda anchors: {2: path_for([5] * 101)}
    with pytest.raises(ValueError, match="moved an exact authorized"):
        replay_frozen_model(**kwargs, output=tmp_path / "moved.json")
    assert not (tmp_path / "moved.checkpoint").exists()


def test_fractional_authorized_seed_survives_replay_and_checkpoint(setup, tmp_path):
    kwargs, _ = setup
    kwargs["initial_anchors"][2][0] = 4.25

    def predict(anchors):
        path = path_for([4] * 101)
        path.samples = path.samples.astype(float)
        path.provisional_samples = path.provisional_samples.astype(float)
        for row, sample in anchors[2].items():
            path.samples[row] = path.provisional_samples[row] = sample
        return {2: path}

    kwargs["predict"] = predict
    output = tmp_path / "fractional.json"
    replay_frozen_model(**kwargs, output=output, config=ReplayConfig(budgets=(0,)))
    stored = np.load(tmp_path / "fractional-step0.npz")
    assert stored["layer2_samples"][0] == 4.25
    assert stored["layer2_provisional_samples"][0] == 4.25


def test_correction_aware_history_is_scoped_and_restored_after_interruption(setup, tmp_path):
    kwargs, _ = setup
    base_predict = kwargs["predict"]

    class Stateful:
        config = SimpleNamespace(correction_aware=True)

        def __init__(self, interrupt=False):
            self.history = np.zeros(101)
            self.interrupt = interrupt

        def export_replay_state(self):
            return {"history": self.history.copy()}

        def restore_replay_state(self, state):
            self.history = state["history"].copy()

        def commit_replay_scope(self, previous_state, order, lo, hi):
            assert order == 2
            self.history[:lo] = previous_state["history"][:lo]
            self.history[hi + 1 :] = previous_state["history"][hi + 1 :]

        def predict(self, anchors):
            self.history += 1
            if self.interrupt and len(anchors[2]) == 5:
                raise InterruptedError("history was updated before interruption")
            return base_predict(anchors)

    first = Stateful(interrupt=True)
    kwargs["predict"] = first.predict
    output = tmp_path / "stateful.json"
    config = ReplayConfig(policy="fixed_spacing", budgets=(0, 1, 2))
    with pytest.raises(InterruptedError):
        replay_frozen_model(**kwargs, output=output, config=config)
    np.testing.assert_array_equal(first.history, np.r_[np.full(46, 2), np.ones(55)])
    reopened = Stateful()
    kwargs["predict"] = reopened.predict
    result = replay_frozen_model(**kwargs, output=output, config=config, resume=True)
    assert [a["row"] for a in result["actions"]] == [20, 40]
    np.testing.assert_array_equal(
        reopened.history, np.r_[np.full(15, 2), np.full(31, 3), np.full(20, 2), np.ones(35)]
    )


@pytest.mark.parametrize(
    "failure_stage", ["predict", "merge", "scope_commit", "score", "npz", "checkpoint"]
)
def test_pending_final_request_and_answer_are_durable_before_fit(
    setup, tmp_path, monkeypatch, failure_stage
):
    import gpr_layer_audit.ml.processed_ml_replay as replay

    kwargs, calls = setup
    base_predict = kwargs["predict"]
    output = tmp_path / f"pending-{failure_stage}.json"
    io = checkpoint_helpers()
    kwargs["checkpoint_io"] = io
    lookups = []
    base_answers = exact_answers
    base_request = request_model_observation
    requests = []
    saved_before_fit = []

    class AnswerMap(dict):
        def get(self, key, default=None):
            lookups.append(key)
            return super().get(key, default)

    def counted_answers(*args):
        return {o: AnswerMap(values) for o, values in base_answers(*args).items()}

    def counted_request(*args, **kwargs):
        requests.append(True)
        return base_request(*args, **kwargs)

    def load_state():
        path = io.checkpoint_path(output)
        return io.load_checkpoint(path, io.checkpoint_header(path)["contract"])

    class Stateful:
        config = SimpleNamespace(correction_aware=True)

        def __init__(self, fail):
            self.history = np.zeros(101)
            self.fail = fail

        def export_replay_state(self):
            return {"history": self.history.copy()}

        def restore_replay_state(self, state):
            self.history = state["history"].copy()

        def commit_replay_scope(self, previous_state, order, lo, hi):
            if self.fail and failure_stage == "scope_commit":
                self.history[:] = 99
                raise InterruptedError("scope commit interrupted")
            self.history[:lo] = previous_state["history"][:lo]
            self.history[hi + 1 :] = previous_state["history"][hi + 1 :]

        def predict(self, anchors):
            if 20 in anchors[2]:
                stored = load_state()
                saved_before_fit.append(stored)
                assert stored["pending_action"] == 0
                assert len(stored["log"]["actions"]) == 1
                assert stored["log"]["actions"][0]["answer_sample"] == 9
                assert stored["revealed_answers"] == {2: {20: 9}}
                assert stored["visited"] == {(2, 20)}
                assert stored["anchors"] == kwargs["initial_anchors"]
                assert stored["corrections"] == {}
                np.testing.assert_array_equal(stored["paths"][2].samples, np.full(101, 4))
                np.testing.assert_array_equal(stored["predictor_state"]["history"], np.ones(101))
            self.history += 1
            if 20 in anchors[2] and self.fail and failure_stage == "predict":
                raise InterruptedError("prediction interrupted")
            return base_predict(anchors)

    base_merge = replay.merge_local_paths
    base_score = replay.score_fixed_cohort
    base_save = np.savez_compressed
    base_checkpoint = io.save_checkpoint

    def interrupted_merge(*args, **kwargs):
        raise InterruptedError("merge interrupted")

    def interrupted_score(*args, **kwargs):
        if kwargs["revealed_answers"][2]:
            raise InterruptedError("scoring interrupted")
        return base_score(*args, **kwargs)

    def interrupted_save(path, *args, **kwargs):
        if path.name.endswith("-step1.npz"):
            raise InterruptedError("NPZ write interrupted")
        return base_save(path, *args, **kwargs)

    def interrupted_checkpoint(path, contract, state):
        if len(state["log"]["steps"]) == 2:
            raise InterruptedError("checkpoint write interrupted")
        return base_checkpoint(path, contract, state)

    monkeypatch.setattr(replay, "exact_answers", counted_answers)
    monkeypatch.setattr(replay, "request_model_observation", counted_request)
    if failure_stage == "merge":
        monkeypatch.setattr(replay, "merge_local_paths", interrupted_merge)
    elif failure_stage == "score":
        monkeypatch.setattr(replay, "score_fixed_cohort", interrupted_score)
    elif failure_stage == "npz":
        monkeypatch.setattr(np, "savez_compressed", interrupted_save)
    elif failure_stage == "checkpoint":
        monkeypatch.setattr(io, "save_checkpoint", interrupted_checkpoint)
    first = Stateful(fail=True)
    kwargs["predict"] = first.predict
    config = ReplayConfig(policy="fixed_spacing", budgets=(0, 1))
    with pytest.raises(InterruptedError):
        replay_frozen_model(**kwargs, output=output, config=config)
    state = load_state()
    assert len(saved_before_fit) == 1
    assert [s["additional_requests"] for s in state["log"]["steps"]] == [0]
    assert state["log"]["actions"][0]["completion_status"] == "pending"
    if failure_stage not in {"score", "npz", "checkpoint"}:
        assert state["log"]["actions"][0]["retrack_status"] == "pending_retry"
        assert state["log"]["actions"][0]["retrack_attempts"][0]["status"] == "interrupted"
    np.testing.assert_array_equal(first.history, np.ones(101))
    np.testing.assert_array_equal(state["predictor_state"]["history"], np.ones(101))
    monkeypatch.setattr(replay, "merge_local_paths", base_merge)
    monkeypatch.setattr(replay, "score_fixed_cohort", base_score)
    monkeypatch.setattr(np, "savez_compressed", base_save)
    monkeypatch.setattr(io, "save_checkpoint", base_checkpoint)
    reopened = Stateful(fail=False)
    kwargs["predict"] = reopened.predict
    result = replay_frozen_model(**kwargs, output=output, config=config, resume=True)
    assert len(requests) == 1 and lookups == [20]
    assert [s["additional_requests"] for s in result["steps"]] == [0, 1]
    assert len(result["actions"]) == 1
    assert result["actions"][0]["completion_status"] == "completed"
    assert result["actions"][0]["retrack_status"] == "applied"
    assert len(result["actions"][0]["retrack_attempts"]) == (
        1 if failure_stage in {"score", "npz", "checkpoint"} else 2
    )
    assert load_state()["pending_action"] is None
    np.testing.assert_array_equal(reopened.history, np.r_[np.full(46, 2), np.ones(55)])


def test_neighbor_rejection_is_revealed_but_never_becomes_active_seed(
    setup, tmp_path, monkeypatch
):
    import gpr_layer_audit.ml.processed_ml_replay as replay

    kwargs, _ = setup
    kwargs["initial_anchors"] = {1: {0: 2, 50: 2, 100: 2}, 2: {0: 9, 50: 9, 100: 9}}
    kwargs["references"] = {
        1: [SimpleNamespace(trace=r, sample=2 if r in {0, 50, 100} else 8) for r in range(101)],
        2: [SimpleNamespace(trace=r, sample=5 if r == 20 else 9) for r in range(101)],
    }
    kwargs["layers"] = [LayerSpec(1, "Asphalt", 1, 13, 1), LayerSpec(2, "Base", 1, 13, 1)]
    kwargs["pulse_samples"] = {1: 4.0, 2: 4.0}
    output = tmp_path / "rejection.json"
    io = checkpoint_helpers()
    calls = []
    rejected_states = []

    def choose(paths, measurement, valid, anchors, dx_m, *, visited, **policy):
        return {"layer_order": 2, "row": 20 if (2, 20) not in visited else 40}

    class Stateful:
        config = SimpleNamespace(correction_aware=True)

        def __init__(self):
            self.history = np.zeros(101)

        def export_replay_state(self):
            return {"history": self.history.copy()}

        def restore_replay_state(self, state):
            self.history = state["history"].copy()

        def commit_replay_scope(self, previous_state, order, lo, hi):
            self.history[:lo] = previous_state["history"][:lo]
            self.history[hi + 1 :] = previous_state["history"][hi + 1 :]

        def predict(self, anchors):
            calls.append({o: dict(a) for o, a in anchors.items()})
            if 40 in anchors[2]:
                assert 20 not in anchors[2]
                path = io.checkpoint_path(output)
                state = io.load_checkpoint(path, io.checkpoint_header(path)["contract"])
                rejected_states.append(state)
                assert state["anchors"] == kwargs["initial_anchors"]
                assert state["corrections"] == {}
                assert state["revealed_answers"][2] == {20: 5, 40: 9}
                np.testing.assert_array_equal(self.history, np.ones(101))
                np.testing.assert_array_equal(state["paths"][2].samples, np.full(101, 9))
            self.history += 1
            result = {}
            for order, seed in anchors.items():
                samples = np.full(101, 8 if order == 1 else 9)
                for row, sample in seed.items():
                    samples[row] = sample
                result[order] = path_for(samples)
            return result

    predictor = Stateful()
    kwargs["predict"] = predictor.predict
    monkeypatch.setattr(replay, "request_model_observation", choose)
    result = replay_frozen_model(
        **kwargs, output=output, config=ReplayConfig(budgets=(0, 1, 2))
    )
    assert len(calls) == 3 and len(rejected_states) == 1
    assert result["actions"][0]["retrack_status"] == "neighbor_observation_conflict"
    assert result["actions"][1]["retrack_status"] == "applied"
    metric = result["steps"][1]["layers"]["2"]
    assert metric["fixed_initial_observations"] == 98
    assert metric["analyst_supplied_rows_no_automatic_credit"] == [20]
    assert metric["revealed_but_inactive_answer_rows"] == [20]
    assert 20 not in {p["row"] for p in metric["evaluation_observations"]}
    final = result["steps"][2]["layers"]["2"]
    assert final["fixed_initial_observations"] == 98
    assert final["analyst_supplied_rows_no_automatic_credit"] == [20, 40]
    assert final["revealed_but_inactive_answer_rows"] == [20]
    path = io.checkpoint_path(output)
    state = io.load_checkpoint(path, io.checkpoint_header(path)["contract"])
    assert state["corrections"] == {2: {40: 9}}
    assert state["anchors"][2] == {0: 9, 50: 9, 100: 9, 40: 9}
    assert state["revealed_answers"][2] == {20: 5, 40: 9}
    assert state["paths"][2].samples[20] == 9
    np.testing.assert_array_equal(
        predictor.history, np.r_[np.ones(15), np.full(51, 2), np.ones(35)]
    )


@pytest.mark.parametrize("write_stage", ["pending", "scored"])
@pytest.mark.parametrize("available", [False, True])
def test_resume_uses_authoritative_checkpoint_after_json_write_failure(
    setup, tmp_path, monkeypatch, write_stage, available
):
    import gpr_layer_audit.ml.processed_ml_replay as replay

    kwargs, calls = setup
    if not available:
        kwargs["references"][2] = [p for p in kwargs["references"][2] if p.trace != 20]
    output = tmp_path / "split-write.json"
    io = checkpoint_helpers()
    atomic_write = io.atomic_write
    request = request_model_observation
    requests = []

    def counted_request(*args, **kwargs):
        requests.append(True)
        return request(*args, **kwargs)

    def fail_json(path, data):
        if path == output:
            payload = json.loads(data)
            has_action = len(payload["actions"]) == 1
            pending = has_action and payload["actions"][0]["completion_status"] == "pending"
            scored = has_action and len(payload["steps"]) == 2
            if (write_stage == "pending" and pending) or (write_stage == "scored" and scored):
                raise OSError("JSON write interrupted after checkpoint commit")
        return atomic_write(path, data)

    monkeypatch.setattr(replay, "request_model_observation", counted_request)
    monkeypatch.setattr(io, "atomic_write", fail_json)
    config = ReplayConfig(policy="fixed_spacing", budgets=(0, 1))
    with pytest.raises(OSError, match="JSON write interrupted"):
        replay_frozen_model(**kwargs, output=output, config=config, checkpoint_io=io)
    monkeypatch.setattr(io, "atomic_write", atomic_write)
    result = replay_frozen_model(
        **kwargs, output=output, config=config, checkpoint_io=io, resume=True
    )
    assert len(requests) == 1
    assert len(calls) == (2 if available else 1)
    assert [s["additional_requests"] for s in result["steps"]] == [0, 1]
    assert len(result["actions"]) == 1
    assert json.loads(output.read_text()) == result
