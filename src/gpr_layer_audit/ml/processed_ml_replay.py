"""Research-only frozen-model interaction through the real processed merge contract.

References live only in the evaluator/answer boundary. Request selection accepts
radar, predictions and already authorized observations. Local correction uses the
same inclusive +/-25 m merge and neighboring-interface guard as the application.
Global re-anchoring is a separately named, equally charged operation; neither
operation updates model weights. Persistence reuses the established trusted-local
replay checkpoint implementation, including its hash and contract checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from gpr_layer_audit.conventional import layer_metrics
from gpr_layer_audit.processing.processed_tracking import guard_local_order, merge_local_paths


@dataclass(frozen=True)
class ReplayConfig:
    policy: str = "uncertainty"
    operation: str = "local_correction"
    budgets: tuple[int, ...] = (0, 1, 2, 4)
    fixed_spacing_slots: int = 4
    radius_m: float = 25.0

    def validate(self):
        if self.policy not in {"uncertainty", "fixed_spacing", "largest_interval_midpoint"}:
            raise ValueError("Unknown research request policy")
        if self.operation not in {"local_correction", "global_model_seed"}:
            raise ValueError("Unknown research correction operation")
        if self.radius_m != 25.0:
            raise ValueError("The local correction contract is exactly +/-25 m")
        if not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("Budgets must be unique and ascending")
        if self.budgets[0] != 0 or any(type(b) is not int or b < 0 for b in self.budgets):
            raise ValueError("Budgets must start at zero and contain nonnegative integers")
        if self.fixed_spacing_slots < max(self.budgets):
            raise ValueError("Fixed spacing schedule must support the full action budget")


def request_model_observation(
    paths, measurement, valid, anchors, dx_m, *, visited=(), policy="uncertainty", fixed_slots=4
):
    """Select an exact working row without access to reference locations or values.

    Fixed spacing uses the same four full-road positions for every budget prefix.
    Midpoint splits the longest interval including extrapolated end intervals.
    Uncertainty uses model entropy, or explicitly uncalibrated 1-confidence, and
    a one-metre exclusion taper around already visited/supplied observations.
    """
    if policy not in {"uncertainty", "fixed_spacing", "largest_interval_midpoint"}:
        raise ValueError("Unknown research request policy")
    rows = len(measurement)
    visited = set(visited)
    measurable = np.any(valid & np.isfinite(measurement) & (measurement != 0), axis=1)
    requests = []
    orders = sorted(paths)
    for order, path in sorted(paths.items()):
        occupied = set(anchors.get(order, {})) | {r for o, r in visited if o == order}
        free = measurable.copy()
        free[list(occupied)] = False
        candidates = np.flatnonzero(free)
        if not len(candidates):
            continue
        if policy == "fixed_spacing":
            scheduled = np.rint(np.linspace(0, rows - 1, fixed_slots + 2)[1:-1]).astype(int)
            chosen = next(((i, int(r)) for i, r in enumerate(scheduled) if free[r]), None)
            if chosen is None:
                continue
            slot, row = chosen
            priority = float(fixed_slots - slot - orders.index(order) / (len(orders) + 1))
            reason = "Predeclared full-road fixed spacing position"
        elif policy == "largest_interval_midpoint":
            boundaries = sorted(occupied | {0, rows - 1})
            intervals = []
            for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
                inside = candidates[(candidates > left) & (candidates < right)]
                if len(inside):
                    middle = (left + right) / 2
                    row = int(inside[np.argmin(abs(inside - middle))])
                    intervals.append(((right - left) * dx_m, -row, row))
            if not intervals:
                # A still-unvisited physical endpoint is a legitimate last request.
                row, priority = int(candidates[0]), 0.0
            else:
                priority, _, row = max(intervals)
            reason = "Midpoint of largest interval between supplied/visited rows and road ends"
        else:
            entropy = path.evidence.get("model_entropy", path.evidence.get("normalized_entropy"))
            score = np.asarray(entropy if entropy is not None else 1 - path.confidence, float)
            if score.shape != (rows,) or not np.all(np.isfinite(score)):
                raise ValueError("Model uncertainty must be a finite per-row array")
            if occupied:
                distance = np.min(abs(candidates[:, None] - np.array(sorted(occupied))), axis=1)
                taper = np.minimum(1.0, distance * dx_m / max(dx_m, 1.0))
            else:
                taper = 1.0
            values = np.maximum(0, score[candidates]) * taper
            row = int(candidates[np.argmax(values)])
            priority = float(np.max(values))
            reason = "Frozen model uncertainty; heuristic ranking without reference access"
        proposed = path.provisional_samples
        if proposed is None:
            proposed = path.samples
        requests.append(
            {
                "layer_order": int(order),
                "row": int(row),
                "priority": float(priority),
                "reason": reason,
                "policy": policy,
                "benefit_is_calibrated": False,
                "candidate_samples": [float(proposed[row])] if proposed[row] >= 0 else [],
            }
        )
    return (
        max(requests, key=lambda q: (q["priority"], -q["layer_order"], -q["row"]))
        if requests
        else None
    )


def exact_answers(references, shape):
    """Build the answer service, rejecting ambiguous or off-grid reference exports."""
    result = {}
    for order, points in references.items():
        values = {}
        for point in points:
            row, sample = point.trace, point.sample
            if int(row) != row or int(sample) != sample:
                raise ValueError("Reference answers require exact native trace/sample coordinates")
            if not 0 <= row < shape[0] or not 0 <= sample < shape[1]:
                raise ValueError("Reference answer outside processed measurement")
            if row in values:
                raise ValueError("Duplicate reference row: establish trustworthy provenance first")
            values[int(row)] = int(sample)
        result[order] = values
    return result


def score_fixed_cohort(
    paths,
    references,
    initial,
    authorized,
    measurement,
    pulse,
    dx_m,
    dt_ns,
    *,
    road_group="unspecified",
    revealed_answers=None,
):
    """Call the untouched timing/signed-lobe scorer; retain initial reviewed N.

    Revealed rows are excluded from automatic numerators, while their membership
    in the initial cohort and the separately identified analyst workload remain.
    """
    from .processed_dense_eval import evaluate_native_predictions

    output = {}
    for order, path in paths.items():
        points = references[order]
        cohort = {int(p.trace) for p in points if p.trace not in initial[order]}
        supplied = dict(initial[order])
        supplied.update(
            authorized[order]
            if revealed_answers is None
            else revealed_answers.get(order, {})
        )
        manual = cohort & (set(supplied) - set(initial[order]))
        metric = layer_metrics(path, points, supplied, measurement, pulse[order], dx_m)
        n = len(cohort)
        correct, wrong = metric["accepted_agree"], metric["accepted"] - metric["accepted_agree"]
        observed = metric["evaluation_observations"]
        wrong_rows = np.sort(
            np.array([p["row"] for p in observed if p["accepted"] and not p["accepted_correct"]])
        )
        groups = np.split(wrong_rows, np.flatnonzero(np.diff(wrong_rows) != 1) + 1)
        # Both established observed-bin footprint and exact endpoint distance.
        endpoint_span = max(
            ((int(g[-1]) - int(g[0])) * dx_m for g in groups if len(g)), default=0.0
        )
        longest = max((len(g) * dx_m for g in groups), default=0.0)
        for observation in observed:
            row = observation["row"]
            proposed = path.provisional_samples
            actual_sample = (proposed if proposed is not None else path.samples)[row]
            observation["proposal_native_sample"] = float(actual_sample)
            observation["distance_to_authorized_seed_m"] = (
                min(abs(row - s) for s in supplied) * dx_m
            )
            observation["distance_to_active_model_seed_m"] = (
                min(abs(row - s) for s in authorized[order]) * dx_m
            )
            observation["initial_region"] = (
                "bracketed" if min(initial[order]) <= row <= max(initial[order]) else "tail"
            )
            observation["proposal_error_ns"] = (
                abs(float(actual_sample) - observation["reference_sample"]) * dt_ns
                if actual_sample >= 0
                else None
            )
        metric.update(
            {
                "fixed_initial_observations": n,
                "correct_automatically_accepted": correct,
                "wrong_automatically_accepted": wrong,
                "correct_automatic_coverage": correct / n if n else None,
                "wrong_automatic_coverage": wrong / n if n else None,
                "legacy_conditional_correct_coverage": metric["correct_coverage"],
                "correct_coverage": correct / n if n else None,
                "coverage_denominator": "fixed_initial_observations",
                "automatic_unresolved": n - correct - wrong,
                "unresolved_after_analyst_answers": n - correct - wrong - len(manual),
                "analyst_supplied_in_initial_cohort": len(manual),
                "analyst_supplied_rows_no_automatic_credit": sorted(manual),
                "active_model_seed_rows": sorted(authorized[order]),
                "revealed_but_inactive_answer_rows": sorted(set(supplied) - set(authorized[order])),
                "correct_proposals_before_gating": metric["proposed_agree"],
                "correct_proposal_coverage_initial_pool": metric["proposed_agree"] / n
                if n
                else None,
                "initial_seed_count": len(initial[order]),
                "longest_contiguous_wrong_accepted_observed_span_m": longest,
                "longest_wrong_observed_endpoint_distance_m": endpoint_span,
                "wrong_span_contract": (
                    "Adjacent reviewed working rows; unknown/answer rows break spans; "
                    "observed-bin footprint, with endpoint distance separately reported"
                ),
            }
        )
        reference_array = np.full(len(measurement), np.nan)
        for point in points:
            reference_array[point.trace] = point.sample
        proposal = path.provisional_samples
        detailed = evaluate_native_predictions(
            measurement,
            reference_array,
            proposal if proposal is not None else path.samples,
            path.visible & (path.samples >= 0),
            accepted_samples=path.samples,
            initial_seeds=initial[order],
            additional_seed_rows=manual,
            native_dx_m=dx_m,
            dt_ns=dt_ns,
            tolerance_samples=max(2, pulse[order] / 4),
            confidence=path.confidence,
            road_group=road_group,
            layer=order,
            include_observations=False,
        )
        if (
            detailed["correct_accepted"],
            detailed["wrong_accepted"],
            detailed["fixed_initial_nonseed_N"],
        ) != (correct, wrong, n):
            raise ValueError("Fixed-cohort evaluation disagrees with the untouched replay scorer")
        metric["native_stratified_evaluation"] = detailed
        output[str(order)] = metric
    return output


def _transitions(previous, current):
    result = {}
    for order, metric in current.items():
        before = {r["row"]: r for r in previous[order]["evaluation_observations"]}
        after = {r["row"]: r for r in metric["evaluation_observations"]}
        shared = before.keys() & after.keys()

        def good(r):
            return r["accepted"] and r["accepted_correct"]

        def bad(r):
            return r["accepted"] and not r["accepted_correct"]

        result[order] = {
            "new_correct_automatic": sum(good(after[r]) and not good(before[r]) for r in shared),
            "lost_correct_automatic": sum(good(before[r]) and not good(after[r]) for r in shared),
            "new_wrong_automatic": sum(bad(after[r]) and not bad(before[r]) for r in shared),
            "repaired_wrong_automatic": sum(bad(before[r]) and good(after[r]) for r in shared),
            "newly_answered_rows_no_automatic_credit": sorted(before.keys() - after.keys()),
        }
    return result


def checkpoint_helpers():
    """Use the actual existing replay serializer without importing torch or a UI."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "replay_seeded_interaction.py"
    spec = importlib.util.spec_from_file_location("_processed_ml_checkpoint_helpers", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_frozen_model(
    predict,
    measurement,
    valid,
    initial_anchors,
    references,
    layers,
    *,
    dt_ns,
    dx_m,
    pulse_samples,
    provenance,
    output,
    config=None,
    resume=False,
    cancel=None,
    checkpoint_io=None,
):
    """Charge each request durably before fitting and checkpoint every prefix.

    ``predict`` accepts only ``{layer: {working_row: native_sample}}``. Its model
    is frozen. Hashes in ``provenance`` bind source/model/preprocessing and survive
    save/reopen; callers validate those files before constructing the predictor.
    """
    config = config or ReplayConfig()
    config.validate()
    predictor = getattr(predict, "__self__", None)
    history_api = predictor is not None and all(
        callable(getattr(predictor, name, None))
        for name in ("export_replay_state", "restore_replay_state", "commit_replay_scope")
    )
    if getattr(getattr(predictor, "config", None), "correction_aware", False) and not history_api:
        raise ValueError("Correction-aware replay requires persisted, scope-preserving history")
    if provenance.get("coordinate_mode") != "processed":
        raise ValueError("Processed-trained replay refuses incompatible/raw coordinates")
    for name in ("input_sha256", "model_sha256", "preprocessing_sha256", "source_sha256"):
        if not isinstance(provenance.get(name), str) or len(provenance[name]) != 64:
            raise ValueError(f"Missing replay fingerprint: {name}")
    if measurement.ndim != 2 or valid.shape != measurement.shape or dt_ns <= 0 or dx_m <= 0:
        raise ValueError("Invalid processed grid")
    if not initial_anchors or set(initial_anchors) != set(references):
        raise ValueError("Every enabled layer requires explicit initial seeds and references")
    initial = {int(o): dict(a) for o, a in initial_anchors.items()}
    for seeds in initial.values():
        if not seeds:
            raise ValueError("Every enabled layer requires initial observations")
        for row, sample in seeds.items():
            if (
                int(row) != row
                or not np.isfinite(sample)
                or not 0 <= row < len(measurement)
                or not 0 <= sample <= measurement.shape[1] - 1
            ):
                raise ValueError("Initial seed requires exact processed trace/sample")
    answers = exact_answers(references, measurement.shape)
    output = Path(output)
    io = checkpoint_io or checkpoint_helpers()
    checkpoint = io.checkpoint_path(output)
    if (output.exists() or checkpoint.exists()) and not resume:
        raise ValueError("Preserve existing replay output; use a new output or explicit resume")
    reference_digest = hashlib.sha256(json.dumps(answers, sort_keys=True).encode()).hexdigest()
    contract = json.loads(
        json.dumps(
            {
                "schema": "processed-ml-replay-v2",
                "request_persistence": "charged exact answers persisted before retracking",
                "provenance": provenance,
                "config": asdict(config),
                "initial_anchors": initial,
                "reference_coordinate_sha256": reference_digest,
                "grid": {"shape": list(measurement.shape), "dt_ns": dt_ns, "dx_m": dx_m},
                "pulse_samples": pulse_samples,
            }
        )
    )
    if resume:
        state = io.load_checkpoint(checkpoint, contract)
        if state["log"].get("completed"):
            io.atomic_write(output, json.dumps(state["log"], indent=2, allow_nan=False).encode())
            return state["log"]
    else:
        state = {
            "paths": None,
            "anchors": {o: dict(a) for o, a in initial.items()},
            "visited": set(),
            "corrections": {},
            "revealed_answers": {o: {} for o in initial},
            "pending_action": None,
            "log": {
                "schema": "processed-ml-interaction-v2",
                "contract": contract,
                "claim": (
                    "Grouped development analyst-reference agreement; experimental frozen weights"
                ),
                "denominator": (
                    "Fixed initial reviewed nonseed cohort; "
                    "revealed answers never receive automatic credit"
                ),
                "query_selection_reads_references": False,
                "road_length_m": (len(measurement) - 1) * dx_m,
                "initial_seed_count": sum(map(len, initial.values())),
                "actions": [],
                "steps": [],
                "reported_budgets": list(config.budgets),
            },
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    log = state["log"]
    if history_api:
        if resume:
            if "predictor_state" not in state:
                raise ValueError("Saved replay is missing predictor history")
            predictor.restore_replay_state(state["predictor_state"])
        else:
            state["predictor_state"] = predictor.export_replay_state()

    def fit(active):
        try:
            paths = predict(active)
            if set(paths) != set(initial):
                raise ValueError("Predictor must return exactly the enabled layers")
            for order, path in paths.items():
                for name in ("samples", "visible", "confidence", "provisional_samples"):
                    if np.asarray(getattr(path, name)).shape != (len(measurement),):
                        raise ValueError(
                            "Predicted path must retain the exact processed trace grid"
                        )
                for row, sample in active[order].items():
                    if path.samples[row] != sample or path.provisional_samples[row] != sample:
                        raise ValueError("Predictor moved an exact authorized sample")
        except BaseException:
            if history_api:
                predictor.restore_replay_state(state["predictor_state"])
            raise
        return paths

    def persist():
        if history_api:
            state["predictor_state"] = predictor.export_replay_state()
        io.save_checkpoint(checkpoint, contract, state)
        io.atomic_write(output, json.dumps(log, indent=2, allow_nan=False).encode())

    def append_score(runtime):
        metrics = score_fixed_cohort(
            state["paths"],
            references,
            initial,
            state["anchors"],
            measurement,
            pulse_samples,
            dx_m,
            dt_ns,
            road_group=provenance.get("physical_road_group", "unspecified"),
            revealed_answers=state["revealed_answers"],
        )
        count = len(log["actions"])
        length_km = log["road_length_m"] / 1000
        step = {
            "additional_requests": count,
            "reported_budget": count in config.budgets,
            "layers": metrics,
            "runtime_s": runtime,
            "actions_per_km": count / length_km if length_km > 0 else None,
            "total_observations_and_requests_per_km": (log["initial_seed_count"] + count)
            / length_km
            if length_km > 0
            else None,
        }
        if log["steps"]:
            step["changes_after_action"] = _transitions(log["steps"][-1]["layers"], metrics)
        log["steps"].append(step)
        np.savez_compressed(
            output.with_name(f"{output.stem}-step{count}.npz"),
            **{
                f"layer{o}_{name}": getattr(path, name)
                for o, path in state["paths"].items()
                for name in (
                    "samples",
                    "visible",
                    "provisional_samples",
                    "confidence",
                    "alternate_samples",
                )
                if getattr(path, name) is not None
            },
        )
        persist()

    if state["paths"] is None:
        start = time.perf_counter()
        state["paths"] = fit({o: dict(a) for o, a in initial.items()})
        append_score(time.perf_counter() - start)
    while state["pending_action"] is not None or len(log["actions"]) < max(config.budgets):
        if cancel and cancel():
            raise InterruptedError("Research replay cancelled")
        if state["pending_action"] is None:
            request = request_model_observation(
                state["paths"],
                measurement,
                valid,
                state["anchors"],
                dx_m,
                visited=state["visited"],
                policy=config.policy,
                fixed_slots=config.fixed_spacing_slots,
            )
            if request is None:
                log["stop_reason"] = "No eligible unvisited request for this policy"
                break
            order, row = request["layer_order"], request["row"]
            action = {
                **request,
                "operation": config.operation,
                "chainage_m": row * dx_m,
                "native_trace": row * int(provenance.get("trace_stride", 1)),
                "completion_status": "pending",
                "retrack_attempts": [],
                "runtime_s": 0.0,
            }
            # Reveal only this exact requested answer and persist its charge before fit.
            # A retry reads the saved action, never the policy or answer service again.
            answer = answers[order].get(row)
            if answer is None:
                action["answer_status"] = "unavailable_reviewed_answer"
            else:
                action.update(answer_status="revealed_exact", answer_sample=answer)
                state["revealed_answers"][order][row] = answer
            state["visited"].add((order, row))
            state["pending_action"] = len(log["actions"])
            log["actions"].append(action)
            persist()

        action = log["actions"][state["pending_action"]]
        order, row = action["layer_order"], action["row"]
        start = time.perf_counter()
        if action["answer_status"] == "revealed_exact":
            answer = action["answer_sample"]
            authorized = {o: dict(a) for o, a in state["anchors"].items()}
            authorized[order][row] = answer
            corrections = {o: dict(a) for o, a in state["corrections"].items()}
            corrections.setdefault(order, {})[row] = answer
            if config.operation == "local_correction":
                lo = max(0, int(np.ceil(row - 25.0 / dx_m)))
                hi = min(len(measurement) - 1, int(np.floor(row + 25.0 / dx_m)))
                active = {o: dict(a) for o, a in initial.items()}
                for o, values in corrections.items():
                    active[o].update({r: s for r, s in values.items() if lo <= r <= hi})
            else:
                lo, hi = 0, len(measurement) - 1
                active = authorized
            action.update(
                affected_rows=[lo, hi],
                affected_layer_orders=[order],
                model_weights_fitted=False,
                local_radius_m=25.0 if config.operation == "local_correction" else None,
            )
            previous_paths = state["paths"]
            previous_anchors = state["anchors"]
            previous_corrections = state["corrections"]
            try:
                regenerated = fit(active)
                if cancel and cancel():
                    raise InterruptedError("Research replay cancelled before correction commit")
                merged = merge_local_paths(
                    previous_paths, regenerated, orders={order}, start_row=lo, stop_row=hi
                )
                try:
                    guard_local_order(merged, previous_paths, layers, active, {order}, lo, hi)
                except ValueError as exc:
                    action.update(
                        retrack_status="neighbor_observation_conflict", failure=str(exc)
                    )
                    if history_api:
                        predictor.restore_replay_state(state["predictor_state"])
                else:
                    if history_api:
                        predictor.commit_replay_scope(state["predictor_state"], order, lo, hi)
                    state["paths"] = merged
                    state["anchors"], state["corrections"] = authorized, corrections
                    action["retrack_status"] = "applied"
            except BaseException as exc:
                state["paths"] = previous_paths
                state["anchors"] = previous_anchors
                state["corrections"] = previous_corrections
                if history_api:
                    predictor.restore_replay_state(state["predictor_state"])
                elapsed = time.perf_counter() - start
                action["retrack_status"] = "pending_retry"
                action["runtime_s"] += elapsed
                action["retrack_attempts"].append(
                    {"status": "interrupted", "failure": str(exc), "runtime_s": elapsed}
                )
                persist()
                raise
            action["retrack_attempts"].append(
                {"status": action["retrack_status"], "runtime_s": time.perf_counter() - start}
            )
        action["runtime_s"] += time.perf_counter() - start
        action["completion_status"] = "completed"
        state["pending_action"] = None
        try:
            append_score(action["runtime_s"])
        except BaseException:
            # Checkpoint is authoritative across score/NPZ/checkpoint/JSON failures.
            # It may contain either the pre-fit pending state or the completed step.
            state = io.load_checkpoint(checkpoint, contract)
            log = state["log"]
            if history_api:
                predictor.restore_replay_state(state["predictor_state"])
            raise
    log["completed"] = True
    log.setdefault("stop_reason", "Requested action budget completed")
    persist()
    return log
