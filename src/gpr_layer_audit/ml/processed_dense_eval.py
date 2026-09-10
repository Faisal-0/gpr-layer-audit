"""Native processed-label evaluation around the unchanged frozen lobe scorer.

This module adds fixed-cohort accounting and stratified summaries. It does not
alter the timing/signed-lobe definition in ``conventional.py``. Development-road
reference agreement is not physical thickness accuracy or untouched-test proof.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gpr_layer_audit import conventional

FROZEN_SCORER_LF_SHA256 = "fc5b2093d7a1efe8abd549e00ea52b290040237ea519225ab20b8fde8f2a307b"


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def scorer_provenance():
    path = Path(inspect.getfile(conventional))
    content = path.read_bytes()
    normalized = hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()
    if normalized != FROZEN_SCORER_LF_SHA256:
        raise ValueError(
            "Frozen conventional scorer changed; separately version and rescore controls"
        )
    return {
        "source": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "normalized_lf_sha256": normalized,
        "agreement_function": "conventional._same_lobe",
        "contract": "abs(native error)<=frozen tolerance AND same signed lobe; no snapping",
    }


def validate_evidence_provenance(
    metadata,
    *,
    source_sha256,
    seed_sha256,
    probabilities_sha256,
    native_trace_indices,
    native_sample_indices,
    preprocessing_version,
    layer,
    road_group,
):
    """Reject same-shape evidence from another road, seed episode, or native grid."""
    expected = {
        "schema": "processed-native-probability-evidence-v1",
        "source_sha256": source_sha256,
        "seed_sha256": seed_sha256,
        "probabilities_sha256": probabilities_sha256,
        "native_trace_indices_sha256": canonical_sha256(
            np.asarray(native_trace_indices, dtype=np.int64).tolist()
        ),
        "native_sample_indices_sha256": canonical_sha256(
            np.asarray(native_sample_indices, dtype=np.int64).tolist()
        ),
        "preprocessing_version": preprocessing_version,
        "layer": int(layer),
        "held_out_group": road_group,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Dense evidence provenance mismatch: {key}")
    for key in ("model_sha256", "weights_sha256"):
        value = metadata.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"Dense evidence requires a model fingerprint: {key}")
    if not metadata.get("training_groups") or road_group in metadata["training_groups"]:
        raise ValueError("Evidence provenance does not establish an out-of-training-road model")
    return expected


def frozen_initial_tolerance(measurement, sample_valid, initial_seeds, dt_ns):
    """Use the established seed-only pulse rule once, before any corrections."""
    from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse

    pulse = resolve_pulse(measurement, sample_valid, initial_seeds, {}, dt_ns, ConventionalConfig())
    return max(2.0, pulse.lobe_samples / 4), pulse.metadata()


def _longest_span(observations, *, trace_step, dx_m):
    wrong = sorted(r["native_trace"] for r in observations if r["wrong_accepted"])
    spans, start, previous = [], None, None
    for trace in wrong:
        if previous is None or trace != previous + trace_step:
            if start is not None:
                spans.append((previous - start + trace_step) * dx_m)
            start = trace
        previous = trace
    if start is not None:
        spans.append((previous - start + trace_step) * dx_m)
    return max(spans, default=0.0), len(spans)


def summarize_observations(observations, *, native_dx_m, dt_ns, trace_step=1):
    n = len(observations)
    accepted = [r for r in observations if r["accepted"]]
    correct = sum(r["accepted_correct"] for r in observations)
    wrong = len(accepted) - correct
    proposed = sum(r["proposed_correct"] for r in observations)
    accepted_errors = [r["absolute_error_samples"] for r in accepted]
    proposal_errors = [
        r["proposal_absolute_error_samples"]
        for r in observations
        if r["proposal_absolute_error_samples"] is not None and not r["manual_answer"]
    ]
    longest, runs = _longest_span(observations, trace_step=trace_step, dx_m=native_dx_m)
    result = {
        "fixed_initial_nonseed_N": n,
        "correct_accepted": correct,
        "wrong_accepted": wrong,
        "accepted": len(accepted),
        "unresolved": n - len(accepted),
        "manual_answers_in_initial_cohort": sum(r["manual_answer"] for r in observations),
        "correct_proposals_before_gating": proposed,
        "accepted_agreement": correct / len(accepted) if accepted else None,
        "correct_coverage": correct / n if n else None,
        "wrong_coverage": wrong / n if n else None,
        "unresolved_coverage": (n - len(accepted)) / n if n else None,
        "correct_proposal_coverage": proposed / n if n else None,
        "wrong_signed_lobe_observations": sum(not r["same_signed_lobe"] for r in accepted),
        "final_gate_correct_proposal_losses": sum(
            r["proposed_correct"] and not r["accepted_correct"] for r in observations
        ),
        "wrong_accepted_observed_runs": runs,
        "longest_contiguous_wrong_accepted_observed_span_m": longest,
        "scored_observation_footprint_m": n * trace_step * native_dx_m,
    }
    for name, errors in (("accepted", accepted_errors), ("proposal", proposal_errors)):
        for statistic, reducer in (
            ("median", np.median),
            ("mean", np.mean),
            ("p95", lambda v: np.percentile(v, 95)),
        ):
            value = float(reducer(errors)) if errors else None
            result[f"{statistic}_{name}_error_samples"] = value
            result[f"{statistic}_{name}_error_ns"] = None if value is None else value * dt_ns
    return result


def evaluate_native_predictions(
    measurement,
    references,
    proposed_samples,
    accepted_mask,
    *,
    initial_seeds,
    native_dx_m,
    dt_ns,
    tolerance_samples,
    road_group,
    layer,
    additional_seed_rows=(),
    native_trace_indices=None,
    accepted_samples=None,
    confidence=None,
    label_valid=None,
    trace_step=None,
    include_observations=True,
):
    """Score one road/layer while preserving the initial reviewed nonseed cohort.

    All arrays use retained rows and full native sample coordinates. ``initial_seeds``
    is a row->sample mapping (or row iterable). Later supplied answers stay in N but
    cannot earn automatic/proposal credit. Thus unresolved means *not automatically
    accepted*, including manually answered rows, which are also reported separately.
    Unknown labels never become absence labels and always break observed wrong runs.
    """
    provenance = scorer_provenance()
    radar = np.asarray(measurement)
    refs = np.asarray(references, dtype=float)
    proposed = np.asarray(proposed_samples, dtype=float)
    accepted = np.asarray(accepted_mask, dtype=bool)
    selected = proposed if accepted_samples is None else np.asarray(accepted_samples, dtype=float)
    if radar.ndim != 2:
        raise ValueError("Measurement must be [retained trace, native sample]")
    n, depth = radar.shape
    if any(v.shape != (n,) for v in (refs, proposed, accepted, selected)):
        raise ValueError("Reference and prediction arrays must match retained rows")
    if not np.isfinite(native_dx_m) or native_dx_m <= 0 or not np.isfinite(dt_ns) or dt_ns <= 0:
        raise ValueError("Positive native coordinate intervals are required")
    if not np.isfinite(tolerance_samples) or tolerance_samples < 2:
        raise ValueError("A frozen native tolerance of at least two samples is required")
    trace = (
        np.arange(n, dtype=np.int64)
        if native_trace_indices is None
        else np.asarray(native_trace_indices)
    )
    if (
        trace.shape != (n,)
        or np.any(trace != trace.astype(np.int64))
        or np.any(np.diff(trace) <= 0)
    ):
        raise ValueError("Native trace indices must be exact, strictly increasing, unique")
    trace = trace.astype(np.int64)
    if trace_step is None:
        # Nonuniform input must declare its nominal observed grid; never infer a
        # large gap as a densely observed road span.
        if n > 2 and len(np.unique(np.diff(trace))) != 1:
            raise ValueError("Nonuniform native trace grids require explicit trace_step")
        trace_step = int(trace[1] - trace[0]) if n > 1 else 1
    if int(trace_step) != trace_step or trace_step < 1:
        raise ValueError("Trace step must be a positive integer native trace stride")
    if n > 1 and np.any(np.diff(trace) % trace_step):
        raise ValueError("Retained rows do not lie on the declared native trace grid")
    known = np.isfinite(refs) & (refs >= 0) & (refs <= depth - 1)
    if label_valid is not None:
        validity = np.asarray(label_valid, dtype=bool)
        if validity.shape != (n,):
            raise ValueError("Label validity must match retained rows")
        if np.any(validity & ~known):
            raise ValueError("Valid target labels cannot have invalid native coordinates")
        known &= validity
    initial = set(initial_seeds)
    manual = set(additional_seed_rows)
    if any(int(r) != r or not 0 <= r < n for r in initial | manual):
        raise ValueError("Seed exclusions must use exact retained native rows")
    if hasattr(initial_seeds, "items"):
        for sample in initial_seeds.values():
            if not np.isfinite(sample) or not 0 <= sample <= depth - 1:
                raise ValueError("Initial seed has invalid native sample coordinate")
    scores = np.ones(n) if confidence is None else np.asarray(confidence, dtype=float)
    if scores.shape != (n,) or np.any(~np.isfinite(scores)):
        raise ValueError("Confidence must be finite and match retained rows")
    initial_native = np.array(sorted(trace[list(initial)]), dtype=int)
    all_native = np.array(sorted(trace[list(initial | manual)]), dtype=int)
    observations = []
    for row in np.flatnonzero(known):
        if row in initial:
            continue
        reference = refs[row]
        sample = selected[row]
        proposal = proposed[row]
        answer = row in manual
        has_proposal = np.isfinite(proposal) and 0 <= proposal <= depth - 1
        has_accepted = bool(
            accepted[row] and not answer and np.isfinite(sample) and 0 <= sample <= depth - 1
        )
        proposal_error = abs(float(proposal) - reference) if has_proposal else None
        error = abs(float(sample) - reference) if has_accepted else None
        same = bool(has_accepted and conventional._same_lobe(radar[row], sample, reference))
        proposal_correct = bool(
            has_proposal
            and not answer
            and proposal_error <= tolerance_samples
            and conventional._same_lobe(radar[row], proposal, reference)
        )
        correct = bool(has_accepted and same and error <= tolerance_samples)
        native = int(trace[row])
        segment = "unseeded"
        if len(initial_native):
            segment = (
                "left_tail"
                if native < initial_native[0]
                else "right_tail"
                if native > initial_native[-1]
                else "bracketed"
            )
        observations.append(
            {
                "row": int(row),
                "native_trace": native,
                "chainage_m": native * native_dx_m,
                "reference_sample": float(reference),
                "proposed_sample": float(proposal) if has_proposal else None,
                "accepted_sample": float(sample) if has_accepted else None,
                "proposed_correct": proposal_correct,
                "accepted": has_accepted,
                "accepted_correct": correct,
                "wrong_accepted": has_accepted and not correct,
                "same_signed_lobe": same,
                "manual_answer": answer,
                "absolute_error_samples": error,
                "proposal_absolute_error_samples": proposal_error,
                "confidence": float(scores[row]),
                "segment": segment,
                "distance_initial_seed_m": (
                    float(np.min(abs(initial_native - native)) * native_dx_m)
                    if len(initial_native)
                    else None
                ),
                "distance_authorized_seed_m": (
                    float(np.min(abs(all_native - native)) * native_dx_m)
                    if len(all_native)
                    else None
                ),
            }
        )
    kwargs = dict(native_dx_m=native_dx_m, dt_ns=dt_ns, trace_step=trace_step)
    segments = {
        name: summarize_observations([r for r in observations if r["segment"] == name], **kwargs)
        for name in ("bracketed", "left_tail", "right_tail", "unseeded")
    }
    bins = (0, 5, 25, 100, 250, float("inf"))
    distance_reports = []
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        rows = [
            r
            for r in observations
            if r["distance_authorized_seed_m"] is not None
            and low <= r["distance_authorized_seed_m"] < high
        ]
        distance_reports.append(
            {
                "lower_m": low,
                "upper_m": high if np.isfinite(high) else None,
                **summarize_observations(rows, **kwargs),
            }
        )
    return {
        "schema": "processed-dense-native-evaluation-v1",
        "claim": "grouped development interpretation agreement; not physical accuracy",
        "road_group": road_group,
        "layer": int(layer),
        "scorer": provenance,
        "tolerance_samples": float(tolerance_samples),
        "native_dx_m": native_dx_m,
        "dt_ns": dt_ns,
        "trace_step": int(trace_step),
        "cohort_sha256": canonical_sha256(
            [[r["native_trace"], r["reference_sample"]] for r in observations]
        ),
        "initial_seed_rows": sorted(int(r) for r in initial),
        "additional_seed_rows": sorted(int(r) for r in manual),
        "initial_seed_count": len(initial),
        "additional_answer_count": len(manual - initial),
        "unresolved_definition": (
            "N minus automatic accepts; supplied answers receive no automatic credit"
        ),
        "span_definition": (
            "consecutive observed retained-grid rows; unknowns break runs; row footprint"
        ),
        **summarize_observations(observations, **kwargs),
        "segments": segments,
        "distance_to_authorized_seed": distance_reports,
        **({"evaluation_observations": observations} if include_observations else {}),
    }


def precision_coverage_curve(report, thresholds=None):
    """Diagnostic sweep; never chooses a threshold using evaluation-road labels."""
    rows = report["evaluation_observations"]
    thresholds = np.linspace(0, 1, 51) if thresholds is None else thresholds
    curve = []
    for threshold in thresholds:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative")
        candidates = [
            r
            for r in rows
            if not r["manual_answer"]
            and r["proposed_sample"] is not None
            and r["confidence"] >= threshold
        ]
        correct = sum(r["proposed_correct"] for r in candidates)
        accepted = len(candidates)
        curve.append(
            {
                "threshold": float(threshold),
                "N": len(rows),
                "accepted": accepted,
                "correct_accepted": correct,
                "wrong_accepted": accepted - correct,
                "accepted_agreement": correct / accepted if accepted else None,
                "correct_coverage": correct / len(rows) if rows else None,
                "wrong_coverage": (accepted - correct) / len(rows) if rows else None,
            }
        )
    return curve


def select_training_threshold(
    reports,
    *,
    training_groups,
    excluded_groups,
    thresholds=None,
    target_agreement=0.95,
    min_accepted=20,
):
    """Select the maximal-coverage operating point from TRAIN-only inner validation.

    Each report must explicitly carry ``split_role='inner_validation'`` and
    ``spatial_buffer_verified=True``. Outer groups are prohibited. Threshold
    selection itself cannot establish that the caller's split provenance is true;
    the manifest audit remains necessary and these assertions are saved.
    """
    training, excluded = set(training_groups), set(excluded_groups)
    if training & excluded or not training or not excluded:
        raise ValueError("Training and outer evaluation road groups must be explicit and disjoint")
    if not 0 < target_agreement <= 1 or min_accepted < 1:
        raise ValueError("Invalid acceptance objective")
    if not reports:
        raise ValueError("No training-road inner-validation predictions supplied")
    for report in reports:
        if report["road_group"] not in training or report["road_group"] in excluded:
            raise ValueError("Threshold selection attempted to read an outer evaluation group")
        if report.get("split_role") != "inner_validation" or not report.get(
            "spatial_buffer_verified"
        ):
            raise ValueError("Threshold selection requires buffered TRAIN-road inner validation")
        if not report["evaluation_observations"]:
            raise ValueError("Each calibration record needs reviewed nonseed observations")
    if {report["road_group"] for report in reports} != training:
        raise ValueError("Calibration reports must cover every declared layer-eligible TRAIN group")
    thresholds = np.linspace(0, 1, 101) if thresholds is None else thresholds
    curves = [precision_coverage_curve(r, thresholds) for r in reports]
    points = []
    for index in range(len(thresholds)):
        by_road = defaultdict(lambda: dict(N=0, accepted=0, correct_accepted=0))
        for report, curve in zip(reports, curves, strict=True):
            for key in ("N", "accepted", "correct_accepted"):
                by_road[report["road_group"]][key] += curve[index][key]
        n = sum(r["N"] for r in by_road.values())
        accepted = sum(r["accepted"] for r in by_road.values())
        correct = sum(r["correct_accepted"] for r in by_road.values())
        macro_coverage = float(
            np.mean([r["correct_accepted"] / r["N"] for r in by_road.values() if r["N"]])
        )
        # Enforce the research precision objective on every contributing road;
        # no long acquisition may conceal a failing training road.
        meets = all(
            r["accepted"] >= min_accepted
            and r["correct_accepted"] / r["accepted"] >= target_agreement
            for r in by_road.values()
        )
        points.append(
            {
                "threshold": float(thresholds[index]),
                "N": n,
                "accepted": accepted,
                "correct_accepted": correct,
                "wrong_accepted": accepted - correct,
                "unresolved": n - accepted,
                "accepted_agreement": correct / accepted if accepted else None,
                "correct_coverage": correct / n if n else None,
                "wrong_coverage": (accepted - correct) / n if n else None,
                "macro_road_correct_coverage": macro_coverage,
                "meets_objective": meets,
                "per_road": {
                    road: {
                        **counts,
                        "wrong_accepted": counts["accepted"] - counts["correct_accepted"],
                        "unresolved": counts["N"] - counts["accepted"],
                        "accepted_agreement": (
                            counts["correct_accepted"] / counts["accepted"]
                            if counts["accepted"]
                            else None
                        ),
                        "correct_coverage": counts["correct_accepted"] / counts["N"],
                        "wrong_coverage": (counts["accepted"] - counts["correct_accepted"])
                        / counts["N"],
                    }
                    for road, counts in sorted(by_road.items())
                },
            }
        )
    eligible = [p for p in points if p["meets_objective"]]
    chosen = (
        max(eligible, key=lambda p: (p["macro_road_correct_coverage"], -p["threshold"]))
        if eligible
        else None
    )
    return {
        "schema": "processed-dense-train-only-threshold-v1",
        "selected_threshold": chosen["threshold"] if chosen else None,
        "selected_operating_point": chosen,
        "objective_met": chosen is not None,
        "target_agreement_each_training_road": target_agreement,
        "minimum_accepts_each_training_road": min_accepted,
        "training_groups": sorted(training),
        "excluded_outer_groups": sorted(excluded),
        "selection": "maximize macro road correct coverage subject to each-road precision",
        "input_cohort_hashes": [r["cohort_sha256"] for r in reports],
        "curve": points,
    }


def aggregate_reports(reports):
    """Pool counts, then give each physical road equal weight in macro rates."""
    by_layer = defaultdict(list)
    for report in reports:
        by_layer[int(report["layer"])].append(report)
    outputs = {}
    keys = (
        "fixed_initial_nonseed_N",
        "correct_accepted",
        "wrong_accepted",
        "accepted",
        "unresolved",
        "correct_proposals_before_gating",
    )
    for layer, values in by_layer.items():
        grouped = defaultdict(list)
        for value in values:
            grouped[value["road_group"]].append(value)
        roads = []
        for road, members in sorted(grouped.items()):
            counts = {key: sum(m[key] for m in members) for key in keys}
            n, accepted = counts["fixed_initial_nonseed_N"], counts["accepted"]
            roads.append(
                {
                    "road_group": road,
                    "records": len(members),
                    **counts,
                    "accepted_agreement": counts["correct_accepted"] / accepted
                    if accepted
                    else None,
                    "correct_coverage": counts["correct_accepted"] / n if n else None,
                    "wrong_coverage": counts["wrong_accepted"] / n if n else None,
                }
            )
        pooled = {key: sum(r[key] for r in roads) for key in keys}
        n, accepted = pooled["fixed_initial_nonseed_N"], pooled["accepted"]
        outputs[str(layer)] = {
            "roads": roads,
            "pooled": {
                **pooled,
                "accepted_agreement": pooled["correct_accepted"] / accepted if accepted else None,
                "correct_coverage": pooled["correct_accepted"] / n if n else None,
                "wrong_coverage": pooled["wrong_accepted"] / n if n else None,
            },
            "macro_road": {
                key: float(np.mean([r[key] for r in roads if r[key] is not None]))
                if any(r[key] is not None for r in roads)
                else None
                for key in ("accepted_agreement", "correct_coverage", "wrong_coverage")
            },
            "roads_with_defined_agreement": sum(r["accepted_agreement"] is not None for r in roads),
        }
    return {
        "schema": "processed-dense-grouped-summary-v1",
        "layers": outputs,
        "independent_unit": "physical road, not adjacent observations or processing variants",
    }


def freeze_failure_atlas(path, entries, *, primary_artifact_hashes):
    """Freeze explicitly selected windows before any challenger is judged."""
    target = Path(path)
    if target.exists():
        raise FileExistsError("Failure atlas already frozen; preserve the initial selection")
    required = {"road_group", "layer", "start_native_trace", "stop_native_trace", "reason"}
    for entry in entries:
        if not required <= entry.keys() or entry["start_native_trace"] > entry["stop_native_trace"]:
            raise ValueError(
                "Atlas windows need physical provenance, exact bounds, and selection reason"
            )
    document = {
        "schema": "processed-dense-fixed-failure-atlas-v1",
        "entries": entries,
        "primary_artifact_hashes": primary_artifact_hashes,
        "selected_before_challenger_judgment": True,
        "semantic_failure_types_require_visual_review": True,
    }
    document["sha256"] = canonical_sha256(document)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    return document


def evaluate_evidence_stages(
    measurement,
    references,
    probabilities,
    *,
    initial_seeds,
    sample_valid,
    native_dx_m,
    dt_ns,
    road_group,
    layer,
    native_trace_indices=None,
    tolerance_samples=None,
    decoder_config=None,
    operating_threshold=None,
    include_observations=True,
):
    """Evaluate direct evidence, dense path, and radar-free seed interpolation.

    Ungated stages deliberately expose every proposal for diagnostics. A separate
    operating gate is emitted only when the caller supplies a frozen TRAIN-derived
    threshold; absence of an operating threshold never implies production approval.
    """
    from dataclasses import replace

    from .processed_dense_decoder import (
        DenseDecoderConfig,
        decode_dense_path,
        direct_predictions,
        seed_interpolation,
    )

    if tolerance_samples is None:
        tolerance_samples, pulse = frozen_initial_tolerance(
            measurement, sample_valid, initial_seeds, dt_ns
        )
    else:
        pulse = {"source": "supplied original frozen comparator tolerance"}
    native_traces = (
        np.arange(len(measurement))
        if native_trace_indices is None
        else np.asarray(native_trace_indices)
    )
    working_dx = (
        float(native_traces[1] - native_traces[0]) * native_dx_m
        if len(native_traces) > 1
        else native_dx_m
    )
    config = replace(decoder_config or DenseDecoderConfig(), acceptance_threshold=0.0)
    direct = direct_predictions(probabilities, sample_valid=sample_valid)
    decoded = decode_dense_path(
        probabilities,
        initial_seeds,
        sample_valid=sample_valid,
        dx_m=working_dx,
        dt_ns=dt_ns,
        config=config,
    )
    kwargs = dict(
        initial_seeds=initial_seeds,
        native_dx_m=native_dx_m,
        dt_ns=dt_ns,
        tolerance_samples=tolerance_samples,
        road_group=road_group,
        layer=layer,
        native_trace_indices=native_traces,
        include_observations=True,
    )
    stages = {}
    for name, result in (("direct_evidence", direct), ("dense_decoder", decoded)):
        report = evaluate_native_predictions(
            measurement,
            references,
            result.proposed_samples,
            result.accepted,
            confidence=result.confidence,
            accepted_samples=result.samples,
            **kwargs,
        )
        report["decoder_diagnostics"] = result.diagnostics
        report["precision_coverage_curve"] = precision_coverage_curve(report)
        if not include_observations:
            report.pop("evaluation_observations")
        stages[name] = report
        threshold = (
            operating_threshold.get(name)
            if isinstance(operating_threshold, dict)
            else operating_threshold
        )
        if threshold is not None:
            gated = evaluate_native_predictions(
                measurement,
                references,
                result.proposed_samples,
                result.accepted & (result.confidence >= threshold),
                confidence=result.confidence,
                accepted_samples=result.samples,
                **kwargs,
            )
            gated["operating_threshold"] = threshold
            if not include_observations:
                gated.pop("evaluation_observations")
            stages[name + "_gated"] = gated
    interpolation = seed_interpolation(len(measurement), initial_seeds)
    stages["seed_interpolation_no_radar"] = evaluate_native_predictions(
        measurement,
        references,
        interpolation,
        interpolation >= 0,
        **kwargs,
    )
    if not include_observations:
        stages["seed_interpolation_no_radar"].pop("evaluation_observations")
    return {
        "schema": "processed-dense-stage-comparison-v1",
        "stages": stages,
        "initial_scoring_pulse": pulse,
        "tolerance_samples": tolerance_samples,
        "production_promotion": False,
        "ungated_stages_are_proposal_diagnostics": True,
    }
