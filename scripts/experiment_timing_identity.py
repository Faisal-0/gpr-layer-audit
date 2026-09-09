"""Post-inference timing/identity error and exact route-objective diagnosis.

No candidate, edge, local score, threshold or production path is changed. The
labelled counterfactual constrains only the inspected row, after recorded inference.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "exports/seeded-tracker/edge-gate-loss/source-e4b9c239079145ea/src"
DEFAULT_INPUT = ROOT / "exports/seeded-tracker/edge-gate-loss/mandiali-short"
DEFAULT_OUTPUT = ROOT / "exports/seeded-tracker/timing-identity"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False, default=scalar), encoding="utf-8"
    )


def lobe_timing(trace, sample):
    from scipy.signal import find_peaks

    from gpr_layer_audit.conventional import _same_lobe

    start, stop = sample, sample
    while start > 0 and _same_lobe(trace, start - 1, sample):
        start -= 1
    while stop + 1 < len(trace) and _same_lobe(trace, stop + 1, sample):
        stop += 1
    amplitude_peak = start + int(np.argmax(abs(trace[start : stop + 1])))
    peaks = find_peaks(abs(trace))[0]
    peaks = peaks[(peaks >= start) & (peaks <= stop)]
    nearest = int(peaks[np.argmin(abs(peaks - sample))]) if len(peaks) else amplitude_peak
    basins = [amplitude_peak]
    if len(peaks):
        if sample <= peaks[0]:
            basins = [int(peaks[0])]
        elif sample >= peaks[-1]:
            basins = [int(peaks[-1])]
        else:
            right = int(np.searchsorted(peaks, sample, side="left"))
            left_peak, right_peak = int(peaks[right - 1]), int(peaks[right])
            between = abs(trace[left_peak : right_peak + 1])
            valley = np.flatnonzero(between == between.min()) + left_peak
            basins = (
                [left_peak]
                if sample < valley[0]
                else [right_peak]
                if sample > valley[-1]
                else [left_peak, right_peak]
            )
    return {
        "lobe_extent_samples": [start, stop],
        "amplitude_peak_sample": amplitude_peak,
        "same_signed_lobe_extrema_samples": peaks.tolist(),
        "nearest_extremum_sample": nearest,
        "offset_from_nearest_extremum_samples": sample - nearest,
        "offset_from_strongest_extremum_samples": sample - amplitude_peak,
        "basin_extremum_samples": basins,
    }


def route_through(index, local):
    prefix, suffix = [], []
    current = index
    while current is not None:
        prefix.append(current)
        current = local["forward_parent"][current]
    current = local["backward_parent"][index]
    while current is not None:
        suffix.append(current)
        current = local["backward_parent"][current]
    return [*reversed(prefix), *suffix]


def route_terms(sequence, local, table, dense_support):
    step, config = local["step"], local["config"]
    nodes = [node for index in sequence for node in local["by_index"][index].nodes]
    rr, cc = np.asarray(nodes, dtype=int).T
    samples = table.samples[rr, cc]
    maps = table.component_maps
    scores = local["scores"][rr, cc]
    terms = {
        "seed_correlation": float(
            step * 0.55 * maps["signed_seed_correlation"][rr, samples].sum(dtype=float)
        ),
        "wavelet": float(step * 0.25 * maps["hybrid_source_wavelet"][rr, samples].sum(dtype=float)),
        "generic_radar": float(
            step * 0.20 * maps["generic_radar_score"][rr, samples].sum(dtype=float)
        ),
        "directed_support": float(step * 0.55 * dense_support[rr, samples].sum(dtype=float)),
    }
    terms["floating_score_residual"] = float(step * scores.sum(dtype=float) - sum(terms.values()))
    terms["internal_correspondence_cost"] = -float(
        step
        * config.transition_cost_per_m
        * sum(local["by_index"][i].transition_cost for i in sequence)
    )
    external_correspondence, gaps = 0.0, 0.0
    for a, b in zip(sequence[:-1], sequence[1:], strict=True):
        left, right = local["by_index"][a], local["by_index"][b]
        distance = max(1, right.start - left.stop) * step
        external_correspondence += (
            config.transition_cost_per_m * (1 - local["edges"][a, b]) * distance
        )
        gaps += config.gap_cost_per_m * max(0, right.start - left.stop - 1) * step
    terms["external_correspondence_cost"] = -float(external_correspondence)
    terms["explicit_gap_cost"] = -float(gaps)
    return {
        "terms": terms,
        "total": float(sum(terms.values())),
        "observed_rows": len(nodes),
        "segment_ids": [int(i) for i in sequence],
        "row_samples": [[int(r), int(table.samples[r, c])] for r, c in nodes],
    }


def diagnose_layer(layer, args, evaluation, control):
    from gpr_layer_audit.conventional import _same_lobe
    from gpr_layer_audit.processing import hybrid, path_ambiguity
    from gpr_layer_audit.processing.conventional_config import ConventionalConfig

    directory = args.input / "lobe-mutual"
    capture = directory / f"layer-{layer}-graph-inputs.pkl.gz"
    with gzip.open(capture, "rb") as handle:
        payload = pickle.load(handle)
    table, measurement = payload["table"], payload["kwargs"]["measurement"]
    graph = np.load(directory / f"layer-{layer}-solver-graph.npz")
    metadata = json.loads((directory / f"layer-{layer}-solver-graph.json").read_text())
    config = ConventionalConfig(**metadata["config"])
    segments = []
    anchors = dict(graph["anchors"].tolist())
    for i, index in enumerate(graph["segment_ids"]):
        nodes = [
            tuple(node) for node in graph["nodes"][graph["offsets"][i] : graph["offsets"][i + 1]]
        ]
        seeds = {r for r, c in nodes if anchors.get(r) == table.samples[r, c]}
        segments.append(hybrid.ReflectorSegment(int(index), nodes, seeds, graph["costs"][i]))
    edges = {(int(a), int(b)): float(q) for a, b, q in graph["edges"]}
    # Recover the exact original directed-support array through the saved link
    # contraction, with its output signature independently checked.
    from experiment_edge_gate_loss import graph_signature, source_hash

    if payload["source_sha256"] != source_hash(args.source)["sha256"]:
        raise ValueError("Different source from captured numerical package")
    code = inspect.getsource(hybrid.correspondence_graph)
    tail = code[
        code.index("    outgoing_local, incoming_local = defaultdict(list), defaultdict(list)\n") :
    ]
    header = (
        "def retained_graph(table, anchors, breaks, pulse_width, step, measurement, "
        "config, links, diagnostics=None):\n"
    )
    namespace = dict(hybrid.correspondence_graph.__globals__)
    exec(compile(header + tail, "<saved-edge-contraction>", "exec"), namespace)
    retained = namespace["retained_graph"](
        table,
        anchors,
        payload["breaks"],
        payload["pulse_width"],
        payload["step"],
        measurement,
        config,
        payload["links"]["filtered"],
    )
    if graph_signature(retained) != payload["graph_signature"]:
        raise AssertionError("Retained-graph parity failed")
    dense_support = retained[3]
    metrics = evaluation["methods"]["seed_hybrid"]["layers"][str(layer)]
    previous = {
        p["row"]: p
        for p in control["methods"]["seed_hybrid"]["layers"][str(layer)]["evaluation_observations"]
    }
    errors = {
        p["row"]: p
        for p in metrics["evaluation_observations"]
        if p["accepted"] and not p["accepted_correct"]
    }
    tolerance = metrics["tolerance_samples"]
    reports = []
    seed_fiducials = [
        {
            "row": r,
            "sample": s,
            **lobe_timing(measurement[r], s),
            "analytic_phase_rad": float(table.component_maps["analytic_phase_rad"][r, s]),
        }
        for r, s in sorted(anchors.items())
    ]

    def capture_marginals(local):
        by_index = local["by_index"]
        for row, point in errors.items():
            selected = int(local["selected"][row])
            if selected < 0:
                continue
            reference = point["reference_sample"]
            correct_candidates, feasible_correct, selected_nodes = [], [], []
            timing_alternatives, family_alternatives = [], []
            for segment in local["segments"]:
                for r, col in segment.nodes:
                    if r != row:
                        continue
                    sample = int(table.samples[r, col])
                    seeds = (
                        local["forward"][segment.index][0]
                        + local["backward"][segment.index][0]
                        - len(segment.seed_rows)
                    )
                    feasible = seeds == local["optimum_count"]
                    score = (
                        local["forward"][segment.index][1]
                        + local["backward"][segment.index][1]
                        - local["local"][segment.index]
                    )
                    correct = abs(sample - reference) <= tolerance and _same_lobe(
                        measurement[r], sample, reference
                    )
                    same_lobe = _same_lobe(measurement[r], sample, selected)
                    item = {
                        "sample": sample,
                        "column": int(col),
                        "segment": int(segment.index),
                        "complete_seed_feasible": bool(feasible),
                        "max_route_objective": float(score),
                        "local_candidate_score": float(local["scores"][r, col]),
                        "reference_consistent": bool(correct),
                        "same_signed_lobe_as_selected": bool(same_lobe),
                        "inside_selected_timing_bucket": bool(
                            abs(sample - selected) <= tolerance
                            and table.polarities[r, col]
                            == table.polarities[r, np.flatnonzero(table.samples[r] == selected)[0]]
                        ),
                        "seed_score": float(
                            table.component_maps["signed_seed_correlation"][r, sample]
                        ),
                        "wavelet_score": float(
                            table.component_maps["hybrid_source_wavelet"][r, sample]
                        ),
                        "generic_score": float(
                            table.component_maps["generic_radar_score"][r, sample]
                        ),
                        "directed_support": float(dense_support[r, sample]),
                        "local_timing": lobe_timing(measurement[r], sample),
                    }
                    if correct:
                        correct_candidates.append(item)
                        if feasible:
                            feasible_correct.append(item)
                    if feasible and sample == selected:
                        selected_nodes.append(item)
                    if feasible and sample != selected:
                        (timing_alternatives if same_lobe else family_alternatives).append(item)
            winner = max(selected_nodes, key=lambda p: p["max_route_objective"])
            selected_route = route_terms(
                route_through(winner["segment"], local), local, table, dense_support
            )
            np.testing.assert_allclose(
                selected_route["total"], winner["max_route_objective"], atol=1e-8
            )
            competitor = (
                max(feasible_correct, key=lambda p: p["max_route_objective"])
                if feasible_correct
                else None
            )
            correct_route, delta = None, None
            if competitor:
                correct_route = route_terms(
                    route_through(competitor["segment"], local), local, table, dense_support
                )
                np.testing.assert_allclose(
                    correct_route["total"], competitor["max_route_objective"], atol=1e-8
                )
                delta = {
                    k: selected_route["terms"][k] - correct_route["terms"][k]
                    for k in selected_route["terms"]
                }
            np.testing.assert_allclose(local["margin"][row], point["path_margin"], atol=1e-10)
            lobe_start, lobe_stop = selected, selected
            while lobe_start > 0 and _same_lobe(measurement[row], lobe_start - 1, selected):
                lobe_start -= 1
            while lobe_stop + 1 < measurement.shape[1] and _same_lobe(
                measurement[row], lobe_stop + 1, selected
            ):
                lobe_stop += 1
            peak = lobe_start + int(np.argmax(abs(measurement[row, lobe_start : lobe_stop + 1])))
            reports.append(
                {
                    **point,
                    "layer": layer,
                    "error_samples": int(selected - reference),
                    "error_ns": float((selected - reference) * metadata["dt_ns"]),
                    "error_kind": "within_signed_lobe_timing"
                    if _same_lobe(measurement[row], selected, reference)
                    else "different_signed_lobe",
                    "newly_wrong_accepted": not (
                        previous[row]["accepted"] and not previous[row]["accepted_correct"]
                    ),
                    "control": previous[row],
                    "tolerance_samples": tolerance,
                    "interval": [
                        min(s.start for s in by_index.values()),
                        max(s.stop for s in by_index.values()),
                    ],
                    "anchors": {str(r): int(s) for r, s in local["anchors"].items()},
                    "lobe_extent_samples": [lobe_start, lobe_stop],
                    "amplitude_peak_sample": peak,
                    "selected_timing": lobe_timing(measurement[row], selected),
                    "reference_timing": lobe_timing(measurement[row], reference),
                    "selected_node": winner,
                    "correct_candidates": correct_candidates,
                    "best_feasible_correct_candidate": competitor,
                    "selected_route": selected_route,
                    "best_correct_at_row_route": correct_route,
                    "objective_selected_minus_correct": delta,
                    "best_same_lobe_other_timing": max(
                        timing_alternatives, key=lambda p: p["max_route_objective"]
                    )
                    if timing_alternatives
                    else None,
                    "best_different_signed_lobe": max(
                        family_alternatives, key=lambda p: p["max_route_objective"]
                    )
                    if family_alternatives
                    else None,
                    "production_margin_competitor_sample": int(local["alternate"][row]),
                    "production_margin_competitor_objective": float(local["competitor_score"][row])
                    if np.isfinite(local["competitor_score"][row])
                    else None,
                    "production_margin_denominator_m": float(local["denominator"][row]),
                    "full_objective_optimum": float(local["optimum"]),
                }
            )

    original = path_ambiguity.component_path_margins
    code = inspect.getsource(original)
    before, after = code.rsplit("    return margin, alternate", 1)
    code = before + "    _capture_marginals(locals())\n    return margin, alternate" + after
    namespace = dict(original.__globals__)
    namespace["_capture_marginals"] = capture_marginals
    exec(compile(code, "<read-only-marginal-objective-capture>", "exec"), namespace)
    path_ambiguity.component_path_margins = namespace["component_path_margins"]
    diagnostics = {}
    try:
        paths = hybrid.solve_complete_intervals(
            segments,
            edges,
            graph["support"],
            table,
            graph["scores"],
            anchors,
            metadata["breaks"],
            metadata["pulse_width_samples"],
            metadata["step_m"],
            config,
            diagnostics,
        )
    finally:
        path_ambiguity.component_path_margins = original
    recorded = np.load(directory / "evaluation-seed_hybrid.npz")[f"layer{layer}_provisional"]
    np.testing.assert_array_equal(paths[0], recorded)
    assert {r["row"] for r in reports} == set(errors)
    reports.sort(key=lambda p: p["row"])
    write(
        args.output / f"layer-{layer}-objective-errors.json",
        {
            "layer": layer,
            "diagnostic_only": True,
            "production_eligible": False,
            "exact_proposal_replay": True,
            "exact_error_margin_replay": True,
            "route_terms_verified_against_exact_max_marginal": True,
            "reference_role": "only this row is constrained in the post-inference counterfactual",
            "initial_operating_seed_timing_fiducials": seed_fiducials,
            "errors": reports,
        },
    )
    return reports, measurement, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.source))
    evaluation = json.loads((args.input / "lobe-mutual/evaluation.json").read_text())
    control = json.loads((args.input / "control/evaluation.json").read_text())
    reports = []
    for layer in (2, 3):
        result, measurement, metadata = diagnose_layer(layer, args, evaluation, control)
        reports.extend(result)
    summary = {
        "diagnostic_only": True,
        "source": str(args.source),
        "script_sha256": sha(__file__),
        "source_evaluation_sha256": sha(args.input / "lobe-mutual/evaluation.json"),
        "error_counts": {
            str(layer): {
                kind: sum(p["layer"] == layer and p["error_kind"] == kind for p in reports)
                for kind in ("within_signed_lobe_timing", "different_signed_lobe")
            }
            for layer in (2, 3)
        },
        "newly_wrong_rows": {
            str(layer): [
                p["row"] for p in reports if p["layer"] == layer and p["newly_wrong_accepted"]
            ]
            for layer in (2, 3)
        },
        "correct_candidate_missing": [p["row"] for p in reports if not p["correct_candidates"]],
        "correct_candidate_without_seed_feasible_route": [
            p["row"]
            for p in reports
            if p["correct_candidates"] and not p["best_feasible_correct_candidate"]
        ],
        "same_signed_lobe_does_not_establish_physical_identity": True,
    }
    write(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
