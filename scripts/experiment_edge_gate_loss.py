"""Isolated paired-edge audit; gate observers receive labels only after inference.

All instrumentation is applied to an in-memory copy of correspondence_graph.
The production source, frozen scoring helpers and retained candidate construction
are untouched. Captures are trusted local experiment artifacts, never user input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "exports/seeded-tracker/edge-gate-loss"
EXECUTED_SCRIPT = Path(__file__).read_bytes()
EXECUTED_SCRIPT_SHA256 = hashlib.sha256(EXECUTED_SCRIPT).hexdigest()
STAGES = (
    "retained",
    "forward_motion",
    "reverse_motion",
    "polarity",
    "phase",
    "registration",
    "local_motion",
    "mutual_lobe_ranking",
    "cosine",
    "dtw_agreement",
    "dtw_slip",
    "reciprocal_route",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def source_hash(source):
    files = {p.relative_to(source).as_posix(): sha(p) for p in sorted(Path(source).rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def lobe_mutual_contenders(score, same_left, same_right, allowed):
    """Reciprocal existing lobe groups, keeping their allowed timing alternatives."""
    from gpr_layer_audit.processing.hybrid import _ranked_lobe_matches

    forward = _ranked_lobe_matches(score, same_right)
    backward = _ranked_lobe_matches(score.T, same_left).T
    left, right = same_left.astype(np.int16), same_right.astype(np.int16)
    return (
        ((left @ forward.astype(np.int16) @ right) > 0)
        & ((left @ backward.astype(np.int16) @ right) > 0)
        & allowed
    )


def freeze_source():
    document = source_hash(ROOT / "src")
    target = WORK / ("source-" + document["sha256"][:16]) / "src"
    if not target.exists():
        shutil.copytree(ROOT / "src", target, ignore=shutil.ignore_patterns("__pycache__"))
    if source_hash(target) != document:
        raise ValueError("Existing source snapshot differs")
    write(target.parent / "manifest.json", document)
    print(target, flush=True)


def instrument(original, observer, mechanism="control"):
    """Insert read-only callbacks at exact executable gate boundaries."""
    source = inspect.getsource(original)
    changes = [
        ("    dtw_cache = {}\n", '    _edge_observer("setup", locals())\n    dtw_cache = {}\n'),
        (
            "            allowed = abs(displacement - predicted) <= drift_radius\n",
            "            allowed = abs(displacement - predicted) <= drift_radius\n"
            '            _edge_observer("forward_motion", locals())\n',
        ),
        (
            "            allowed &= table.polarities[left, ii, None] "
            "== table.polarities[right, js][None, :]\n",
            '            _edge_observer("reverse_motion", locals())\n'
            "            allowed &= table.polarities[left, ii, None] "
            "== table.polarities[right, js][None, :]\n"
            '            _edge_observer("polarity", locals())\n',
        ),
        (
            "            allowed &= np.minimum(phase_distance, 8 - phase_distance) <= 1\n",
            "            allowed &= np.minimum(phase_distance, 8 - phase_distance) <= 1\n"
            '            _edge_observer("phase", locals())\n',
        ),
        (
            "            motion = table.component_maps\n",
            '            _edge_observer("registration", locals())\n'
            "            motion = table.component_maps\n",
        ),
        (
            "            score = similarity - 0.08 * abs(displacement - predicted) "
            "/ max(budget, 1)\n",
            '            _edge_observer("local_motion", locals())\n'
            "            score = similarity - 0.08 * abs(displacement - predicted) "
            "/ max(budget, 1)\n",
        ),
        (
            "            contenders &= _ranked_lobe_matches(score.T, same_left).T & allowed\n",
            "            contenders &= _ranked_lobe_matches(score.T, same_left).T & allowed\n"
            '            _edge_observer("mutual_lobe_ranking", locals())\n',
        ),
        (
            "            eligible_pairs = contenders & (similarity >= 0.60)\n",
            "            eligible_pairs = contenders & (similarity >= 0.60)\n"
            '            _edge_observer("cosine", locals())\n',
        ),
        (
            "                agreement, slip = dtw_cache[key]\n",
            "                agreement, slip = dtw_cache[key]\n"
            '                _edge_observer("dtw", locals())\n',
        ),
        (
            "    if config and config.directed_correspondence:\n"
            "        from .directed_support import reciprocal_route_filter\n",
            '    _edge_observer("prefilter", locals())\n'
            "    if config and config.directed_correspondence:\n"
            "        from .directed_support import reciprocal_route_filter\n",
        ),
        (
            "    outgoing_local, incoming_local = defaultdict(list), defaultdict(list)\n",
            '    _edge_observer("filtered", locals())\n'
            "    outgoing_local, incoming_local = defaultdict(list), defaultdict(list)\n",
        ),
    ]
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError("Instrumentation target changed: " + old)
        source = source.replace(old, new)
    if mechanism == "lobe-mutual":
        old = (
            "            contenders = _ranked_lobe_matches(score, same_right)\n"
            "            contenders &= _ranked_lobe_matches(score.T, same_left).T & allowed\n"
        )
        new = (
            "            contenders = _lobe_mutual_contenders(\n"
            "                score, same_left, same_right, allowed\n"
            "            )\n"
        )
        if source.count(old) != 1:
            raise ValueError("Mutual-ranking replacement target changed")
        source = source.replace(old, new)
    elif mechanism != "control":
        raise ValueError("Unknown isolated mechanism")
    namespace = dict(original.__globals__)
    namespace["_edge_observer"] = observer
    namespace["_lobe_mutual_contenders"] = lobe_mutual_contenders
    exec(compile(source, "<isolated-edge-gate-observer>", "exec"), namespace)
    return namespace[original.__name__], source


def graph_signature(result):
    segments, edges, support, dense = result
    digest = hashlib.sha256()
    for segment in segments:
        digest.update(
            repr(
                (segment.index, segment.nodes, sorted(segment.seed_rows), segment.transition_cost)
            ).encode()
        )
    digest.update(repr(sorted(edges.items())).encode())
    digest.update(support.tobytes())
    digest.update(dense.tobytes())
    return digest.hexdigest()


def run(args):
    from experiment_identity_path import capture as capture_solver
    from seeded_eval import validate_frozen_helpers

    from gpr_layer_audit.conventional import evaluate_reference
    from gpr_layer_audit.processing import hybrid

    case = json.loads((ROOT / "benchmarks/seeded-evaluation-inputs.json").read_text())["cases"][
        args.case
    ]
    for field in ("dzt", "dzx"):
        if sha(ROOT / case[field]) != case[field + "_sha256"]:
            raise ValueError("Frozen input changed: " + field)
    if sha(ROOT / case["seed_source"]) != case["seed_sha256"]:
        raise ValueError("Frozen seeds changed")
    validate_frozen_helpers(args.source)
    output = args.output or WORK / args.case / args.mechanism
    if (output / "evaluation.json").exists():
        raise ValueError("Completed evaluation exists; preserve it and choose a fresh --output")
    output.mkdir(parents=True, exist_ok=True)
    (output / "executed_experiment_script.py").write_bytes(EXECUTED_SCRIPT)
    (output / "lobe_mutual_helper.py").write_text(
        inspect.getsource(lobe_mutual_contenders), encoding="utf-8"
    )
    config = json.loads(args.config.read_text())
    original = hybrid.correspondence_graph
    original_solver = hybrid.solve_complete_intervals
    capture_index = 1
    active = {}

    def observer(stage, local):
        if stage in ("prefilter", "filtered"):
            active[stage] = local["links"].copy()

    transformed, source = instrument(original, observer, args.mechanism)
    (output / "instrumented_correspondence_graph.py").write_text(source, encoding="utf-8")

    def wrapped(*values, **kwargs):
        nonlocal capture_index
        capture_index += 1
        active.clear()
        started = time.perf_counter()
        # Only radar, initial operating seeds and fixed configuration reach this
        # call; the evaluator owns its already audited reference observations.
        result = transformed(*values, **kwargs)
        payload = {
            "table": values[0],
            "anchors": values[1],
            "breaks": values[2],
            "pulse_width": values[3],
            "step": values[4],
            "kwargs": {k: v for k, v in kwargs.items() if k != "diagnostics"},
            "graph_signature": graph_signature(result),
            "links": active.copy(),
            "source_sha256": source_hash(args.source)["sha256"],
            "radar_only": True,
            "layer": capture_index,
        }
        with gzip.open(output / f"layer-{capture_index}-graph-inputs.pkl.gz", "wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        write(
            output / f"layer-{capture_index}-capture.json",
            {
                "graph_signature": payload["graph_signature"],
                "radar_only": True,
                "runtime_s": time.perf_counter() - started,
                "links_before_route_filter": len(active["prefilter"]),
                "retained_links": len(active["filtered"]),
                "segments": len(result[0]),
                "segment_edges": len(result[1]),
                "candidate_nodes": int(values[0].valid.sum()),
            },
        )
        print(f"Captured layer {capture_index}", flush=True)
        return result

    hybrid.correspondence_graph = wrapped

    def wrapped_solver(*values, **kwargs):
        ordered = list(values)
        for name in ("step", "config", "diagnostics"):
            if len(ordered) < 11 and name in kwargs:
                ordered.append(kwargs[name])
        capture_solver(output / f"layer-{capture_index}-solver-graph.npz", ordered, case["dt_ns"])
        return original_solver(*values, **kwargs)

    hybrid.solve_complete_intervals = wrapped_solver
    started = time.perf_counter()
    try:
        result = evaluate_reference(
            ROOT / case["dzx"],
            output=output / "evaluation.json",
            stride=case["stride"],
            methods=["seed_hybrid"],
            seed_source=ROOT / case["seed_source"],
            config=config,
        )
    finally:
        hybrid.correspondence_graph = original
        hybrid.solve_complete_intervals = original_solver
    result["edge_gate_experiment"] = {
        "mechanism": args.mechanism,
        "source": str(args.source),
        "source_manifest": source_hash(args.source),
        "script_sha256": EXECUTED_SCRIPT_SHA256,
        "instrumentation_sha256": sha(output / "instrumented_correspondence_graph.py"),
        "input_case": case,
        "config_sha256": sha(args.config),
        "runtime_wall_s": time.perf_counter() - started,
        "references": "frozen scorer only; graph captures contain radar and initial seeds only",
    }
    write(output / "evaluation.json", result)
    print(
        json.dumps(
            {
                order: {
                    k: values[k]
                    for k in (
                        "accepted",
                        "accepted_agree",
                        "proposed_agree",
                        "correct_coverage",
                        "reflector_switches",
                    )
                }
                for order, values in result["methods"]["seed_hybrid"]["layers"].items()
            }
        ),
        flush=True,
    )


class GateObserver:
    """Post-inference observer; it never modifies a table, score, mask or link."""

    def __init__(self, payload, reviewed):
        from gpr_layer_audit.conventional import _same_lobe

        self.payload = payload
        self.table = payload["table"]
        self.measurement = payload["kwargs"]["measurement"]
        self.tolerance = max(2.0, payload["pulse_width"] / 4)
        self.reviewed = {int(p["row"]): int(p["reference_sample"]) for p in reviewed}
        self.reviewed.update(payload["anchors"])
        self.correct = np.zeros_like(self.table.valid)
        for row, sample in self.reviewed.items():
            for col in np.flatnonzero(self.table.valid[row]):
                candidate = int(self.table.samples[row, col])
                self.correct[row, col] = abs(candidate - sample) <= self.tolerance and _same_lobe(
                    self.measurement[row], candidate, sample
                )
        self.pairs = {}
        self.rows = []
        self.admissible = None
        self.distances = None
        self.region = None
        self.filter_links = None

    def __call__(self, stage, local):
        if stage == "setup":
            self.admissible = local["admissible"].copy()
            self.distances, self.region = local["distances"], local["region"].copy()
            return
        if stage in ("prefilter", "filtered"):
            if stage == "filtered":
                self.filter_links = {(a, b) for a, b, _, _ in local["links"]}
                for key, group in self.pairs.items():
                    for index, (a, b) in enumerate(group["cols"]):
                        if (
                            not group["first_failure"][index]
                            and ((key[0], a), (key[1], b)) not in self.filter_links
                        ):
                            group["first_failure"][index] = STAGES.index("reciprocal_route")
            return
        key = (local["left"], local["right"])
        if stage == "forward_motion":
            ii, js = local["ii"], local["js"]
            aa, bb = np.nonzero(self.correct[key[0], ii, None] & self.correct[key[1], js][None, :])
            if len(aa):
                self.pairs[key] = {
                    "matrix": (aa, bb),
                    "cols": np.stack((ii[aa], js[bb]), axis=1),
                    "first_failure": np.zeros(len(aa), dtype=np.int8),
                    "cosine": local["similarity"][aa, bb].copy(),
                    "displacement": local["displacement"][aa, bb].copy(),
                    "agreement": np.full(len(aa), np.nan),
                    "slip": np.full(len(aa), np.nan),
                    "same_lobe_rank_rescue": np.zeros(len(aa), dtype=bool),
                }
        group = self.pairs.get(key)
        if group is None:
            return
        aa, bb = group["matrix"]
        if stage == "dtw":
            indices = np.flatnonzero((aa == local["a"]) & (bb == local["b"]))
            for index in indices:
                group["agreement"][index], group["slip"][index] = local["agreement"], local["slip"]
                if local["agreement"] < 0.75:
                    group["first_failure"][index] = STAGES.index("dtw_agreement")
                elif local["slip"] > self.tolerance:
                    group["first_failure"][index] = STAGES.index("dtw_slip")
            return
        mask = local["allowed"]
        if stage == "mutual_lobe_ranking":
            from gpr_layer_audit.processing.hybrid import _ranked_lobe_matches

            mask = local["contenders"]
            # Diagnostic counterfactual: does an allowed timing pair join the same
            # pair of endpoint signed lobes as a reciprocal contender? No scores
            # or graph edges are changed by this calculation.
            same_left, same_right = local["same_left"], local["same_right"]
            forward = _ranked_lobe_matches(local["score"], same_right)
            backward = _ranked_lobe_matches(local["score"].T, same_left).T
            lifted = (
                (
                    same_left.astype(np.int16)
                    @ forward.astype(np.int16)
                    @ same_right.astype(np.int16)
                )
                > 0
            ) & (
                (
                    same_left.astype(np.int16)
                    @ backward.astype(np.int16)
                    @ same_right.astype(np.int16)
                )
                > 0
            )
            group["same_lobe_rank_rescue"] = (
                lifted[aa, bb] & local["allowed"][aa, bb] & ~mask[aa, bb]
            )
        elif stage == "cosine":
            mask = local["eligible_pairs"]
        failed = ~mask[aa, bb] & (group["first_failure"] == 0)
        group["first_failure"][failed] = STAGES.index(stage)

    def report(self, output):
        counts, group_counts, distance_counts = Counter(), Counter(), {}
        points = []
        candidates_missing, admissibility_missing, barrier = 0, 0, 0
        for right in sorted(self.reviewed):
            for gap in self.distances:
                left = right - gap
                if left not in self.reviewed:
                    continue
                if self.region[left] != self.region[right] or any(
                    left < r < right for r in self.payload["anchors"]
                ):
                    barrier += 1
                    continue
                if not self.correct[left].any() or not self.correct[right].any():
                    candidates_missing += 1
                    continue
                if (
                    not (self.correct[left] & self.admissible[left]).any()
                    or not (self.correct[right] & self.admissible[right]).any()
                ):
                    admissibility_missing += 1
                    continue
                group = self.pairs[left, right]
                failures = group["first_failure"]
                first_loss = 0 if np.any(failures == 0) else int(failures.max())
                group_counts[STAGES[first_loss]] += 1
                distance_counts.setdefault(str(gap), Counter())[STAGES[first_loss]] += 1
                for index, (a, b) in enumerate(group["cols"]):
                    failed = int(failures[index])
                    counts[STAGES[failed]] += 1
                    points.append(
                        [
                            left,
                            int(a),
                            right,
                            int(b),
                            failed,
                            int(self.table.samples[left, a]),
                            int(self.table.samples[right, b]),
                            float(group["cosine"][index]),
                            float(group["agreement"][index]),
                            float(group["slip"][index]),
                            int(group["same_lobe_rank_rescue"][index]),
                        ]
                    )
        columns = [
            "left_row",
            "left_column",
            "right_row",
            "right_column",
            "first_failure_id",
            "left_sample",
            "right_sample",
            "cosine",
            "dtw_agreement",
            "dtw_slip",
            "same_lobe_rank_rescue",
        ]
        np.savez_compressed(output / "paired-edges.npz", values=np.asarray(points), columns=columns)
        examples = []
        for stage in STAGES[1:]:
            stage_groups = [
                (k, v)
                for k, v in self.pairs.items()
                if np.all(v["first_failure"] != 0)
                and int(v["first_failure"].max()) == STAGES.index(stage)
            ]
            for (left, right), group in sorted(
                stage_groups, key=lambda item: (item[0][1] - item[0][0], item[0])
            )[:5]:
                examples.append(
                    {
                        "first_irreversible_gate": stage,
                        "left_row": left,
                        "right_row": right,
                        "reference_samples": [self.reviewed[left], self.reviewed[right]],
                        "correct_candidate_pairs": group["cols"].tolist(),
                        "candidate_sample_pairs": [
                            [int(self.table.samples[left, a]), int(self.table.samples[right, b])]
                            for a, b in group["cols"]
                        ],
                        "phase_pairs": [
                            [
                                int(self.table.phase_classes[left, a]),
                                int(self.table.phase_classes[right, b]),
                            ]
                            for a, b in group["cols"]
                        ],
                        "polarity_pairs": [
                            [
                                int(self.table.polarities[left, a]),
                                int(self.table.polarities[right, b]),
                            ]
                            for a, b in group["cols"]
                        ],
                        "cosine": group["cosine"].tolist(),
                    }
                )
        nonseed = set(self.reviewed) - set(self.payload["anchors"])
        return {
            "diagnostic_only": True,
            "production_eligible": False,
            "scoring_contract": "same signed lobe and abs(error)<=max(2, initial pulse width/4)",
            "timing_tolerance_samples": self.tolerance,
            "denominators": {
                "candidate_pairs": (
                    "Every pair of retained reference-consistent candidates at an actual matching "
                    "distance, with both rows reviewed or operating anchors; "
                    "structural/seed barriers excluded"
                ),
                "row_pairs": (
                    "Unique reviewed/anchor row pair at an actual matching distance; category is "
                    "last gate reached by any correct pair, retained if any survives all gates"
                ),
                "unknown_rows": "not scored, never absent or background",
            },
            "reviewed_nonseed_rows": len(nonseed),
            "correct_candidate_retained_nonseed_rows": int(
                sum(self.correct[r].any() for r in nonseed)
            ),
            "row_pairs_blocked_by_structural_or_seed_barriers": barrier,
            "row_pairs_missing_a_correct_candidate": candidates_missing,
            "row_pairs_missing_after_admissibility": admissibility_missing,
            "candidate_pair_first_losses": dict(counts),
            "row_pair_first_losses": dict(group_counts),
            "row_pair_first_losses_by_gap_rows": {k: dict(v) for k, v in distance_counts.items()},
            "same_lobe_rank_rescuable_correct_timing_pairs": int(
                sum(v["same_lobe_rank_rescue"].sum() for v in self.pairs.values())
            ),
            "row_pairs_lost_at_ranking_with_same_lobe_reciprocal_alternative": int(
                sum(
                    np.all(v["first_failure"] != 0)
                    and int(v["first_failure"].max()) == STAGES.index("mutual_lobe_ranking")
                    and v["same_lobe_rank_rescue"].any()
                    for v in self.pairs.values()
                )
            ),
            "failure_examples": examples,
            "stage_ids": dict(enumerate(STAGES)),
        }


def diagnose(args):
    from gpr_layer_audit.processing import hybrid

    input_dir = args.input or WORK / args.case / args.mechanism
    evaluation = json.loads((input_dir / "evaluation.json").read_text())
    for layer in args.layers:
        capture = input_dir / f"layer-{layer}-graph-inputs.pkl.gz"
        with gzip.open(capture, "rb") as handle:
            payload = pickle.load(handle)
        if payload["source_sha256"] != source_hash(args.source)["sha256"]:
            raise ValueError("Diagnostic source differs from captured numerical source")
        reviewed = evaluation["methods"]["seed_hybrid"]["layers"][str(layer)][
            "evaluation_observations"
        ]
        observer = GateObserver(payload, reviewed)
        transformed, source = instrument(hybrid.correspondence_graph, observer, args.mechanism)
        output = input_dir / f"layer-{layer}-diagnostic"
        output.mkdir(parents=True, exist_ok=True)
        (output / "instrumented_correspondence_graph.py").write_text(source, encoding="utf-8")
        started = time.perf_counter()
        result = transformed(
            payload["table"],
            payload["anchors"],
            payload["breaks"],
            payload["pulse_width"],
            payload["step"],
            **payload["kwargs"],
        )
        signature = graph_signature(result)
        if signature != payload["graph_signature"]:
            raise AssertionError("Observer changed original retained graph output")
        report = observer.report(output)
        report.update(
            graph_signature=signature,
            exact_graph_parity=True,
            runtime_wall_s=time.perf_counter() - started,
            source_sha256=payload["source_sha256"],
            capture_sha256=sha(capture),
            evaluation_sha256=sha(input_dir / "evaluation.json"),
            script_sha256=sha(__file__),
        )
        write(output / "report.json", report)
        print(
            json.dumps(
                {
                    "layer": layer,
                    "row_pair_first_losses": report["row_pair_first_losses"],
                    "exact_graph_parity": True,
                }
            ),
            flush=True,
        )


def oracle(args):
    """Replay captured links, then solve a labelled diagnostic with costs zero."""
    from experiment_identity_path import capture as capture_solver
    from experiment_identity_path import replay

    from gpr_layer_audit.processing import hybrid
    from gpr_layer_audit.processing.identity_path import IdentityGeometry

    input_dir = args.input or WORK / args.case / args.mechanism
    evaluation = json.loads((input_dir / "evaluation.json").read_text())
    for layer in args.layers:
        with gzip.open(input_dir / f"layer-{layer}-graph-inputs.pkl.gz", "rb") as handle:
            payload = pickle.load(handle)
        if payload["source_sha256"] != source_hash(args.source)["sha256"]:
            raise ValueError("Oracle source differs from captured numerical source")
        original = hybrid.correspondence_graph
        source = inspect.getsource(original)
        boundary = "    outgoing_local, incoming_local = defaultdict(list), defaultdict(list)\n"
        tail = source[source.index(boundary) :]
        header = (
            "def captured_contraction(table, anchors, breaks, pulse_width, step, "
            "measurement, config, links, diagnostics=None):\n"
        )
        namespace = dict(original.__globals__)
        exec(compile(header + tail, "<captured-original-contraction>", "exec"), namespace)
        graph = namespace["captured_contraction"](
            payload["table"],
            payload["anchors"],
            payload["breaks"],
            payload["pulse_width"],
            payload["step"],
            payload["kwargs"]["measurement"],
            payload["kwargs"]["config"],
            payload["links"]["filtered"],
        )
        if graph_signature(graph) != payload["graph_signature"]:
            raise AssertionError("Captured-link replay differs from original retained graph")
        graph_path = input_dir / f"layer-{layer}-diagnostic-graph.npz"
        capture_solver(
            graph_path,
            [
                *graph[:3],
                payload["table"],
                np.zeros_like(payload["table"].samples, dtype=float),
                payload["anchors"],
                payload["breaks"],
                payload["pulse_width"],
                payload["step"],
                payload["kwargs"]["config"],
                {},
            ],
            evaluation["edge_gate_experiment"]["input_case"]["dt_ns"],
        )
        metadata = json.loads(graph_path.with_suffix(".json").read_text())
        metadata["local_scores"] = "zero placeholders; ONLY correct-only diagnostic oracle eligible"
        metadata["exact_graph_parity"] = True
        write(graph_path.with_suffix(".json"), metadata)
        replay(
            graph_path,
            IdentityGeometry(),
            input_dir / f"layer-{layer}-correct-only-oracle.json",
            input_dir / "evaluation.json",
            layer=layer,
            correct_only=True,
        )
        print(
            f"Layer {layer}: exact captured-link graph replay and diagnostic oracle complete",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze-source", "run", "diagnose", "oracle", "self-test"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"))
    parser.add_argument("--mechanism", choices=("control", "lobe-mutual"), default="control")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmarks/conventional-motion-calibrated-development.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    args = parser.parse_args()
    if args.mode == "freeze-source":
        freeze_source()
        return
    if args.source is None or (args.case is None and args.mode != "self-test"):
        parser.error("--source and --case are required")
    args.source = args.source.resolve()
    sys.path.insert(0, str(args.source))
    if args.mode == "self-test":
        from gpr_layer_audit.processing.hybrid import _ranked_lobe_matches

        score = np.array([[0.9, 0.8], [1.0, 0.9]])
        same = np.ones((2, 2), dtype=bool)
        allowed = np.ones_like(same)
        prior = _ranked_lobe_matches(score, same) & _ranked_lobe_matches(score.T, same).T
        fixed = lobe_mutual_contenders(score, same, same, allowed)
        assert not prior[0, 1] and fixed[0, 1] and np.all(fixed)
        distinct = np.eye(2, dtype=bool)
        np.testing.assert_array_equal(
            lobe_mutual_contenders(score, distinct, distinct, allowed),
            _ranked_lobe_matches(score, distinct) & _ranked_lobe_matches(score.T, distinct).T,
        )
        allowed[0, 1] = False
        assert not lobe_mutual_contenders(score, same, same, allowed)[0, 1]
        assert not lobe_mutual_contenders(np.full_like(score, -np.inf), same, same, allowed).any()
        print("4 numerical contracts passed", flush=True)
    elif args.mode == "run":
        run(args)
    elif args.mode == "oracle":
        oracle(args)
    else:
        diagnose(args)


if __name__ == "__main__":
    main()
