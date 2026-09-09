"""One isolated seed-derived extremum-state experiment on frozen road inputs.

Run self-test before run; summarize loads scoring references only after inference.
Production and frozen source are never edited.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "exports/seeded-tracker/extremum-states"
SOURCE = ROOT / "exports/seeded-tracker/edge-gate-loss/source-e4b9c239079145ea/src"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def bootstrap(source):
    sys.path.insert(0, str(source))
    from gpr_layer_audit.processing import hybrid, path_ambiguity, seed_graph

    return hybrid, path_ambiguity, seed_graph


def self_test(args):
    from extremum_state_model import basin_map, clone, peak_closure

    hybrid, ambiguity, _ = bootstrap(args.source)
    from types import SimpleNamespace

    from gpr_layer_audit.processing.conventional_config import ConventionalConfig
    from gpr_layer_audit.processing.path_objective import local_score, transition_penalty

    mapping, valleys = basin_map(np.array([0, 1, 3, 5, 4, 2, 3, 4, 2, 0, -1, -3, -2, 0]))
    assert list(mapping[1:5]) == [3] * 4
    assert mapping[5] == -1 and valleys[5] == (3, 7)
    assert list(mapping[6:9]) == [7] * 3
    assert list(mapping[10:13]) == [11] * 3
    allowed = np.ones(len(mapping), bool)
    closed = peak_closure([(4, 4, [4]), (5, 5, [5])], mapping, valleys, allowed)
    assert closed[0][2] == [3, 4] and closed[1][2] == [3, 5, 7]
    allowed[3] = False
    assert peak_closure([(4, 4, [4])], mapping, valleys, allowed)[0][2] == [4]
    plateau, _ = basin_map([0, 1, 3, 3, 1, 0])
    assert plateau[2] == plateau[3] == 2

    # Enumerate every route on a tiny split/shoulder DAG including a gap.
    table = SimpleNamespace(
        samples=np.array([[10, -1], [20, 21], [30, 32], [40, -1]]),
        valid=np.ones((4, 2), bool),
        polarities=np.ones((4, 2)),
        component_maps={},
    )
    nodes = [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
    segments = [
        hybrid.ReflectorSegment(i, [node], {node[0]} if node[0] in (0, 3) else set())
        for i, node in enumerate(nodes)
    ]
    edges = {
        (0, 1): 0.98,
        (0, 2): 0.92,
        (1, 3): 0.98,
        (2, 3): 0.93,
        (1, 4): 0.94,
        (2, 4): 0.96,
        (3, 5): 0.97,
        (4, 5): 0.96,
        (0, 3): 0.70,
    }
    scores = np.array([[0.6, 0], [0.9, 0.8], [0.9, 0.83], [0.6, 0]])
    config = ConventionalConfig()
    anchors = {0: 10, 3: 40}
    strict, _ = clone(
        ambiguity.component_path_margins,
        [("    tolerance = max(2, pulse_width / 4)\n", "    tolerance = 0\n")],
    )
    routes = []

    def visit(sequence):
        tail = sequence[-1]
        if tail == 5:
            total = sum(
                local_score(segments[i], scores, 1, config.transition_cost_per_m) for i in sequence
            )
            total -= sum(
                transition_penalty(
                    segments[a],
                    segments[b],
                    edges[a, b],
                    1,
                    config.transition_cost_per_m,
                    config.gap_cost_per_m,
                )
                for a, b in zip(sequence[:-1], sequence[1:], strict=True)
            )
            members = np.full(4, -1)
            for i in sequence:
                r, c = nodes[i]
                members[r] = table.samples[r, c]
            emitted = members.copy()
            emitted[1] = 20 if emitted[1] >= 0 else -1  # shoulder 21 is basin 20.
            routes.append((total, sequence, emitted))
        for a, b in edges:
            if a == tail:
                visit([*sequence, b])

    visit([0])
    best = max(routes, key=lambda item: item[0])
    selected = hybrid._solve_component(
        segments,
        edges,
        np.ones((4, 2)),
        table,
        scores,
        anchors,
        [],
        29,
        step=1,
        config=config,
        require_all=True,
    )[0]
    expected_members = np.full(4, -1)
    for i in best[1]:
        r, c = nodes[i]
        expected_members[r] = table.samples[r, c]
    np.testing.assert_array_equal(selected, expected_members)
    emitted_table = SimpleNamespace(**vars(table))
    emitted_table.samples = table.samples.copy()
    emitted_table.samples[1, 1] = 20
    margins, alternate = strict(
        segments, edges, emitted_table, scores, anchors, best[2], 29, step=1, config=config
    )
    for row in (1, 2):
        rivals = [route for route in routes if route[2][row] != best[2][row]]
        rival = max(rivals, key=lambda item: item[0])
        different = np.flatnonzero(rival[2] != best[2])
        group = next(
            g for g in np.split(different, np.flatnonzero(np.diff(different) > 1) + 1) if row in g
        )
        expected = min(1, (best[0] - rival[0]) / len(group))
        np.testing.assert_allclose(margins[row], expected, atol=1e-12)
        assert alternate[row] == rival[2][row]
    # Chain contraction preserves exactly the expanded objective and seed cuts.
    chain_table = SimpleNamespace(
        samples=np.array([[10], [20], [30], [40]]), valid=np.ones((4, 1), bool)
    )
    links = [((r, 0), (r + 1, 0), 0.9 + 0.01 * r, 1) for r in range(3)]
    nxt = {a: b for a, b, _, _ in links}
    prev = {b: a for a, b, _, _ in links}
    contracted, membership = hybrid._contract_links(chain_table, anchors, links, nxt, prev)
    chain_scores = np.array([[0.6], [0.9], [0.9], [0.6]])
    internal = sum(
        local_score(s, chain_scores, 1, config.transition_cost_per_m) for s in contracted
    )
    external = sum(
        (1 - q) * config.transition_cost_per_m
        for a, b, q, _ in links
        if membership[a] != membership[b]
    )
    expected = (
        chain_scores.sum() - sum(1 - q for _, _, q, _ in links) * config.transition_cost_per_m
    )
    np.testing.assert_allclose(internal - external, expected, atol=1e-12)
    report = {
        "passed": True,
        "enumerated_routes": len(routes),
        "contracts": [
            "unique signed extremum and shoulder emission",
            "secondary peak isolation",
            "valley preserves two modes and unresolved emission",
            "peak closure before cap",
            "peak validity exclusion",
            "plateau determinism",
            "exact emitted-mode max-marginals versus exhaustive routes",
            "explicit gap competitor",
            "selected path equals exhaustive optimum",
            "chain contraction objective parity",
        ],
    }
    write(WORK / "self-test.json", report)
    print(json.dumps(report), flush=True)


def run(args):
    from experiment_edge_gate_loss import graph_signature, instrument, source_hash
    from extremum_state_model import CONTRACT, install
    from seeded_eval import validate_frozen_helpers

    hybrid, ambiguity, seed_graph = bootstrap(args.source)
    from gpr_layer_audit.conventional import evaluate_reference

    if not json.loads((WORK / "self-test.json").read_text())["passed"]:
        raise ValueError("Self-test must pass first")
    case = json.loads((ROOT / "benchmarks/seeded-evaluation-inputs.json").read_text())["cases"][
        args.case
    ]
    for field in ("dzt", "dzx"):
        assert sha(ROOT / case[field]) == case[field + "_sha256"]
    assert sha(ROOT / case["seed_source"]) == case["seed_sha256"]
    helpers = validate_frozen_helpers(args.source)
    output = args.output or WORK / args.case / args.graph
    if (output / "evaluation.json").exists():
        raise ValueError("Preserve completed experiment; choose a fresh output")
    output.mkdir(parents=True, exist_ok=True)
    # Written before any inference or access to reference scoring results.
    write(output / "contract.json", CONTRACT)
    for filename in ("experiment_extremum_states.py", "extremum_state_model.py"):
        (output / filename).write_bytes((ROOT / "scripts" / filename).read_bytes())
    captures = {}

    def observer(stage, values):
        if stage in ("prefilter", "filtered"):
            captures[stage] = values["links"].copy()

    graph, graph_source = instrument(hybrid.correspondence_graph, observer, args.graph)
    (output / "executed-correspondence_graph.py").write_text(graph_source, encoding="utf-8")
    layer = 1

    def wrapped_graph(*values, **kwargs):
        nonlocal layer
        layer += 1
        started = time.perf_counter()
        result = graph(*values, **kwargs)
        payload = {
            "table": values[0],
            "anchors": values[1],
            "breaks": values[2],
            "pulse_width": values[3],
            "step": values[4],
            "kwargs": {k: v for k, v in kwargs.items() if k != "diagnostics"},
            "links": captures.copy(),
            "graph_signature": graph_signature(result),
            "radar_only": True,
        }
        with gzip.open(output / f"layer-{layer}-graph-inputs.pkl.gz", "wb") as handle:
            pickle.dump(payload, handle, protocol=5)
        print(
            f"Layer {layer}: {values[0].valid.sum()} states, "
            f"{len(captures['filtered'])} edges, {time.perf_counter() - started:.1f}s",
            flush=True,
        )
        return result

    hybrid.correspondence_graph = wrapped_graph
    original_solver = hybrid.solve_complete_intervals

    def wrapped_solver(*values, **kwargs):
        table = values[3]
        result = original_solver(*values, **kwargs)
        paths = np.asarray(result)
        np.savez_compressed(
            output / f"layer-{layer}-selected-members.npz",
            paths=paths,
            samples=table.samples,
            scores=values[4],
            emissions=table.component_maps["extremum_emission"],
        )
        return result

    hybrid.solve_complete_intervals = wrapped_solver
    seeds, _ = install(seed_graph, hybrid, ambiguity, output)
    started = time.perf_counter()
    result = evaluate_reference(
        ROOT / case["dzx"],
        output=output / "evaluation.json",
        stride=case["stride"],
        methods=["seed_hybrid"],
        seed_source=ROOT / case["seed_source"],
        config=json.loads(args.config.read_text()),
    )
    result["extremum_experiment"] = {
        "contract": CONTRACT,
        "graph": args.graph,
        "source_manifest": source_hash(args.source),
        "scripts": {p.name: sha(p) for p in output.glob("*.py")},
        "config_sha256": sha(args.config),
        "input_case": case,
        "frozen_helpers": helpers,
        "runtime_wall_s": time.perf_counter() - started,
        "operating_seeds": seeds,
        "acceleration": "Frozen source already dispatches exact compiled batch DTW",
        "references": "frozen scorer only",
    }
    write(output / "evaluation.json", result)
    print(
        json.dumps(
            {
                k: {
                    f: v[f]
                    for f in ("accepted", "accepted_agree", "proposed_agree", "correct_coverage")
                }
                for k, v in result["methods"]["seed_hybrid"]["layers"].items()
            }
        ),
        flush=True,
    )


def summarize(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bootstrap(args.source)
    from gpr_layer_audit.io.dzt import DZTFile

    result = {}
    for graph in ("control", "lobe-mutual"):
        after_path = WORK / args.case / graph / "evaluation.json"
        if not after_path.exists():
            continue
        before_path = (
            ROOT / "exports/seeded-tracker/edge-gate-loss" / args.case / graph / "evaluation.json"
        )
        before, after = [json.loads(p.read_text()) for p in (before_path, after_path)]
        measurement = DZTFile(after["audit"]["dzt_path"]).channel()[:: after["stride"]]
        figure, axes = plt.subplots(2, 2, figsize=(17, 8), sharex=True, sharey="row")
        layers = {}
        for row, layer in enumerate(("2", "3")):
            layers[layer] = {}
            for label, record in (("before", before), ("after", after)):
                data = record["methods"]["seed_hybrid"]["layers"][layer]
                layers[layer][label] = {
                    name: data[name]
                    for name in (
                        "accepted",
                        "accepted_agree",
                        "proposed_agree",
                        "correct_coverage",
                        "reflector_switches",
                    )
                }
                observations = data["evaluation_observations"]
                if observations:
                    layers[layer][label]["observation_schema"] = list(observations[0])
                ax = axes[row, int(label == "after")]
                low = min(p["reference_sample"] for p in observations) - 25
                high = max(p["reference_sample"] for p in observations) + 25
                scale = np.quantile(abs(measurement[:, max(0, low) : high]), 0.99)
                ax.imshow(
                    measurement.T,
                    origin="upper",
                    aspect="auto",
                    cmap="gray",
                    vmin=-scale,
                    vmax=scale,
                )
                ax.scatter(
                    [p["row"] for p in observations],
                    [p["reference_sample"] for p in observations],
                    s=8,
                    c="cyan",
                    label="Reviewed",
                )
                for accepted, color, title in (
                    (False, "orange", "Proposal"),
                    (True, "lime", "Accepted"),
                ):
                    usable = [
                        p
                        for p in observations
                        if p.get("proposed_sample", -1) >= 0 and bool(p.get("accepted")) == accepted
                    ]
                    ax.scatter(
                        [p["row"] for p in usable],
                        [p["proposed_sample"] for p in usable],
                        s=10,
                        c=color,
                        label=title,
                    )
                ax.set_ylim(high, low)
                ax.set_title(
                    f"Layer {layer}, {graph}, {label}: "
                    f"{data['accepted_agree']}/{data['accepted']} accepted agree"
                )
                ax.set_xlabel("Stored trace row")
                ax.set_ylabel("Native sample")
                ax.legend(fontsize=7, loc="upper right")
            earlier = {
                p["row"]: p
                for p in before["methods"]["seed_hybrid"]["layers"][layer][
                    "evaluation_observations"
                ]
            }
            current = after["methods"]["seed_hybrid"]["layers"][layer]["evaluation_observations"]
            layers[layer]["transitions"] = {
                "new_correct_acceptances": [
                    p["row"]
                    for p in current
                    if p["accepted_correct"] and not earlier[p["row"]]["accepted_correct"]
                ],
                "lost_correct_acceptances": [
                    p["row"]
                    for p in current
                    if not p["accepted_correct"] and earlier[p["row"]]["accepted_correct"]
                ],
                "new_wrong_acceptances": [
                    p["row"]
                    for p in current
                    if p["accepted"]
                    and not p["accepted_correct"]
                    and not (
                        earlier[p["row"]]["accepted"] and not earlier[p["row"]]["accepted_correct"]
                    )
                ],
                "removed_wrong_acceptances": [
                    p["row"]
                    for p in current
                    if not (p["accepted"] and not p["accepted_correct"])
                    and earlier[p["row"]]["accepted"]
                    and not earlier[p["row"]]["accepted_correct"]
                ],
                "all_after_wrong_acceptances": [
                    p for p in current if p["accepted"] and not p["accepted_correct"]
                ],
            }
        figure.tight_layout()
        figure.savefig(WORK / args.case / f"{graph}-overlay.png", dpi=150)
        plt.close(figure)
        result[graph] = {
            "before": str(before_path),
            "after": str(after_path),
            "hashes": {"before": sha(before_path), "after": sha(after_path)},
            "layers": layers,
        }
    write(WORK / args.case / "comparison.json", result)
    print(json.dumps(result), flush=True)


def audit(args):
    """Rebuild captured contraction and independently replay objective/margins."""
    from experiment_edge_gate_loss import graph_signature
    from extremum_state_model import clone, emission_view, emit_path

    hybrid, ambiguity, _ = bootstrap(args.source)
    from gpr_layer_audit.processing.path_objective import local_score, transition_penalty

    graph_source = inspect.getsource(hybrid.correspondence_graph)
    tail = graph_source[
        graph_source.index(
            "    outgoing_local, incoming_local = defaultdict(list), defaultdict(list)\n"
        ) :
    ]
    namespace = dict(hybrid.correspondence_graph.__globals__)
    exec(
        compile(
            "def rebuild(table, anchors, breaks, pulse_width, step, "
            "measurement, config, links, diagnostics=None):\n"
            + tail,
            "<captured-contraction-audit>",
            "exec",
        ),
        namespace,
    )
    strict, _ = clone(
        ambiguity.component_path_margins,
        [("    tolerance = max(2, pulse_width / 4)\n", "    tolerance = 0\n")],
    )
    summary = {}
    for graph in ("control", "lobe-mutual"):
        directory = WORK / args.case / graph
        evaluation = json.loads((directory / "evaluation.json").read_text())
        summary[graph] = {}
        for layer in (2, 3):
            with gzip.open(directory / f"layer-{layer}-graph-inputs.pkl.gz", "rb") as handle:
                capture = pickle.load(handle)
            table, anchors = capture["table"], capture["anchors"]
            config, step, width = (
                capture["kwargs"]["config"],
                capture["step"],
                capture["pulse_width"],
            )
            rebuilt = namespace["rebuild"](
                table,
                anchors,
                capture["breaks"],
                width,
                step,
                capture["kwargs"]["measurement"],
                config,
                capture["links"]["filtered"],
            )
            assert graph_signature(rebuilt) == capture["graph_signature"]
            segments, edges = rebuilt[:2]
            with np.load(directory / f"layer-{layer}-selected-members.npz") as data:
                members, scores = data["paths"][0], data["scores"]
            emissions = emit_path(table, members)
            replay_margin = np.zeros(len(members))
            boundaries = sorted({0, len(members), *capture["breaks"]})
            intervals = []
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                seed_rows = sorted(r for r in anchors if start <= r < stop)
                if not seed_rows:
                    continue
                for left, right in [
                    (start, seed_rows[0]),
                    *zip(seed_rows[:-1], seed_rows[1:], strict=True),
                    (seed_rows[-1], stop - 1),
                ]:
                    required = {r: s for r, s in anchors.items() if left <= r <= right}
                    subset = [s for s in segments if s.start >= left and s.stop <= right]
                    by_id = {s.index: s for s in subset}
                    connections = {
                        (a, b): q for (a, b), q in edges.items() if a in by_id and b in by_id
                    }
                    ordered = sorted(subset, key=lambda s: (s.start, s.stop, s.index))
                    locals_ = {
                        s.index: local_score(s, scores, step, config.transition_cost_per_m)
                        for s in subset
                    }
                    incoming = {}
                    for (a, b), quality in connections.items():
                        incoming.setdefault(b, []).append((a, quality))
                    forward = {}
                    for segment in ordered:
                        count = len(segment.seed_rows)
                        choices = [(count, locals_[segment.index])]
                        for a, quality in incoming.get(segment.index, []):
                            b = segment.index
                            if any(by_id[a].stop < r < segment.start for r in required):
                                continue
                            penalty = transition_penalty(
                                by_id[a],
                                segment,
                                quality,
                                step,
                                config.transition_cost_per_m,
                                config.gap_cost_per_m,
                            )
                            choices.append(
                                (forward[a][0] + count, forward[a][1] + locals_[b] - penalty)
                            )
                        forward[segment.index] = max(choices)
                    optimum_count, optimum = max(forward.values(), default=(0, -np.inf))
                    feasible = optimum_count == len(required)
                    item = {
                        "left": left,
                        "right": right,
                        "required_seeds": len(required),
                        "feasible": feasible,
                    }
                    if feasible:
                        selected = [
                            s
                            for s in ordered
                            if all(members[r] == table.samples[r, c] for r, c in s.nodes)
                        ]
                        objective = sum(locals_[s.index] for s in selected)
                        for a, b in zip(selected[:-1], selected[1:], strict=True):
                            objective -= transition_penalty(
                                a,
                                b,
                                connections[a.index, b.index],
                                step,
                                config.transition_cost_per_m,
                                config.gap_cost_per_m,
                            )
                        assert abs(objective - optimum) < 1e-5, (
                            graph,
                            layer,
                            left,
                            right,
                            objective,
                            optimum,
                        )
                        values, _ = strict(
                            subset,
                            connections,
                            emission_view(table),
                            scores,
                            required,
                            emissions,
                            width,
                            step=step,
                            config=config,
                        )
                        replay_margin[left : right + 1] = values[left : right + 1]
                        item.update(
                            selected_objective=objective,
                            exact_optimum=optimum,
                            delta=objective - optimum,
                            selected_segments=len(selected),
                        )
                    intervals.append(item)
            layer_result = evaluation["methods"]["seed_hybrid"]["layers"][str(layer)]
            differences = [
                abs(p["path_margin"] - replay_margin[p["row"]])
                for p in layer_result["evaluation_observations"]
            ]
            assert max(differences, default=0) < 1e-10
            with gzip.open(
                ROOT
                / "exports/seeded-tracker/edge-gate-loss"
                / args.case
                / graph
                / f"layer-{layer}-graph-inputs.pkl.gz",
                "rb",
            ) as handle:
                before = pickle.load(handle)
            added, removed = 0, 0
            for row in range(len(members)):
                original = set(before["table"].samples[row, before["table"].valid[row]]) - {-1}
                current = set(table.samples[row, table.valid[row]]) - {-1}
                added += len(current - original)
                removed += len(original - current)
            summary[graph][layer] = {
                "capture_graph_signature_parity": True,
                "observed_margin_max_error": max(differences, default=0),
                "intervals": intervals,
                "retained_member_nodes_before": int(
                    (before["table"].valid & (before["table"].samples >= 0)).sum()
                ),
                "retained_member_nodes_after": int((table.valid & (table.samples >= 0)).sum()),
                "added_member_nodes": added,
                "removed_member_nodes": removed,
                "member_links_before": len(before["links"]["filtered"]),
                "member_links_after": len(capture["links"]["filtered"]),
                "contracted_segments": len(segments),
                "contracted_edges": len(edges),
                "selected_measured_members": int((members >= 0).sum()),
                "selected_emitted_extrema": int((emissions >= 0).sum()),
                "selected_unresolved_fiducials": int(((members >= 0) & (emissions < 0)).sum()),
                "selected_nonzero_member_offsets": int(
                    ((members >= 0) & (emissions >= 0) & (members != emissions)).sum()
                ),
            }
    write(WORK / args.case / "objective-audit.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("self-test", "run", "summarize", "audit"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--case", choices=("mandiali-short", "gujrat-second"), default="mandiali-short"
    )
    parser.add_argument("--graph", choices=("control", "lobe-mutual"), default="control")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmarks/conventional-motion-calibrated-development.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    {"self-test": self_test, "run": run, "summarize": summarize, "audit": audit}[args.mode](args)


if __name__ == "__main__":
    main()
