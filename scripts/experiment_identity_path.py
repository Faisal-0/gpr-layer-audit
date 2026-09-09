"""Development-only same-input comparison of the history-aware path objective.

Inference receives frozen operating seeds and radar only. Frozen reviewed picks
reach the existing scorer after tracking. ``--replay`` reuses a captured retained
graph for cheap objective diagnosis; it does not claim whole-tracker accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--source", type=Path, default=ROOT / "src")
SOURCE = BOOTSTRAP.parse_known_args()[0].source.resolve()
sys.path.insert(0, str(SOURCE))

from gpr_layer_audit.conventional import evaluate_reference, write_json  # noqa: E402
from gpr_layer_audit.processing import hybrid  # noqa: E402
from gpr_layer_audit.processing.conventional_config import ConventionalConfig  # noqa: E402
from gpr_layer_audit.processing.identity_path import (  # noqa: E402
    IdentityGeometry,
    immutable_seed_modes,
    solve_identity_intervals,
)


def capture(path, args, dt, identity=None):
    segments, edges, support, table, scores, anchors, breaks, width, step, config, _ = args
    nodes = [node for segment in segments for node in segment.nodes]
    offsets = np.r_[0, np.cumsum([len(s.nodes) for s in segments])]
    arrays = dict(
        samples=table.samples,
        valid=table.valid,
        polarities=table.polarities,
        scores=scores,
        support=support,
        anchors=np.array(sorted(anchors.items()), dtype=int),
        segment_ids=np.array([s.index for s in segments]),
        nodes=np.asarray(nodes),
        offsets=offsets,
        costs=np.array([s.transition_cost for s in segments]),
        edges=np.array([(a, b, q) for (a, b), q in edges.items()]),
    )
    lobes = table.component_maps.get("hybrid_lobe_id")
    if lobes is not None:
        arrays["lobes"] = lobes
    if identity is not None:
        arrays["seed_scores"], arrays["mode_scores"] = identity[:2]
        arrays["mode_change_rows"] = np.array(sorted(identity[2]), dtype=int)
    np.savez_compressed(path, **arrays)
    write_json(
        path.with_suffix(".json"),
        {
            "dt_ns": dt,
            "step_m": step,
            "pulse_width_samples": width,
            "breaks": list(breaks),
            "config": asdict(config),
            "graph_inputs": "radar and operating seeds only; no scoring references",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )


def replay(
    path, geometry, output, oracle_reference=None, layer=None, persistent=False, correct_only=False
):
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    anchors = dict(arrays["anchors"].tolist())
    table = SimpleNamespace(
        **{name: arrays[name] for name in ("samples", "valid", "polarities")}, component_maps={}
    )
    if "lobes" in arrays:
        table.component_maps["hybrid_lobe_id"] = arrays["lobes"]
    segments = []
    for i, index in enumerate(arrays["segment_ids"]):
        nodes = [tuple(n) for n in arrays["nodes"][arrays["offsets"][i] : arrays["offsets"][i + 1]]]
        seed_rows = {r for r, c in nodes if anchors.get(r) == table.samples[r, c]}
        segments.append(hybrid.ReflectorSegment(int(index), nodes, seed_rows, arrays["costs"][i]))
    edges = {(int(a), int(b)): float(q) for a, b, q in arrays["edges"]}
    scores = arrays["scores"]
    oracle = None
    if oracle_reference:
        from gpr_layer_audit.conventional import _same_lobe
        from gpr_layer_audit.io.dzt import DZTFile

        if layer is None:
            raise ValueError("An oracle requires an explicit layer")
        reference = json.loads(oracle_reference.read_text())
        reviewed = reference["methods"]["seed_hybrid"]["layers"][str(layer)]
        measurement = DZTFile(reference["audit"]["dzt_path"]).channel()[:: reference["stride"]]
        tolerance = max(2, metadata["pulse_width_samples"] / 4)
        scores = np.zeros_like(scores)
        correct = np.zeros_like(table.valid)
        retained = 0
        for point in reviewed["evaluation_observations"]:
            row, sample = point["row"], point["reference_sample"]
            for col in np.flatnonzero(table.valid[row]):
                candidate = table.samples[row, col]
                correct[row, col] = abs(candidate - sample) <= tolerance and _same_lobe(
                    measurement[row], candidate, sample
                )
            scores[row] = np.where(correct[row], 100, -100)
            retained += int(np.any(correct[row]))
        oracle = dict(
            diagnostic_only=True,
            production_eligible=False,
            purpose="Oracle local scores on the unchanged retained graph",
            reviewed_nonseed_observations=len(reviewed["evaluation_observations"]),
            reference_candidate_retained=retained,
            reference_source_sha256=hashlib.sha256(oracle_reference.read_bytes()).hexdigest(),
        )
        if correct_only:
            reviewed_rows = {point["row"] for point in reviewed["evaluation_observations"]}
            segments = [
                segment
                for segment in segments
                if all(row not in reviewed_rows or correct[row, col] for row, col in segment.nodes)
            ]
            scores = correct.astype(float)
            geometry = IdentityGeometry()
            oracle["purpose"] = (
                "Maximum correct reviewed observations on the retained graph with every "
                "known wrong node forbidden and all correspondence/geometry costs zero"
            )
            oracle["unknown_rows"] = "neutral, neither observations nor absence"
    report = {}
    start = time.perf_counter()
    mode_options = {}
    if persistent:
        if "mode_scores" not in arrays:
            raise ValueError("This capture does not contain immutable seed mode evidence")
        scores = scores - 0.55 * arrays["seed_scores"]
        mode_options = dict(
            mode_scores=0.55 * arrays["mode_scores"], mode_change_rows=arrays["mode_change_rows"]
        )
    replay_config = ConventionalConfig(**metadata["config"])
    if correct_only:
        replay_config = replace(replay_config, transition_cost_per_m=0, gap_cost_per_m=0)
    paths = solve_identity_intervals(
        segments,
        edges,
        arrays["support"],
        table,
        scores,
        anchors,
        metadata["breaks"],
        metadata["pulse_width_samples"],
        metadata["step_m"],
        replay_config,
        report,
        dt_ns=metadata["dt_ns"],
        geometry=geometry,
        **mode_options,
    )
    margins, alternate = report.pop("path_margin"), report.pop("path_alternate")
    legacy_comparison = None
    if oracle is None:
        legacy_diagnostics = {}
        legacy = hybrid.solve_complete_intervals(
            segments,
            edges,
            arrays["support"],
            table,
            arrays["scores"],
            anchors,
            metadata["breaks"],
            metadata["pulse_width_samples"],
            metadata["step_m"],
            ConventionalConfig(**metadata["config"]),
            legacy_diagnostics,
        )
        legacy_comparison = {
            "same_retained_graph_and_local_scores": True,
            "changed_selected_rows": int(np.count_nonzero(paths[0] != legacy[0])),
            "legacy_observed_rows": int(np.count_nonzero(legacy[0] >= 0)),
            "maximum_margin_difference": float(
                np.max(abs(margins - legacy_diagnostics["path_margin"]))
            ),
        }
    if oracle is not None:
        agreed, wrong, missing = 0, 0, 0
        for point in reviewed["evaluation_observations"]:
            row, sample = point["row"], point["reference_sample"]
            prediction = paths[0][row]
            if prediction < 0:
                missing += 1
            elif abs(prediction - sample) <= tolerance and _same_lobe(
                measurement[row], prediction, sample
            ):
                agreed += 1
            else:
                wrong += 1
        oracle.update(
            oracle_proposed_agree=agreed, oracle_proposed_wrong=wrong, oracle_unresolved=missing
        )
    np.savez_compressed(
        output.with_suffix(".npz"), paths=np.array(paths), margins=margins, alternate=alternate
    )
    write_json(
        output,
        {
            "geometry": asdict(geometry),
            "runtime_s": time.perf_counter() - start,
            "capture": str(path),
            "diagnostics": report,
            "proposed_rows": int(np.count_nonzero(paths[0] >= 0)),
            "oracle": oracle,
            "legacy_comparison": legacy_comparison,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"))
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "benchmarks/conventional-motion-calibrated-development.json",
    )
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--persistent", action="store_true")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--oracle-reference", type=Path)
    parser.add_argument("--correct-only-oracle", action="store_true")
    parser.add_argument("--layer", type=int, choices=(2, 3))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geometry = (
        IdentityGeometry(**json.loads(args.geometry.read_text()))
        if args.geometry
        else IdentityGeometry(curvature_weight=0.05, curvature_scale_ns_per_m2=1.0)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.replay:
        if args.correct_only_oracle and not args.oracle_reference:
            parser.error("--correct-only-oracle requires --oracle-reference")
        if args.persistent and args.oracle_reference:
            parser.error("Keep oracle label scores separate from deployable mode experiments")
        replay(
            args.replay,
            geometry,
            args.output,
            args.oracle_reference,
            args.layer,
            args.persistent,
            args.correct_only_oracle,
        )
        return
    if args.oracle_reference:
        parser.error("Oracle scores are confined to --replay graph diagnostics")
    if args.case is None:
        parser.error("--case or --replay is required")
    manifest = json.loads((ROOT / "benchmarks/seeded-evaluation-inputs.json").read_text())
    case = manifest["cases"][args.case]
    for field, hash_field in (
        ("dzt", "dzt_sha256"),
        ("dzx", "dzx_sha256"),
        ("seed_source", "seed_sha256"),
    ):
        if hashlib.sha256((ROOT / case[field]).read_bytes()).hexdigest() != case[hash_field]:
            raise ValueError(f"Frozen input changed: {field}")
    config = json.loads(args.config.read_text())
    config.update(interval_seed_scoring=True, distinct_path_inference=True)
    original = hybrid.solve_complete_intervals
    from gpr_layer_audit.processing import interval_templates

    original_templates = interval_templates.interval_seed_scores
    identities = {}

    def template_scores(
        table, prototypes, correlations, anchors, breaks, measurement, valid, step, **kwargs
    ):
        result = original_templates(
            table, prototypes, correlations, anchors, breaks, measurement, valid, step, **kwargs
        )
        modes, changes, metadata = immutable_seed_modes(
            table, prototypes, correlations, anchors, measurement, valid
        )
        identities[id(table)] = result[0], modes, changes, metadata
        return result

    interval_templates.interval_seed_scores = template_scores
    count = 1
    solver_diagnostics = []

    def wrapped(*values, **kwargs):
        nonlocal count
        count += 1
        # The production call supplies step/config/diagnostics as named arguments.
        ordered = list(values)
        for name in ("step", "config", "diagnostics"):
            if len(ordered) < 11 and name in kwargs:
                ordered.append(kwargs[name])
        capture(
            args.output.with_name(args.output.stem + f"-graph-{count}.npz"),
            ordered,
            case["dt_ns"],
            identities.get(id(values[3])),
        )
        if args.capture_only:
            return original(*values, **kwargs)
        mode_options = {}
        if args.persistent:
            seed_scores, modes, changes, _ = identities[id(values[3])]
            values = list(values)
            values[4] = values[4] - 0.55 * seed_scores
            mode_options = dict(mode_scores=0.55 * modes, mode_change_rows=changes)
        result = solve_identity_intervals(
            *values, **kwargs, dt_ns=case["dt_ns"], geometry=geometry, **mode_options
        )
        report = kwargs.get("diagnostics", ordered[-1])
        solver_diagnostics.append(
            {"layer": count, "intervals": report.get("identity_objective", [])}
        )
        return result

    hybrid.solve_complete_intervals = wrapped
    try:
        result = evaluate_reference(
            ROOT / case["dzx"],
            output=args.output,
            stride=case["stride"],
            methods=["seed_hybrid"],
            seed_source=ROOT / case["seed_source"],
            config=config,
        )
        result["identity_experiment"] = {
            "geometry": asdict(geometry),
            "capture_only": args.capture_only,
            "persistent_modes": args.persistent,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "import_source": str(SOURCE),
            "solves": solver_diagnostics,
        }
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    order: {
                        key: metrics[key]
                        for key in (
                            "accepted",
                            "accepted_agree",
                            "proposed_agree",
                            "correct_coverage",
                            "reflector_switches",
                        )
                    }
                    for order, metrics in result["methods"]["seed_hybrid"]["layers"].items()
                }
            ),
            flush=True,
        )
    finally:
        hybrid.solve_complete_intervals = original
        interval_templates.interval_seed_scores = original_templates


if __name__ == "__main__":
    main()
