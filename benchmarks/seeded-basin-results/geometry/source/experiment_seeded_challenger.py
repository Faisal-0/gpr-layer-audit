"""Run the independent packet comparator with the frozen processed-road protocol.

Example: .venv/Scripts/python scripts/experiment_seeded_challenger.py
    --case gujrat-second --output exports/seeded-tracker/challenger/gujrat.json
Only explicit operating seeds enter inference; DZX observations enter scoring
after paths have been returned. No threshold is fitted by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if "--source-root" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--source-root") + 1])

from gpr_layer_audit.conventional import _overlay, layer_metrics, peak_process_memory, write_json
from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file
from gpr_layer_audit.io.dzx import read_dzx
from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

ROOT = Path(__file__).resolve().parents[1]


def run_case(case, output, *, config=None, stride=None, layers=(1, 2, 3)):
    from gpr_layer_audit.processing.seeded_challenger import PacketConfig, pick_seeded_packet

    config = config or PacketConfig()
    manifest_path = ROOT / "benchmarks/seeded-evaluation-inputs.json"
    manifest = json.loads(manifest_path.read_text())
    source_info = manifest["cases"][case]
    source = ROOT / source_info["dzt"]
    if fingerprint_file(source) != source_info["dzt_sha256"]:
        raise ValueError("Input DZT fingerprint changed")
    seed_path = ROOT / source_info["seed_source"]
    if fingerprint_file(seed_path) != source_info["seed_sha256"]:
        raise ValueError("Seed fingerprint changed")
    seed_document = json.loads(seed_path.read_text())
    stride = stride or source_info["stride"]
    anchors = {}
    for order, points in seed_document["observations"].items():
        if int(order) not in layers:
            continue
        if any(p["trace"] % stride for p in points):
            raise ValueError("Native seeds are incompatible with stride; no snapping")
        anchors[int(order)] = {p["trace"] // stride: p["sample"] for p in points}
    radar_file = DZTFile(source)
    data = np.asarray(radar_file.channel()[::stride], np.float32).copy()
    valid = processed_boundary_mask(data)
    dt_ns = radar_file.header.sample_interval_ns
    dx_m = radar_file.header.distance_per_trace_m * stride
    surface = int(np.clip(round(-radar_file.header.position_ns / dt_ns), 0, data.shape[1] - 1))
    pulses = {
        order: resolve_pulse(data, valid, seeds, {}, dt_ns, ConventionalConfig())
        for order, seeds in anchors.items()
    }
    paths, runtimes = {}, {}
    for order, seeds in anchors.items():
        start = time.perf_counter()
        paths[order] = pick_seeded_packet(
            data,
            seeds,
            valid=valid,
            dt_ns=dt_ns,
            dx_m=dx_m,
            pulse_width_samples=pulses[order].lobe_samples,
            context_radius=pulses[order].context_radius,
            reference_surface=surface,
            config=config,
        )
        runtimes[order] = time.perf_counter() - start
        print(f"{case} layer {order}: inference {runtimes[order]:.2f}s", flush=True)
    # Evaluation-only labels are opened only after inference returns.
    reference_path = ROOT / source_info["dzx"]
    if fingerprint_file(reference_path) != source_info["dzx_sha256"]:
        raise ValueError("Reference DZX fingerprint changed")
    references = {
        int(source_info["reference_label_mapping"][str(group.number)]): [
            replace(p, trace=p.trace // stride)
            for p in group.picks
            if p.channel == 0 and p.trace % stride == 0
        ]
        for group in read_dzx(reference_path).layers
        if str(group.number) in source_info["reference_label_mapping"]
        and int(source_info["reference_label_mapping"][str(group.number)]) in paths
    }
    metrics = {}
    for order, path in paths.items():
        metrics[order] = layer_metrics(
            path,
            references[order],
            anchors[order],
            data,
            pulses[order].lobe_samples,
            dx_m,
        )
        metrics[order]["runtime_s"] = runtimes[order]
        metrics[order]["bracketed_and_tail"] = {}
        for name, subset in (
            (
                "bracketed",
                [
                    p
                    for p in references[order]
                    if min(anchors[order]) <= p.trace <= max(anchors[order])
                ],
            ),
            (
                "tail",
                [
                    p
                    for p in references[order]
                    if not min(anchors[order]) <= p.trace <= max(anchors[order])
                ],
            ),
        ):
            values = layer_metrics(
                path, subset, anchors[order], data, pulses[order].lobe_samples, dx_m
            )
            metrics[order]["bracketed_and_tail"][name] = {
                k: v for k, v in values.items() if k != "evaluation_observations"
            }
    output = Path(output)
    snapshot = output.with_name(output.stem + "-source")
    snapshot.mkdir(parents=True, exist_ok=True)
    for source_file in (
        Path(__file__),
        ROOT / "src/gpr_layer_audit/processing/seeded_challenger.py",
    ):
        shutil.copy2(source_file, snapshot / source_file.name)
    result = {
        "schema": "dense-seeded-packet-experiment-v1",
        "case": case,
        "usage": "development",
        "config": asdict(config),
        "source": source_info,
        "stride": stride,
        "manifest_sha256": fingerprint_file(manifest_path),
        "layers": metrics,
        "runtime_s": sum(runtimes.values()),
        "process_peak_memory_bytes": peak_process_memory(),
        "implementation_sha256": fingerprint_file(
            ROOT / "src/gpr_layer_audit/processing/seeded_challenger.py"
        ),
        "config_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "source_snapshot": str(snapshot),
        "scoring_tolerance": {
            order: max(2, pulse.lobe_samples / 4) for order, pulse in pulses.items()
        },
        "claims": "Interpretation-reference agreement only; physical accuracy not established",
    }
    write_json(output, result)
    np.savez_compressed(
        output.with_suffix(".npz"),
        sample_validity=valid,
        **{
            f"layer{order}_{key}": value
            for order, path in paths.items()
            for key, value in (
                ("accepted", path.samples),
                ("provisional", path.provisional_samples),
                ("margin", path.evidence["hybrid_path_margin"]),
                ("correlation", path.evidence["packet_seed_correlation"]),
            )
        },
    )
    _overlay(output.with_suffix(".png"), data, references, anchors, {"dense_packet": paths}, dx_m)
    print(
        json.dumps(
            {
                order: {
                    k: v[k]
                    for k in ("accepted", "accepted_agree", "correct_coverage", "proposed_agree")
                }
                for order, v in metrics.items()
            },
            indent=2,
        )
    )
    return result


def capture_graph(case, output, *, config=None, stride=None):
    """Capture an unmodified backend's observation DAG; labels stay in scorer."""
    from gpr_layer_audit.conventional import backend_fingerprint, evaluate_reference
    from gpr_layer_audit.processing import hybrid

    info = json.loads((ROOT / "benchmarks/seeded-evaluation-inputs.json").read_text())["cases"][
        case
    ]
    stride = stride or info["stride"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_contract, original_graph = hybrid._contract_links, hybrid.correspondence_graph
    captured = {}
    seed_doc = json.loads((ROOT / info["seed_source"]).read_text())
    expected = {
        int(order): {p["trace"] // stride: p["sample"] for p in points}
        for order, points in seed_doc["observations"].items()
    }

    def contract(table, anchors, links, *args):
        captured["links"] = list(links)
        return original_contract(table, anchors, links, *args)

    def graph(table, anchors, breaks, pulse_width, step, *args, **kwargs):
        result = original_graph(table, anchors, breaks, pulse_width, step, *args, **kwargs)
        order = next(k for k, v in expected.items() if v == anchors)
        np.savez_compressed(
            output / f"graph-{order}.npz",
            samples=table.samples,
            valid=table.valid,
            waveforms=table.waveforms,
            phase_classes=table.phase_classes,
            polarities=table.polarities,
            measurement=kwargs["measurement"],
            signal_valid=table.component_maps.get("sample_validity"),
            seeds=np.asarray(sorted(anchors.items()), int),
            links=np.asarray([(a[0], a[1], b[0], b[1], q, d) for a, b, q, d in captured["links"]]),
            correspondence=result[3],
        )
        write_json(
            output / f"graph-{order}.json",
            {
                "case": case,
                "layer": order,
                "backend_sha256": backend_fingerprint(),
                "config": asdict(kwargs["config"]),
                "pulse_width_samples": pulse_width,
                "dx_m": step,
                "stride": stride,
            },
        )
        return result

    hybrid._contract_links, hybrid.correspondence_graph = contract, graph
    try:
        evaluate_reference(
            ROOT / info["dzx"],
            output=output / "evaluation.json",
            methods=["seed_hybrid"],
            stride=stride,
            config=config,
            seed_source=ROOT / info["seed_source"],
        )
    finally:
        hybrid._contract_links, hybrid.correspondence_graph = original_contract, original_graph


def diagnose_graph(capture, output):
    """Evaluation-only retained-route oracle; it never writes inference inputs."""
    from gpr_layer_audit.conventional import _same_lobe

    capture = Path(capture)
    evaluation = json.loads((capture / "evaluation.json").read_text())
    result = {"schema": "retained-route-diagnostic-v1", "oracle_only": True, "layers": {}}
    for file in capture.glob("graph-*.npz"):
        order = file.stem.split("-")[-1]
        arrays = np.load(file)
        samples, valid = arrays["samples"], arrays["valid"]
        radar, links = arrays["measurement"], arrays["links"]
        layer = evaluation["methods"]["seed_hybrid"]["layers"][order]
        tolerance = layer["tolerance_samples"]
        seeds = {int(r): int(s) for r, s in arrays["seeds"]}
        references = {o["row"]: o["reference_sample"] for o in layer["evaluation_observations"]}
        known = {**references, **seeds}
        correct = {}
        for row, reference in known.items():
            correct[row] = {
                (row, int(col))
                for col in np.flatnonzero(valid[row])
                if abs(samples[row, col] - reference) <= tolerance
                and _same_lobe(radar[row], samples[row, col], reference)
            }
        all_nodes = {(r, int(c)) for r in range(len(samples)) for c in np.flatnonzero(valid[r])}
        allowed = {n for n in all_nodes if n[0] not in known or n in correct[n[0]]}
        incoming = {}
        for ar, ac, br, bc, _quality, _gap in links:
            source, target = (int(ar), int(ac)), (int(br), int(bc))
            if source in allowed and target in allowed:
                incoming.setdefault(target, []).append(source)
        intervals = []
        for left, right in zip(sorted(seeds)[:-1], sorted(seeds)[1:], strict=True):
            values = {n: 0 for n in correct[left]}
            for node in sorted(n for n in allowed if left < n[0] <= right):
                preceding = [values[n] for n in incoming.get(node, ()) if n in values]
                if preceding:
                    values[node] = max(preceding) + int(node[0] in references)
            best = max((values.get(n, -1) for n in correct[right]), default=-1)
            held = [r for r in references if left < r < right]
            intervals.append(
                {
                    "start_row": left,
                    "stop_row": right,
                    "scored_observations": len(held),
                    "reference_consistent_endpoint_route_exists": best >= 0,
                    "oracle_recoverable_observations": max(0, best),
                    "oracle_policy": (
                        "remove incorrect known-row nodes; score correct visits; "
                        "skipped rows remain gaps"
                    ),
                }
            )
        missing_candidates = [r for r in references if not correct[r]]
        no_incident_correct = [
            r for r in references if correct[r] and not any(n in incoming for n in correct[r])
        ]
        result["layers"][order] = {
            "scored_observations": len(references),
            "candidate_retained": sum(bool(correct[r]) for r in references),
            "missing_candidate_rows": missing_candidates,
            "retained_candidate_without_reference_consistent_incoming": no_incident_correct,
            "intervals": intervals,
            "source_graph_sha256": fingerprint_file(file),
            "source_evaluation_sha256": fingerprint_file(capture / "evaluation.json"),
            "adjacent_first_loss": adjacent_first_loss(arrays, known, tolerance),
        }
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def adjacent_first_loss(arrays, known, tolerance):
    """Offline classification for the captured default graph's adjacent pairs.

    This is explicitly local diagnostics, not an end-to-end route oracle. It
    identifies failed necessary raw-packet checks. Motion may have rejected an
    edge earlier; this is not a claimed decomposition of that gate's internals.
    """
    from collections import Counter

    from gpr_layer_audit.conventional import _same_lobe

    samples, valid = arrays["samples"], arrays["valid"]
    radar, waveforms = arrays["measurement"], arrays["waveforms"]
    seeds = {int(r): int(s) for r, s in arrays["seeds"]}
    retained = {(int(a), int(ac), int(b), int(bc)) for a, ac, b, bc, _q, _d in arrays["links"]}
    counts, examples = Counter(), []
    for right in sorted(known):
        left = right - 1
        if left not in known:
            continue
        ii, jj = np.flatnonzero(valid[left]), np.flatnonzero(valid[right])
        if left in seeds:
            ii = ii[samples[left, ii] == seeds[left]]
        if right in seeds:
            jj = jj[samples[right, jj] == seeds[right]]
        good_left = [
            i
            for i in ii
            if abs(samples[left, i] - known[left]) <= tolerance
            and _same_lobe(radar[left], samples[left, i], known[left])
        ]
        good_right = [
            j
            for j in jj
            if abs(samples[right, j] - known[right]) <= tolerance
            and _same_lobe(radar[right], samples[right, j], known[right])
        ]
        counts["reviewed_adjacent_pairs"] += 1
        if not good_left or not good_right:
            reason = "candidate_missing"
        else:
            polarity = (
                arrays["polarities"][left, good_left, None]
                == arrays["polarities"][right, good_right][None, :]
            )
            phase = abs(
                arrays["phase_classes"][left, good_left, None].astype(int)
                - arrays["phase_classes"][right, good_right][None, :].astype(int)
            )
            compatible = polarity & (np.minimum(phase, 8 - phase) <= 1)
            correlation = waveforms[left, good_left] @ waveforms[right, good_right].T
            if not np.any(polarity):
                reason = "polarity_hard_gate"
            elif not np.any(compatible):
                reason = "phase_hard_gate"
            elif not np.any(compatible & (correlation >= 0.6)):
                reason = "cosine_floor"
            elif not any(
                (left, int(i), right, int(j)) in retained for i in good_left for j in good_right
            ):
                reason = "motion_ranking_dtw_or_route_filter"
            else:
                reason = "retained_correct_local_link"
        counts[reason] += 1
        if reason != "retained_correct_local_link" and len(examples) < 20:
            examples.append({"left_row": left, "right_row": right, "reason": reason})
    return {
        "counts": dict(counts),
        "first_failures": examples,
        "scope": "adjacent reviewed pairs; non-local recovery is evaluated separately",
        "classification": (
            "candidate, polarity, phase, raw packet cosine, residual correspondence gates"
        ),
    }


def replay_graph(capture, output, *, retained_lobes=3, near_best=0.15):
    """Seed-only graph sensitivity; reused candidates never contain reference picks."""
    from gpr_layer_audit.processing import hybrid
    from gpr_layer_audit.processing.continuation import local_phase_motion

    capture, output = Path(capture), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_ranked, original_contract = hybrid._ranked_lobe_matches, hybrid._contract_links
    state = {}

    def ranked(scores, same_lobe, count=3):
        remaining = scores.copy()
        selected = np.zeros(scores.shape, bool)
        best, rows = np.max(scores, axis=1), np.arange(len(scores))
        for _ in range(retained_lobes):
            winner = np.argmax(remaining, axis=1)
            value = remaining[rows, winner]
            valid = np.isfinite(value) & (value >= best - near_best)
            selected[rows[valid], winner[valid]] = True
            remaining = np.where(same_lobe[winner], -np.inf, remaining)
        return selected

    def contract(table, anchors, links, *args):
        state["links"] = list(links)
        return original_contract(table, anchors, links, *args)

    hybrid._ranked_lobe_matches, hybrid._contract_links = ranked, contract
    try:
        for file in capture.glob("graph-*.npz"):
            arrays = dict(np.load(file, allow_pickle=False))
            meta = json.loads(file.with_suffix(".json").read_text())
            width, radar = meta["pulse_width_samples"], arrays["measurement"]
            maps = local_phase_motion(radar, width)
            maps["sample_validity"] = arrays["signal_valid"]
            table = SimpleNamespace(
                **{
                    k: arrays[k]
                    for k in ("samples", "valid", "waveforms", "polarities", "phase_classes")
                },
                component_maps=maps,
                dense_radar_score=np.zeros_like(radar),
            )
            result = hybrid.correspondence_graph(
                table,
                {int(r): int(s) for r, s in arrays["seeds"]},
                (),
                width,
                meta["dx_m"],
                measurement=radar,
                config=ConventionalConfig(**meta["config"]),
                displacement_samples=width,
            )
            links = np.asarray([(a[0], a[1], b[0], b[1], q, d) for a, b, q, d in state["links"]])
            same = links.shape == arrays["links"].shape and np.allclose(links, arrays["links"])
            if retained_lobes == 3 and near_best == 0.15 and not same:
                raise ValueError("Reconstructed graph differs; sensitivity comparison is invalid")
            arrays["links"], arrays["correspondence"] = links, result[3]
            np.savez_compressed(output / file.name, **arrays)
            meta["retention_sensitivity"] = {
                "retained_lobes": retained_lobes,
                "near_best": near_best,
                "identical_to_source": bool(same),
                "links": len(links),
            }
            write_json(output / file.with_suffix(".json").name, meta)
            print(file.name, meta["retention_sensitivity"], flush=True)
    finally:
        hybrid._ranked_lobe_matches, hybrid._contract_links = original_ranked, original_contract
    # Reference file copied only after graph inference completes, for separate diagnosis.
    shutil.copy2(capture / "evaluation.json", output / "evaluation.json")


def make_generous_source(source_root, output):
    """Create an isolated falsification snapshot; never alter the live backend."""
    source_root, output = Path(source_root), Path(output)
    if output.exists():
        raise ValueError("Experiment source destination already exists; preserve it")
    shutil.copytree(source_root, output, ignore=shutil.ignore_patterns("__pycache__"))
    target = output / "gpr_layer_audit/processing/hybrid.py"
    code = target.read_text()
    ranked_prefix = "def _ranked_lobe_matches(scores, same_lobe, count=3):"
    phase_prefix = "allowed &= np.minimum(phase_distance, 8 - phase_distance) <= 1"
    replacements = {
        ranked_prefix: ranked_prefix.replace("count=3", "count=6"),
        "value >= best - 0.15": "value >= best - 0.30",
        phase_prefix: phase_prefix.replace("<= 1", "<= 4"),
        "allowed &= ~reliable | (": "allowed &= np.ones_like(allowed) | (",
    }
    for before, after in replacements.items():
        if code.count(before) != 1:
            raise ValueError("Frozen source does not match the reviewed pruning experiment")
        code = code.replace(before, after)
    compile(code, str(target), "exec")
    target.write_text(code)
    write_json(
        output / "experiment.json",
        {
            "role": "isolated development pruning falsification; not promoted",
            "source_root": str(source_root.resolve()),
            "replacements": replacements,
            "hybrid_sha256": fingerprint_file(target),
            "preserved": (
                "polarity, physical displacement, cosine/DTW floors, "
                "gap/seed barriers, objective, acceptance"
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--capture-graph", action="store_true")
    parser.add_argument("--diagnose-graph", type=Path)
    parser.add_argument("--replay-graph", type=Path)
    parser.add_argument("--retained-lobes", type=int, default=3)
    parser.add_argument("--near-best", type=float, default=0.15)
    parser.add_argument("--make-generous-source", action="store_true")
    args = parser.parse_args()
    if args.make_generous_source:
        make_generous_source(args.source_root, args.output)
    elif args.replay_graph:
        replay_graph(
            args.replay_graph,
            args.output,
            retained_lobes=args.retained_lobes,
            near_best=args.near_best,
        )
    elif args.diagnose_graph:
        diagnose_graph(args.diagnose_graph, args.output)
    elif args.capture_graph:
        capture_graph(
            args.case,
            args.output,
            config=json.loads(args.config.read_text()) if args.config else {},
            stride=args.stride,
        )
    else:
        from gpr_layer_audit.processing.seeded_challenger import PacketConfig

        settings = (
            PacketConfig(**json.loads(args.config.read_text())) if args.config else PacketConfig()
        )
        run_case(
            args.case, args.output, config=settings, stride=args.stride, layers=tuple(args.layers)
        )
