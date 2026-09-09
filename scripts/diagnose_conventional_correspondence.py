"""Capture the seed-only observation DAG for offline correspondence failure analysis.

Reviewed observations reach only the benchmark scorer, never the graph capture.
The compressed arrays allow diagnosing the same graph without repeating features.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpr_layer_audit.conventional import backend_fingerprint, evaluate_reference, write_json
from gpr_layer_audit.processing import hybrid
from gpr_layer_audit.processing.continuation import local_phase_motion
from gpr_layer_audit.processing.conventional_config import ConventionalConfig
from gpr_layer_audit.processing.directed_support import directed_seed_support


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path, nargs="?")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--integrated-motion", action="store_true")
    parser.add_argument("--near-links", action="store_true")
    parser.add_argument("--context-radius-m", type=float, default=0.0)
    parser.add_argument("--minimum-motion-margin", type=float, default=0.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original_graph, original_contract = hybrid.correspondence_graph, hybrid._contract_links
    captured = {}
    config = {
        "integrated_motion": args.integrated_motion,
        "waveform_context_radius_m": args.context_radius_m,
        "minimum_motion_margin": args.minimum_motion_margin,
    }
    if args.near_links:
        config["matching_distances_m"] = (0.05, 0.125, 0.25, 0.5, 1, 2, 5, 10, 25)

    def contract(table, anchors, links, *rest):
        captured["links"] = list(links)
        return original_contract(table, anchors, links, *rest)

    def graph(table, anchors, breaks, pulse_width, step, *rest, **kwargs):
        result = original_graph(table, anchors, breaks, pulse_width, step, *rest, **kwargs)
        number = captured.get("number", 1) + 1
        captured["number"] = number
        measurement = kwargs["measurement"]
        observable = {
            r: hybrid.seed_lobe_supported(measurement, r, s, pulse_width)
            for r, s in anchors.items()
        }
        nodes = [n for segment in result[0] for n in segment.nodes]
        links = captured["links"]
        support, forward, backward = directed_seed_support(
            nodes, links, anchors, table.samples, breaks, observable
        )
        summaries = []
        for row, sample in sorted(anchors.items()):
            cols = np.flatnonzero(table.valid[row] & (table.samples[row] == sample))
            incident = [(a[0], b[0], q) for a, b, q, _ in links if a[0] == row or b[0] == row]
            summaries.append(
                {
                    "row": row,
                    "sample": sample,
                    "observable": bool(observable[row]),
                    "candidate_count": len(cols),
                    "incident_links": incident,
                }
            )
        write_json(
            args.output / f"graph-{number}.json",
            {
                "backend_source_sha256": backend_fingerprint(),
                "config": asdict(kwargs["config"]),
                "graph_replay_only": args.replay is not None,
                "replay_source": str(args.replay.resolve()) if args.replay else None,
                "seeds": summaries,
                "pulse_width_samples": pulse_width,
                "horizontal_step_m": step,
                "nodes": len(nodes),
                "links": len(links),
                "supported_rows": int(np.count_nonzero(np.max(support, axis=1) >= 0.8)),
                "forward_rows": int(np.count_nonzero(np.max(forward, axis=1) >= 0.8)),
                "backward_rows": int(np.count_nonzero(np.max(backward, axis=1) >= 0.8)),
            },
        )
        np.savez_compressed(
            args.output / f"graph-{number}.npz",
            samples=table.samples,
            valid=table.valid,
            waveforms=table.waveforms,
            phase_classes=table.phase_classes,
            polarities=table.polarities,
            measurement=measurement,
            signal_valid=table.component_maps.get("sample_validity", np.isfinite(measurement)),
            breaks=np.array(sorted(breaks), dtype=int),
            seeds=np.array(sorted(anchors.items())),
            links=np.array([(a[0], a[1], b[0], b[1], q, d) for a, b, q, d in links]),
            support=support,
            forward=forward,
            backward=backward,
        )
        return result

    hybrid._contract_links, hybrid.correspondence_graph = contract, graph
    try:
        if args.replay:
            for source in sorted(args.replay.glob("graph-*.npz")):
                arrays = np.load(source)
                metadata = json.loads(source.with_suffix(".json").read_text())
                measurement = arrays["measurement"]
                width = metadata["pulse_width_samples"]
                table = SimpleNamespace(
                    **{
                        key: arrays[key]
                        for key in ("samples", "valid", "waveforms", "phase_classes", "polarities")
                    },
                    dense_radar_score=np.zeros_like(measurement),
                    component_maps=local_phase_motion(measurement, width),
                )
                table.component_maps["sample_validity"] = (
                    arrays["signal_valid"] if "signal_valid" in arrays else np.isfinite(measurement)
                )
                graph(
                    table,
                    {int(r): int(s) for r, s in arrays["seeds"]},
                    arrays["breaks"].tolist() if "breaks" in arrays else (),
                    width,
                    metadata["horizontal_step_m"],
                    measurement=measurement,
                    config=ConventionalConfig(**config),
                )
            return
        if args.reference is None:
            parser.error("reference or --replay is required")
        evaluate_reference(
            args.reference,
            output=args.output / "evaluation.json",
            stride=args.stride,
            methods=["seed_hybrid"],
            config=config,
        )
    finally:
        hybrid._contract_links, hybrid.correspondence_graph = original_contract, original_graph


if __name__ == "__main__":
    main()
