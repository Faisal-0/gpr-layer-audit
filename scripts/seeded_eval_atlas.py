"""Post-inference reviewed checkpoint atlas; never imported by production tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exports/seeded-tracker/baseline-source/src"))

from gpr_layer_audit.conventional import _same_lobe  # noqa: E402
from gpr_layer_audit.io.dzt import DZTFile  # noqa: E402


def analyze(reference, graph, layer):
    run = json.loads(reference.read_text())
    metadata_path = graph.with_suffix(".json")
    graph_metadata = json.loads(metadata_path.read_text())
    graph_digest = hashlib.sha256(graph.read_bytes()).hexdigest()
    if graph_metadata.get("sha256") != graph_digest:
        raise ValueError("Graph capture hash differs from its metadata")
    if (
        hashlib.sha256(Path(run["audit"]["dzt_path"]).read_bytes()).hexdigest()
        != run["audit"]["dzt_sha256"]
    ):
        raise ValueError("Radar fingerprint differs from the evaluated input")
    values = run["methods"]["seed_hybrid"]["layers"][str(layer)]
    radar = DZTFile(run["audit"]["dzt_path"]).channel()[:: run["stride"]]
    step = run["audit"]["dx_m"] * run["stride"]
    with np.load(graph) as archive:
        table, valid = archive["samples"], archive["valid"]
        expected_anchors = {
            (p["trace"] // run["stride"], p["sample"])
            for p in run["seed_support"]["observations"][str(layer)]
        }
        if set(map(tuple, archive["anchors"].tolist())) != expected_anchors:
            raise ValueError("Graph anchors differ from the evaluated native seeds")
        if (
            len(table) != len(radar)
            or graph_metadata["dt_ns"] != run["audit"]["dt_ns"]
            or graph_metadata["step_m"] != step
        ):
            raise ValueError("Graph and scored radar coordinate systems differ")
        if (
            graph_metadata["pulse_width_samples"]
            != run["scoring_pulses"][str(layer)]["lobe_samples"]
        ):
            raise ValueError("Graph pulse differs from frozen initial seed width")
        all_nodes, offsets, supports = archive["nodes"], archive["offsets"], archive["support"]
        node_segments = {}
        for i, segment in enumerate(archive["segment_ids"]):
            for row, col in all_nodes[offsets[i] : offsets[i + 1]]:
                node_segments[int(row), int(col)] = (int(segment), float(supports[i]))
    predictions = np.load(reference.with_name(reference.stem + "-seed_hybrid.npz"))
    accepted = predictions[f"layer{layer}_accepted"]
    checkpoints = []
    for observation in values["evaluation_observations"]:
        row, sample = observation["row"], observation["reference_sample"]
        cols = [
            int(c)
            for c in np.flatnonzero(valid[row])
            if abs(table[row, c] - sample) <= values["tolerance_samples"]
            and _same_lobe(radar[row], table[row, c], sample)
        ]
        segments = [node_segments[row, c] for c in cols if (row, c) in node_segments]
        support = max((s for _, s in segments), default=0)
        if not cols:
            stage = "candidate_missing"
        elif not segments:
            stage = "candidate_lost_before_contracted_graph"
        elif support <= 0:
            stage = "zero_segment_seed_support"
        elif not observation["proposed_correct"]:
            stage = "selected_route_wrong_or_missing"
        elif not observation["accepted_correct"]:
            stage = "correct_proposal_rejected_by_final_gate"
        else:
            stage = "correct_accepted"
        lo, hi = max(0, sample - 50), min(radar.shape[1], sample + 51)
        context_peak = max(float(np.max(abs(radar[row, lo:hi]))), 1)
        checkpoints.append(
            {
                **observation,
                "native_trace": row * run["stride"],
                "distance_m": row * step,
                "first_observed_loss_stage": stage,
                "matching_candidate_samples": [int(table[row, c]) for c in cols],
                "retained_segment_ids": [s for s, _ in segments],
                "maximum_matching_segment_support": support,
                "reference_amplitude_over_context_max": float(abs(radar[row, sample]))
                / context_peak,
                "accepted_sample": int(accepted[row]),
                "bracketed": min(values["seed_rows"]) < row < max(values["seed_rows"]),
            }
        )
    selected = []
    for stage in dict.fromkeys(p["first_observed_loss_stage"] for p in checkpoints):
        available = [p for p in checkpoints if p["first_observed_loss_stage"] == stage]
        selected.append(min(available, key=lambda p: p["reference_amplitude_over_context_max"]))
    wrong = [p for p in checkpoints if p["accepted"] and not p["accepted_correct"]]
    if wrong:
        selected.append(max(wrong, key=lambda p: abs(p["accepted_sample"] - p["reference_sample"])))
    selected = sorted({p["row"]: p for p in selected}.values(), key=lambda p: p["row"])
    predictions.close()
    return (
        {
            "layer": layer,
            "reviewed_nonseed_observations": len(checkpoints),
            "stage_counts": dict(Counter(p["first_observed_loss_stage"] for p in checkpoints)),
            "selected_checkpoints": selected,
            "all_checkpoint_stages": [
                {"native_trace": p["native_trace"], "stage": p["first_observed_loss_stage"]}
                for p in checkpoints
            ],
            "reference": str(reference),
            "graph": str(graph),
            "graph_sha256": graph_digest,
            "graph_metadata": graph_metadata,
            "evaluation_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "input_sha256": run["audit"]["dzt_sha256"],
            "backend_source_sha256": run["backend_source_sha256"],
            "inference_config": run["config"],
            "identity_experiment": run.get("identity_experiment"),
            "scoring_tolerance_samples": values["tolerance_samples"],
            "full_operating_seed_context": run["seed_support"]["observations"][str(layer)],
        },
        run,
        radar,
        step,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--graph-base", type=Path, required=True)
    parser.add_argument("--graph-subbase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    layers = []
    for layer, graph in ((2, args.graph_base), (3, args.graph_subbase)):
        result, run, radar, step = analyze(args.reference, graph, layer)
        layers.append(result)
    asphalt_control = None
    asphalt = run["methods"]["seed_hybrid"]["layers"].get("1")
    if asphalt is not None:
        correct = {p["row"]: p for p in asphalt["evaluation_observations"] if p["accepted_correct"]}
        ordered = np.array(sorted(correct), dtype=int)
        if len(ordered):
            groups = np.split(ordered, np.flatnonzero(np.diff(ordered) > 1) + 1)
            longest = max(groups, key=len)
            row = int(longest[len(longest) // 2])
            with np.load(
                args.reference.with_name(args.reference.stem + "-seed_hybrid.npz")
            ) as arrays:
                accepted_sample = int(arrays["layer1_accepted"][row])
            asphalt_control = {
                **correct[row],
                "native_trace": row * run["stride"],
                "distance_m": row * step,
                "accepted_sample": accepted_sample,
                "first_observed_loss_stage": "established_shallow_easy_control",
                "matching_candidate_samples": [],
                "maximum_matching_segment_support": None,
                "selection_rule": "Midpoint of longest observed agreeing accepted shallow run",
                "contiguous_agreeing_observed_span_m": len(longest) * step,
                "deep_graph_diagnosis": "Not applicable to the established shallow engine",
                "full_operating_seed_context": run["seed_support"]["observations"]["1"],
            }
    document = {
        "schema": "seeded-failure-atlas-v1",
        "claim": "Development-only post-inference diagnostic; not deployable evidence",
        "limits": (
            "Segment support is a bottleneck heuristic, not semantic probability. "
            "Graph capture starts after candidate construction and cannot separate earlier "
            "candidate generation from pruning. A graph-supported candidate does not prove "
            "a complete reference-consistent route. No absence/permanent-disappearance "
            "labels are available, so those states are not manufactured from missing picks."
        ),
        "layers": layers,
        "asphalt_easy_control": asphalt_control,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [(r["layer"], point) for r in layers for point in r["selected_checkpoints"]]
    if asphalt_control is not None:
        selected.insert(0, (1, asphalt_control))
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 2.6 * len(selected)), squeeze=False)
    for (left, right), (layer, point) in zip(axes, selected, strict=True):
        row, sample = point["row"], point["reference_sample"]
        xlo, xhi = max(0, row - round(3 / step)), min(len(radar), row + round(3 / step) + 1)
        shown_samples = [sample]
        if point["proposed_sample"] >= 0:
            shown_samples.append(point["proposed_sample"])
        if point["accepted"]:
            shown_samples.append(point["accepted_sample"])
        ylo, yhi = max(0, min(shown_samples) - 20), min(radar.shape[1], max(shown_samples) + 21)
        limit = max(float(np.percentile(abs(radar[xlo:xhi, ylo:yhi]), 97)), 1)
        left.imshow(
            radar[xlo:xhi, ylo:yhi].T,
            cmap="gray",
            aspect="auto",
            vmin=-limit,
            vmax=limit,
            extent=(xlo * step, xhi * step, yhi, ylo),
        )
        left.scatter([row * step], [sample], marker="+", s=90, c="cyan", label="Reference")
        seeds = run["seed_support"]["observations"][str(layer)]
        local = [p for p in seeds if xlo <= p["trace"] // run["stride"] < xhi]
        left.scatter(
            [p["trace"] * run["audit"]["dx_m"] for p in local],
            [p["sample"] for p in local],
            marker="x",
            c="magenta",
            label="Seeds",
        )
        left.axvline(row * step, color="cyan", lw=0.6)
        left.set(
            title=(
                f"Layer {layer}, native trace {point['native_trace']}:\n"
                + point["first_observed_loss_stage"].replace("_", " ")
            ),
            xlabel="Distance (m)",
            ylabel="Stored sample",
        )
        right.plot(radar[row, ylo:yhi], np.arange(ylo, yhi), color="black", lw=1)
        right.axhline(sample, color="#009bb8", label="Reviewed lobe")
        for value in point["matching_candidate_samples"]:
            right.axhline(value, color="green", lw=0.5, alpha=0.5)
        if point["proposed_sample"] >= 0:
            right.axhline(point["proposed_sample"], color="orange", label="Proposed")
        if point["accepted"]:
            color = "green" if point["accepted_correct"] else "red"
            right.axhline(point["accepted_sample"], color=color, ls="--", label="Accepted")
        right.set_ylim(yhi, ylo)
        right.set(
            title=(
                "Established shallow engine; deep graph diagnosis not applicable"
                if point["maximum_matching_segment_support"] is None
                else f"Candidate count {len(point['matching_candidate_samples'])}; "
                f"max segment support {point['maximum_matching_segment_support']:.3f}"
            ),
            xlabel="Stored amplitude",
            ylabel="Stored sample",
        )
        right.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".png"), dpi=120)
    plt.close(fig)
    print(json.dumps({r["layer"]: r["stage_counts"] for r in layers}))


if __name__ == "__main__":
    main()
