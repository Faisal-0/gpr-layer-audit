"""Post-inference dense-route diagnosis; oracle references never enter a tracker.

Reads an already scored immutable patchnet experiment and writes a separate report.
Oracle coverage counts only reviewed nonseed observations and respects exact anchors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter1d

from gpr_layer_audit.conventional import _same_lobe
from gpr_layer_audit.io.dzt import DZTFile


def maximum_correct_route(correct, surface, anchors, max_shift):
    """Exact label-only oracle on the retained dense displacement graph.

    All positions include latent gaps just as the comparator does. Only the
    supplied correct-observation mask earns credit; latent positions earn none.
    This is an upper bound, not a deployable path or a score calibration.
    """
    scores = np.zeros(correct.shape[1], np.int32)
    impossible = -len(correct) - 1
    for row, reward in enumerate(correct):
        if row:
            scores = maximum_filter1d(scores, 2 * max_shift + 1, mode="constant", cval=impossible)
        scores = scores + reward
        scores[: surface + 1] = impossible
        if row in anchors:
            selected = scores[anchors[row]]
            scores[:] = impossible
            scores[anchors[row]] = selected
    value = int(scores.max())
    return value if value >= 0 else None


def compact_metrics(rows, dx, dt):
    accepted = [p for p in rows if p["accepted"]]
    good = sum(p["accepted_correct"] for p in accepted)
    wrong = len(accepted) - good
    switches = 0
    current = longest = 0
    previous = -2
    for point in sorted(rows, key=lambda p: p["row"]):
        if point["accepted"] and not point["accepted_correct"]:
            current = current + 1 if point["row"] == previous + 1 else 1
            longest = max(longest, current)
        else:
            current = 0
        previous = point["row"]
        switches += point.get("accepted_wrong_signed_lobe", False)
    errors = [abs(p["proposed_sample"] - p["reference_sample"]) for p in accepted]
    count = len(rows)
    return {
        "reviewed_nonseed_denominator": count,
        "accepted_correct": good,
        "accepted_wrong": wrong,
        "unresolved": count - len(accepted),
        "proposed_correct": sum(p["proposed_correct"] for p in rows),
        "accepted_agreement": good / len(accepted) if accepted else None,
        "correct_coverage": good / count if count else None,
        "wrong_coverage": wrong / count if count else None,
        "unresolved_coverage": (count - len(accepted)) / count if count else None,
        "accepted_wrong_signed_lobe_observations": int(switches),
        "longest_adjacent_wrong_sample_footprint_m": longest * dx,
        "median_accepted_error_ns": float(np.median(errors) * dt) if errors else None,
        "correct_sample_footprint_m": good * dx,
        "wrong_sample_footprint_m": wrong * dx,
    }


def run(source, output):
    from experiment_patchnet_dense import read, sha, write

    if output.exists():
        raise ValueError("Preserve an existing diagnostic")
    output.mkdir(parents=True)
    manifest = read(source / "jamshoro-predictions/prediction-manifest.json")
    evaluation = read(source / "jamshoro-predictions/evaluation.json")
    for name, expected in manifest["prediction_hashes"].items():
        assert sha(source / "jamshoro-predictions" / name) == expected
    entry = manifest["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    data = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)[::4]
    dx, dt = entry["dx_m"] * 4, entry["dt_ns"]
    surface = int(np.floor(-entry["origin_ns"] / dt))
    max_shift = int(np.ceil(2 * dx / dt))
    result = {
        "claim": "Development interpretation only; exact graph oracle is not inference",
        "script_sha256": sha(__file__),
        "evaluation_sha256": sha(source / "jamshoro-predictions/evaluation.json"),
        "model_sha256": manifest["model_sha256"],
        "dzt_sha256": entry["dzt_sha256"],
        "missing_evidence": {
            "correspondence_survived": "Not instrumented; legacy zero is not a graph loss",
            "measurement_support": "Use saved candidate/visibility arrays, not legacy default zero",
        },
        "denominators": "Observed nonseed grid rows; unknowns excluded, seeds never automatic",
        "layers": {},
    }
    for order in (2, 3):
        records = evaluation["results"][f"classical-{order}"]["evaluation_observations"]
        seeds = {
            int(row): sample for row, sample in manifest["layers"][str(order)]["seeds"].items()
        }
        arrays = np.load(source / f"jamshoro-predictions/classical-layer{order}.npz")
        candidate = np.isfinite(arrays["candidate"])
        correct = np.zeros(data.shape, bool)
        tolerance = evaluation["results"][f"classical-{order}"]["tolerance_samples"]
        for point in records:
            row, sample = point["row"], point["reference_sample"]
            columns = np.flatnonzero(candidate[row])
            columns = columns[abs(columns - sample) <= tolerance]
            for column in columns:
                correct[row, column] = _same_lobe(data[row], column, sample)
        diagnostic = {
            "original_maximum_step_samples": max_shift,
            "correct_candidate_rows": int(correct.any(axis=1).sum()),
            "original_graph_maximum_correct_rows": maximum_correct_route(
                correct, surface, seeds, max_shift
            ),
            "double_step_oracle_maximum_correct_rows": maximum_correct_route(
                correct, surface, seeds, 2 * max_shift
            ),
            "arms": {},
        }
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, sharey=True)
        vmax = np.percentile(abs(data[:, surface + 1 :]), 99)
        for axis, arm in zip(axes, ("classical", "learned"), strict=True):
            arm_records = evaluation["results"][f"{arm}-{order}"]["evaluation_observations"]
            arrays = np.load(source / f"jamshoro-predictions/{arm}-layer{order}.npz")
            assert np.array_equal(candidate, np.isfinite(arrays["candidate"]))
            for p in arm_records:
                p["accepted_wrong_signed_lobe"] = bool(
                    p["accepted"]
                    and not _same_lobe(data[p["row"]], p["proposed_sample"], p["reference_sample"])
                )
            metrics = compact_metrics(arm_records, dx, dt)
            metrics["bracketed"] = compact_metrics(
                [p for p in arm_records if min(seeds) <= p["row"] <= max(seeds)], dx, dt
            )
            metrics["tails"] = compact_metrics(
                [p for p in arm_records if not min(seeds) <= p["row"] <= max(seeds)], dx, dt
            )
            rr = np.array([p["row"] for p in arm_records])
            exists = arrays["provisional"][rr] >= 0
            correlation = arrays["correlation"][rr]
            margin = arrays["margin"][rr]
            metrics["gate_diagnosis"] = {
                "measured_proposals": int(exists.sum()),
                "measured_above_correlation_gate": int((exists & (correlation >= 0.5)).sum()),
                "measured_above_margin_gate": int((exists & (margin >= 0.02)).sum()),
                "maximum_reviewed_margin": float(margin.max()),
                "correct_proposals_above_correlation_gate": sum(
                    p["proposed_correct"] and correlation[i] >= 0.5
                    for i, p in enumerate(arm_records)
                ).item(),
            }
            diagnostic["arms"][arm] = metrics
            axis.imshow(
                data.T,
                origin="upper",
                aspect="auto",
                cmap="gray",
                vmin=-vmax,
                vmax=vmax,
                extent=(
                    0,
                    len(data) * dx,
                    data.shape[1] * dt + entry["origin_ns"],
                    entry["origin_ns"],
                ),
            )
            reference = np.array([p["reference_sample"] for p in arm_records])
            axis.scatter(
                rr * dx,
                reference * dt + entry["origin_ns"],
                s=2,
                c="#55c5ff",
                label="Reviewed reference (scoring only)",
            )
            good = np.array([p["accepted_correct"] for p in arm_records])
            accepted = np.array([p["accepted"] for p in arm_records])
            for mask, color, label in (
                (exists & ~accepted, "#f9b44a", "Unresolved proposal"),
                (accepted & good, "#21d36b", "Accepted agreeing"),
                (accepted & ~good, "#ff3d55", "Accepted wrong"),
            ):
                axis.scatter(
                    rr[mask] * dx,
                    arrays["provisional"][rr[mask]] * dt + entry["origin_ns"],
                    s=3,
                    c=color,
                    label=label,
                )
            axis.scatter(
                np.array(list(seeds)) * dx,
                np.array(list(seeds.values())) * dt + entry["origin_ns"],
                marker="*",
                s=95,
                c="white",
                edgecolors="black",
                label="Operating seed",
            )
            axis.set_title(
                f"Jamshoro Layer {order} / {arm}: 0 accepted; "
                f"{metrics['proposed_correct']}/{len(records)} agreeing proposals"
            )
            axis.set_ylabel("Header time (ns)")
            axis.set_ylim(data.shape[1] * dt + entry["origin_ns"], 0)
        axes[0].legend(loc="upper right", fontsize=8, ncol=3)
        axes[-1].set_xlabel("Native trace distance (m); gaps without reviewed rows have no score")
        fig.tight_layout()
        fig.savefig(output / f"layer{order}-comparison.png", dpi=160)
        plt.close(fig)
        result["layers"][str(order)] = diagnostic
        print(order, json.dumps(diagnostic), flush=True)
    write(output / "diagnosis.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("exports/seeded-tracker/patchnet-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
