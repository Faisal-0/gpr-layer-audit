"""Post-inference development diagnosis; oracle labels never enter a tracker run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from seeded_eval import sha

from gpr_layer_audit.conventional import _same_lobe
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.processing.seeded_challenger import dense_interval_dp


def oracle_recoverable(data, candidate, references, seeds, tolerance, shift, surface):
    """Maximum reviewed agreement on unchanged displacement graph, diagnostic only.

    Replace the entire local objective by one unit of reward for each agreeing
    eligible observation. Every latent state and displacement edge is retained;
    seed and surface constraints are unchanged. This deliberately uses labels,
    so its routes and scores are separate from production predictions.
    """
    rows, samples = data.shape
    reward = np.zeros(data.shape, np.float32)
    retained = 0
    for row, sample in references.items():
        if row in seeds:
            continue
        eligible = np.flatnonzero(np.isfinite(candidate[row]))
        good = [
            c for c in eligible if abs(c - sample) <= tolerance and _same_lobe(data[row], c, sample)
        ]
        reward[row, good] = 1
        retained += bool(good)
    bounds = sorted(seeds)
    spans = [(0, bounds[0]), *zip(bounds[:-1], bounds[1:], strict=False), (bounds[-1], rows - 1)]
    total = 0
    records = []
    route = np.full(rows, -1, np.int32)
    for left, right in spans:
        if left == right:
            continue
        unary = -reward[left : right + 1].astype(float)
        unary[:, : surface + 1] = np.inf
        transition = np.zeros((right - left, 2 * shift + 1, samples), np.float32)
        picked, costs = dense_interval_dp(unary, transition, seeds.get(left), seeds.get(right))
        if np.any(picked < 0):
            raise ValueError("Frozen native seed route is infeasible even with oracle scores")
        gain = int(round(-costs[-1, picked[-1]]))
        total += gain
        route[left : right + 1] = picked
        records.append(
            {
                "left": left,
                "right": right,
                "maximum_correct": gain,
                "bracketed": left in seeds and right in seeds,
            }
        )
    return {
        "candidate_retained": retained,
        "maximum_graph_correct": total,
        "retained_route_all_correct_feasible": total == retained,
        "intervals": records,
    }, route


def summaries(points, seeds, data, dx, dt, tolerance):
    output = {}
    for name, subset in (
        ("all", points),
        ("bracketed", [p for p in points if min(seeds) < p["row"] < max(seeds)]),
        ("tails", [p for p in points if p["row"] < min(seeds) or p["row"] > max(seeds)]),
    ):
        n = len(subset)
        accepted = sum(p["accepted"] for p in subset)
        correct = sum(p["accepted_correct"] for p in subset)
        counts = {"correct": 0, "same_lobe_timing_error": 0, "wrong_signed_lobe": 0, "hidden": 0}
        errors = []
        for p in subset:
            row, sample, target = p["row"], p["proposed_sample"], p["reference_sample"]
            if sample < 0:
                key = "hidden"
            elif not _same_lobe(data[row], sample, target):
                key = "wrong_signed_lobe"
            elif abs(sample - target) > tolerance:
                key = "same_lobe_timing_error"
            else:
                key = "correct"
            counts[key] += 1
            if sample >= 0:
                errors.append(abs(sample - target) * dt)
        bad_rows = sorted(p["row"] for p in subset if p["accepted"] and not p["accepted_correct"])
        longest, current, previous = 0, 0, -2
        for row in bad_rows:
            current = current + 1 if row == previous + 1 else 1
            longest = max(longest, current)
            previous = row
        output[name] = {
            "reviewed_nonseed_rows": n,
            "accepted": accepted,
            "accepted_correct": correct,
            "accepted_precision": correct / accepted if accepted else None,
            "correct_accepted_coverage": correct / n if n else None,
            "wrong_accepted_coverage": (accepted - correct) / n if n else None,
            "unresolved_coverage": (n - accepted) / n if n else None,
            "proposal_categories": counts,
            "proposal_absolute_time_error_ns_quantiles": np.quantile(errors, [0.5, 0.95]).tolist()
            if errors
            else None,
            "longest_contiguous_wrong_accepted_footprint_m": longest * dx,
            "semantic_switch_events": None,
        }
    return output


def run(folder):
    destination = folder / "diagnosis"
    if destination.exists():
        raise ValueError("Preserve existing diagnosis")
    destination.mkdir()
    contract = json.loads((folder / "contract.json").read_text())
    predictions = json.loads((folder / "prediction.json").read_text())
    for name, digest in predictions["hashes"].items():
        assert sha(folder / name) == digest
    evaluation = json.loads((folder / "evaluation.json").read_text())["results"]
    entry = contract["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    data = np.asarray(DZTFile(entry["dzt"]).channel())[:: contract["stride"]]
    dt, dx, origin = entry["dt_ns"], entry["dx_m"] * contract["stride"], entry["origin_ns"]
    surface = int(np.floor(-origin / dt))
    shift = json.loads((folder / "edges.json").read_text())["max_shift"]
    report = {
        "script_sha256": sha(__file__),
        "input_sha256": entry["dzt_sha256"],
        "prediction_manifest_sha256": sha(folder / "prediction.json"),
        "evaluation_sha256": sha(folder / "evaluation.json"),
        "layers": {},
        "claim": "Development interpretation, one road; no independent-road uncertainty claim",
        "oracle": "Labels used only here after predictions; replaces objective, never deployable",
        "switch_limit": "Wrong signed lobe counts observations, not semantic transitions",
        "length_limit": "Observed footprints only; unknown labels are not background",
    }
    for order in (2, 3):
        seeds = {int(r): s for r, s in contract["layers"][str(order)]["seeds"].items()}
        control = evaluation[f"control-layer{order}"]
        points = control["evaluation_observations"]
        references = {p["row"]: p["reference_sample"] for p in points}
        candidates = np.load(folder / f"control-layer{order}.npz")["candidate"]
        oracle, route = oracle_recoverable(
            data, candidates, references, seeds, control["tolerance_samples"], shift, surface
        )
        np.savez_compressed(destination / f"DIAGNOSTIC-ORACLE-layer{order}.npz", route=route)
        result = {"diagnostic_oracle": oracle, "arms": {}}
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=True)
        vmax = np.percentile(abs(data), 99)
        for axis, arm in zip(axes, ("control", "ncc", "cnn"), strict=True):
            metrics = evaluation[f"{arm}-layer{order}"]
            arm_points = metrics["evaluation_observations"]
            result["arms"][arm] = summaries(
                arm_points, seeds, data, dx, dt, metrics["tolerance_samples"]
            )
            previous = {p["row"]: p["proposed_correct"] for p in points}
            gains = [
                p["row"] for p in arm_points if p["proposed_correct"] and not previous[p["row"]]
            ]
            losses = [
                p["row"] for p in arm_points if not p["proposed_correct"] and previous[p["row"]]
            ]
            result["arms"][arm]["gains_over_control_rows"] = gains
            result["arms"][arm]["losses_from_control_rows"] = losses
            result["arms"][arm]["blocks_50m"] = [
                {
                    "start_m": float(start),
                    **summaries(
                        [p for p in arm_points if start <= p["row"] * dx < start + 50],
                        seeds,
                        data,
                        dx,
                        dt,
                        metrics["tolerance_samples"],
                    )["all"],
                }
                for start in np.arange(0, len(data) * dx, 50)
            ]
            rr = np.array([p["row"] for p in arm_points])
            pp = np.array([p["proposed_sample"] for p in arm_points])
            tt = np.array([p["reference_sample"] for p in arm_points])
            accepted = np.array([p["accepted"] for p in arm_points])
            good = np.array([p["accepted_correct"] for p in arm_points])
            axis.imshow(
                data.T,
                cmap="gray",
                aspect="auto",
                vmin=-vmax,
                vmax=vmax,
                extent=(
                    -dx / 2,
                    (len(data) - 0.5) * dx,
                    origin + (data.shape[1] - 0.5) * dt,
                    origin - dt / 2,
                ),
            )
            axis.scatter(rr * dx, tt * dt + origin, s=3, c="#46b8f0", label="Reviewed")
            for mask, color, label in (
                ((pp >= 0) & ~accepted, "#e4a025", "Unresolved proposal"),
                (accepted & good, "#20c55e", "Accepted agreeing"),
                (accepted & ~good, "#ef3b45", "Accepted wrong"),
            ):
                axis.scatter(rr[mask] * dx, pp[mask] * dt + origin, s=4, c=color, label=label)
            axis.scatter(
                np.array(list(seeds)) * dx,
                np.array(list(seeds.values())) * dt + origin,
                marker="*",
                s=65,
                color="white",
                edgecolors="black",
                label="Native seed",
            )
            axis.set_title(
                f"Layer {order}, {arm}: "
                f"{metrics['proposed_agree']}/{len(points)} correct proposals; "
                f"{metrics['accepted_agree']}/{metrics['accepted']} agreeing accepted"
            )
            axis.set_ylabel("Header time (ns)")
            axis.set_ylim(
                max(tt.max(), pp.max()) * dt + origin + 0.5,
                min(tt.min(), min(seeds.values())) * dt + origin - 0.5,
            )
        axes[0].legend(ncol=5, fontsize=8)
        axes[-1].set_xlabel("Native distance (m); all available data is development evidence")
        fig.tight_layout()
        fig.savefig(destination / f"layer{order}-comparison.png", dpi=150)
        plt.close(fig)
        report["layers"][str(order)] = result
        print("oracle", order, oracle, flush=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (destination / "executed-diagnosis.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    run(parser.parse_args().folder.resolve())
