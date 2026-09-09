"""Separate proposal identity and timing failures after immutable guide inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from extremum_state_model import basin_map
from seeded_eval import sha

from gpr_layer_audit.conventional import _same_lobe
from gpr_layer_audit.io.dzt import DZTFile


def diagnose(folder):
    if (folder / "diagnosis.json").exists():
        raise ValueError("Preserve existing diagnosis")
    evaluation = json.loads((folder / "evaluation.json").read_text())
    manifest = evaluation["manifest"]
    entry = manifest["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    data = np.asarray(DZTFile(entry["dzt"]).channel())[::4]
    dt, dx, origin = entry["dt_ns"], entry["dx_m"] * 4, entry["origin_ns"]
    result = {
        "claim": "Post-inference development diagnosis; projection is not a deployable result",
        "script_sha256": sha(__file__),
        "evaluation_sha256": sha(folder / "evaluation.json"),
        "no_old_acceptance_reused_for_projection": True,
        "layers": {},
    }
    for order in (2, 3):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True)
        arms = {}
        for axis, weight in zip(axes, (0, 1), strict=True):
            name = f"guide{weight}-layer{order}"
            assert sha(folder / f"{name}.npz") == manifest["prediction_hashes"][f"{name}.npz"]
            records = evaluation["results"][name]["evaluation_observations"]
            tolerance = evaluation["results"][name]["tolerance_samples"]
            stats = dict(
                reviewed_nonseed=len(records),
                measured=0,
                correct=0,
                timing_only=0,
                wrong_signed_lobe=0,
                missing=0,
                projected_correct=0,
            )
            for point in records:
                row, proposed, reference = (
                    point["row"],
                    point["proposed_sample"],
                    point["reference_sample"],
                )
                if proposed < 0:
                    stats["missing"] += 1
                    continue
                stats["measured"] += 1
                same_lobe = _same_lobe(data[row], proposed, reference)
                correct = same_lobe and abs(proposed - reference) <= tolerance
                stats["correct"] += correct
                stats["timing_only"] += same_lobe and not correct
                stats["wrong_signed_lobe"] += not same_lobe
                emitted = basin_map(data[row])[0][proposed]
                stats["projected_correct"] += (
                    emitted >= 0
                    and abs(emitted - reference) <= tolerance
                    and _same_lobe(data[row], emitted, reference)
                )
            stats = {k: int(v) for k, v in stats.items()}
            assert stats["correct"] == evaluation["results"][name]["proposed_agree"]
            values = np.load(folder / f"{name}.npz")
            rr = np.array([p["row"] for p in records])
            stats["maximum_reviewed_margin"] = float(values["margin"][rr].max())
            seeds = {int(r): s for r, s in manifest["layers"][str(order)]["seeds"].items()}
            stats["bracketed_reviewed"] = sum(min(seeds) < r < max(seeds) for r in rr)
            stats["bracketed_reviewed"] = int(stats["bracketed_reviewed"])
            arms[str(weight)] = stats
            axis.imshow(
                data.T,
                cmap="gray",
                aspect="auto",
                vmin=-np.percentile(abs(data), 99),
                vmax=np.percentile(abs(data), 99),
                extent=(0, len(data) * dx, data.shape[1] * dt + origin, origin),
            )
            axis.scatter(
                rr * dx,
                np.array([p["reference_sample"] for p in records]) * dt + origin,
                c="#40b7e5",
                s=4,
                label="Reviewed reference (scoring only)",
            )
            exists = values["provisional"][rr] >= 0
            axis.scatter(
                rr[exists] * dx,
                values["provisional"][rr[exists]] * dt + origin,
                c="#e7a023",
                s=3,
                label="Unresolved proposal; none accepted",
            )
            axis.scatter(
                np.array(list(seeds)) * dx,
                np.array(list(seeds.values())) * dt + origin,
                marker="*",
                c="white",
                edgecolors="black",
                s=85,
                label="Operating seed",
            )
            axis.set_title(
                f"Jamshoro Layer {order}, guide {weight}: "
                f"{stats['correct']}/{len(records)} correct proposals; zero accepted"
            )
            axis.set_ylabel("Header time (ns)")
            axis.set_ylim(data.shape[1] * dt + origin, 0)
        axes[0].legend(loc="upper right", fontsize=8, ncol=3)
        axes[-1].set_xlabel("Native trace distance (m); unknown rows are unscored")
        fig.tight_layout()
        fig.savefig(folder / f"layer{order}-guide-comparison.png", dpi=160)
        plt.close(fig)
        result["layers"][str(order)] = arms
    (folder / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "source" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    diagnose(parser.parse_args().folder)
