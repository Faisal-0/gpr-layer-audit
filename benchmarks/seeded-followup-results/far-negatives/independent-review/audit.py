"""Read-only independent audit; writes only this directory's reports.

Run from the repository: .venv/Scripts/python.exe PATH/TO/THIS/audit.py
Requires the frozen v2 and prepared v3 artifacts. Adds prediction checks if ready.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
ROOT = HERE.parents[3]
OLD = NEW.with_name("patchnet-v2-mask")
sys.path.insert(0, str(ROOT / "scripts"))
import experiment_patchnet_far_negatives as far
from gpr_layer_audit.ml.processed_correspondence import interval_weights

base = far.base


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_array(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def sampler_checks():
    rng = np.random.default_rng(104729)
    seed_checks = 0
    for _ in range(1000):
        anchors = np.sort(rng.choice(100000, 3, replace=False))
        rows = np.r_[rng.integers(-100, 100100, 30), (anchors[:-1] + anchors[1:]) / 2]
        expected = np.array([min(anchors, key=lambda r: (abs(r-row), r)) for row in rows])
        actual = anchors[np.argmax(interval_weights(rows, anchors), axis=1)]
        np.testing.assert_array_equal(actual, expected)
        seed_checks += len(rows)
    for _ in range(1000):
        candidates = np.arange(100)
        targets = rng.integers(-1, 2, (100, 2)).astype(float)
        reference = int(rng.integers(100))
        width = int(rng.integers(1, 15))
        correlations = rng.normal(size=100)
        actual = far.negative_indices(candidates, targets, reference, width, correlations)
        eligible = np.flatnonzero(targets[:, 0] == 0)
        near = sorted((i for i in eligible if abs(i-reference) <= 4*width),
                      key=lambda i: abs(i-reference))[:3]
        remote = sorted((i for i in eligible if abs(i-reference) > 4*width),
                        key=lambda i: -correlations[i])[:3]
        # Equal-distance near ties need not be stable; their distance multiset must agree.
        assert sorted(abs(actual[:len(near)] - reference)) == sorted(abs(np.array(near) - reference))
        assert list(actual[len(near):len(near)+len(remote)]) == remote
        assert len(actual) == min(6, len(eligible)) == len(set(actual))
        assert np.all(targets[actual, 0] == 0)
    c = np.arange(100)
    y = np.zeros((100, 2))
    ncc = np.ones(100)
    selected = far.negative_indices(c, y, 50, 2, ncc)
    assert list(selected[3:]) == [0, 1, 2]  # Stable far-score ties.
    return {"nearest_seed_cases_including_midpoint_ties": seed_checks,
            "random_sampler_cases": 1000, "stable_far_ncc_tie": True}


def source_checks():
    contract = read(NEW / "contract.json")
    manifest = read(NEW / "sampling-source.json")
    assert contract["dependencies"] == far.dependency_contract(ROOT)
    for name, expected in manifest["scripts"].items():
        assert sha(NEW / "source" / name) == expected
    for relative, expected in manifest["package"]["python_files"].items():
        assert sha(NEW / "source/package" / relative) == expected
    assert hashlib.sha256(far.prepared_source().encode()).hexdigest() == contract["prepare_function_sha256"]
    assert sha(ROOT / "scripts/experiment_patchnet_far_negatives.py") == contract["sampling_runner_sha256"]
    original_contract = copy.deepcopy(base.CONTRACT)
    original_sha = base.sha
    reached = []

    def tampered_hash(path):
        if Path(path).name == "patchnet_model.py":
            return "0" * 64
        return original_sha(path)

    try:
        with patch.object(base, "sha", tampered_hash), patch.object(base, "train", lambda: reached.append(True)):
            try:
                far.run("train", NEW, ROOT)
            except ValueError as exc:
                assert "contract" in str(exc)
            else:
                raise AssertionError("Changed model source hash was not rejected")
        assert not reached
    finally:
        base.CONTRACT = original_contract
    return {"source_snapshot_files": len(manifest["scripts"]) + len(manifest["package"]["python_files"]),
            "model_source_mutation_guard_rejects": True,
            "package_sha256": contract["dependencies"]["package"]["sha256"],
            "runtime": contract["dependencies"]["runtime"]}


def training_checks():
    assert read(OLD / "inputs.json") == read(NEW / "inputs.json")
    old_reports = read(OLD / "training-data.json")["reports"]
    new_reports = read(NEW / "training-data.json")["reports"]
    for old, new in zip(old_reports, new_reports, strict=True):
        for field in ("file", "layer", "sampled_rows", "rows_without_training_pair", "positive_timing", "tolerance_samples", "pulse"):
            assert old[field] == new[field]
    snapshots = []
    for directory in (OLD, NEW):
        info = read(directory / "training-data.json")
        assert sha(directory / "training.npz") == info["training_sha256"]
        with np.load(directory / "training.npz", allow_pickle=False) as data:
            y, partner, seeds = (data[name] for name in ("y", "partner", "seeds"))
            x = data["x"]
            positive = y[:, 0] == 1
            snapshots.append({"pairs": len(y), "positive_count": int(positive.sum()),
                              "positive_x_sha256": digest_array(x[positive]),
                              "positive_y_sha256": digest_array(y[positive]),
                              "positive_partner_sha256": digest_array(partner[positive]),
                              "seed_patches_sha256": digest_array(seeds),
                              "seed_patch_count": len(seeds),
                              "unknown_targets": int((y < 0).any(axis=1).sum())})
            del x
    for field in snapshots[0]:
        if field != "pairs":
            assert snapshots[0][field] == snapshots[1][field]
    return {"frozen_inputs_and_seeds_exact": True, "old": snapshots[0], "new": snapshots[1],
            "report_pair_deltas": [{"file": Path(n["file"]).name, "layer": n["layer"],
                                    "delta": n["pairs"]-o["pairs"]}
                                   for o, n in zip(old_reports, new_reports, strict=True)]}


def explain_missing_negative():
    entry = next(e for e in read(NEW / "inputs.json") if e["road"] == "mandiali" and "_002" in e["dzt"])
    radar = base.DZTFile(entry["dzt"])
    data = np.asarray(radar.channel(), np.float32)
    valid = base.processed_boundary_mask(data)
    dt, dx = entry["dt_ns"], entry["dx_m"]
    surface = int(np.floor(-entry["origin_ns"] / dt))
    observable = base.native_candidates(data, valid, surface)
    anchors = {int(k): v for k, v in entry["seeds"]["2"].items()}
    pulse = base.resolve_pulse(data, valid, anchors, {}, dt, base.ConventionalConfig())
    tolerance = max(2, pulse.lobe_samples / 4)
    bank, packet_valid = base.dense._packet_correlations(data, valid, anchors, pulse.context_radius)
    layer = next(layer for layer in base.read_dzx(entry["dzx"]).layers if layer.number + 1 == 2)
    reviewed = sorted((p for p in layer.picks if p.channel == 0 and p.trace not in anchors), key=lambda p: p.trace)
    records = []
    for point in reviewed:
        candidates = np.flatnonzero(observable[point.trace])
        target = base.targets(data[point.trace], candidates, point.sample, tolerance)
        positive = np.flatnonzero(target[:, 0] == 1)
        old_negative = np.flatnonzero((target[:, 0] == 0) & (abs(candidates-point.sample) <= 4*pulse.lobe_samples))
        if not len(positive) or not len(old_negative):
            continue
        positive = positive[np.argsort(abs(candidates[positive]-point.sample))[:3]]
        partner = min(anchors, key=lambda r: (abs(r-point.trace), r))
        new_negative = far.negative_indices(candidates, target, point.sample, pulse.lobe_samples, bank[partner][point.trace, candidates])
        old_negative = old_negative[np.argsort(abs(candidates[old_negative]-point.sample))[:6]]
        losses = {}
        for name, negative in (("old", old_negative), ("new", new_negative)):
            chosen = np.r_[positive, negative]
            x, usable = base.patches(data, valid, np.full(len(chosen), point.trace), candidates[chosen], dt, dx)
            losses[name] = [{"sample": int(candidates[chosen[i]]),
                             "target": target[chosen[i]].tolist(),
                             "patch_support_fraction": float(x[i, 1].mean()),
                             "packet_valid": bool(packet_valid[point.trace, candidates[chosen[i]]])}
                            for i in np.flatnonzero(~usable)]
        if len(losses["old"]) != len(losses["new"]):
            records.append({"trace": point.trace, "reference_sample": point.sample,
                            "partner_seed_trace": partner, **losses})
    return {"file": Path(entry["dzt"]).name, "layer": 2, "changed_validity_rows": records}


def prediction_checks():
    directory = NEW / "jamshoro-predictions"
    if not (directory / "evaluation.json").exists():
        return {"ready": False}
    provenance = read(directory / "prediction-manifest.json")
    for name, expected in provenance["prediction_hashes"].items():
        assert sha(directory / name) == expected
    candidate_checks = []
    seed_checks = 0
    radar = np.asarray(base.DZTFile(provenance["input"]["dzt"]).channel())[::4]
    metadata = base.read_dzx(provenance["input"]["dzx"])
    independent = {}
    for order in (2, 3):
        anchors = {int(row): sample for row, sample in provenance["layers"][str(order)]["seeds"].items()}
        tolerance = max(2, provenance["layers"][str(order)]["lobe_samples"] / 4)
        layer = next(layer for layer in metadata.layers if layer.number + 1 == order)
        references = [p for p in layer.picks if p.channel == 0 and p.trace % 4 == 0 and p.trace // 4 not in anchors]
        for arm in ("classical", "learned"):
            with np.load(directory / f"{arm}-layer{order}.npz") as new, np.load(OLD / "jamshoro-predictions" / f"{arm}-layer{order}.npz") as old:
                np.testing.assert_array_equal(new["candidate"], old["candidate"])
                candidate_checks.append(f"{arm}-{order}")
                if arm == "classical":
                    for key in old.files:
                        np.testing.assert_array_equal(new[key], old[key])
                for row, sample in provenance["layers"][str(order)]["seeds"].items():
                    assert new["samples"][int(row)] == sample
                    seed_checks += 1
                regions = {}
                for region in ("all", "bracketed", "tail"):
                    counts = dict(reviewed=0, correct_proposals=0, accepted=0, accepted_correct=0,
                                  wrong_lobe_proposals=0, same_lobe_timing_misses=0)
                    for point in references:
                        row, reference = point.trace // 4, point.sample
                        bracketed = min(anchors) < row < max(anchors)
                        if (region == "bracketed" and not bracketed) or (region == "tail" and bracketed):
                            continue
                        counts["reviewed"] += 1
                        sign = np.sign(radar[row, reference])

                        def same_lobe(sample):
                            left, right = sorted((int(sample), reference))
                            return sign != 0 and np.all(np.sign(radar[row, left:right+1]) == sign)

                        proposal = new["provisional"][row]
                        if proposal >= 0:
                            family = same_lobe(proposal)
                            agree = family and abs(proposal-reference) <= tolerance
                            counts["correct_proposals"] += int(agree)
                            counts["wrong_lobe_proposals"] += int(not family)
                            counts["same_lobe_timing_misses"] += int(family and not agree)
                        accepted = new["samples"][row] >= 0 and new["visible"][row]
                        counts["accepted"] += int(accepted)
                        if accepted:
                            sample = new["samples"][row]
                            counts["accepted_correct"] += int(same_lobe(sample) and abs(sample-reference) <= tolerance)
                    regions[region] = counts
                independent[f"{arm}-{order}"] = regions
    evaluation = read(directory / "evaluation.json")
    old_evaluation = read(OLD / "jamshoro-predictions/evaluation.json")
    summary = {}
    for key, metrics in evaluation["results"].items():
        summary[key] = {name: value for name, value in metrics.items() if not isinstance(value, (dict, list))}
        for independent_key, reported_key in (("reviewed", "observations_excluding_seeds"),
                                               ("correct_proposals", "proposed_agree"),
                                               ("accepted", "accepted"),
                                               ("accepted_correct", "accepted_agree")):
            assert independent[key]["all"][independent_key] == metrics[reported_key]
        if key.startswith("classical"):
            assert metrics == old_evaluation["results"][key]
    return {"ready": True, "candidate_parity_arms": candidate_checks,
            "seed_array_checks": seed_checks, "classical_predictions_and_metrics_exact": True,
            "metrics": summary, "independent_signed_lobe_rescore": independent}


def main():
    report = {"audit_source_sha256": sha(Path(__file__)),
              "claim": "Historical development roads; interpretation agreement only",
              "sampler": sampler_checks(), "source": source_checks(),
              "training": training_checks(), "training_pair_difference": explain_missing_negative(),
              "prediction": prediction_checks(),
              "recommendation": "Reject promotion and Gujrat transfer; base proposals regress, subbase remains poor, and both layers accept zero automatic observations. Preserve the isolated experiment.",
              "denominator": "Reviewed nonseed native traces divisible by four; bracketed versus tail evaluated separately. Unknown rows excluded, no continuous-road or physical-thickness claim.",
              "limitations": [
                  "inputs.json is not enforced by the stage guard; this audit independently confirms exact equality to frozen v2 inputs.",
                  "Six negatives are selected before patch validity filtering; actual retained count can be lower.",
                  "No active-query policy is implemented or modified by this experiment.",
                  "Local CNN scores and interval-normalized dense margins remain uncalibrated.",
                  "Legacy graph-loss zero fields are unavailable instrumentation, not measured losses.",
              ]}
    (HERE / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
