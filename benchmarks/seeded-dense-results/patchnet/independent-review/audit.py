"""Independent read-only contract audit; writes its own review artifact only."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from patchnet_model import native_candidates
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.io.dzx import read_dzx

OUT = Path(__file__).resolve().parent
EXP = OUT.parent
PRED = EXP / "jamshoro-predictions"

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

manifest = read(PRED / "prediction-manifest.json")
evaluation = read(PRED / "evaluation.json")
source = read(EXP / "source.json")
checks = {
    "source_scripts_match_snapshot": all(sha(ROOT / "scripts" / key) == value for key, value in source["scripts"].items()),
    "prediction_hashes_match": all(sha(PRED / key) == value for key, value in manifest["prediction_hashes"].items()),
    "model_hash_matches": sha(EXP / "model.pt") == manifest["model_sha256"],
}
data = np.asarray(DZTFile(manifest["input"]["dzt"]).channel())[::4]
metadata = read_dzx(manifest["input"]["dzx"])
results = {}
for layer in (2, 3):
    anchors = {int(k): v for k, v in manifest["layers"][str(layer)]["seeds"].items()}
    tolerance = max(2, manifest["layers"][str(layer)]["lobe_samples"] / 4)
    points = next(x for x in metadata.layers if x.number + 1 == layer).picks
    points = [p for p in points if p.channel == 0 and p.trace % 4 == 0 and p.trace // 4 not in anchors]
    with np.load(PRED / f"classical-layer{layer}.npz") as source_arrays:
        first = {key: source_arrays[key] for key in source_arrays.files}
    with np.load(PRED / f"learned-layer{layer}.npz") as source_arrays:
        second = {key: source_arrays[key] for key in source_arrays.files}
    checks[f"layer{layer}_common_candidates"] = bool(np.array_equal(first["candidate"], second["candidate"], equal_nan=True))
    for arm, arrays in (("classical", first), ("learned", second)):
        key = f"{arm}-{layer}"
        checks[f"{key}_exact_seeds"] = all(arrays["samples"][r] == s and arrays["visible"][r] for r, s in anchors.items())
        stats = {}
        for region in ("all", "bracketed", "tail"):
            counts = dict(reviewed=0, proposed=0, proposed_correct=0, proposed_wrong_lobe=0,
                          proposed_same_lobe_timing_miss=0, accepted=0, accepted_correct=0,
                          correct_candidate_retained=0)
            margins = []
            for point in points:
                row, sample = point.trace // 4, point.sample
                bracketed = min(anchors) < row < max(anchors)
                if region == "bracketed" and not bracketed or region == "tail" and bracketed:
                    continue
                counts["reviewed"] += 1
                trace = data[row]
                sign = np.sign(trace[sample])
                def same_lobe(candidate):
                    left, right = sorted((int(candidate), sample))
                    return sign != 0 and np.all(np.sign(trace[left:right+1]) == sign)
                proposal = arrays["provisional"][row]
                if proposal >= 0:
                    counts["proposed"] += 1
                    family = bool(same_lobe(proposal))
                    correct = family and abs(proposal - sample) <= tolerance
                    counts["proposed_correct"] += int(correct)
                    counts["proposed_wrong_lobe"] += int(not family)
                    counts["proposed_same_lobe_timing_miss"] += int(family and not correct)
                accepted = arrays["samples"][row] >= 0 and arrays["visible"][row]
                counts["accepted"] += int(accepted)
                if accepted:
                    picked = arrays["samples"][row]
                    counts["accepted_correct"] += int(same_lobe(picked) and abs(picked-sample) <= tolerance)
                candidates = np.flatnonzero(np.isfinite(arrays["candidate"][row]))
                candidates = candidates[np.abs(candidates-sample) <= tolerance]
                counts["correct_candidate_retained"] += int(any(same_lobe(c) for c in candidates))
                margins.append(arrays["margin"][row])
            counts["margin_max"] = float(np.max(margins))
            counts["margin_q50_q90_q99"] = np.quantile(margins, [.5, .9, .99]).tolist()
            counts["rows_passing_fixed_margin"] = int(np.sum(np.array(margins) >= .02))
            counts["unresolved"] = counts["reviewed"]-counts["accepted"]
            stats[region] = counts
        published = evaluation["results"][key]
        checks[f"{key}_scoring_reproduced"] = all(stats["all"][ours] == published[theirs] for ours, theirs in (
            ("reviewed", "observations_excluding_seeds"), ("proposed_correct", "proposed_agree"),
            ("accepted", "accepted"), ("accepted_correct", "accepted_agree"),
            ("correct_candidate_retained", "candidate_retained")))
        results[key] = stats

probe = np.array([[0,1,2,3,4,5,6,7,0]], np.float32)
mask = np.ones_like(probe, bool)
mask[0,4] = False
before = native_candidates(probe, mask, -1)
probe[0,4] = 1e30
after = native_candidates(probe, mask, -1)
from gpr_layer_audit.processing.seeded_challenger import dense_interval_dp
normalization_case = []
for rows in (101, 1001):
    _, margins = dense_interval_dp(np.tile([0., .065, .1], (rows, 1)), np.zeros((rows - 1, 3, 3)), 0, 0)
    difference = float(margins[rows // 2, 2] - margins[rows // 2, 0])
    normalization_case.append({"rows": rows, "total_detour_cost": difference, "normalized_margin": difference / (rows * .1), "passes_fixed_gate": difference / (rows * .1) >= .02})
report = {
    "checks": checks,
    "all_contract_checks_pass": all(checks.values()),
    "results": results,
    "tests": ["python -m pytest tests/test_patchnet_experiment.py -q: 3 passed", "python -m pytest tests/test_seeded_challenger.py -q: 13 passed"],
    "denominator": "Reviewed nonseed native rows divisible by 4 only; .1m observation spacing, unknown rows excluded, no continuous full-road-coverage claim",
    "reported_zero_fields_are_unavailable": ["correspondence_survived", "correspondence_survival", "evaluation_observations[].correspondence", "evaluation_observations[].measurement_support"],
    "length_normalization_counterexample": normalization_case,
    "masked_neighbor_candidate_counterexample": {
        "invalid_index": 4,
        "before_valid_candidates": np.flatnonzero(before[0]).tolist(),
        "after_valid_candidates": np.flatnonzero(after[0]).tolist(),
        "meaning": "find_peaks sees invalid amplitudes before center masking; unsupported neighbors can change candidate retention. Both arms share this, so comparative scoring remains valid."
    },
    "recommendation": "Reject application integration and Gujrat transfer: zero accepted coverage in both arms, base proposal accuracy regresses, subbase remains 3.39%. Keep isolated reproducible experiment.",
    "limitations": [
        "No untouched-road validation; Jamshoro historical layer identities only.",
        "Candidate retention is local availability, not end-to-end correct-route feasibility.",
        "Training nearest-seed pairing and fixed NCC/CNN blend were not calibrated or proven to preserve reflector identity.",
        "Exact dense margins normalized by entire seed interval are uncalibrated; long-interval gates can fail mechanically even where a local proposal is correct.",
        "No production change or interaction replay is justified by this learned arm."
    ]
}
(OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"checks": checks, "results": results, "counterexample": report["masked_neighbor_candidate_counterexample"]}, indent=2))
