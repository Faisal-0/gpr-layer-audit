"""Acceptance calibration and promotion gates. Scores are not probabilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .processing.conventional_config import resolve_config


def calibrate_acceptance(paths, config, *, freeze=False):
    base = resolve_config(config)
    observations = {1: [], 2: [], 3: []}
    sources = []
    backends = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        group, split = data["partition"]
        if split not in ("development", "calibration"):
            raise ValueError("Held-out data cannot calibrate acceptance")
        if resolve_config(data["config"]).fingerprint != base.fingerprint:
            raise ValueError("Calibration inputs must use identical conventional settings")
        backends.add(data.get("backend_source_sha256"))
        for order, values in data["methods"]["seed_hybrid"]["layers"].items():
            if split == "calibration" and int(order) == 3:
                continue
            observations[int(order)].extend(values.get("evaluation_observations", []))
        sources.append(
            {
                "path": str(Path(path).resolve()),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "group": group,
                "split": split,
            }
        )
    thresholds, outcomes = {}, {}
    if len(backends) != 1 or None in backends:
        raise ValueError("Calibration requires one recorded backend source fingerprint")
    for order, rows in observations.items():
        choices = []
        for support in (0.8, 0.85, 0.9, 0.95, 0.99):
            for margin in (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32):
                retained = [
                    r
                    for r in rows
                    if r["proposed_sample"] >= 0
                    and r["correspondence"] >= support
                    and r["path_margin"] >= margin
                    and r["measurement_support"] >= base.minimum_measurement_support
                ]
                accuracy = (
                    sum(r["proposed_correct"] for r in retained) / len(retained) if retained else 0
                )
                if len(retained) >= 30 and accuracy >= 0.95:
                    choices.append((len(retained), accuracy, support, margin))
        if choices:
            n, accuracy, support, margin = max(choices)
            thresholds[str(order)] = {
                "minimum_correspondence": support,
                "minimum_path_margin": margin,
            }
            outcomes[str(order)] = {
                "status": "development_calibrated",
                "accepted_observations": n,
                "accepted_agreement": accuracy,
                "confidence_is_probability": False,
            }
        else:
            outcomes[str(order)] = {
                "status": "insufficient_correct_supported_observations",
                "observations": len(rows),
            }
    updated = {**config, "acceptance_by_layer": thresholds}
    complete = all(str(order) in thresholds for order in base.hybrid_layers)
    if freeze and not complete:
        raise ValueError(
            "Cannot freeze acceptance: an enabled layer has insufficient "
            "correct supported observations"
        )
    return {
        "schema": "conventional-calibration-v1",
        "config": updated,
        "layers": outcomes,
        "sources": sources,
        "settings_frozen": bool(freeze and complete),
        "configuration_sha256": resolve_config(updated).fingerprint,
        "backend_source_sha256": next(iter(backends)),
        "promoted_layers": [],
        "subbase_status": "experimental",
    }


def promotion_gate(group_metrics, *, layer_order):
    """Require three independent adequately observed roads and a useful improvement."""
    reasons = []
    seen = set()
    for result in group_metrics:
        group = result["physical_road_group"]
        if group in seen:
            reasons.append("duplicate_physical_road_group")
        seen.add(group)
        if (
            result.get("split") != "held_out"
            or result.get("observations", 0) < 30
            or not result.get("distributed_observations")
        ):
            reasons.append("insufficient_distributed_held_out_observations")
        agreement = result.get("accepted_pick_agreement") or 0
        baseline_agreement = result.get("baseline_accepted_pick_agreement") or 0
        if agreement < 0.95 or not result.get("selected_lobe_identity_checked"):
            reasons.append("accepted_correctness_gate_failed")
        if agreement < baseline_agreement:
            reasons.append("accepted_correctness_reduced")
        gain = (result.get("correct_coverage") or 0) - (
            result.get("baseline_correct_coverage") or 0
        )
        before, after = result.get("baseline_corrections"), result.get("corrections")
        fewer = before is not None and after is not None and before > 0 and after <= 0.8 * before
        if gain < 0.1 and not fewer:
            reasons.append("coverage_or_correction_improvement_gate_failed")
    if len(seen) < 3:
        reasons.append("fewer_than_three_independent_road_groups")
    if layer_order == 3:
        reasons.append("subbase_experimental_pending_independent_evidence")
    return {"eligible": not reasons, "reasons": sorted(set(reasons)), "groups": sorted(seen)}
