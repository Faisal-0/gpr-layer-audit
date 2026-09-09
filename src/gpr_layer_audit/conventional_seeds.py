"""Reusable native observations for controlled processed-radar comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .io.dzt import fingerprint_file


def native_support(audit, label_mapping, selected):
    """Export support only; withheld reference observations are never copied here."""
    return {
        "schema": "conventional-native-seeds-v1",
        "mode": "processed",
        "dzx_sha256": audit["metadata"]["source_sha256"],
        "dzt_sha256": audit["dzt_sha256"],
        "reference_label_mapping": label_mapping,
        "observations": {
            str(order): [asdict(p) for p in picks] for order, picks in selected.items()
        },
    }


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Seed coordinates and strides must be integers; snapping is prohibited")
    return value


def load_support(path, audit, label_mapping, stride, mode):
    """Read explicit seeds or recover prior evaluation seeds without tracker output.

    Legacy evaluation samples come from the matching DZX observations at recorded
    seed rows, never from proposed/accepted paths or accuracy metrics.
    """
    if mode != "processed":
        raise ValueError("Reused native seeds currently require processed-input mode")
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") == "conventional-native-seeds-v1":
        support = document
        basis = "explicit_native_observations"
    else:
        if document.get("schema") != "conventional-evaluation-v2":
            raise ValueError("Unsupported seed-source schema")
        if document.get("mode") != "processed" or document.get("transform") is not None:
            raise ValueError("Seed-source evaluation must use processed coordinates")
        prior_audit = document["audit"]
        # Always validate the enclosing evaluation, even when it has explicit support.
        if (
            prior_audit["metadata"]["source_sha256"] != audit["metadata"]["source_sha256"]
            or prior_audit["dzt_sha256"] != audit["dzt_sha256"]
            or document["reference_label_mapping"] != label_mapping
        ):
            raise ValueError("Seed source fingerprint or layer mapping differs")
        support = document.get("seed_support")
        basis = "explicit_native_observations"
        if support is None:
            prior_stride = _integer(document["stride"])
            if prior_stride < 1:
                raise ValueError("Seed-source stride must be positive")
            selections = [
                {
                    order: sorted(_integer(r) * prior_stride for r in layer["seed_rows"])
                    for order, layer in method["layers"].items()
                }
                for method in document["methods"].values()
                if method.get("layers")
            ]
            if not selections or any(s != selections[0] for s in selections):
                raise ValueError("Seed-source methods have missing or inconsistent support")
            selected = {}
            for group in audit["metadata"]["layers"]:
                order = label_mapping.get(str(group["number"]), group["number"] + 1)
                if str(order) not in selections[0]:
                    continue
                by_row = {p["trace"]: p for p in group["picks"] if p["channel"] == 0}
                try:
                    selected[str(order)] = [by_row[r] for r in selections[0][str(order)]]
                except KeyError as exc:
                    raise ValueError("Prior seed is not an exact reviewed observation") from exc
            support = {
                "schema": "conventional-native-seeds-v1",
                "mode": "processed",
                "dzx_sha256": prior_audit["metadata"]["source_sha256"],
                "dzt_sha256": prior_audit["dzt_sha256"],
                "reference_label_mapping": document["reference_label_mapping"],
                "observations": selected,
            }
            basis = "prior_seed_rows_times_stride_and_exact_DZX_observation"
    if (
        support.get("schema") != "conventional-native-seeds-v1"
        or support.get("mode") != "processed"
    ):
        raise ValueError("Seed source requires native processed observations")
    if (
        support.get("dzx_sha256") != audit["metadata"]["source_sha256"]
        or support.get("dzt_sha256") != audit["dzt_sha256"]
        or support.get("reference_label_mapping") != label_mapping
    ):
        raise ValueError("Seed source fingerprint or layer mapping differs")
    expected = {}
    for group in audit["metadata"]["layers"]:
        order = label_mapping.get(str(group["number"]), group["number"] + 1)
        picks = [p for p in group["picks"] if p["channel"] == 0]
        if order in (1, 2, 3) and picks:
            if str(order) in expected:
                raise ValueError("Layer mapping combines multiple reviewed interfaces")
            expected[str(order)] = {(p["trace"], p["sample"], 0) for p in picks}
    if set(support["observations"]) != set(expected):
        raise ValueError("Seed source must supply every evaluated interface")
    anchors = {}
    for order, picks in support["observations"].items():
        rows = {}
        if not picks:
            raise ValueError("Seed source contains an empty interface")
        for p in picks:
            trace, sample, channel = (_integer(p[k]) for k in ("trace", "sample", "channel"))
            if (trace, sample, channel) not in expected[order]:
                raise ValueError("Seed is not an exact reviewed observation")
            if trace % stride:
                raise ValueError(
                    "Native seed is incompatible with retained stride grid; snapping is prohibited"
                )
            if trace in rows:
                raise ValueError("Seed source has duplicate observations")
            rows[trace] = sample
        anchors[int(order)] = rows
    return anchors, {
        "path": str(path.resolve()),
        "sha256": fingerprint_file(path),
        "basis": basis,
        "coordinate_policy": "exact native trace/sample; incompatible grids rejected",
    }
