from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from gpr_layer_audit.ml.dataset import (
    ANNOTATION_FIELDS,
    annotation_reason,
    assign_splits,
    audit_dataset,
    build_dataset,
    make_targets,
    physical_road_group,
    transform_samples,
)
from gpr_layer_audit.ml.evaluation import calibrate_acceptance, promotion_report


def test_repeat_portion_and_duplicate_files_share_split():
    assert physical_road_group("NORTH/DASKA PASRUR BACK.PRJ/a.DZT") == "daska"
    assert physical_road_group("NORTH/SOHAL 2ND PORTION GUJRAT.PRJ/a.DZT") == "sohal-gujrat"
    rows = [
        {"sha256": h, "road_group": g}
        for h, g in [("a", "one"), ("a", "two"), ("b", "two"), ("c", "three"), ("d", "four")]
    ]
    assign_splits(rows)
    assert len({r["split"] for r in rows[:3]}) == 1
    assert len({r["road_group"] for r in rows[:3]}) == 1


def label():
    return {
        "source_sha256": "x",
        "trace_index": "3",
        "layer_order": "2",
        "sample_raw": "45",
        "visibility": "visible",
        "verified": "true",
        "origin": "manual",
        "training_use": "allowed",
        "selected_lobe": "negative_trough",
        "pulse_width_samples": "7",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_use", "prohibited"),
        ("origin", "interpolated"),
        ("origin", "auto"),
        ("verified", "false"),
        ("sample_raw", "nan"),
        ("selected_lobe", "unknown"),
    ],
)
def test_uncertain_or_withheld_labels_cannot_train(field, value):
    row = {**label(), field: value}
    assert annotation_reason(row, {"x": {"traces": 10, "samples": 128}}) is not None


def test_sample_transform_round_trip():
    raw, surface = np.array([45.0, 48.5]), np.array([14, 19])
    aligned = transform_samples(raw, surface, 16)
    np.testing.assert_allclose(transform_samples(aligned, surface, 16, inverse=True), raw)


def test_sparse_masks_do_not_teach_unlabeled_background():
    labels = [
        {**label(), "row": 3, "sample_aligned": 45},
        {**label(), "row": 5, "visibility": "not_visible"},
        {**label(), "row": 7, "visibility": "absent"},
    ]
    target, valid, visible, observed = make_targets((10, 128), labels, 2)
    assert not valid[0].any() and not valid[5].any()
    assert valid[3].all() and valid[7].all()
    assert target[3, 45] == 1 and visible[3] == 1
    assert observed[5] and visible[5] == 0


def test_audit_and_build_label_coordinates(synthetic_acquisition, tmp_path):
    road, _, _ = synthetic_acquisition
    audit = audit_dataset(road.parent)
    source = next(s for s in audit["sources"] if s["path"] == str(road.resolve()))
    annotation = {
        **label(),
        "source_sha256": source["sha256"],
        "trace_index": "30",
        "sample_raw": "136",
    }
    labels_path = tmp_path / "annotations.csv"
    with labels_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerow(annotation)
    audit = audit_dataset(road.parent, annotations=labels_path)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit))
    built = build_dataset(path, tmp_path / "cache")
    assert len(built["chunks"]) == 1
    chunk = built["chunks"][0]
    record = chunk["labels"][0]
    surface = np.load(chunk["surface_path"])
    raw = transform_samples(
        record["sample_aligned"],
        surface[record["row"]],
        chunk["reference_surface_sample"],
        inverse=True,
    )
    assert raw == 136
    assert chunk["split"] == audit["annotations"][0]["split"]


def test_training_refuses_insufficient_data_without_torch(tmp_path):
    from gpr_layer_audit.ml.training import train_model

    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"eligibility": {"2": {"eligible_for_pilot": False}}, "chunks": []}))
    with pytest.raises(ValueError, match="eligible pilot"):
        train_model(path, tmp_path / "model")


def test_promotion_requires_each_road_and_coverage_gain():
    records = [
        {
            "road_group": g,
            "layer_order": 2,
            "expected_visible": True,
            "accepted": True,
            "correct": True,
        }
        for g in ("a", "b", "c")
        for _ in range(30)
    ]
    baseline = [{**r, "accepted": i % 2 == 0} for i, r in enumerate(records)]
    assert promotion_report(records, baseline)["2"]["passed"]
    assert not promotion_report(records[:60], baseline[:60])["2"]["passed"]
    assert not promotion_report(records, records)["2"]["passed"]
    with pytest.raises(ValueError, match="test set"):
        calibrate_acceptance({"split": "test"})


def test_evaluation_only_labels_are_preserved_but_never_built_for_training(
    synthetic_acquisition, tmp_path
):
    road, _, _ = synthetic_acquisition
    initial = audit_dataset(road.parent)
    source = next(s for s in initial["sources"] if s["path"] == str(road.resolve()))
    annotation = {
        **label(),
        "source_sha256": source["sha256"],
        "trace_index": "30",
        "training_use": "evaluation_only",
        "origin": "checkpoint",
    }
    path = tmp_path / "evaluation.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerow(annotation)
    audited = audit_dataset(road.parent, annotations=path)
    assert not audited["annotations"]
    assert len(audited["evaluation_annotations"]) == 1
    assert audited["evaluation_annotations"][0]["split"] == "test"
    manifest = tmp_path / "audit.json"
    manifest.write_text(json.dumps(audited))
    assert not build_dataset(manifest, tmp_path / "cache")["chunks"]
