from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.ml.dataset import (
    ANNOTATION_FIELDS,
    PROCESSED_ANNOTATION_FIELDS,
    annotation_reason,
    build_dataset,
    export_confirmed_annotations,
)


def confirmed_result(*, processed):
    result = SimpleNamespace(
        source=SimpleNamespace(fingerprint="known-source"),
        chainage_m=np.array([0.0, 0.1]),
        picks=[SimpleNamespace(chainage_m=0.1, trace_index=4)],
        surface_samples_raw=np.array([12.0, 15.0]),
        reference_surface_sample=10.0,
        parameters={"input_mode": "processed" if processed else "raw"},
    )
    station = SimpleNamespace(
        station_id="observation",
        chainage_m=0.1,
        user_confirmed={2: True, 3: True},
        visibility={2: "visible", 3: "not_visible"},
        samples={2: 45.5, 3: None},
        role="correction",
        selected_lobe={2: "negative_trough"},
        pulse_width_samples={2: 7},
    )
    return result, station


def export_rows(tmp_path, *, processed):
    result, station = confirmed_result(processed=processed)
    path = tmp_path / "observations.csv"
    assert (
        export_confirmed_annotations(
            result,
            [station],
            path,
            current_station_ids={("observation", 2), ("observation", 3)},
        )
        == 2
    )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_raw_export_keeps_legacy_schema_and_inverse_surface_coordinates(tmp_path):
    fields, rows = export_rows(tmp_path, processed=False)
    assert fields == list(ANNOTATION_FIELDS)
    assert rows[0]["sample_raw"] == "50.5"
    assert rows[1]["sample_raw"] == ""
    assert all(row["training_use"] == "allowed" for row in rows)
    sources = {"known-source": {"traces": 10, "samples": 128}}
    assert all(annotation_reason(row, sources) is None for row in rows)


def test_processed_export_preserves_native_sample_and_visibility_without_raw_transform(tmp_path):
    fields, rows = export_rows(tmp_path, processed=True)
    assert fields == list(PROCESSED_ANNOTATION_FIELDS)
    assert rows[0]["sample_processed"] == "45.5"
    assert rows[0]["trace_index"] == "4"
    assert rows[1]["sample_processed"] == "" and rows[1]["visibility"] == "not_visible"
    assert all(row["sample_raw"] == "" and row["input_mode"] == "processed" for row in rows)
    assert all(row["training_use"] == "incompatible_processed_coordinates" for row in rows)
    assert all(row["origin"] == "manual_correction" and row["verified"] == "true" for row in rows)


@pytest.mark.parametrize("permission", ["allowed", "evaluation_only", "prohibited"])
def test_known_source_and_authorization_cannot_make_processed_coordinates_raw(tmp_path, permission):
    _, rows = export_rows(tmp_path, processed=True)
    sources = {"known-source": {"traces": 10, "samples": 128}}
    for row in rows:
        changed = {**row, "training_use": permission, "sample_raw": "45.5"}
        assert annotation_reason(changed, sources, evaluation=True) == (
            "incompatible_processed_coordinates"
        )
    # Either explicit coordinate marker independently prevents silent reinterpretation.
    changed.pop("input_mode")
    changed["sample_processed"] = "45.5"
    assert annotation_reason(changed, sources, evaluation=True) == (
        "incompatible_processed_coordinates"
    )


def test_raw_dataset_builder_rejects_processed_export_before_opening_known_source(tmp_path):
    _, rows = export_rows(tmp_path, processed=True)
    source = {
        "sha256": "known-source",
        "traces": 10,
        "samples": 128,
        "road_group": "one-road",
        "split": "train",
        "path": str(tmp_path / "must-not-be-opened.DZT"),
    }
    annotations = [
        {**row, "training_use": "allowed", "road_group": "one-road", "split": "train"}
        for row in rows
    ]
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps({"schema_version": 1, "sources": [source], "annotations": annotations})
    )
    with pytest.raises(ValueError, match="incompatible_processed_coordinates"):
        build_dataset(path, tmp_path / "cache")


def test_processed_provenance_cannot_fall_back_to_raw_when_input_mode_is_missing(tmp_path):
    result, station = confirmed_result(processed=True)
    result.parameters = {"coordinate_provenance": {"mode": "processed"}}
    del result.surface_samples_raw
    del result.reference_surface_sample
    path = tmp_path / "processed.csv"
    export_confirmed_annotations(result, [station], path, current_station_ids={("observation", 2)})
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["sample_raw"] == "" and row["sample_processed"] == "45.5"
