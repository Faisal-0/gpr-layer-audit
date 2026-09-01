"""Pure metrics and evidence-schema tests for the blinded evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.models import LabelOrigin, ReferencePoint
from scripts.evaluate_blinded_multiroad import (
    _load_frozen_case,
    _observation,
    _path_arrays,
    _path_is_visible,
    _selective_risk_report,
    _tracking_orders,
    _wilson_interval,
)


def _reference(chainage: float, order: int, thickness: float, origin=LabelOrigin.MANUAL):
    return ReferencePoint(
        road_id="test-road",
        line_id="test-road",
        chainage_m=chainage,
        layer_order=order,
        cumulative_depth_mm=thickness * order,
        individual_thickness_mm=thickness,
        label_origin=origin,
        source_path=Path("reference.xlsx"),
        source_sheet="Sheet1",
        source_cell=f"A{int(chainage) + 1}",
    )


def _path(samples, *, deep=None, gap=None, visible=None):
    samples = np.asarray(samples, dtype=np.int32)
    evidence = {
        "graph_selected_sample": samples.astype(float),
        "canonical_event_sample": samples.astype(float),
    }
    if deep is not None:
        evidence["deep_identity_support"] = np.asarray(deep, dtype=float)
    if gap is not None:
        evidence["seed_gap_support"] = np.asarray(gap, dtype=float)
    return SimpleNamespace(
        samples=samples,
        alternate_samples=samples.copy(),
        visible=(
            samples >= 0
            if visible is None
            else np.asarray(visible, dtype=bool)
        ),
        interpolated=np.zeros(samples.shape, dtype=bool),
        confidence=np.ones(samples.shape, dtype=float),
        design_conflict=np.zeros(samples.shape, dtype=bool),
        evidence=evidence,
    )


def test_wilson_interval_is_bounded_and_empty_is_unavailable():
    interval = _wilson_interval(5, 10)

    assert interval is not None
    assert 0.0 < interval[0] < 0.5 < interval[1] < 1.0
    assert _wilson_interval(0, 0) is None
    with pytest.raises(ValueError, match="between zero"):
        _wilson_interval(11, 10)


def test_path_arrays_persist_identity_support_and_acceptance_evidence():
    full = {
        2: _path(
            [120, 121],
            deep=[1.0, 0.0],
            gap=[1.0, 1.0],
        )
    }

    result = _path_arrays(full, (2,))[2]

    assert np.array_equal(result["deep_identity_support"], [1.0, 0.0])
    assert np.array_equal(result["seed_gap_support"], [1.0, 1.0])
    assert np.array_equal(result["visible"], [True, True])
    assert np.array_equal(result["confidence"], [1.0, 1.0])


def test_selective_report_requires_both_interfaces_and_deep_support():
    chainage = np.arange(0.0, 101.0, 10.0)
    samples = np.full(chainage.shape, 130, dtype=np.int32)
    samples[7] = -1
    deep = np.ones(chainage.shape, dtype=float)
    deep[5] = 0.0
    paths = {
        1: _path(np.full(chainage.shape, 90, dtype=np.int32)),
        2: _path(samples, deep=deep, gap=np.ones(chainage.shape)),
    }
    case = {"stations": [{"chainage_m": 0.0}, {"chainage_m": 100.0}]}
    points = [
        _reference(40.0, 2, 40.0),
        _reference(50.0, 2, 40.0),
        _reference(70.0, 2, 40.0),
        _reference(20.0, 2, 40.0, LabelOrigin.INTERPOLATED),
    ]

    result = _selective_risk_report(
        case,
        points,
        chainage,
        paths,
        80.0,
        2,
        {"status": "accepted", "fitted_mm_per_sample": 1.0},
    )

    assert result["status"] == "evaluated"
    assert result["eligible_manual_checkpoint_count"] == 3
    assert result["accepted_count"] == 1
    assert result["coverage"] == pytest.approx(1 / 3)
    assert result["review_count"] == 1
    assert result["unresolved_count"] == 1
    assert result["within_target_precision"] == pytest.approx(1.0)
    assert result["median_absolute_error_mm"] == pytest.approx(0.0)
    states = [item["state"] for item in result["checkpoints"]]
    assert states == ["accepted", "review", "unresolved"]


def test_subbase_acceptance_requires_identity_support_on_base_and_subbase():
    chainage = np.arange(0.0, 101.0, 10.0)
    paths = {
        1: _path(np.full(chainage.shape, 90, dtype=np.int32)),
        2: _path(
            np.full(chainage.shape, 130, dtype=np.int32),
            deep=np.zeros(chainage.shape),
            gap=np.ones(chainage.shape),
        ),
        3: _path(
            np.full(chainage.shape, 170, dtype=np.int32),
            deep=np.ones(chainage.shape),
            gap=np.ones(chainage.shape),
        ),
    }
    case = {"stations": [{"chainage_m": 0.0}, {"chainage_m": 100.0}]}
    points = [_reference(40.0, 3, 40.0)]

    result = _selective_risk_report(
        case,
        points,
        chainage,
        paths,
        80.0,
        3,
        {"status": "accepted", "fitted_mm_per_sample": 1.0},
    )

    assert result["accepted_count"] == 0
    assert result["review_count"] == 1
    checkpoint = result["checkpoints"][0]
    assert checkpoint["upper_identity_support_ok"] is False
    assert checkpoint["state"] == "review"


def test_nonfinite_or_explicitly_hidden_paths_are_not_visible_measurements():
    invalid = {
        "samples": np.asarray([np.nan, np.inf, 120.0]),
        "canonical": np.asarray([120.0, 120.0, np.inf]),
        "visible": np.asarray([True, True, False]),
    }
    paths = {2: invalid}

    assert not _path_is_visible(invalid, 0)
    assert not _path_is_visible(invalid, 1)
    assert not _path_is_visible(invalid, 2)
    assert _observation(paths, 2, 0) == (None, False)
    assert _observation(paths, 2, 1) == (None, False)
    assert _observation(paths, 2, 2) == (None, False)


def test_selective_report_rejects_reversed_interface_ordering():
    chainage = np.arange(0.0, 101.0, 10.0)
    paths = {
        1: _path(np.full(chainage.shape, 130, dtype=np.int32)),
        2: _path(
            np.full(chainage.shape, 120, dtype=np.int32),
            deep=np.ones(chainage.shape),
            gap=np.ones(chainage.shape),
        ),
    }
    result = _selective_risk_report(
        {"stations": [{"chainage_m": 0.0}, {"chainage_m": 100.0}]},
        [_reference(40.0, 2, 40.0)],
        chainage,
        paths,
        80.0,
        2,
        {"status": "accepted", "fitted_mm_per_sample": 1.0},
    )

    assert result["accepted_count"] == 0
    assert result["coverage"] == 0.0
    assert result["unresolved_count"] == 1
    assert result["checkpoints"][0]["state"] == "unresolved"


def test_legacy_deep_npz_is_explicitly_unavailable_not_accepted(tmp_path):
    case_id = "legacy"
    chainage = np.arange(0.0, 101.0, 10.0)
    np.savez_compressed(
        tmp_path / f"{case_id}-paths.npz",
        chainage_m=chainage,
        layer_1_samples=np.full(chainage.shape, 90, dtype=np.int32),
        layer_1_alternate=np.full(chainage.shape, 90, dtype=np.int32),
        layer_1_graph=np.full(chainage.shape, 90, dtype=float),
        layer_1_canonical=np.full(chainage.shape, 90, dtype=float),
        layer_2_samples=np.full(chainage.shape, 130, dtype=np.int32),
        layer_2_alternate=np.full(chainage.shape, 130, dtype=np.int32),
        layer_2_graph=np.full(chainage.shape, 130, dtype=float),
        layer_2_canonical=np.full(chainage.shape, 130, dtype=float),
    )
    (tmp_path / f"{case_id}-summary.json").write_text(
        json.dumps({"dropout_audit": []}), encoding="utf-8"
    )
    loaded_chainage, paths, _ = _load_frozen_case(case_id, (1, 2), tmp_path)
    points = [_reference(40.0, 2, 40.0)]
    result = _selective_risk_report(
        {"stations": [{"chainage_m": 0.0}, {"chainage_m": 100.0}]},
        points,
        loaded_chainage,
        paths,
        80.0,
        2,
        {"status": "accepted", "fitted_mm_per_sample": 1.0},
    )

    assert "deep_identity_support" in result["missing_evidence_fields"]
    assert "seed_gap_support" in result["missing_evidence_fields"]
    assert result["status"] == "unavailable"
    assert result["accepted_count"] == 0
    assert result["coverage"] is None


def test_old_manifest_without_validation_layers_defaults_to_asphalt_and_base():
    assert _tracking_orders({"case_id": "daska"}) == (1, 2)
