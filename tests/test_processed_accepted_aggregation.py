"""Processed reports must not resurrect proposals as layer measurements."""

from __future__ import annotations

import csv
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from openpyxl import load_workbook

from gpr_layer_audit.export import audit
from gpr_layer_audit.models import (
    AcquisitionFileSet,
    AnalysisResult,
    CalibrationDiagnostics,
    DielectricSource,
    InterfacePick,
    LayerSpec,
    PickSource,
    PickStatus,
    ReviewIssue,
    TrackingEvidence,
    TrackingProvenance,
    VisibilityState,
)
from gpr_layer_audit.processing.pipeline import (
    AnalysisOptions,
    _aggregate_results,
    resolve_review_issue,
)
from gpr_layer_audit.processing.processed_tracking import _refresh_result

DT = 0.1171875


def pick(order=1, row=0, sample=50, **changes):
    values = dict(
        layer_order=order,
        layer_name=f"Layer {order}",
        trace_index=row,
        chainage_m=row * 0.025,
        sample_index=float(sample),
        twtt_ns=sample * DT,
        amplitude=1.0,
        confidence=0.9,
        status=PickStatus.ACCEPTED,
        evidence=TrackingEvidence(deep_identity_support=1.0, seed_gap_support=1.0),
    )
    values.update(changes)
    return InterfacePick(**values)


def refreshed(tmp_path, picks, orders=(1, 2, 3), interval=1.0):
    layers = [LayerSpec(o, f"Layer {o}", 1, 511) for o in orders]
    chainage = np.array(sorted({item.chainage_m for item in picks}))
    radar = np.zeros((len(chainage), 512), dtype=np.float32)
    result = AnalysisResult(
        source=AcquisitionFileSet(tmp_path / "processed.DZT"),
        header=SimpleNamespace(
            position_ns=0.0,
            sample_interval_ns=DT,
            antenna="test",
            trace_count=len(chainage),
            samples_per_trace=512,
            range_ns=60.0,
            distance_per_trace_m=0.025,
        ),
        stack_size=1,
        chainage_m=chainage,
        calibrated_radargram=radar,
        surface_samples_raw=np.zeros(len(chainage)),
        reference_surface_sample=0,
        picks=picks,
        thickness=[],
        review_issues=[],
        diagnostics=CalibrationDiagnostics(False),
        parameters={
            "input_mode": "processed",
            "dielectric_by_layer": {
                str(o): {"value": 8.0, "source": "analyst"} for o in orders
            },
        },
    )
    _refresh_result(result, AnalysisOptions(layer_specs=layers, report_interval_m=interval))
    return result


def assert_withheld(item):
    assert np.isnan(item.twtt_ns)
    assert item.thickness_mm is None
    assert item.uncertainty_low_mm is None
    assert item.uncertainty_high_mm is None


def test_review_proposal_cannot_reappear_in_report_export_or_profile(tmp_path):
    proposal = pick(status=PickStatus.REVIEW, twtt_ns=float("nan"))
    original = deepcopy(proposal)
    result = refreshed(tmp_path, [proposal], orders=(1,))
    thickness = result.thickness[0]
    assert thickness.status == PickStatus.REVIEW
    assert thickness.bottom_sample == 50  # The proposal remains available for review.
    assert_withheld(thickness)
    assert result.profile[0].individual_thickness_mm is None
    assert result.profile[0].cumulative_depth_mm is None
    assert proposal.sample_index == original.sample_index
    assert proposal.status == original.status
    assert np.isnan(proposal.twtt_ns)

    csv_path, workbook_path = tmp_path / "thickness.csv", tmp_path / "audit.xlsx"
    audit._csv(result, csv_path)
    audit._workbook(result, workbook_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert np.isnan(float(row["twtt_ns"]))
    assert row["thickness_mm"] == ""
    workbook = load_workbook(workbook_path, read_only=True, data_only=True)
    for name, fields in (
        ("Thickness Results", ("twtt_ns", "thickness_mm")),
        ("Layer Profiles", ("individual_thickness_mm", "cumulative_depth_mm")),
    ):
        headers, values = list(workbook[name].values)[:2]
        row = dict(zip(headers, values, strict=True))
        assert all(row[field] is None for field in fields)
    workbook.close()

    # The strict processed repair must not silently alter the raw comparator.
    legacy = _aggregate_results(
        [proposal], [LayerSpec(1, "Asphalt", 1, 511)], result.header, 1.0,
        {1: (8.0, DielectricSource.ANALYST)}, 0,
    )[0]
    assert legacy.twtt_ns == 5.859375
    assert legacy.thickness_mm == pytest.approx(310.5253, abs=0.0001)


@pytest.mark.parametrize("invalid", [
    {"status": PickStatus.REVIEW},
    {"status": PickStatus.UNRESOLVED},
    {"visibility": VisibilityState.NOT_VISIBLE},
    {"visibility": VisibilityState.ABSENT},
    {"visibility": VisibilityState.UNCERTAIN},
    {"twtt_ns": float("nan")},
    {"sample_index": -1.0},
    {"interpolated": True},
    {"source": PickSource.INTERPOLATED},
    {"provenance": TrackingProvenance.INTERPOLATED},
])
def test_partly_accepted_bin_withholds_measurement_and_upper_dependency(tmp_path, invalid):
    result = refreshed(tmp_path, [
        pick(1, 0, 50), pick(1, 1, 52, **invalid),
        pick(2, 0, 100), pick(2, 1, 102),
    ])
    for item in result.thickness:
        assert_withheld(item)
    base = result.thickness[1]
    assert base.status == PickStatus.ACCEPTED
    assert base.bottom_sample == 101
    assert all(p.cumulative_depth_mm is None for p in result.profile)


@pytest.mark.parametrize("orders", [(2,), (3,), (1, 3)])
def test_missing_immediate_interface_is_not_replaced_by_surface_or_previous_spec(tmp_path, orders):
    result = refreshed(tmp_path, [pick(o, sample=o * 50) for o in orders], orders=orders)
    for item, profile in zip(result.thickness, result.profile, strict=True):
        assert item.status == PickStatus.ACCEPTED
        if item.layer_order > 1:
            assert_withheld(item)
            assert item.top_sample == -1
            assert profile.individual_thickness_mm is None
            assert profile.cumulative_depth_mm is None
        else:
            assert item.twtt_ns == 50 * DT
            assert profile.individual_thickness_mm is not None


@pytest.mark.parametrize("mismatch", ["chainage", "trace", "missing", "duplicate"])
def test_upper_and_lower_must_cover_identical_native_observations(tmp_path, mismatch):
    upper = [pick(1, 0, 50), pick(1, 1, 60)]
    lower = [pick(2, 0, 100), pick(2, 1, 110)]
    if mismatch == "chainage":
        lower[1].chainage_m += 0.001
    elif mismatch == "trace":
        lower[1].trace_index += 1
    elif mismatch == "missing":
        lower.pop()
    else:
        lower[1].chainage_m = lower[0].chainage_m
        lower[1].trace_index = lower[0].trace_index
    result = refreshed(tmp_path, upper + lower)
    assert result.thickness[0].thickness_mm is not None
    assert_withheld(result.thickness[1])


def test_paired_gaps_do_not_require_unrelated_upper_clicks(tmp_path):
    result = refreshed(tmp_path, [
        pick(1, row, sample, status=PickStatus.REVIEW, twtt_ns=float("nan"))
        for row, sample in enumerate((20, 30, 40))
    ] + [
        pick(2, row, sample) for row, sample in enumerate((50, 100, 150))
    ] + [
        pick(3, row, sample, status=PickStatus.HIGH_CONFIDENCE)
        for row, sample in enumerate((150, 110, 160))
    ])
    asphalt, base, subbase = result.thickness
    assert_withheld(asphalt)
    assert_withheld(base)
    assert base.status == PickStatus.ACCEPTED
    assert subbase.status == PickStatus.HIGH_CONFIDENCE
    # Median paired gaps [100, 10, 10] differs from the difference of medians [150-100].
    assert subbase.twtt_ns == 10 * DT
    assert subbase.thickness_mm is not None
    assert result.profile[2].individual_thickness_mm == subbase.thickness_mm
    assert result.profile[2].cumulative_depth_mm is None


def test_crossing_at_one_observation_cannot_be_hidden_by_bin_medians(tmp_path):
    result = refreshed(tmp_path, [
        pick(1, 0, 50), pick(1, 1, 150), pick(1, 2, 50),
        pick(2, 0, 100), pick(2, 1, 140), pick(2, 2, 100),
    ])
    assert result.thickness[1].bottom_sample > result.thickness[1].top_sample
    assert_withheld(result.thickness[1])


def test_unresolved_bin_does_not_erase_neighboring_accepted_bin(tmp_path):
    result = refreshed(tmp_path, [
        pick(1, 0, 50), pick(1, 1, 55, status=PickStatus.REVIEW, twtt_ns=float("nan")),
    ], orders=(1,), interval=0.025)
    assert result.thickness[0].status == PickStatus.ACCEPTED
    assert result.thickness[0].twtt_ns == 50 * DT
    assert result.profile[0].individual_thickness_mm is not None
    assert_withheld(result.thickness[1])
    assert result.profile[1].individual_thickness_mm is None


@pytest.mark.parametrize("action", ["absent", "not_visible", "anomaly"])
@pytest.mark.parametrize("mode", ["processed", "raw"])
def test_public_review_keeps_processed_measurement_rules(tmp_path, action, mode):
    result = refreshed(tmp_path, [
        pick(1, 0, 50), pick(1, 1, 52, visibility=VisibilityState.ABSENT),
        pick(2, 0, 100), pick(2, 1, 102),
    ], orders=(1, 2))
    assert_withheld(result.thickness[0])
    result.parameters.update(input_mode=mode, processed_stride=1)
    result.interpretation_input_radargram = np.zeros((2, 512), np.float32)
    result.sample_validity = np.ones((2, 512), bool)
    result.processed_paths = {
        2: SimpleNamespace(
            samples=np.array([100, 102]), visible=np.ones(2, bool),
            confidence=np.ones(2), evidence={}, provenance={},
            provisional_samples=np.array([100, 102]), feature=np.zeros((2, 512)),
            candidate_components={"audit_candidate_rank": np.full((2, 512), np.nan)},
        )
    }
    result.review_issues = [ReviewIssue("review", 2, "Base", 0, 0.025, [], action)]
    options = AnalysisOptions(
        layer_specs=[LayerSpec(o, f"Layer {o}", 1, 511) for o in (1, 2)],
        query_layer_orders=[],
    )
    resolve_review_issue(result, options, "review", action)
    if mode == "processed":
        assert_withheld(result.thickness[0])
        assert result.profile[0].individual_thickness_mm is None
        np.testing.assert_array_equal(result.processed_paths[2].samples, -1)
        assert not result.processed_paths[2].visible.any()
    else:
        # This fix does not change the established raw aggregation comparator.
        assert result.thickness[0].twtt_ns == 51 * DT
        assert result.thickness[0].thickness_mm is not None
