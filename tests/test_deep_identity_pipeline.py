from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gpr_layer_audit.models import (
    DielectricSource,
    InterfacePick,
    LayerSpec,
    PickStatus,
    ReviewIssue,
    TrackingEvidence,
    TrackingProvenance,
)
from gpr_layer_audit.processing.pipeline import (
    AnalysisOptions,
    _aggregate_results,
    _interface_picks,
    resolve_review_issue,
)
from gpr_layer_audit.processing.seed_graph import SeedConditionedPath


def _path(
    samples: tuple[int, int],
    *,
    deep_identity: float = 0.0,
    seed_gap: float | None = None,
) -> SeedConditionedPath:
    row_count = len(samples)
    sample_array = np.asarray(samples, dtype=np.int32)
    evidence = {
        "signal_score": np.full(row_count, 0.95),
        "joint_hypothesis_support": np.full(row_count, 0.95),
        "forward_backward_agreement": np.full(row_count, 0.95),
        "neighborhood_support": np.full(row_count, 0.95),
        "cycle_slip_risk": np.full(row_count, 0.05),
        "branch_multimodality": np.full(row_count, 0.05),
        "edge_condition": np.zeros(row_count),
        "preprocessing_agreement": np.full(row_count, 0.95),
        "local_snr": np.full(row_count, 4.0),
        "alternative_cycle_margin": np.full(row_count, 0.95),
        "tracklet_support": np.full(row_count, 0.95),
        "canonical_event_sample": sample_array.astype(float),
        "deep_identity_support": np.full(row_count, deep_identity),
        "seed_gap_support": np.full(
            row_count, deep_identity if seed_gap is None else seed_gap
        ),
    }
    return SeedConditionedPath(
        samples=sample_array,
        confidence=np.full(row_count, 0.95),
        feature=np.zeros((row_count, 200), dtype=np.float32),
        alternate_samples=np.full(row_count, -1, dtype=np.int32),
        visible=np.ones(row_count, dtype=bool),
        interpolated=np.zeros(row_count, dtype=bool),
        evidence=evidence,
        signal_only_samples=sample_array.copy(),
        design_guided_samples=np.full(row_count, -1, dtype=np.int32),
        design_conflict=np.zeros(row_count, dtype=bool),
        design_constrained=False,
    )


def _pipeline_outputs(deep_identity: float, seed_gap: float | None = None):
    layers = LayerSpec.defaults()[:2]
    paths = {
        1: _path((110, 111)),
        2: _path(
            (150, 151), deep_identity=deep_identity, seed_gap=seed_gap
        ),
    }
    picks = _interface_picks(
        paths,
        layers,
        np.zeros((2, 200), dtype=float),
        reference_surface_sample=80,
        sample_interval_ns=0.03,
        trace_centres=np.asarray([0.0, 1.0]),
        chainage=np.asarray([0.0, 1.0]),
        anchors={1: {0: 110}, 2: {0: 150}},
        confidence_threshold=0.45,
    )
    thickness = _aggregate_results(
        picks,
        layers,
        SimpleNamespace(sample_interval_ns=0.03),
        report_interval_m=10.0,
        dielectric_by_layer={
            1: (7.0, DielectricSource.ANALYST),
            2: (7.0, DielectricSource.ANALYST),
        },
        reference_surface_sample=80,
    )
    return picks, thickness


@pytest.mark.parametrize(
    ("deep_identity", "seed_gap"),
    ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
)
def test_both_deep_identity_signals_are_required_for_automatic_thickness(
    deep_identity: float, seed_gap: float
):
    picks, thickness = _pipeline_outputs(deep_identity, seed_gap)

    base_seed, base_auto = [item for item in picks if item.layer_order == 2]
    assert base_seed.status == PickStatus.ACCEPTED
    assert np.isfinite(base_seed.twtt_ns)
    assert base_auto.status == PickStatus.REVIEW
    assert np.isnan(base_auto.twtt_ns)
    assert "Deep reflector identity" in (base_auto.review_reason or "")
    assert base_auto.evidence.deep_identity_support == deep_identity
    assert base_auto.evidence.seed_gap_support == seed_gap

    base_thickness = next(item for item in thickness if item.layer_order == 2)
    assert base_thickness.status == PickStatus.REVIEW
    assert np.isnan(base_thickness.twtt_ns)
    assert base_thickness.thickness_mm is None


def test_deep_identity_evidence_allows_supported_automatic_thickness():
    picks, thickness = _pipeline_outputs(deep_identity=1.0)

    base_seed, base_auto = [item for item in picks if item.layer_order == 2]
    assert base_seed.status == PickStatus.ACCEPTED
    assert base_auto.status == PickStatus.HIGH_CONFIDENCE
    assert np.isfinite(base_auto.twtt_ns)
    assert base_auto.evidence.deep_identity_support == 1.0
    assert base_auto.evidence.seed_gap_support == 1.0

    base_thickness = next(item for item in thickness if item.layer_order == 2)
    assert base_thickness.status == PickStatus.HIGH_CONFIDENCE
    assert np.isfinite(base_thickness.twtt_ns)
    assert base_thickness.thickness_mm is not None


def test_asphalt_acceptance_is_independent_of_deep_identity_fields():
    picks, _ = _pipeline_outputs(deep_identity=0.0)

    asphalt_auto = [item for item in picks if item.layer_order == 1][1]
    assert asphalt_auto.status == PickStatus.HIGH_CONFIDENCE
    assert np.isfinite(asphalt_auto.twtt_ns)


def test_manual_review_confirmation_promotes_only_the_frozen_interval():
    layers = LayerSpec.defaults()[:2]
    picks = [
        InterfacePick(
            1,
            "Asphalt",
            row,
            chainage,
            110.0,
            0.9,
            1.0,
            0.9,
            PickStatus.HIGH_CONFIDENCE,
            selected_lobe_sample=110.0,
        )
        for row, chainage in enumerate((0.2, 1.2))
    ]
    picks.extend(
        InterfacePick(
            2,
            "Base",
            row,
            chainage,
            -1.0,
            float("nan"),
            float("nan"),
            0.0,
            PickStatus.UNRESOLVED,
            evidence=TrackingEvidence(
                guided_graph_selected_sample=display,
                canonical_event_sample=canonical,
            ),
        )
        for row, (chainage, display, canonical) in enumerate(
            ((0.2, 148.0, 150.0), (1.2, 149.0, 151.0))
        )
    )
    issue = ReviewIssue(
        "confirm-one-bin",
        2,
        "Base",
        0.2,
        0.2,
        ["Direct family support is missing"],
        "Confirm after inspection",
    )
    result = SimpleNamespace(
        picks=picks,
        review_issues=[issue],
        calibrated_radargram=np.zeros((2, 200)),
        reference_surface_sample=80,
        header=SimpleNamespace(sample_interval_ns=0.03),
        candidate_events=[],
        anomaly_regions=[],
        parameters={
            "required_seed_orders": [2],
            "dielectric_by_layer": {
                "1": {"value": 7.0, "source": "analyst"},
                "2": {"value": 7.0, "source": "analyst"},
            },
        },
    )
    options = AnalysisOptions(
        layer_specs=layers,
        report_interval_m=1.0,
        auto_fine_retrack=False,
    )

    resolve_review_issue(result, options, issue.issue_id, "accept")

    inside, outside = [item for item in result.picks if item.layer_order == 2]
    assert inside.status == PickStatus.ACCEPTED
    assert inside.provenance == TrackingProvenance.MANUAL_CORRECTION
    assert inside.selected_lobe_sample == 148.0
    assert inside.sample_index == 150.0
    assert np.isfinite(inside.twtt_ns)
    assert outside.sample_index == -1.0
    base = [item for item in result.thickness if item.layer_order == 2]
    assert base[0].status == PickStatus.ACCEPTED
    assert base[0].thickness_mm is not None
    assert base[1].thickness_mm is None
