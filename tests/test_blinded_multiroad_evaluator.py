from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_blinded_multiroad import (
    _assess_scale,
    _dropout_assessment,
    _release_assessment,
    _thickness_evaluation,
)


def _scale_record(
    value: float,
    *,
    local: bool = True,
    implied_dielectric: float = 7.0,
) -> dict[str, object]:
    return {
        "eligible_local_reference": local,
        "mm_per_sample": value,
        "implied_dielectric": implied_dielectric,
    }


def test_scale_gate_requires_two_local_references():
    result = _assess_scale(
        [_scale_record(1.7), _scale_record(1.8, local=False)]
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "insufficient_local_manual_scale_references"
    assert result["fitted_mm_per_sample"] is None


def test_scale_gate_rejects_inconsistent_local_ratios():
    result = _assess_scale(
        [_scale_record(1.5), _scale_record(2.0), _scale_record(2.7)]
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "local_scale_ratios_exceed_relative_deviation_limit"
    assert result["fitted_mm_per_sample"] is None


def test_scale_gate_accepts_consistent_local_ratios():
    result = _assess_scale(
        [_scale_record(1.70), _scale_record(1.74), _scale_record(1.84)]
    )

    assert result["status"] == "accepted"
    assert result["fitted_mm_per_sample"] == pytest.approx(1.74)
    assert result["maximum_relative_deviation"] < 0.15


def test_scale_gate_rejects_consistent_but_nonphysical_conversion():
    result = _assess_scale(
        [
            _scale_record(7.0, implied_dielectric=0.4),
            _scale_record(7.1, implied_dielectric=0.38),
            _scale_record(7.2, implied_dielectric=0.37),
        ]
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "implied_dielectric_outside_physical_bounds"


def test_dropout_gate_only_assesses_declared_validation_layers():
    audit = [
        {
            "layer_order": 1,
            "within_one_pulse_width": True,
            "absolute_sample_difference": 1.0,
        },
        {
            "layer_order": 1,
            "within_one_pulse_width": True,
            "absolute_sample_difference": 0.0,
        },
        {
            "layer_order": 2,
            "within_one_pulse_width": False,
            "absolute_sample_difference": 14.0,
        },
    ]

    result = _dropout_assessment(audit, {1})

    assert set(result) == {"1"}
    assert result["1"]["all_within_one_pulse"] is True


def _release_case(case_id: str, checkpoints: int = 30) -> dict[str, object]:
    values = [
        {
            "automatically_visible": index < 27,
            "within_release_target": index < 26,
        }
        for index in range(checkpoints)
    ]
    return {
        "case_id": case_id,
        "thickness_validation_by_layer": {
            "1": {
                "status": "evaluated",
                "eligible_held_out_manual_checkpoints": checkpoints,
                "checkpoints": values,
            }
        },
    }


def test_release_gate_requires_three_roads_with_30_checkpoints_each():
    result = _release_assessment(
        [_release_case("a"), _release_case("b"), _release_case("short", 29)]
    )["1"]

    assert result["field_accuracy_established"] is False
    assert result["qualifying_road_count"] == 2
    assert result["supplementary_or_excluded_roads"] == [
        {"case_id": "short", "held_out_manual_checkpoints": 29}
    ]


def test_release_gate_aggregates_only_qualifying_roads():
    result = _release_assessment(
        [
            _release_case("a"),
            _release_case("b"),
            _release_case("c"),
            _release_case("short", 10),
        ]
    )["1"]

    assert result["field_accuracy_established"] is True
    assert result["held_out_manual_checkpoints"] == 90
    assert result["automatically_visible_checkpoints"] == 81
    assert result["aggregate_automatic_coverage_percent"] == pytest.approx(90.0)
    assert result["aggregate_visible_within_release_target_percent"] == pytest.approx(
        26 / 27 * 100
    )


def test_release_gate_excludes_declared_development_case():
    development = _release_case("development")
    development["release_eligible"] = False
    development["release_exclusion_reason"] = "already revealed"

    result = _release_assessment(
        [_release_case("a"), _release_case("b"), development]
    )["1"]

    assert result["qualifying_road_count"] == 2
    assert result["field_accuracy_established"] is False
    assert result["supplementary_or_excluded_roads"][-1] == {
        "case_id": "development",
        "held_out_manual_checkpoints": 30,
        "release_exclusion_reason": "already revealed",
    }


def test_deeper_thickness_is_withheld_when_upstream_identity_is_unstable():
    result = _thickness_evaluation(
        {"stations": []},
        [],
        np.asarray([0.0, 1.0]),
        {},
        100,
        2,
        {"all_within_one_pulse": True},
        {"status": "accepted", "fitted_mm_per_sample": 2.0},
        False,
    )

    assert result["status"] == "not_evaluated"
    assert result["reasons"] == [
        "upstream_interface_seed_identity_dropout_failed"
    ]
