"""Post-selection reliability checks; never changes the chosen radar event."""

from __future__ import annotations

import numpy as np

from gpr_layer_audit.models import AnalysisResult, PickStatus, SeedStation, VisibilityState


def _same_observation(left, right, pulse_width_samples: float) -> bool:
    if left is None or right is None or left.anomaly or right.anomaly:
        return False
    left_hidden = left.sample_index < 0 or left.visibility in {
        VisibilityState.ABSENT,
        VisibilityState.NOT_VISIBLE,
    }
    right_hidden = right.sample_index < 0 or right.visibility in {
        VisibilityState.ABSENT,
        VisibilityState.NOT_VISIBLE,
    }
    if left_hidden or right_hidden:
        return left_hidden and right_hidden
    left_display = left.selected_lobe_sample if left is not None else None
    right_display = right.selected_lobe_sample if right is not None else None
    return bool(
        left.sample_index >= 0
        and right.sample_index >= 0
        and left_display is not None
        and right_display is not None
        and abs(left.sample_index - right.sample_index) <= pulse_width_samples
        and abs(left_display - right_display) <= pulse_width_samples
        and (left.polarity == 0 or right.polarity == 0 or left.polarity == right.polarity)
    )


def apply_seed_dropout_check(
    result: AnalysisResult,
    alternative: AnalysisResult,
    station: SeedStation | str,
    *,
    pulse_width_samples: float = 7.0,
) -> list[dict]:
    """Demote picks that depend on one station, using an independently refitted run.

    Family integers are local to each fit and cannot be compared across fits.
    Compare canonical observation, displayed lobe and polarity instead. A lost
    pick is unstable, not a zero-error agreement. Numerical support is not a
    calibrated probability, and passing dropout is necessary but not sufficient.
    """
    station_id = station.station_id if isinstance(station, SeedStation) else str(station)
    station_chainage = station.chainage_m if isinstance(station, SeedStation) else None
    station_orders = (
        set(station.samples) | set(station.visibility)
        if isinstance(station, SeedStation)
        else set()
    )
    lookup = {(p.layer_order, round(p.chainage_m, 6)): p for p in alternative.picks}
    audit = result.parameters.setdefault("seed_dropout_audit", [])
    first_run = not audit
    summaries = []
    for order in sorted({p.layer_order for p in result.picks}):
        unstable_chainages = []
        checked = 0
        demoted = 0
        for pick in (p for p in result.picks if p.layer_order == order):
            other = lookup.get((order, round(pick.chainage_m, 6)))
            if pick.sample_index < 0 or pick.anomaly:
                continue
            checked += 1
            display = pick.selected_lobe_sample
            other_display = other.selected_lobe_sample if other is not None else None
            stable = _same_observation(pick, other, pulse_width_samples)
            prior = 1.0 if first_run else pick.evidence.drop_seed_stability
            pick.evidence.drop_seed_stability = min(prior, float(stable))
            pick.drop_seed_stability = pick.evidence.drop_seed_stability
            if not stable:
                unstable_chainages.append(pick.chainage_m)
                if other is not None and other_display is not None and other.sample_index >= 0:
                    # Preserve the actual rival from the independently fitted
                    # run for the two-family review, not a fabricated offset.
                    old = pick.competing_family_sample
                    old_separation = (
                        abs(old - display) if old is not None and display is not None else 0
                    )
                    separation = abs(other_display - display) if display is not None else 0
                    if separation > old_separation:
                        pick.competing_family_sample = other_display
                        pick.competing_family_id = f"dropout:{station_id}:{other.event_family_id}"
                    pick.evidence.branch_multimodality = max(
                        pick.evidence.branch_multimodality,
                        min(1.0, separation / max(2 * pulse_width_samples, 1.0)),
                    )
                if pick.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}:
                    demoted += 1
                    pick.status = PickStatus.REVIEW
                    pick.visibility = VisibilityState.UNCERTAIN
                pick.review_reason = "Reflector identity changes when a seed station is withheld"
        station_pick = None
        independent_pick = None
        if station_chainage is not None and order in station_orders:
            layer_picks = [p for p in result.picks if p.layer_order == order]
            if layer_picks:
                station_pick = min(
                    layer_picks, key=lambda item: abs(item.chainage_m - station_chainage)
                )
                independent_pick = lookup.get(
                    (order, round(station_pick.chainage_m, 6))
                )
        station_inconsistent = bool(
            station_pick is not None
            and not _same_observation(
                station_pick, independent_pick, pulse_width_samples
            )
        )
        summaries.append(
            {
                "station_id": station_id,
                "station_chainage_m": station_chainage,
                "layer_order": order,
                "checked_visible_picks": checked,
                "unstable_picks": len(unstable_chainages),
                "demoted_picks": demoted,
                "unstable_chainages_m": unstable_chainages,
                "station_inconsistent": station_inconsistent,
                "withheld_manual_sample": (
                    station_pick.selected_lobe_sample if station_pick is not None else None
                ),
                "independent_sample": (
                    independent_pick.selected_lobe_sample
                    if independent_pick is not None and independent_pick.sample_index >= 0
                    else None
                ),
            }
        )
    audit.extend(summaries)
    result.parameters.setdefault("event_family_tracker", {})["drop_seed_measure"] = (
        "independently_refitted_leave_one_station_out"
    )
    result.parameters["seed_dropout_pulse_width_samples"] = float(pulse_width_samples)
    result.parameters["seed_dropout_gate_passed"] = not any(
        item["unstable_picks"] for item in audit
    )
    # Chainages are retained for grouping and audit, not interpolated into a
    # corrected path. A later user decision can disambiguate the region.
    assert all(np.isfinite(x) for item in summaries for x in item["unstable_chainages_m"])
    return summaries
