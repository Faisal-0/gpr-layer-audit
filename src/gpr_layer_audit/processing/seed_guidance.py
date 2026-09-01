"""Conservative geometric guidance derived from sparse manual layer seeds.

The guide in this module is deliberately not radar evidence.  It can only
produce a bounded, non-positive adjustment for an existing candidate.  In
particular, it must not be used to create candidates, mark an interface as
visible, or increase the confidence assigned to a radar observation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

GuideMode = Literal["absolute", "gap"]


@dataclass(frozen=True, slots=True)
class SeedGuide:
    """A soft per-row expectation in sample coordinates.

    ``values`` contains either absolute interface samples or inter-layer gaps,
    according to ``mode``.  It is ``NaN`` on an ambiguous regime-change span:
    a scalar coordinate there would invent a transition location that was not
    observed.  ``lower_bounds`` and ``upper_bounds`` retain the admissible
    envelope for scoring and review.  ``uncertainty`` is a scale for soft
    scoring, not a statistical confidence interval.  Extrapolation is
    explicitly marked so callers can expose it in audit output if desired.
    """

    values: NDArray[np.float64]
    uncertainty: NDArray[np.float64]
    extrapolated: NDArray[np.bool_]
    seed_rows: NDArray[np.int64]
    seed_values: NDArray[np.float64]
    mode: GuideMode
    ambiguous: NDArray[np.bool_] | None = None
    lower_bounds: NDArray[np.float64] | None = None
    upper_bounds: NDArray[np.float64] | None = None


def derive_seed_guide(
    row_count: int,
    seed_rows: Mapping[int, float] | Sequence[int] | NDArray[np.integer],
    seed_values: Sequence[float] | NDArray[np.floating] | None = None,
    *,
    mode: GuideMode = "absolute",
    minimum_seeds: int = 2,
    base_uncertainty: float = 2.0,
    interpolation_growth: float = 0.04,
    extrapolation_growth: float = 0.10,
    regime_change_allowance: float = 0.50,
    regime_change_threshold_samples: float = 7.0,
) -> SeedGuide | None:
    """Derive an exact-at-seeds, piecewise-linear soft guide.

    Two independent observations are required by default.  Between seeds the
    uncertainty grows with distance from the nearest observation. Smooth spans
    retain piecewise-linear interpolation.  If two adjacent observations differ
    by more than ``regime_change_threshold_samples`` (about one pulse by
    default), their unobserved interval is instead kept as a full admissible
    envelope; no transition location is inferred. Outside the observed span,
    the guide is held constant and uncertainty grows with distance; this avoids
    unsafe runaway linear extrapolation.

    ``mode="absolute"`` interprets values as interface sample indices.
    ``mode="gap"`` interprets them as lower-minus-upper interface sample gaps.
    The interpolation is intentionally identical for both coordinate systems.
    """

    if row_count < 0:
        raise ValueError("row_count must be non-negative")
    if mode not in ("absolute", "gap"):
        raise ValueError("mode must be 'absolute' or 'gap'")
    if minimum_seeds < 2:
        raise ValueError("minimum_seeds must be at least two")
    parameters = (
        base_uncertainty,
        interpolation_growth,
        extrapolation_growth,
        regime_change_allowance,
    )
    if not all(np.isfinite(value) and value >= 0 for value in parameters):
        raise ValueError("uncertainty parameters must be finite and non-negative")
    if base_uncertainty == 0:
        raise ValueError("base_uncertainty must be positive")
    if (
        not np.isfinite(regime_change_threshold_samples)
        or regime_change_threshold_samples <= 0
    ):
        raise ValueError("regime_change_threshold_samples must be finite and positive")

    rows, values = _normalise_seeds(seed_rows, seed_values)
    if len(rows) and (rows[0] < 0 or rows[-1] >= row_count):
        raise ValueError("seed rows must lie within the output row range")
    if mode == "gap" and np.any(values < 0):
        raise ValueError("gap seeds must be non-negative")
    if len(rows) < minimum_seeds:
        return None
    if row_count == 0:
        return SeedGuide(
            values=np.empty(0, dtype=np.float64),
            uncertainty=np.empty(0, dtype=np.float64),
            extrapolated=np.empty(0, dtype=bool),
            seed_rows=rows,
            seed_values=values,
            mode=mode,
            ambiguous=np.empty(0, dtype=bool),
            lower_bounds=np.empty(0, dtype=np.float64),
            upper_bounds=np.empty(0, dtype=np.float64),
        )
    output_rows = np.arange(row_count, dtype=np.float64)
    guide = np.interp(output_rows, rows, values).astype(np.float64, copy=False)
    lower_bounds = guide.copy()
    upper_bounds = guide.copy()
    ambiguous = np.zeros(row_count, dtype=bool)
    uncertainty = np.full(row_count, float(base_uncertainty), dtype=np.float64)

    # np.interp safely holds endpoint values outside the seed span.  Increase
    # uncertainty there rather than extending a sparsely observed slope.
    before = output_rows < rows[0]
    after = output_rows > rows[-1]
    uncertainty[before] += extrapolation_growth * (rows[0] - output_rows[before])
    uncertainty[after] += extrapolation_growth * (output_rows[after] - rows[-1])

    for left, right, left_value, right_value in zip(
        rows[:-1], rows[1:], values[:-1], values[1:], strict=True
    ):
        interior = (output_rows > left) & (output_rows < right)
        if not np.any(interior):
            continue
        offset = output_rows[interior] - left
        span = float(right - left)
        nearest_seed_distance = np.minimum(offset, span - offset)
        fraction = offset / span
        uncertainty[interior] += interpolation_growth * nearest_seed_distance
        if abs(right_value - left_value) > regime_change_threshold_samples:
            # The sparse endpoints support both regimes, but do not reveal
            # where the change occurred.  Preserve every plausible value
            # between them and make a reviewer-visible unknown rather than a
            # fabricated linear midpoint.
            ambiguous[interior] = True
            lower_bounds[interior] = min(left_value, right_value)
            upper_bounds[interior] = max(left_value, right_value)
        else:
            # A smooth, sub-pulse change can still be a local slope.  Widen
            # gently in its unobserved middle without treating it as a gate.
            change_shape = 4.0 * fraction * (1.0 - fraction)
            uncertainty[interior] += (
                regime_change_allowance * abs(right_value - left_value) * change_shape
            )

    # Preserve manual observations exactly, including a seed-observed regime
    # change, and give each the irreducible manual-pick uncertainty only.
    guide[rows] = values
    uncertainty[rows] = base_uncertainty
    lower_bounds[rows] = values
    upper_bounds[rows] = values
    guide[ambiguous] = np.nan
    extrapolated = (output_rows < rows[0]) | (output_rows > rows[-1])
    return SeedGuide(
        values=guide,
        uncertainty=uncertainty,
        extrapolated=extrapolated,
        seed_rows=rows,
        seed_values=values,
        mode=mode,
        ambiguous=ambiguous,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )


def score_candidates_against_guide(
    candidate_samples: NDArray[np.integer] | NDArray[np.floating],
    guide: SeedGuide | None,
    *,
    upper_samples: NDArray[np.integer] | NDArray[np.floating] | None = None,
    maximum_penalty: float = 0.35,
) -> NDArray[np.float32]:
    """Return a bounded non-positive emission adjustment for candidates.

    Candidate arrays may be ``(rows,)`` or ``(rows, candidates)``.  For an
    absolute guide they are compared directly with its values.  For a gap
    guide, ``upper_samples`` is required and lower-minus-upper is compared with
    the guide.  A one-dimensional upper path broadcasts across candidate
    columns.

    Negative/non-finite samples represent unavailable or no-pick states and
    receive zero adjustment.  Thus this function never removes a state and its
    result can never increase an emission or radar confidence.
    """

    candidates = np.asarray(candidate_samples, dtype=np.float64)
    if candidates.ndim not in (1, 2):
        raise ValueError("candidate_samples must be a one- or two-dimensional array")
    if not np.isfinite(maximum_penalty) or maximum_penalty < 0:
        raise ValueError("maximum_penalty must be finite and non-negative")
    adjustment = np.zeros(candidates.shape, dtype=np.float32)
    if guide is None or candidates.shape[0] == 0 or maximum_penalty == 0:
        return adjustment
    if candidates.shape[0] != len(guide.values):
        raise ValueError("candidate row count must match guide row count")

    observed = candidates
    valid = np.isfinite(candidates) & (candidates >= 0)
    if guide.mode == "gap":
        if upper_samples is None:
            raise ValueError("upper_samples is required when scoring a gap guide")
        upper = np.asarray(upper_samples, dtype=np.float64)
        if candidates.ndim == 2 and upper.ndim == 1:
            upper = upper[:, None]
        try:
            upper = np.broadcast_to(upper, candidates.shape)
        except ValueError as error:
            raise ValueError("upper_samples cannot be broadcast to candidate shape") from error
        valid &= np.isfinite(upper) & (upper >= 0)
        observed = candidates - upper
        valid &= observed >= 0
    elif upper_samples is not None:
        raise ValueError("upper_samples is only valid with a gap guide")

    scale = guide.uncertainty if candidates.ndim == 1 else guide.uncertainty[:, None]
    # Saturating Gaussian loss is gentle near the guide and cannot become a
    # hard gate far from it.  Importantly, its best possible adjustment is 0.
    scale_values = np.broadcast_to(scale, candidates.shape)
    lower_values, upper_values = _guide_bounds(guide, candidates.shape)
    adjustment[valid] = soft_guide_adjustment(
        observed[valid],
        np.zeros(np.count_nonzero(valid), dtype=np.float64),
        scale_values[valid],
        maximum_penalty=maximum_penalty,
        lower_bound=lower_values[valid],
        upper_bound=upper_values[valid],
    )
    return adjustment


def guide_conflict_mask(
    candidate_samples: NDArray[np.integer] | NDArray[np.floating],
    guide: SeedGuide | None,
    *,
    upper_samples: NDArray[np.integer] | NDArray[np.floating] | None = None,
    pulse_width_samples: float = 7.0,
    uncertainty_multiple: float = 2.5,
) -> NDArray[np.bool_]:
    """Flag selected events that need another manual identity observation.

    This is a conservative presentation gate, not a detector.  It never changes
    a candidate or its radar confidence.  A conflict means that an otherwise
    selected event lies well outside both the pulse-scale tolerance and the
    guide's deliberately widening uncertainty.  Confirmed seed rows are always
    exempt because the analyst's observation is authoritative there.
    """

    candidates = np.asarray(candidate_samples, dtype=np.float64)
    if candidates.ndim != 1:
        raise ValueError("candidate_samples must be a one-dimensional array")
    if not np.isfinite(pulse_width_samples) or pulse_width_samples <= 0:
        raise ValueError("pulse_width_samples must be finite and positive")
    if not np.isfinite(uncertainty_multiple) or uncertainty_multiple <= 0:
        raise ValueError("uncertainty_multiple must be finite and positive")
    conflict = np.zeros(len(candidates), dtype=bool)
    if guide is None or not len(candidates):
        return conflict
    if len(candidates) != len(guide.values):
        raise ValueError("candidate row count must match guide row count")

    observed = candidates.copy()
    valid = np.isfinite(candidates) & (candidates >= 0)
    if guide.mode == "gap":
        if upper_samples is None:
            raise ValueError("upper_samples is required when checking a gap guide")
        upper = np.asarray(upper_samples, dtype=np.float64)
        if upper.shape != candidates.shape:
            raise ValueError("upper_samples must match candidate_samples")
        valid &= np.isfinite(upper) & (upper >= 0)
        observed -= upper
        valid &= observed >= 0
    elif upper_samples is not None:
        raise ValueError("upper_samples is only valid with a gap guide")

    lower_bounds, upper_bounds = _guide_bounds(guide, candidates.shape)
    deviation = np.maximum(
        lower_bounds - observed,
        observed - upper_bounds,
    )
    limit = np.maximum(
        2.0 * float(pulse_width_samples),
        float(uncertainty_multiple) * guide.uncertainty,
    )
    conflict[valid] = deviation[valid] > limit[valid]
    conflict[guide.seed_rows] = False
    return conflict


def soft_guide_adjustment(
    observed: NDArray[np.floating] | float,
    expected: NDArray[np.floating] | float,
    uncertainty: NDArray[np.floating] | float,
    *,
    maximum_penalty: float = 0.35,
    lower_bound: NDArray[np.floating] | float | None = None,
    upper_bound: NDArray[np.floating] | float | None = None,
) -> NDArray[np.float32]:
    """Score already-valid values against a guide without positive bonuses.

    When both bounds are given, all observed values within that envelope are
    equally admissible and only distance outside it is penalized.
    """

    observed_array, expected_array, uncertainty_array = np.broadcast_arrays(
        np.asarray(observed, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        np.asarray(uncertainty, dtype=np.float64),
    )
    if not np.isfinite(maximum_penalty) or maximum_penalty < 0:
        raise ValueError("maximum_penalty must be finite and non-negative")
    if np.any(~np.isfinite(uncertainty_array)) or np.any(uncertainty_array <= 0):
        raise ValueError("uncertainty must be finite and positive")
    if (lower_bound is None) != (upper_bound is None):
        raise ValueError("lower_bound and upper_bound must be supplied together")
    if lower_bound is None:
        distance = np.abs(observed_array - expected_array)
    else:
        lower_array, upper_array = np.broadcast_arrays(
            np.asarray(lower_bound, dtype=np.float64),
            np.asarray(upper_bound, dtype=np.float64),
        )
        lower_array = np.broadcast_to(lower_array, observed_array.shape)
        upper_array = np.broadcast_to(upper_array, observed_array.shape)
        if (
            np.any(~np.isfinite(lower_array))
            or np.any(~np.isfinite(upper_array))
            or np.any(lower_array > upper_array)
        ):
            raise ValueError("guide bounds must be finite and ordered")
        distance = np.maximum(lower_array - observed_array, observed_array - upper_array)
        distance = np.maximum(distance, 0.0)
    z = np.minimum(distance / uncertainty_array, 12.0)
    penalty = maximum_penalty * (1.0 - np.exp(-0.5 * np.square(z)))
    return np.asarray(-penalty, dtype=np.float32)


def _guide_bounds(
    guide: SeedGuide,
    shape: tuple[int, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Broadcast a guide's finite scoring envelope to candidate shape."""

    lower = guide.values if guide.lower_bounds is None else guide.lower_bounds
    upper = guide.values if guide.upper_bounds is None else guide.upper_bounds
    if len(shape) == 2:
        lower = lower[:, None]
        upper = upper[:, None]
    try:
        lower_values = np.broadcast_to(lower, shape)
        upper_values = np.broadcast_to(upper, shape)
    except ValueError as error:
        raise ValueError("guide bounds cannot be broadcast to candidate shape") from error
    if (
        np.any(~np.isfinite(lower_values))
        or np.any(~np.isfinite(upper_values))
        or np.any(lower_values > upper_values)
    ):
        raise ValueError("guide bounds must be finite and ordered")
    return lower_values, upper_values


def _normalise_seeds(
    seed_rows: Mapping[int, float] | Sequence[int] | NDArray[np.integer],
    seed_values: Sequence[float] | NDArray[np.floating] | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    if isinstance(seed_rows, Mapping):
        if seed_values is not None:
            raise ValueError("seed_values must be omitted when seed_rows is a mapping")
        pairs = list(seed_rows.items())
        try:
            raw_rows = np.asarray([item[0] for item in pairs], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("seed_rows must contain numeric row indices") from error
        if raw_rows.ndim != 1 or not np.all(np.isfinite(raw_rows)):
            raise ValueError("seed_rows must be a finite one-dimensional sequence")
        if not np.all(raw_rows == np.floor(raw_rows)):
            raise ValueError("seed_rows must contain integer row indices")
        rows = raw_rows.astype(np.int64)
        try:
            values = np.asarray([item[1] for item in pairs], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("seed values must be numeric") from error
        order = np.argsort(rows, kind="stable")
        rows, values = rows[order], values[order]
    else:
        if seed_values is None:
            raise ValueError("seed_values is required when seed_rows is not a mapping")
        try:
            raw_rows = np.asarray(seed_rows, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("seed_rows must contain numeric row indices") from error
        if raw_rows.ndim != 1 or not np.all(np.isfinite(raw_rows)):
            raise ValueError("seed_rows must be a finite one-dimensional sequence")
        if not np.all(raw_rows == np.floor(raw_rows)):
            raise ValueError("seed_rows must contain integer row indices")
        rows = raw_rows.astype(np.int64)
        try:
            values = np.asarray(seed_values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("seed values must be numeric") from error
        if values.ndim != 1:
            raise ValueError("seed_values must be a one-dimensional sequence")
        if len(rows) != len(values):
            raise ValueError("seed_rows and seed_values must have equal length")
        order = np.argsort(rows, kind="stable")
        rows, values = rows[order], values[order]
    if not np.all(np.isfinite(values)):
        raise ValueError("seed values must be finite")
    if len(rows) and np.any(np.diff(rows) == 0):
        raise ValueError("each seed row must be unique")
    return rows, values
