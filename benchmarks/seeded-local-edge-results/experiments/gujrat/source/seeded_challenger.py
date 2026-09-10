"""Independent dense seeded packet DP for falsifying sparse-graph bottlenecks.

This research comparator retains every sample position, uses immutable endpoint
packets and exact forward/backward dynamic programming, and never reads labels.
Margins are objective differences, not probabilities. Missing signal has an
explicit latent state but can never become a reported observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, uniform_filter1d
from scipy.signal import fftconvolve

from .seed_graph import SeedConditionedPath


@dataclass(frozen=True)
class PacketConfig:
    geometry_weight: float = 0.15
    slope_scale_ns_per_m: float = 0.2
    max_slope_ns_per_m: float = 2.0
    context_m: float = 0.1
    minimum_correlation: float = 0.5
    minimum_margin: float = 0.02
    gap_cost: float = 0.65
    geometry: str = "tensor"
    guide_weight: float = 0.0
    guide_width_lobes: float = 1.0


def tensor_geometry(measurement, valid, dt_ns, dx_m, context_m=0.1):
    """Structure-tensor plane slope in ns/m; no interpolation through gaps.

    Input/output axes are [trace, time sample]. Derivatives have physical units.
    The linear-plane estimate uses -<Ix It>/<It It>; coherence and a dilated
    validity mask prevent zero-filled regions from acquiring geometry support.
    """
    data = np.where(valid, measurement, 0).astype(float)
    ix, it = np.gradient(data, dx_m, dt_ns)
    sigma = (max(0.5, context_m / dx_m), 1.0)
    xx = gaussian_filter(ix * ix, sigma)
    tt = gaussian_filter(it * it, sigma)
    xt = gaussian_filter(ix * it, sigma)
    slope = -xt / np.maximum(tt, np.finfo(float).tiny)
    strength = xx + tt
    confidence = np.sqrt((xx - tt) ** 2 + 4 * xt**2) / np.maximum(strength, 1e-20)
    support = minimum_filter(valid, size=(2 * int(np.ceil(3 * sigma[0])) + 1, 7))
    support &= strength > np.max(strength, initial=0) * 1e-12
    confidence = np.where(support, confidence, 0)
    return np.where(support, slope, 0), confidence


def pyseistr_geometry(measurement, valid, dt_ns, dx_m, *, implementation="c", rect=(5, 5, 1)):
    """Optional pinned pyseistr adapter; upstream slope is samples per trace.

    pyseistr expects [time sample, trace] and returns the same shape. The C
    version accepts a mask; the Python version explicitly does not. Unsupported
    neighborhoods are withheld after estimation, and no output fills signal.
    See docs/upstream-seeded-pyseistr.md for revision, license and contract tests.
    """
    from pyseistr.dip2d import dip2d, dip2dc

    data, valid = np.asarray(measurement), np.asarray(valid, bool)
    if data.ndim != 2 or valid.shape != data.shape or min(dt_ns, dx_m) <= 0:
        raise ValueError("Require [trace,sample] data, equal validity and positive physical units")
    if implementation not in ("c", "python"):
        raise ValueError("implementation must be c or python")
    if implementation == "python" and not np.all(valid):
        raise ValueError("Upstream pure Python dip2d does not implement masks")
    kwargs = dict(niter=3, liter=10, order=2, rect=list(rect), verb=0)
    transposed = np.where(valid, data, 0).T.astype(np.float32)
    if implementation == "c":
        dip = dip2dc(transposed, mask=valid.T.astype(np.float32), **kwargs)
    else:
        dip = dip2d(transposed, **kwargs)
    slope = np.asarray(dip).T * dt_ns / dx_m
    if slope.shape != data.shape:
        raise ValueError("Upstream dip shape differs from input")
    support = minimum_filter(valid, size=(2 * rect[1] + 1, 2 * rect[0] + 5))
    support &= np.isfinite(slope)
    return np.where(support, slope, 0), support.astype(float)


def _packet_correlations(measurement, valid, anchors, radius):
    size = 2 * radius + 1
    data = np.where(valid, measurement, 0).astype(float)
    sums = uniform_filter1d(data, size, axis=1, mode="constant") * size
    squared = uniform_filter1d(data * data, size, axis=1, mode="constant") * size
    energy = np.sqrt(np.maximum(squared - sums * sums / size, 1e-20))
    packet_valid = minimum_filter(valid, size=(1, size), mode="constant", cval=False)
    bank = {}
    for row, sample in sorted(anchors.items()):
        if not radius <= sample < data.shape[1] - radius or not packet_valid[row, sample]:
            bank[row] = np.zeros(data.shape, np.float32)
            continue
        template = data[row, sample - radius : sample + radius + 1].copy()
        template -= template.mean()
        template /= max(float(np.linalg.norm(template)), 1e-20)
        corr = fftconvolve(data, template[None, ::-1], axes=1, mode="same") / energy
        bank[row] = np.clip(corr, -1, 1).astype(np.float32)
    return bank, packet_valid


def dense_interval_dp(unary, transition, left_sample=None, right_sample=None, cancel=None):
    """Exact min-sum chain with stationary displacement offsets and edge costs.

    ``transition[row,offset,column]`` is the cost from column-offset at row to
    column at row+1. Offset index zero means -radius. Infinities forbid edges.
    Returns path and exact total-path min-marginals under identical constraints.
    This standalone numerical contract is also tested by exhaustive enumeration.
    """
    unary = np.asarray(unary, float).copy()
    rows, samples = unary.shape
    if transition.shape[0] != rows - 1 or transition.shape[2] != samples:
        raise ValueError("Transition array does not match unary shape")
    offsets = np.arange(transition.shape[1]) - transition.shape[1] // 2
    if left_sample is not None:
        value = unary[0, left_sample]
        unary[0] = np.inf
        unary[0, left_sample] = value
    if right_sample is not None:
        value = unary[-1, right_sample]
        unary[-1] = np.inf
        unary[-1, right_sample] = value
    forward = np.full((rows, samples), np.inf)
    backward = np.full_like(forward, np.inf)
    parents = np.full((rows, samples), -1, np.int32)
    forward[0] = unary[0]
    columns = np.arange(samples)
    source = columns[None, :] - offsets[:, None]
    in_range = (source >= 0) & (source < samples)
    source_safe = np.clip(source, 0, samples - 1)
    for row in range(1, rows):
        if cancel and row % 32 == 0 and cancel():
            raise InterruptedError("Packet tracking cancelled")
        costs = forward[row - 1, source_safe] + transition[row - 1]
        costs[~in_range] = np.inf
        winner = np.argmin(costs, axis=0)
        forward[row] = unary[row] + costs[winner, columns]
        parents[row] = source_safe[winner, columns]
    terminal = int(np.argmin(forward[-1]))
    if not np.isfinite(forward[-1, terminal]):
        return np.full(rows, -1, np.int32), np.full_like(forward, np.inf)
    path = np.full(rows, terminal, np.int32)
    for row in range(rows - 1, 0, -1):
        path[row - 1] = parents[row, path[row]]
    backward[-1] = 0
    target = columns[None, :] + offsets[:, None]
    target_valid = (target >= 0) & (target < samples)
    target_safe = np.clip(target, 0, samples - 1)
    for row in range(rows - 2, -1, -1):
        costs = (
            transition[row, np.arange(len(offsets))[:, None], target_safe]
            + unary[row + 1, target_safe]
            + backward[row + 1, target_safe]
        )
        costs[~target_valid] = np.inf
        backward[row] = np.min(costs, axis=0)
    return path, forward + backward


def pick_seeded_packet(
    measurement,
    anchors,
    *,
    valid,
    dt_ns,
    dx_m,
    pulse_width_samples,
    context_radius=None,
    reference_surface=0,
    breaks=(),
    config=None,
    cancel=None,
):
    """Return the application path contract for one layer from operating seeds only."""
    config = config or PacketConfig()
    data, valid = np.asarray(measurement, np.float32), np.asarray(valid, bool)
    if data.ndim != 2 or data.shape != valid.shape or min(dt_ns, dx_m) <= 0:
        raise ValueError("Require radar and validity [trace,sample], dt_ns>0, dx_m>0")
    if not np.all(np.isfinite(data[valid])):
        raise ValueError("Valid measurements must be finite")
    rows, samples = data.shape
    anchors = dict(sorted(anchors.items()))
    if any(not (0 <= r < rows and 0 <= s < samples) for r, s in anchors.items()):
        raise ValueError("Exact seed coordinates lie outside radar")
    radius = context_radius or max(3, round(1.5 * pulse_width_samples))
    correlations, packet_valid = _packet_correlations(data, valid, anchors, radius)
    if config.geometry == "tensor":
        slope, slope_support = tensor_geometry(data, valid, dt_ns, dx_m, config.context_m)
    elif config.geometry == "pyseistr":
        slope, slope_support = pyseistr_geometry(data, valid, dt_ns, dx_m)
    elif config.geometry == "none":
        slope, slope_support = np.zeros_like(data), np.zeros_like(data)
    else:
        raise ValueError("Unknown packet geometry")
    slope = np.clip(slope, -config.max_slope_ns_per_m, config.max_slope_ns_per_m)
    local_peak = np.sqrt(
        np.maximum(0, uniform_filter1d(data.astype(float) ** 2, 2 * radius + 1, axis=1))
    )
    central_support = np.abs(data) / np.maximum(local_peak, 1e-20)
    observable = packet_valid & (central_support >= 0.2) & (local_peak > 0)
    proposal = np.full(rows, -1, np.int32)
    alternate = proposal.copy()
    margin = np.zeros(rows)
    selected_correlation = np.zeros(rows)
    intervals = []
    boundaries = sorted({0, rows, *(int(x) for x in breaks if 0 < x < rows)})
    max_shift = min(samples - 1, max(1, int(np.ceil(config.max_slope_ns_per_m * dx_m / dt_ns))))
    offsets = np.arange(-max_shift, max_shift + 1)
    expected = slope[:-1] * slope_support[:-1] * dx_m / dt_ns
    # Half-sample dead zone prevents grid quantization being mistaken for curvature.
    residual = np.maximum(np.abs(offsets[None, :, None] - expected[:, None, :]) - 0.5, 0)
    normalized = residual * dt_ns / max(config.slope_scale_ns_per_m * dx_m, dt_ns / 2)
    robust = np.where(normalized <= 1, normalized**2 / 2, normalized - 0.5)
    transition = config.geometry_weight * robust * dx_m
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        seeds = [r for r in anchors if start <= r < stop]
        if not seeds:
            continue
        spans = [
            (start, seeds[0]),
            *zip(seeds[:-1], seeds[1:], strict=False),
            (seeds[-1], stop - 1),
        ]
        for left, right in spans:
            if left == right:
                continue
            rr = np.arange(left, right + 1)
            left_anchor = left if left in anchors else right
            right_anchor = right if right in anchors else left
            weight = (rr - left) / (right - left)
            corr = (1 - weight[:, None]) * correlations[left_anchor][rr] + weight[
                :, None
            ] * correlations[right_anchor][rr]
            # Gap has a finite cost and retained latent position; no numerical fill
            # of measurement is used to score it, and it remains non-observable.
            unary = np.where(observable[rr], 1 - corr, config.gap_cost) * dx_m
            if config.guide_weight:
                # An explicit prior from operating endpoints, never a measurement
                # or evaluation reference. Latent gap positions remain withheld.
                guide = (1 - weight) * anchors[left_anchor] + weight * anchors[right_anchor]
                distance = np.abs(np.arange(samples)[None, :] - guide[:, None])
                distance /= max(1, config.guide_width_lobes * pulse_width_samples)
                guide_cost = np.where(distance <= 1, distance**2 / 2, distance - 0.5)
                unary += config.guide_weight * guide_cost * dx_m
            unary[:, : max(0, reference_surface + 1)] = np.inf
            picked, costs = dense_interval_dp(
                unary,
                transition[left:right],
                anchors.get(left),
                anchors.get(right),
                cancel,
            )
            finite = picked >= 0
            proposal[rr[finite]] = picked[finite]
            for i in np.flatnonzero(finite):
                row, sample = rr[i], picked[i]
                signs = np.sign(data[row])
                changes = np.r_[True, (signs[1:] != signs[:-1]) | (signs[1:] == 0)]
                lobes = np.cumsum(changes)
                other = costs[i].copy()
                other[lobes == lobes[sample]] = np.inf
                competitor = int(np.argmin(other))
                if np.isfinite(other[competitor]):
                    alternate[row] = competitor
                    # Objective per metre, normalized by full conditioned interval.
                    margin[row] = max(0, other[competitor] - costs[i, sample]) / (
                        (right - left + 1) * dx_m
                    )
                else:
                    # Lost competitors are not certainty; their absence gives no gate credit.
                    margin[row] = 0
                selected_correlation[row] = corr[i, sample]
            intervals.append(
                {
                    "start_row": left,
                    "stop_row": right,
                    "bracketed": left in anchors and right in anchors,
                }
            )
    proposed_rows = np.flatnonzero(proposal >= 0)
    accepted = np.zeros(rows, bool)
    accepted[proposed_rows] = (
        observable[proposed_rows, proposal[proposed_rows]]
        & (selected_correlation[proposed_rows] >= config.minimum_correlation)
        & (margin[proposed_rows] >= config.minimum_margin)
    )
    # Latent gaps are not displayed as measured proposals either.
    hidden = proposed_rows[~observable[proposed_rows, proposal[proposed_rows]]]
    proposal[hidden] = -1
    for row, sample in anchors.items():
        proposal[row] = sample
        accepted[row] = bool(valid[row, sample])
    output = np.where(accepted, proposal, -1).astype(np.int32)
    candidate = np.where(observable, 0.0, np.nan).astype(np.float32)
    return SeedConditionedPath(
        samples=output,
        confidence=margin,
        feature=np.zeros(rows, np.float32),
        alternate_samples=alternate,
        visible=accepted,
        interpolated=np.zeros(rows, bool),
        evidence={
            "hybrid_path_margin": margin,
            "packet_seed_correlation": selected_correlation,
            "measurement_support": accepted.astype(float),
        },
        signal_only_samples=proposal.copy(),
        design_guided_samples=proposal.copy(),
        design_conflict=np.zeros(rows, bool),
        design_constrained=False,
        candidate_components={"audit_candidate_rank": candidate},
        provisional_samples=proposal,
        provenance={
            "backend": "dense_seeded_packet_research",
            "config": asdict(config),
            "margin_kind": "uncalibrated interval-normalized exact min-sum difference",
            "all_sample_states_retained": True,
            "gap_is_not_measurement": True,
            "intervals": intervals,
            "original_seed_templates_preserved": True,
        },
    )
