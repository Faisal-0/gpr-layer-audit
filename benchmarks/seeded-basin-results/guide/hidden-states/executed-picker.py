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
    _capture(proposal.copy())
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
