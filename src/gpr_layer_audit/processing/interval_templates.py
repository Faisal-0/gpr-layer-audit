"""Seed-region waveform scoring without mixing distant reflector identities."""

from __future__ import annotations

import numpy as np


def interval_seed_scores(
    table, prototypes, correlations, anchors, breaks, measurement, valid, step, *, cancel=None
):
    """Interpolate observable template scores between the two bracketing seeds.

    Original seed waveforms remain independent. Interpolation blends *scores*,
    never waveform samples or reflector depths. Explicitly different regimes or
    selected lobes remain competing hypotheses, with their maximum retained.
    Structural breaks isolate banks; tails use only their nearest endpoint seed.
    Candidate generation and directed correspondence are untouched.
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Positive horizontal sampling required")
    if len(prototypes) != len(correlations):
        raise ValueError("Each seed template needs its own correlation evidence")
    rows = len(table.samples)
    columns = np.clip(table.samples, 0, measurement.shape[1] - 1)
    rr = np.arange(rows)[:, None]
    observable = table.valid & (table.samples >= 0) & valid[rr, columns]
    scores = np.zeros_like(table.samples, dtype=np.float32)
    banks = []
    for index, prototype in enumerate(prototypes):
        if cancel and cancel():
            raise InterruptedError("Analysis cancelled")
        # The established template builder stores its working row in chainage_m.
        # Resolve to metres here instead of treating that field as physical distance.
        row = int(prototype.chainage_m)
        sample, radius = prototype.sample_index, prototype.radius_samples
        if (
            row not in anchors
            or anchors[row] != sample
            or row != prototype.chainage_m
            or sample < radius
            or sample + radius >= measurement.shape[1]
            or not np.all(valid[row, sample - radius : sample + radius + 1])
        ):
            continue
        compatible = np.sign(measurement[rr, columns]) == prototype.polarity
        values = np.where(
            observable & compatible, np.clip(correlations[index][rr, columns], 0, 1), 0
        )
        banks.append((row, index, values))
    bounds = sorted({0, rows, *(int(b) for b in breaks if 0 < b < rows)})
    regions = []
    for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
        local = sorted((b for b in banks if start <= b[0] < stop), key=lambda b: b[0])
        if not local:
            continue
        locations = np.asarray([b[0] * step for b in local])
        coordinates = np.arange(start, stop) * step
        right = np.clip(np.searchsorted(locations, coordinates), 0, len(local) - 1)
        left = np.maximum(right - 1, 0)
        left[coordinates >= locations[-1]] = len(local) - 1
        for li, ri in sorted(set(zip(left.tolist(), right.tolist(), strict=True))):
            if cancel and cancel():
                raise InterruptedError("Analysis cancelled")
            use = np.arange(start, stop)[(left == li) & (right == ri)]
            lrow, lindex, lvalues = local[li]
            rrow, rindex, rvalues = local[ri]
            lp, rp = prototypes[lindex], prototypes[rindex]
            different_identity = (
                lp.regime_id != rp.regime_id
                or lp.polarity != rp.polarity
                or (
                    lp.selected_lobe != rp.selected_lobe
                    and "unknown" not in (lp.selected_lobe, rp.selected_lobe)
                )
            )
            if li == ri:
                scores[use] = lvalues[use]
                policy = "endpoint_template"
            elif different_identity:
                scores[use] = np.maximum(lvalues[use], rvalues[use])
                policy = "competing_explicit_regimes"
            else:
                weight = (use * step - lrow * step) / max(step, (rrow - lrow) * step)
                scores[use] = (1 - weight[:, None]) * lvalues[use] + weight[:, None] * rvalues[use]
                policy = "bracketing_seed_score_interpolation"
            regions.append(
                {
                    "start_row": int(use[0]),
                    "stop_row": int(use[-1]),
                    "left_seed_row": lrow,
                    "right_seed_row": rrow,
                    "policy": policy,
                }
            )
    return scores, {
        "policy": "interval_seed_scores_v1",
        "regions": regions,
        "original_seed_templates_preserved": True,
        "evaluation_observations_used": False,
    }
