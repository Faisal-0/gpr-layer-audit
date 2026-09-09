"""Compose measured adjacent displacement instead of extrapolating one local slope."""

from __future__ import annotations

import numpy as np


def contextual_waveforms(
    table,
    measurement,
    maps,
    width,
    step,
    radius_m,
    *,
    valid=None,
    breaks=(),
    cancel=None,
    minimum_margin=0.0,
):
    """Average reciprocal, aligned neighbouring snippets as extra matching context.

    Original amplitudes, candidate coordinates and seed reference templates remain
    untouched. Each contributing snippet must be observable and preserve central
    polarity. This context alone cannot declare a measurement or a seed identity.
    """
    radius = max(0, round(radius_m / step))
    if radius == 0:
        return table.waveforms
    distances = list(range(1, min(radius, len(measurement) - 1) + 1))
    predictions = compose_motion(
        maps, distances, valid=valid, cancel=cancel, minimum_margin=minimum_margin
    )
    output = table.waveforms.copy()
    counts = np.ones(table.samples.shape, np.float32)
    row_ids, col_ids = np.nonzero(table.valid & (table.samples >= 0))
    region = np.searchsorted(sorted(breaks), np.arange(len(measurement)), side="right")
    half = table.waveforms.shape[-1] // 2
    offsets = np.arange(-half, half + 1)
    for gap, fields in predictions.items():
        for direction, reverse, sign in (("forward", "backward", 1), ("backward", "forward", -1)):
            for start in range(0, len(row_ids), 2048):
                if cancel and cancel():
                    raise InterruptedError("Analysis cancelled")
                rr, cc = row_ids[start : start + 2048], col_ids[start : start + 2048]
                samples = table.samples[rr, cc]
                target_rows = rr + sign * gap
                target_samples = np.rint(
                    samples + fields[f"motion_{direction}_shift"][rr, samples]
                ).astype(int)
                inside = (
                    (target_rows >= 0)
                    & (target_rows < len(measurement))
                    & (target_samples >= half)
                    & (target_samples < measurement.shape[1] - half)
                )
                rr, cc, samples = rr[inside], cc[inside], samples[inside]
                tr, ts = target_rows[inside], target_samples[inside]
                quality = np.minimum(
                    fields[f"motion_{direction}_support"][rr, samples],
                    fields[f"motion_{reverse}_support"][tr, ts],
                )
                reciprocal = abs(ts + fields[f"motion_{reverse}_shift"][tr, ts] - samples)
                usable = (quality >= 0.7) & (reciprocal <= max(2, width / 4))
                usable &= region[rr] == region[tr]
                usable &= measurement[rr, samples] != 0
                usable &= np.sign(measurement[rr, samples]) == np.sign(measurement[tr, ts])
                if valid is not None:
                    usable &= np.all(valid[tr[:, None], ts[:, None] + offsets], axis=1)
                rr, cc, tr, ts, quality = (
                    rr[usable],
                    cc[usable],
                    tr[usable],
                    ts[usable],
                    quality[usable],
                )
                snippets = measurement[tr[:, None], ts[:, None] + offsets].copy()
                snippets -= snippets.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(snippets, axis=1, keepdims=True)
                snippets = np.divide(
                    snippets, norms, out=np.zeros_like(snippets), where=norms > 1e-7
                )
                weights = quality**2 * (norms[:, 0] > 1e-7)
                output[rr, cc] += weights[:, None] * snippets
                counts[rr, cc] += weights
    output /= counts[:, :, None]
    norms = np.linalg.norm(output, axis=2, keepdims=True)
    return np.divide(output, norms, out=np.zeros_like(output), where=norms > 1e-7)


def compose_motion(maps, distances, *, valid=None, cancel=None, minimum_margin=0.0):
    """Independent forward/backward flow composition, with bottleneck observability.

    Binary composition visits intermediate waveform coordinates. Missing signal
    invalidates the prediction; it never becomes an observed graph node. Returned
    shifts are total displacements, not per-trace slopes.
    """
    distances = sorted(set(int(d) for d in distances if d > 0))
    if not distances:
        return {}
    shape = maps["motion_forward_shift"].shape
    rows, samples = shape
    rr, cc = np.indices(shape)
    result = {d: {} for d in distances}
    for direction, sign in (("forward", 1), ("backward", -1)):
        shift = np.asarray(maps[f"motion_{direction}_shift"], np.float32).copy()
        quality = np.asarray(maps[f"motion_{direction}_support"], np.float32).copy()
        if minimum_margin > 0:
            margin = maps.get(f"motion_{direction}_margin", np.zeros(shape, np.float32))
            quality *= margin >= minimum_margin
        if valid is not None:
            quality *= valid
        outside = (rr + sign < 0) | (rr + sign >= rows)
        quality[outside] = 0
        powers = [(shift, quality)]
        length = 1
        while 2 * length <= max(distances):
            if cancel and cancel():
                raise InterruptedError("Analysis cancelled")
            shift, quality = powers[-1]
            target_row = rr + sign * length
            target_sample = cc + shift
            inside = (
                (target_row >= 0)
                & (target_row < rows)
                & (target_sample >= 0)
                & (target_sample <= samples - 1)
            )
            at_row = np.clip(target_row, 0, rows - 1)
            at_sample = np.clip(np.rint(target_sample).astype(int), 0, samples - 1)
            powers.append(
                (
                    shift + shift[at_row, at_sample],
                    np.where(inside, np.minimum(quality, quality[at_row, at_sample]), 0),
                )
            )
            length *= 2
        for distance in distances:
            total = np.zeros(shape, np.float32)
            support = np.ones(shape, np.float32)
            travelled = 0
            for bit, (shift, quality) in enumerate(powers):
                if not distance & (1 << bit):
                    continue
                target_row = rr + sign * travelled
                target_sample = cc + total
                inside = (
                    (target_row >= 0)
                    & (target_row < rows)
                    & (target_sample >= 0)
                    & (target_sample <= samples - 1)
                )
                at_row = np.clip(target_row, 0, rows - 1)
                at_sample = np.clip(np.rint(target_sample).astype(int), 0, samples - 1)
                support = np.where(inside, np.minimum(support, quality[at_row, at_sample]), 0)
                total += shift[at_row, at_sample]
                travelled += 1 << bit
            endpoint_row, endpoint_sample = rr + sign * distance, cc + total
            inside = (
                (endpoint_row >= 0)
                & (endpoint_row < rows)
                & (endpoint_sample >= 0)
                & (endpoint_sample <= samples - 1)
            )
            if valid is not None:
                inside &= valid[
                    np.clip(endpoint_row, 0, rows - 1),
                    np.clip(np.rint(endpoint_sample).astype(int), 0, samples - 1),
                ]
            result[distance][f"motion_{direction}_shift"] = total
            result[distance][f"motion_{direction}_support"] = np.where(inside, support, 0)
    return result
