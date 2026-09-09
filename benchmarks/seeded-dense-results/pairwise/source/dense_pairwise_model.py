"""Research adjacent-packet edges, with no retained correspondence pruning."""

from __future__ import annotations

import inspect

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def pairwise_cost(data, valid, radius, max_shift, dx_m):
    """Signed local NCC edge cost in the dense solver's [row, offset, target] axes.

    The unit-weight term has the same cost-per-metre scale as seed NCC. Unsupported
    packet pairs use the comparator's fixed gap cost; they supply no correlation.
    No top-k, reciprocal rank, phase or displacement alternative is discarded.
    """
    valid = np.asarray(valid, bool) & np.isfinite(data)
    data = np.where(valid, data, 0).astype(float)
    size = 2 * radius + 1
    sums = uniform_filter1d(data, size, axis=1, mode="constant") * size
    energy = np.sqrt(
        np.maximum(
            uniform_filter1d(data * data, size, axis=1, mode="constant") * size
            - sums * sums / size,
            0,
        )
    )
    support = minimum_filter1d(valid, size, axis=1, mode="constant", cval=False)
    support &= energy > 1e-10
    rows, samples = data.shape
    output = np.full((rows - 1, 2 * max_shift + 1, samples), 0.65 * dx_m)
    for index, offset in enumerate(range(-max_shift, max_shift + 1)):
        target = np.arange(max(0, offset), min(samples, samples + offset))
        source = target - offset
        shifted = np.zeros((rows - 1, samples))
        shifted[:, target] = data[:-1, source] * data[1:, target]
        dot = uniform_filter1d(shifted, size, axis=1, mode="constant") * size
        dot = dot[:, target] - sums[:-1, source] * sums[1:, target] / size
        denominator = energy[:-1, source] * energy[1:, target]
        cosine = np.clip(dot / np.maximum(denominator, 1e-20), -1, 1)
        both = support[:-1, source] & support[1:, target]
        output[:, index, target] = np.where(both, 1 - cosine, 0.65) * dx_m
    return output


def instrument(picker):
    """Change only the dense transition objective in a source-checked local copy."""
    source = inspect.getsource(picker)
    original = "    transition = config.geometry_weight * robust * dx_m\n"
    if source.count(original) != 1:
        raise ValueError("Dense comparator source changed; review the instrumentation")
    updated = source.replace(
        original,
        original + "    transition += _pairwise_cost(data, valid, radius, max_shift, dx_m)\n",
    )
    namespace = dict(picker.__globals__, _pairwise_cost=pairwise_cost)
    exec(compile(updated, "<isolated-pairwise-picker>", "exec"), namespace)
    return namespace[picker.__name__]
