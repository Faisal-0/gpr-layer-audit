"""Use motion confidence as evidence weight, not as a zero-motion observation."""

from __future__ import annotations

import inspect

import numpy as np


def geometry_cost(slope, support, offsets, dt, dx, scale, weight):
    expected = slope[:-1] * dx / dt
    residual = np.maximum(abs(offsets[None, :, None] - expected[:, None, :]) - 0.5, 0)
    normalized = residual * dt / max(scale * dx, dt / 2)
    robust = np.where(normalized <= 1, normalized**2 / 2, normalized - 0.5)
    return weight * robust * dx * support[:-1, None, :]


def confidence_picker(original):
    code = inspect.getsource(original)
    before = (
        "    expected = slope[:-1] * slope_support[:-1] * dx_m / dt_ns\n"
        "    # Half-sample dead zone prevents grid quantization being mistaken for curvature.\n"
        "    residual = np.maximum(np.abs(offsets[None, :, None] "
        "- expected[:, None, :]) - 0.5, 0)\n"
        "    normalized = residual * dt_ns / max(config.slope_scale_ns_per_m * dx_m, dt_ns / 2)\n"
        "    robust = np.where(normalized <= 1, normalized**2 / 2, normalized - 0.5)\n"
        "    transition = config.geometry_weight * robust * dx_m\n"
    )
    after = (
        "    transition = _geometry_cost(slope, slope_support, offsets, dt_ns, dx_m,\n"
        "                                config.slope_scale_ns_per_m, config.geometry_weight)\n"
    )
    if code.count(before) != 1:
        raise ValueError("Review changed dense geometry before instrumentation")
    code = code.replace(before, after)
    namespace = dict(original.__globals__, _geometry_cost=geometry_cost)
    exec(compile(code, "<dense-confidence-weight>", "exec"), namespace)
    return namespace[original.__name__], code
