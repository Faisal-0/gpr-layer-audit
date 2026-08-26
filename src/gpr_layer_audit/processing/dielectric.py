from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from gpr_layer_audit.models import DielectricSource

LIGHT_SPEED_M_PER_S = 299_792_458.0


def surface_reflection_dielectric(
    surface_amplitude: NDArray[np.floating], plate_amplitude: float
) -> NDArray[np.float64]:
    result = np.full(np.asarray(surface_amplitude).shape, np.nan, dtype=float)
    if not np.isfinite(plate_amplitude) or abs(plate_amplitude) < 1e-9:
        return result
    ratio = np.abs(np.asarray(surface_amplitude, dtype=float) / plate_amplitude)
    valid = (ratio > 0.01) & (ratio < 0.85)
    result[valid] = ((1.0 + ratio[valid]) / (1.0 - ratio[valid])) ** 2
    result[(result < 2.0) | (result > 30.0)] = np.nan
    if np.count_nonzero(np.isfinite(result)) >= 5:
        filled = result.copy()
        finite = np.isfinite(filled)
        x = np.arange(len(filled))
        filled[~finite] = np.interp(x[~finite], x[finite], filled[finite])
        result = median_filter(filled, size=min(31, max(3, len(filled) // 20 * 2 + 1)))
    return result


def recursive_dielectric(
    upper_dielectric: NDArray[np.floating],
    surface_ratio: NDArray[np.floating],
    interface_ratio: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Conservative two-interface amplitude estimate with physical rejection.

    This implements the commonly published transmission-corrected reflection
    relationship. Values are only retained where the input ratios and resulting
    permittivity are finite and physically plausible.
    """

    eps = np.asarray(upper_dielectric, dtype=float)
    a1 = np.asarray(surface_ratio, dtype=float)
    a2 = np.asarray(interface_ratio, dtype=float)
    numerator = 1.0 - a1**2 + a2
    denominator = 1.0 - a1 + a2
    with np.errstate(divide="ignore", invalid="ignore"):
        output = eps * (numerator / denominator) ** 2
    output[(output < 2.0) | (output > 30.0) | ~np.isfinite(output)] = np.nan
    return output


def thickness_from_twtt_mm(twtt_ns: float, dielectric: float) -> float:
    if twtt_ns < 0 or dielectric <= 0 or not math.isfinite(dielectric):
        raise ValueError("TWTT and dielectric must be finite and physically valid")
    seconds = twtt_ns * 1e-9
    return LIGHT_SPEED_M_PER_S * seconds / (2.0 * math.sqrt(dielectric)) * 1000.0


def resolve_dielectric(
    reflection_value: float | None,
    analyst_value: float | None,
    scan_value: float | None,
    accept_scan_value: bool,
) -> tuple[float | None, DielectricSource]:
    if reflection_value is not None and np.isfinite(reflection_value):
        return float(reflection_value), DielectricSource.REFLECTION
    if analyst_value is not None and 1.0 < analyst_value <= 40.0:
        return analyst_value, DielectricSource.ANALYST
    if accept_scan_value and scan_value is not None and 1.0 < scan_value <= 40.0:
        return scan_value, DielectricSource.ASSUMED_SCAN
    return None, DielectricSource.UNRESOLVED
