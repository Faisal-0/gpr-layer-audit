"""Display and measurement time coordinates, separate from graph sample indices.

Processed DZT samples retain their header time origin. Raw analysis keeps the
existing calibrated display origin and its independently estimated surface.
"""

from __future__ import annotations

from gpr_layer_audit.models import AnalysisResult


def sample_time_ns(result: AnalysisResult, sample):
    """Time at a sample centre on the displayed radargram (scalar or array)."""
    origin = (
        result.header.position_ns if result.parameters.get("input_mode") == "processed" else 0.0
    )
    return origin + sample * result.header.sample_interval_ns


def sample_from_time_ns(result: AnalysisResult, time_ns):
    """Invert the display mapping without quantizing or changing native samples."""
    return (time_ns - sample_time_ns(result, 0)) / result.header.sample_interval_ns


def measurement_zero_sample(result: AnalysisResult) -> float:
    """Exact zero of reported TWTT; never use this float for signal indexing."""
    if result.parameters.get("input_mode") == "processed":
        return -result.header.position_ns / result.header.sample_interval_ns
    return float(result.reference_surface_sample)


def sample_twtt_ns(result: AnalysisResult, canonical_sample):
    """Measurement time uses the canonical event, independently of its drawn lobe."""
    if result.parameters.get("input_mode") == "processed":
        return sample_time_ns(result, canonical_sample)
    return (canonical_sample - result.reference_surface_sample) * result.header.sample_interval_ns


def image_time_bounds_ns(result: AnalysisResult) -> tuple[float, float]:
    """Raster edge times; processed pixel centres coincide with native samples."""
    if result.parameters.get("input_mode") == "processed":
        count = result.calibrated_radargram.shape[1]
        return sample_time_ns(result, -0.5), sample_time_ns(result, count - 0.5)
    return 0.0, float(result.header.range_ns)
