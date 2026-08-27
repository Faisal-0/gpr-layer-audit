from __future__ import annotations

import math

import numpy as np

from gpr_layer_audit.models import (
    DesignSegment,
    DielectricSource,
    LayerDesign,
    LayerSpec,
    SearchCorridor,
    SeedStation,
)

from .dielectric import LIGHT_SPEED_M_PER_S


def samples_per_mm(sample_interval_ns: float, dielectric: float) -> float:
    light_speed_mm_ns = LIGHT_SPEED_M_PER_S * 1e-6
    return 2.0 * math.sqrt(dielectric) / (light_speed_mm_ns * sample_interval_ns)


def _design_value(
    layer: LayerSpec,
    chainage_m: float,
    quick: dict[int, LayerDesign],
    segments: list[DesignSegment],
) -> tuple[float | None, float, str]:
    segment = next(
        (
            item
            for item in segments
            if item.layer_name.casefold() == layer.name.casefold()
            and item.start_chainage_m <= chainage_m <= item.end_chainage_m
        ),
        None,
    )
    if segment is not None:
        tolerance = max(
            abs(segment.tolerance_low_mm or 0.0),
            abs(segment.tolerance_high_mm or 0.0),
            segment.design_thickness_mm * 0.40,
        )
        return segment.design_thickness_mm, tolerance, "schedule"
    item = quick.get(layer.order)
    if item is None or item.thickness_mm is None:
        return None, 0.0, "seed"
    return item.thickness_mm, item.thickness_mm * item.tolerance_fraction, item.source


def _seed_gaps(
    stations: list[SeedStation],
    order: int,
    surface_sample: int,
    chainage_m: np.ndarray,
    preceding_interface: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    locations: list[float] = []
    gaps: list[float] = []
    for station in stations:
        current = station.visible_sample(order)
        previous = surface_sample if order == 1 else station.visible_sample(order - 1)
        if previous is None and order > 1:
            previous = float(
                np.interp(station.chainage_m, chainage_m, preceding_interface)
            )
        if current is None or previous is None or current <= previous:
            continue
        locations.append(station.chainage_m)
        gaps.append(float(current - previous))
    return np.asarray(locations), np.asarray(gaps)


def build_search_corridors(
    chainage_m: np.ndarray,
    reference_surface_sample: int,
    sample_interval_ns: float,
    layers: list[LayerSpec],
    designs: list[LayerDesign],
    segments: list[DesignSegment],
    dielectric_by_layer: dict[int, tuple[float | None, DielectricSource]],
    stations: list[SeedStation],
    *,
    pulse_width_samples: float = 7.0,
) -> dict[int, SearchCorridor]:
    quick = {item.layer_order: item for item in designs}
    output: dict[int, SearchCorridor] = {}
    cumulative_low = np.full(len(chainage_m), float(reference_surface_sample))
    cumulative_centre = cumulative_low.copy()
    cumulative_high = cumulative_low.copy()
    for layer in sorted((item for item in layers if item.analysis_enabled), key=lambda x: x.order):
        seed_outlier_chainages: list[float] = []
        low = np.full(len(chainage_m), np.nan)
        centre = np.full(len(chainage_m), np.nan)
        high = np.full(len(chainage_m), np.nan)
        sources: set[str] = set()
        design_item = quick.get(layer.order)
        resolved_epsilon = dielectric_by_layer.get(layer.order, (None, None))[0]
        base_epsilon = (
            design_item.dielectric
            if design_item and design_item.dielectric
            else resolved_epsilon or 7.0
        )
        for row, distance in enumerate(chainage_m):
            thickness, tolerance, source = _design_value(layer, float(distance), quick, segments)
            if thickness is None:
                continue
            sources.add(source)
            minimum_thickness = max(1.0, thickness - tolerance)
            maximum_thickness = thickness + tolerance
            epsilon_low = float(np.clip(base_epsilon * 0.70, 3.0, 15.0))
            epsilon_high = float(np.clip(base_epsilon * 1.30, 3.0, 15.0))
            centre[row] = thickness * samples_per_mm(sample_interval_ns, base_epsilon)
            low[row] = minimum_thickness * samples_per_mm(sample_interval_ns, epsilon_low)
            high[row] = maximum_thickness * samples_per_mm(sample_interval_ns, epsilon_high)
        seed_locations, seed_values = _seed_gaps(
            stations,
            layer.order,
            reference_surface_sample,
            chainage_m,
            cumulative_centre,
        )
        if len(seed_values):
            finite_design = np.isfinite(centre)
            if np.any(finite_design):
                expected_at_seeds = np.interp(
                    seed_locations,
                    chainage_m[finite_design],
                    centre[finite_design],
                )
                residuals = seed_values - expected_at_seeds
                consensus = np.ones(len(residuals), dtype=bool)
                if len(residuals) >= 3:
                    residual_median = float(np.median(residuals))
                    residual_mad = 1.4826 * float(
                        np.median(np.abs(residuals - residual_median))
                    )
                    consensus = np.abs(residuals - residual_median) <= max(
                        2.0 * pulse_width_samples, 2.5 * residual_mad
                    )
                    if np.count_nonzero(consensus) < 2:
                        nearest = np.argsort(np.abs(residuals - residual_median))[:2]
                        consensus[:] = False
                        consensus[nearest] = True
                consensus_residuals = residuals[consensus]
                correction = float(np.median(consensus_residuals))
                centre[finite_design] += correction
                spread_values = consensus_residuals
                sources.add("seed_residual_calibrated")
                if np.any(~consensus):
                    sources.add("local_seed_outlier")
                    seed_outlier_chainages = [
                        float(value) for value in seed_locations[~consensus]
                    ]
            else:
                centre[:] = float(np.median(seed_values))
                finite_design = np.ones(len(chainage_m), dtype=bool)
                spread_values = seed_values
                sources.add("seed_family_bounded")
            spread = 1.4826 * float(
                np.median(np.abs(spread_values - np.median(spread_values)))
            )
            designed_width = np.where(
                np.isfinite(centre), 0.20 * np.abs(centre), 0.0
            )
            half_width = np.maximum.reduce(
                [
                    np.full(len(chainage_m), 2.0 * pulse_width_samples),
                    np.full(len(chainage_m), 3.0 * spread),
                    designed_width,
                ]
            )
            low = centre - half_width
            high = centre + half_width
            # Confirmed seeds calibrate the reflector family and corridor width.
            # They deliberately do not form an interpolated whole-road depth guide.
        finite = np.isfinite(centre)
        if not np.any(finite):
            continue
        minimum_half_width = 2.0 * pulse_width_samples
        low[finite] = np.minimum(low[finite], centre[finite] - minimum_half_width)
        high[finite] = np.maximum(high[finite], centre[finite] + minimum_half_width)
        gap_low = np.maximum(low, layer.min_gap_samples)
        gap_centre = np.maximum(centre, layer.min_gap_samples)
        gap_high = np.maximum(high, gap_centre + 2.0)
        cumulative_low = cumulative_low + np.where(finite, gap_low, 0.0)
        cumulative_centre = cumulative_centre + np.where(finite, gap_centre, 0.0)
        cumulative_high = cumulative_high + np.where(finite, gap_high, 0.0)
        output[layer.order] = SearchCorridor(
            layer_order=layer.order,
            chainage_m=chainage_m.copy(),
            lower_sample=np.where(finite, cumulative_low, np.nan),
            centre_sample=np.where(finite, cumulative_centre, np.nan),
            upper_sample=np.where(finite, cumulative_high, np.nan),
            gap_lower_samples=gap_low,
            gap_centre_samples=gap_centre,
            gap_upper_samples=gap_high,
            source="+".join(sorted(sources)) or "design",
            seed_outlier_chainages_m=seed_outlier_chainages,
        )
    return output
