"""Versioned physical settings. Research scores remain uncalibrated until frozen evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True, slots=True)
class ConventionalConfig:
    version: str = "conventional-v2"
    signal_validity: bool = True
    directed_correspondence: bool = True
    integrated_motion: bool = False
    whole_trace_registration: bool = False
    waveform_context_radius_m: float = 0.0
    minimum_motion_margin: float = 0.0
    adapt_seed_templates: bool = False
    interval_seed_scoring: bool = False
    distinct_path_inference: bool = False
    template_update_support: float = 0.90
    template_update_path_margin: float = 0.10
    template_update_min_rows: int = 3
    template_update_gap_m: float = 0.5
    template_minimum_seed_similarity: float = 0.60
    template_adaptation_weight: float = 0.20
    complete_paths: bool = True
    path_acceptance: bool = True
    physical_pulse: bool = True
    hybrid_layers: tuple[int, ...] = (2, 3)
    matching_distances_m: tuple[float, ...] = (1, 2, 5, 10, 25)
    fallback_lobe_ns: float = 0.205078125
    template_context_ns: float = 0.615234375
    displacement_ns: float = 0.205078125
    lateral_context_m: float = 2.8
    transition_cost_per_m: float = 1
    gap_cost_per_m: float = 0.1
    minimum_correspondence: float = 0.80
    minimum_path_margin: float = 0.02
    minimum_measurement_support: float = 0.015
    alignment_before_stacking: bool = False
    promoted_layers: tuple[int, ...] = ()
    acceptance_by_layer: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.version != "conventional-v2":
            raise ValueError("Unsupported conventional configuration version")
        if not np.isfinite(self.waveform_context_radius_m) or self.waveform_context_radius_m < 0:
            raise ValueError("Waveform context radius must be finite and nonnegative")
        if not np.isfinite(self.minimum_motion_margin) or not 0 <= self.minimum_motion_margin <= 1:
            raise ValueError("Motion margin must be between zero and one")
        if (
            not 0.8 <= self.template_update_support <= 1
            or not 0 < self.template_update_path_margin <= 1
            or not 0.5 <= self.template_minimum_seed_similarity <= 1
            or not 0 <= self.template_adaptation_weight <= 0.35
            or not isinstance(self.template_update_min_rows, int)
            or self.template_update_min_rows < 3
        ):
            raise ValueError("Invalid guarded template-adaptation settings")
        for name in (
            "fallback_lobe_ns",
            "template_context_ns",
            "displacement_ns",
            "lateral_context_m",
            "template_update_gap_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.promoted_layers:
            raise ValueError(
                "Promotion requires independent field evidence; this configuration is experimental"
            )
        for order, threshold in self.acceptance_by_layer.items():
            if (
                int(order) not in (1, 2, 3)
                or not 0.5 <= threshold.get("minimum_correspondence", 0.8) <= 1
                or not 0 < threshold.get("minimum_path_margin", 0.02) <= 1
            ):
                raise ValueError("Invalid per-layer acceptance thresholds")
        if any(order not in (1, 2, 3) for order in self.hybrid_layers):
            raise ValueError("Unknown hybrid layer")
        if not 0.5 <= self.minimum_correspondence <= 1 or not 0 < self.minimum_path_margin <= 1:
            raise ValueError("Invalid conventional acceptance thresholds")
        if (
            not self.matching_distances_m
            or any(not np.isfinite(x) or x <= 0 for x in self.matching_distances_m)
            or min(self.transition_cost_per_m, self.gap_cost_per_m) < 0
        ):
            raise ValueError("Invalid physical graph settings")

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def resolve_config(value=None):
    return value if isinstance(value, ConventionalConfig) else ConventionalConfig(**(value or {}))


@dataclass(frozen=True, slots=True)
class PulseDescription:
    lobe_ns: float
    packet_ns: float
    context_ns: float
    displacement_ns: float
    dt_ns: float
    source: str
    fallback_used: bool

    @property
    def lobe_samples(self):
        return max(1.0, self.lobe_ns / self.dt_ns)

    @property
    def context_radius(self):
        return max(2, round(self.context_ns / self.dt_ns / 2))

    def metadata(self):
        return {
            **asdict(self),
            "lobe_samples": self.lobe_samples,
            "packet_samples": self.packet_ns / self.dt_ns,
            "packet_estimator": (
                "three-times seeded lobe width proxy; "
                "not an independently measured packet boundary"
            ),
            "context_radius_samples": self.context_radius,
            "displacement_samples": self.displacement_ns / self.dt_ns,
        }


def resolve_pulse(measurement, valid, anchors, metadata, dt_ns, config):
    """Estimate only at supplied support seeds; withheld picks never reach this function."""
    if not np.isfinite(dt_ns) or dt_ns <= 0:
        raise ValueError("Positive sample interval required")
    widths, packets, verified = [], [], []
    for row, sample in sorted(anchors.items()):
        item = (metadata or {}).get(row, {})
        if item.get("pulse_estimation_use") is False:
            # A local observation may constrain identity without recutting every
            # original seed template and changing the candidate graph's scale.
            continue
        if item.get("verified") or item.get("source_coordinate_verified"):
            value = item.get("selected_lobe_width_ns")
            if value is None and item.get("pulse_width_samples"):
                value = float(item["pulse_width_samples"]) * dt_ns
            if value is not None and np.isfinite(value) and value > 0:
                verified.append(float(value))
        if not valid[row, sample] or measurement[row, sample] == 0:
            continue
        trace = measurement[row]
        polarity = np.sign(trace[sample])
        left = right = sample
        while left > 0 and valid[row, left - 1] and np.sign(trace[left - 1]) == polarity:
            left -= 1
        while (
            right + 1 < len(trace)
            and valid[row, right + 1]
            and np.sign(trace[right + 1]) == polarity
        ):
            right += 1
        if 1 < right - left + 1 < len(trace) / 4:
            widths.append((right - left + 1) * dt_ns)
            packets.append(3 * (right - left + 1) * dt_ns)
    source = (
        "verified_seed_metadata"
        if verified
        else "seed_waveform_zero_crossings"
        if widths
        else "physical_fallback"
    )
    lobe = float(np.median(verified or widths)) if verified or widths else config.fallback_lobe_ns
    packet = max(lobe, float(np.median(packets)) if packets else 3 * lobe)
    return PulseDescription(
        lobe,
        packet,
        max(packet, config.template_context_ns),
        max(lobe, config.displacement_ns),
        dt_ns,
        source,
        not (verified or widths),
    )
