"""Coordinate-explicit optional evidence shared by all hybrid execution paths.

This module deliberately has no dependency on a learning framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PREPROCESSING_VERSION = "seed-hybrid-conventional-v2"


@dataclass(slots=True)
class LearnedEvidence:
    likelihood: np.ndarray
    visibility: np.ndarray
    valid: np.ndarray
    layer_order: int
    weight: float = 0.0
    provenance: dict = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        if self.layer_order not in (1, 2, 3):
            raise ValueError("Unknown learned interface")
        if self.likelihood.shape != shape or self.valid.shape != shape:
            raise ValueError("Learned evidence must use the working trace/sample grid")
        if self.visibility.shape != (shape[0],):
            raise ValueError("Visibility must contain one value per trace")
        if self.valid.dtype != np.bool_:
            raise ValueError("Evidence validity must be boolean")
        if not 0 <= self.weight <= 0.35:
            raise ValueError("ML evidence weight must be bounded to [0, .35]")
        for values in (self.likelihood, self.visibility):
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError("Learned scores must be finite values in [0, 1]")


@dataclass(slots=True)
class HybridEvidence:
    """Arrays always have trace first, original working sample second."""

    measurement: np.ndarray
    sample_interval_ns: float
    horizontal_step_m: float
    learned: dict[int, LearnedEvidence] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    valid: np.ndarray | None = None
    coordinate_transforms: list[dict] = field(default_factory=list)
    resolved_pulses: dict[int, dict] = field(default_factory=dict)

    def validate(self, shape: tuple[int, int]) -> None:
        if self.measurement.shape != shape:
            raise ValueError("Measurement evidence does not match the working radargram")
        if not np.isfinite(self.sample_interval_ns) or self.sample_interval_ns <= 0:
            raise ValueError("A positive sample interval is required")
        if not np.isfinite(self.horizontal_step_m) or self.horizontal_step_m <= 0:
            raise ValueError("A positive horizontal interval is required")
        if self.valid is not None and (self.valid.shape != shape or self.valid.dtype != np.bool_):
            raise ValueError("Sample validity must match the working radar grid")
        finite = np.isfinite(self.measurement)
        if np.any(~finite if self.valid is None else (~finite & self.valid)):
            raise ValueError("Non-finite measurement marked as valid")
        for order, evidence in self.learned.items():
            if order != evidence.layer_order:
                raise ValueError("Learned interface key mismatch")
            evidence.validate(shape)
