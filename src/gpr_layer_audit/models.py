from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np


class PickStatus(StrEnum):
    HIGH_CONFIDENCE = "high_confidence"
    REVIEW = "review"
    UNRESOLVED = "unresolved"
    ACCEPTED = "accepted"


class PickSource(StrEnum):
    AUTO = "auto"
    ANCHOR = "anchor"
    MANUAL = "manual"
    INTERPOLATED = "interpolated"


class DielectricSource(StrEnum):
    REFLECTION = "reflection"
    REFLECTION_RECURSIVE = "reflection_recursive"
    ANALYST = "analyst"
    CORE = "core"
    ASSUMED_SCAN = "assumed_scan"
    UNRESOLVED = "unresolved"


@dataclass(slots=True)
class AcquisitionFileSet:
    dzt_path: Path
    dzg_path: Path | None = None
    dzx_path: Path | None = None
    antenna: str | None = None
    antenna_serial: str | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        self.dzt_path = Path(self.dzt_path)
        if self.dzg_path is None:
            candidate = self.dzt_path.with_suffix(".DZG")
            self.dzg_path = candidate if candidate.exists() else None
        elif self.dzg_path:
            self.dzg_path = Path(self.dzg_path)
        if self.dzx_path is None:
            candidate = self.dzt_path.with_suffix(".DZX")
            self.dzx_path = candidate if candidate.exists() else None
        elif self.dzx_path:
            self.dzx_path = Path(self.dzx_path)


@dataclass(slots=True)
class LayerSpec:
    order: int
    name: str
    min_offset_samples: int
    max_offset_samples: int
    min_gap_samples: int = 8
    analysis_enabled: bool = True
    audit_enabled: bool = True
    start_chainage_m: float | None = None
    end_chainage_m: float | None = None
    dielectric: float | None = None
    dielectric_source: DielectricSource = DielectricSource.UNRESOLVED

    @classmethod
    def defaults(cls) -> list[LayerSpec]:
        return [
            cls(1, "Asphalt", 18, 70, 8),
            cls(2, "Base course", 55, 165, 14),
            cls(3, "Sub-base course", 110, 285, 18),
        ]


@dataclass(slots=True)
class DesignSegment:
    road_id: str
    start_chainage_m: float
    end_chainage_m: float
    layer_name: str
    design_thickness_mm: float
    tolerance_low_mm: float | None = None
    tolerance_high_mm: float | None = None


@dataclass(slots=True)
class InterfacePick:
    layer_order: int
    layer_name: str
    trace_index: int
    chainage_m: float
    sample_index: float
    twtt_ns: float
    amplitude: float
    confidence: float
    status: PickStatus
    source: PickSource = PickSource.AUTO
    latitude: float | None = None
    longitude: float | None = None
    polarity: int = 0


@dataclass(slots=True)
class ThicknessResult:
    layer_order: int
    layer_name: str
    chainage_m: float
    start_chainage_m: float
    end_chainage_m: float
    top_sample: float
    bottom_sample: float
    twtt_ns: float
    dielectric: float | None
    dielectric_source: DielectricSource
    thickness_mm: float | None
    uncertainty_low_mm: float | None
    uncertainty_high_mm: float | None
    confidence: float
    status: PickStatus
    latitude: float | None = None
    longitude: float | None = None
    design_thickness_mm: float | None = None
    deviation_mm: float | None = None
    deviation_percent: float | None = None
    compliance: str | None = None


@dataclass(slots=True)
class ReviewIssue:
    issue_id: str
    layer_order: int
    layer_name: str
    start_chainage_m: float
    end_chainage_m: float
    reasons: list[str]
    suggested_action: str
    status: str = "open"


@dataclass(slots=True)
class CalibrationDiagnostics:
    valid_for_dielectric: bool
    messages: list[str] = field(default_factory=list)
    reference_surface_sample: int = 0
    plate_peak_sample: int = 0
    plate_peak_amplitude: float = 0.0
    surface_amplitude_median: float = 0.0
    gain_compatible: bool = False
    clipping_fraction: float = 0.0
    preprocessing_steps: list[str] = field(default_factory=list)
    preprocessing_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReferenceDiagnostic:
    layer_order: int
    chainage_m: float
    reference_interface_depth_mm: float
    measured_interface_depth_mm: float | None
    absolute_error_mm: float | None
    interpolated_reference: bool
    pick_status: PickStatus
    within_release_target: bool | None


@dataclass(slots=True)
class AnalysisResult:
    source: AcquisitionFileSet
    header: Any
    stack_size: int
    chainage_m: np.ndarray
    calibrated_radargram: np.ndarray
    surface_samples_raw: np.ndarray
    reference_surface_sample: int
    picks: list[InterfacePick]
    thickness: list[ThicknessResult]
    review_issues: list[ReviewIssue]
    diagnostics: CalibrationDiagnostics
    gps_latitude: np.ndarray | None = None
    gps_longitude: np.ndarray | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    reference_diagnostics: list[ReferenceDiagnostic] = field(default_factory=list)
    interpretation_input_radargram: np.ndarray | None = None
    matched_template: np.ndarray | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "source": {
                "dzt": str(self.source.dzt_path),
                "dzg": str(self.source.dzg_path) if self.source.dzg_path else None,
                "dzx": str(self.source.dzx_path) if self.source.dzx_path else None,
                "fingerprint": self.source.fingerprint,
            },
            "stack_size": self.stack_size,
            "trace_bins": int(len(self.chainage_m)),
            "diagnostics": asdict(self.diagnostics),
            "parameters": self.parameters,
            "manual_reference": {
                "diagnostic_points": len(self.reference_diagnostics),
                "non_interpolated_points": sum(
                    not item.interpolated_reference for item in self.reference_diagnostics
                ),
            },
        }
