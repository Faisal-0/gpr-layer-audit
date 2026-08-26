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
    SEED = "seed"
    MANUAL = "manual"
    INTERPOLATED = "interpolated"


class DielectricSource(StrEnum):
    REFLECTION = "reflection"
    REFLECTION_RECURSIVE = "reflection_recursive"
    ANALYST = "analyst"
    CORE = "core"
    ASSUMED_SCAN = "assumed_scan"
    UNRESOLVED = "unresolved"


class VisibilityState(StrEnum):
    VISIBLE = "visible"
    UNCERTAIN = "uncertain"
    NOT_VISIBLE = "not_visible"
    ABSENT = "absent"


class TrackingProvenance(StrEnum):
    SIGNAL_ONLY = "signal_only"
    DESIGN_AGREEMENT = "design_assisted_agreement"
    DESIGN_CONFLICT = "design_conflict"
    MANUAL_CORRECTION = "manual_correction"
    INTERPOLATED = "interpolated"


class LabelOrigin(StrEnum):
    MANUAL = "manual"
    INTERPOLATED = "interpolated"
    FORMULA = "formula"
    EXTRAPOLATED = "extrapolated"
    UNKNOWN = "unknown"


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
class SurveyLine:
    survey_id: str
    dzt_path: Path
    dzg_path: Path | None
    dzx_path: Path | None
    is_calibration: bool
    trace_count: int
    samples_per_trace: int
    range_ns: float
    scans_per_meter: float
    antenna: str
    gain_signature: str
    warnings: list[str] = field(default_factory=list)
    project_path: Path | None = None


@dataclass(slots=True)
class CalibrationCandidate:
    road_survey_id: str
    calibration_survey_id: str
    compatibility_score: float
    gain_compatible: bool
    problems: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SurveyCatalog:
    root: Path
    surveys: list[SurveyLine]
    calibration_candidates: list[CalibrationCandidate]
    reference_files: list[Path] = field(default_factory=list)
    design_files: list[Path] = field(default_factory=list)
    files_by_type: dict[str, list[Path]] = field(default_factory=dict)
    orphan_files: list[Path] = field(default_factory=list)

    @property
    def roads(self) -> list[SurveyLine]:
        return [item for item in self.surveys if not item.is_calibration]

    @property
    def calibrations(self) -> list[SurveyLine]:
        return [item for item in self.surveys if item.is_calibration]


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
            cls(3, "Sub-base course", 90, 285, 18),
        ]


@dataclass(slots=True)
class LayerDefinition(LayerSpec):
    """Public layer definition used by seed files and catalog workflows."""


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
class DesignPrior:
    layer_order: int
    layer_name: str
    start_chainage_m: float
    end_chainage_m: float
    design_thickness_mm: float
    weight: float = 0.20
    sigma_fraction: float = 0.30
    tolerance_low_mm: float | None = None
    tolerance_high_mm: float | None = None


@dataclass(slots=True)
class ReferencePoint:
    road_id: str
    line_id: str
    chainage_m: float
    layer_order: int
    cumulative_depth_mm: float
    individual_thickness_mm: float | None
    label_origin: LabelOrigin
    source_path: Path
    source_sheet: str
    source_cell: str

    @property
    def trustworthy(self) -> bool:
        return self.label_origin == LabelOrigin.MANUAL


@dataclass(slots=True)
class SeedStation:
    station_id: str
    chainage_m: float
    samples: dict[int, float] = field(default_factory=dict)
    visibility: dict[int, VisibilityState] = field(default_factory=dict)
    role: str = "initial"

    def visible_sample(self, layer_order: int) -> float | None:
        if self.visibility.get(layer_order, VisibilityState.VISIBLE) != VisibilityState.VISIBLE:
            return None
        return self.samples.get(layer_order)


@dataclass(slots=True)
class TrackingEvidence:
    signal_score: float = 0.0
    absolute_strength: float = 0.0
    seed_correlation: float = 0.0
    phase_score: float = 0.0
    coherence_score: float = 0.0
    candidate_margin: float = 0.0
    forward_backward_agreement: float = 0.0
    perturbation_stability: float = 0.0
    design_score: float = 0.0
    local_snr: float = 0.0


@dataclass(slots=True)
class InterfacePath:
    layer_order: int
    samples: np.ndarray
    confidence: np.ndarray
    visibility: list[VisibilityState]
    interpolated: np.ndarray
    evidence: list[TrackingEvidence]
    signal_only_samples: np.ndarray | None = None
    design_guided_samples: np.ndarray | None = None


@dataclass(slots=True)
class ReviewRegion:
    layer_order: int
    layer_name: str
    start_chainage_m: float
    end_chainage_m: float
    priority: float
    reasons: list[str]
    suggested_chainage_m: float | None = None


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
    visibility: VisibilityState = VisibilityState.VISIBLE
    provenance: TrackingProvenance = TrackingProvenance.SIGNAL_ONLY
    signal_only_sample: float | None = None
    design_guided_sample: float | None = None
    interpolated: bool = False
    evidence: TrackingEvidence = field(default_factory=TrackingEvidence)


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
    priority: float = 0.0
    suggested_chainage_m: float | None = None


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
    display_radargrams: dict[str, np.ndarray] = field(default_factory=dict)
    seed_stations: list[SeedStation] = field(default_factory=list)
    proposed_seed_chainages: list[float] = field(default_factory=list)
    signal_only_paths: dict[int, np.ndarray] = field(default_factory=dict)
    design_guided_paths: dict[int, np.ndarray] = field(default_factory=dict)
    benchmark_summary: dict[str, Any] = field(default_factory=dict)

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
            "seeds": [
                {
                    "station_id": station.station_id,
                    "chainage_m": station.chainage_m,
                    "samples": station.samples,
                    "visibility": {
                        str(order): str(value) for order, value in station.visibility.items()
                    },
                    "role": station.role,
                }
                for station in self.seed_stations
            ],
            "review": {
                "groups": len(self.review_issues),
                "unresolved_picks": sum(
                    item.status == PickStatus.UNRESOLVED for item in self.picks
                ),
                "design_conflicts": sum(
                    item.provenance == TrackingProvenance.DESIGN_CONFLICT
                    for item in self.picks
                ),
            },
            "benchmark_summary": self.benchmark_summary,
            "manual_reference": {
                "diagnostic_points": len(self.reference_diagnostics),
                "non_interpolated_points": sum(
                    not item.interpolated_reference for item in self.reference_diagnostics
                ),
            },
        }
