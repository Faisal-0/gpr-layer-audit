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
    DESIGN_CONSTRAINED = "design_constrained"
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
    waveform_compatible: bool = True
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
class LayerDesign:
    layer_order: int
    layer_name: str
    thickness_mm: float | None
    dielectric: float | None = None
    tolerance_fraction: float = 0.40
    start_chainage_m: float = 0.0
    end_chainage_m: float | None = None
    source: str = "quick_entry"


@dataclass(slots=True)
class SearchCorridor:
    layer_order: int
    chainage_m: np.ndarray
    lower_sample: np.ndarray
    centre_sample: np.ndarray
    upper_sample: np.ndarray
    gap_lower_samples: np.ndarray
    gap_centre_samples: np.ndarray
    gap_upper_samples: np.ndarray
    source: str
    expanded: bool = False
    seed_outlier_chainages_m: list[float] = field(default_factory=list)


@dataclass(slots=True)
class CandidateEvent:
    layer_order: int
    chainage_m: float
    sample_index: int
    rank: int
    radar_score: float
    design_tiebreak: float
    polarity: int
    signed_amplitude: float = 0.0
    analytic_phase_rad: float = 0.0
    phase_class: int = 0
    reflectivity_strength: float = 0.0
    lateral_semblance: float = 0.0
    residual_improvement: float = 0.0
    waveform_correlation: float = 0.0
    signed_waveform_correlation: float = 0.0
    canonical_sample_index: float | None = None
    event_id: str | None = None
    selected_lobe: str = "unknown"
    pulse_width_samples: float = 0.0
    prototype_id: str | None = None
    competing_event_ids: list[str] = field(default_factory=list)
    branch_scores: dict[str, float] = field(default_factory=dict)
    lobe_samples: list[int] = field(default_factory=list)
    lobe_offsets: list[float] = field(default_factory=list)
    event_family_id: str | None = None
    competing_family_id: str | None = None
    alternative_cycle_margin: float = 0.0
    branch_agreement: float = 0.0
    graph_selected: bool = False
    joint_hypothesis_count: int = 0
    packet_id: str | None = None
    spatial_lineage_id: int | None = None
    seed_reachable: bool | None = None


@dataclass(slots=True)
class AnomalyRegion:
    start_chainage_m: float
    end_chainage_m: float
    score: float
    kind: str = "structural_anomaly"
    status: str = "automatic"


@dataclass(slots=True)
class LayerProfilePoint:
    layer_order: int
    layer_name: str
    chainage_m: float
    cumulative_depth_mm: float | None
    individual_thickness_mm: float | None
    cumulative_low_mm: float | None
    cumulative_high_mm: float | None
    individual_low_mm: float | None
    individual_high_mm: float | None
    confidence: float
    status: PickStatus
    interpolated: bool = False
    anomaly: bool = False
    design_thickness_mm: float | None = None


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
    user_confirmed: dict[int, bool] = field(default_factory=dict)
    phase_class: dict[int, int] = field(default_factory=dict)
    analytic_phase_rad: dict[int, float] = field(default_factory=dict)
    polarity: dict[int, int] = field(default_factory=dict)
    selected_lobe: dict[int, str] = field(default_factory=dict)
    canonical_samples: dict[int, float] = field(default_factory=dict)
    pulse_width_samples: dict[int, float] = field(default_factory=dict)
    event_ids: dict[int, str] = field(default_factory=dict)
    regime_ids: dict[int, str] = field(default_factory=dict)
    competing_samples: dict[int, list[float]] = field(default_factory=dict)
    preview_status: dict[int, str] = field(default_factory=dict)
    preview_start_chainage_m: dict[int, float] = field(default_factory=dict)
    preview_end_chainage_m: dict[int, float] = field(default_factory=dict)
    warnings: dict[int, str] = field(default_factory=dict)
    family_ids: dict[int, str] = field(default_factory=dict)
    competing_family_ids: dict[int, str] = field(default_factory=dict)
    preview_paths: dict[int, dict[str, list[float]]] = field(default_factory=dict)

    def visible_sample(self, layer_order: int) -> float | None:
        if self.visibility.get(layer_order, VisibilityState.VISIBLE) != VisibilityState.VISIBLE:
            return None
        if not self.user_confirmed.get(layer_order, False):
            return None
        return self.samples.get(layer_order)


@dataclass(slots=True)
class ConfirmedSeed:
    station_id: str
    layer_order: int
    chainage_m: float
    sample_index: float | None
    visibility: VisibilityState
    user_confirmed: bool
    phase_class: int | None = None
    analytic_phase_rad: float | None = None
    polarity: int | None = None
    selected_lobe: str | None = None
    canonical_sample_index: float | None = None
    pulse_width_samples: float | None = None
    event_id: str | None = None
    regime_id: str = "default"
    competing_samples: list[float] = field(default_factory=list)
    preview_start_chainage_m: float | None = None
    preview_end_chainage_m: float | None = None
    preview_status: str = "not_run"
    warning: str | None = None


@dataclass(slots=True)
class WaveformPrototype:
    layer_order: int
    station_id: str
    chainage_m: float
    sample_index: int
    real_waveform: np.ndarray
    quadrature_waveform: np.ndarray
    polarity: int
    phase_class: int
    radius_samples: int
    analytic_phase_rad: float = 0.0
    selected_lobe: str = "unknown"
    canonical_offset_samples: float = 0.0
    event_id: str | None = None
    regime_id: str = "default"
    propagated_rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int32))


@dataclass(slots=True)
class ReflectionEventPacket:
    """One physical reflection hypothesis containing all observable wavelet lobes."""

    layer_order: int
    row_index: int
    event_id: str
    canonical_sample_index: float
    selected_sample_index: int
    negative_trough_sample: int | None
    positive_peak_sample: int | None
    zero_crossing_samples: tuple[float, ...]
    selected_lobe: str
    polarity: int
    analytic_phase_rad: float
    phase_class: int
    pulse_width_samples: float
    prototype_id: str | None = None
    competing_event_ids: tuple[str, ...] = ()
    lobe_samples: tuple[int, ...] = ()
    lobe_offsets: tuple[float, ...] = ()
    complex_waveform: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.complex64)
    )
    branch_scores: dict[str, float] = field(default_factory=dict)
    competing_samples: tuple[float, ...] = ()
    regime_id: str = "default"


@dataclass(slots=True)
class PhaseLockedTracklet:
    """A seed-conditioned continuation of one reflection-event family."""

    tracklet_id: str
    layer_order: int
    seed_station_id: str
    regime_id: str
    rows: np.ndarray
    samples: np.ndarray
    canonical_samples: np.ndarray
    support: np.ndarray
    phase_classes: np.ndarray
    polarities: np.ndarray
    cycle_slip_risk: np.ndarray
    stopped_reason: str | None = None


@dataclass(slots=True)
class EventFamilyHypothesis:
    """One persistent reflector-family interpretation across a road span."""

    family_id: str
    layer_order: int
    regime_id: str
    samples: np.ndarray
    score: float
    posterior_weight: float
    source_tracklet_ids: tuple[str, ...] = ()
    competing_family_id: str | None = None


@dataclass(slots=True)
class RegimeBoundary:
    chainage_m: float
    layer_order: int
    left_regime_id: str
    right_regime_id: str
    reason: str
    user_confirmed: bool = False


@dataclass(slots=True)
class PathReliability:
    """Evidence support, not a calibrated probability of field accuracy."""

    layer_order: int
    support: np.ndarray
    alternative_cycle_margin: np.ndarray
    seed_distance_support: np.ndarray
    drop_seed_stability: np.ndarray
    joint_hypothesis_support: np.ndarray
    preprocessing_agreement: np.ndarray
    cycle_slip_risk: np.ndarray


@dataclass(slots=True)
class ValidationCheckpoint:
    """Blinded radar-only event identity used for validation, never training."""

    checkpoint_id: str
    layer_order: int
    chainage_m: float
    sample_index: float | None
    visibility: VisibilityState
    user_confirmed: bool
    canonical_sample_index: float | None = None
    selected_lobe: str | None = None
    event_family_id: str | None = None
    regime_id: str = "default"
    pulse_width_samples: float = 7.0
    source: str = "radar_only"


@dataclass(slots=True)
class RetentionAuditRecord:
    """Stage at which a blinded event was retained or lost by the tracker."""

    checkpoint_id: str
    layer_order: int
    chainage_m: float
    expected_sample: float | None
    expected_canonical_sample: float | None
    expected_visibility: VisibilityState
    corridor_includes_expected: bool
    candidate_generated: bool
    candidate_rank: int | None
    graph_selected: bool
    selected_sample: float | None
    selected_canonical_sample: float | None
    selected_family_id: str | None
    confidence: float
    visible: bool
    accepted: bool
    loss_stage: str
    sample_error: float | None = None
    canonical_sample_error: float | None = None
    alternative_cycle_margin: float = 0.0
    branch_agreement: float = 0.0


@dataclass(slots=True)
class ReflectivityEvent:
    layer_order: int
    row_index: int
    sample_index: int
    signed_amplitude: float
    analytic_phase_rad: float
    phase_class: int
    polarity: int
    reflectivity_strength: float
    lateral_semblance: float
    residual_improvement: float
    waveform_correlation: float
    signed_waveform_correlation: float
    design_tiebreak: float
    radar_score: float


@dataclass(slots=True)
class TrackHypothesis:
    layer_order: int
    samples: np.ndarray
    score: float
    posterior_weight: float
    no_pick: np.ndarray


@dataclass(slots=True)
class LayerTrackResult:
    layer_order: int
    selected_samples: np.ndarray
    confidence: np.ndarray
    hypotheses: list[TrackHypothesis]
    events_by_row: list[list[ReflectivityEvent]]
    stripped_input: bool
    review_mask: np.ndarray


@dataclass(slots=True)
class ReviewSpan:
    layer_order: int
    start_chainage_m: float
    end_chainage_m: float
    priority: float
    reason_codes: list[str]
    suggested_chainage_m: float
    neighboring_anchor_ids: tuple[str | None, str | None] = (None, None)


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
    ensemble_agreement: float = 0.0
    preprocessing_agreement: float = 0.0
    design_tiebreak: float = 0.0
    hypothesis_agreement: float = 0.0
    neighborhood_support: float = 0.0
    residual_improvement: float = 0.0
    waveform_similarity: float = 0.0
    reflectivity_strength: float = 0.0
    phase_cycle_agreement: float = 0.0
    path_margin: float = 0.0
    edge_condition: float = 0.0
    canonical_event_sample: float = -1.0
    selected_lobe_code: float = 0.0
    alternative_cycle_margin: float = 0.0
    seed_distance_support: float = 0.0
    drop_seed_stability: float = 0.0
    joint_hypothesis_support: float = 0.0
    tracklet_support: float = 0.0
    cycle_slip_risk: float = 0.0
    branch_multimodality: float = 0.0
    event_family_index: float = -1.0
    regime_index: float = 0.0
    graph_selected_sample: float = -1.0
    pre_gate_confidence: float = 0.0
    spatial_lineage_index: float = -1.0
    seed_reachable: float = 0.0
    lineage_break: float = 0.0


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
    corridor_lower_sample: float | None = None
    corridor_centre_sample: float | None = None
    corridor_upper_sample: float | None = None
    selected_candidate_rank: int | None = None
    anomaly: bool = False
    canonical_event_sample: float | None = None
    selected_lobe_sample: float | None = None
    selected_lobe: str | None = None
    event_family_id: str | None = None
    competing_family_id: str | None = None
    competing_family_sample: float | None = None
    regime_id: str = "default"
    alternative_cycle_margin: float = 0.0
    branch_agreement: float = 0.0
    drop_seed_stability: float = 0.0
    review_reason: str | None = None


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
    interpolated: bool = False
    anomaly: bool = False


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
    reference_individual_thickness_mm: float | None = None
    measured_individual_thickness_mm: float | None = None
    individual_absolute_error_mm: float | None = None
    individual_within_release_target: bool | None = None


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
    search_corridors: dict[int, SearchCorridor] = field(default_factory=dict)
    candidate_events: list[CandidateEvent] = field(default_factory=list)
    anomaly_regions: list[AnomalyRegion] = field(default_factory=list)
    profile: list[LayerProfilePoint] = field(default_factory=list)
    retention_audit: list[RetentionAuditRecord] = field(default_factory=list)

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
                    "user_confirmed": {
                        str(order): bool(value)
                        for order, value in station.user_confirmed.items()
                    },
                    "phase_class": {
                        str(order): int(value) for order, value in station.phase_class.items()
                    },
                    "analytic_phase_rad": {
                        str(order): float(value)
                        for order, value in station.analytic_phase_rad.items()
                    },
                    "polarity": {
                        str(order): int(value) for order, value in station.polarity.items()
                    },
                    "selected_lobe": {
                        str(order): value for order, value in station.selected_lobe.items()
                    },
                    "canonical_samples": {
                        str(order): float(value)
                        for order, value in station.canonical_samples.items()
                    },
                    "pulse_width_samples": {
                        str(order): float(value)
                        for order, value in station.pulse_width_samples.items()
                    },
                    "event_ids": {
                        str(order): value for order, value in station.event_ids.items()
                    },
                    "family_ids": {
                        str(order): value for order, value in station.family_ids.items()
                    },
                    "competing_family_ids": {
                        str(order): value
                        for order, value in station.competing_family_ids.items()
                    },
                    "regime_ids": {
                        str(order): value for order, value in station.regime_ids.items()
                    },
                    "competing_samples": {
                        str(order): [float(sample) for sample in values]
                        for order, values in station.competing_samples.items()
                    },
                    "preview_status": {
                        str(order): value for order, value in station.preview_status.items()
                    },
                    "preview_bounds_m": {
                        str(order): [
                            station.preview_start_chainage_m.get(order),
                            station.preview_end_chainage_m.get(order),
                        ]
                        for order in set(station.preview_start_chainage_m)
                        | set(station.preview_end_chainage_m)
                    },
                    "warnings": {
                        str(order): value for order, value in station.warnings.items()
                    },
                    "preview_paths": {
                        str(order): paths for order, paths in station.preview_paths.items()
                    },
                }
                for station in self.seed_stations
            ],
            "review": {
                "groups": len(self.review_issues),
                "unresolved_picks": sum(
                    item.status == PickStatus.UNRESOLVED for item in self.picks
                ),
                "design_conflicts": sum(
                    item.provenance == TrackingProvenance.DESIGN_CONFLICT for item in self.picks
                ),
                "per_layer": {
                    str(order): {
                        "high_confidence_fraction": sum(
                            item.layer_order == order
                            and item.status in {PickStatus.HIGH_CONFIDENCE, PickStatus.ACCEPTED}
                            for item in self.picks
                        )
                        / max(1, sum(item.layer_order == order for item in self.picks)),
                        "review_fraction": sum(
                            item.layer_order == order and item.status == PickStatus.REVIEW
                            for item in self.picks
                        )
                        / max(1, sum(item.layer_order == order for item in self.picks)),
                        "unresolved_fraction": sum(
                            item.layer_order == order and item.status == PickStatus.UNRESOLVED
                            for item in self.picks
                        )
                        / max(1, sum(item.layer_order == order for item in self.picks)),
                    }
                    for order in sorted({item.layer_order for item in self.picks})
                },
            },
            "benchmark_summary": self.benchmark_summary,
            "retention_audit": {
                "checkpoints": len(self.retention_audit),
                "retained": sum(
                    item.loss_stage in {"retained", "retained_absence"}
                    for item in self.retention_audit
                ),
                "loss_stages": {
                    stage: sum(item.loss_stage == stage for item in self.retention_audit)
                    for stage in sorted({item.loss_stage for item in self.retention_audit})
                },
            },
            "anomalies": [asdict(item) for item in self.anomaly_regions],
            "manual_reference": {
                "diagnostic_points": len(self.reference_diagnostics),
                "non_interpolated_points": sum(
                    not item.interpolated_reference for item in self.reference_diagnostics
                ),
            },
        }
