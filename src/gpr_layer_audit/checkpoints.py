from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gpr_layer_audit.models import (
    AnalysisResult,
    PickStatus,
    RetentionAuditRecord,
    ValidationCheckpoint,
    VisibilityState,
)

CHECKPOINT_SCHEMA_VERSION = 1


def checkpoint_document(
    survey_id: str, checkpoints: list[ValidationCheckpoint]
) -> dict:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "survey_id": survey_id,
        "selection_source": "radar_only",
        "training_use": "prohibited",
        "checkpoints": [
            {
                "checkpoint_id": item.checkpoint_id,
                "layer_order": item.layer_order,
                "chainage_m": item.chainage_m,
                "sample_index": item.sample_index,
                "canonical_sample_index": item.canonical_sample_index,
                "visibility": item.visibility.value,
                "user_confirmed": item.user_confirmed,
                "selected_lobe": item.selected_lobe,
                "event_family_id": item.event_family_id,
                "regime_id": item.regime_id,
                "pulse_width_samples": item.pulse_width_samples,
                "source": item.source,
            }
            for item in checkpoints
        ],
    }


def save_checkpoint_file(
    path: str | Path, survey_id: str, checkpoints: list[ValidationCheckpoint]
) -> None:
    Path(path).write_text(
        json.dumps(checkpoint_document(survey_id, checkpoints), indent=2),
        encoding="utf-8",
    )


def load_checkpoint_file(path: str | Path) -> tuple[str, list[ValidationCheckpoint]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema {document.get('schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    if document.get("selection_source") != "radar_only":
        raise ValueError("Validation checkpoints must be captured from radar evidence.")
    survey_id = str(document.get("survey_id") or "").strip()
    if not survey_id:
        raise ValueError("Checkpoint file is missing survey_id.")
    raw_items = document.get("checkpoints")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Checkpoint file must contain at least one checkpoint.")
    output: list[ValidationCheckpoint] = []
    identifiers: set[str] = set()
    for raw in raw_items:
        checkpoint_id = str(raw.get("checkpoint_id") or "").strip()
        if not checkpoint_id or checkpoint_id in identifiers:
            raise ValueError(f"Missing or duplicate checkpoint id: {checkpoint_id!r}")
        identifiers.add(checkpoint_id)
        chainage = float(raw["chainage_m"])
        if not math.isfinite(chainage) or chainage < 0:
            raise ValueError(f"Invalid checkpoint chainage: {chainage}")
        visibility = VisibilityState(raw.get("visibility", VisibilityState.VISIBLE))
        confirmed = raw.get("user_confirmed") is True
        if not confirmed:
            raise ValueError(f"Checkpoint {checkpoint_id!r} is not user-confirmed.")
        sample_value = raw.get("sample_index")
        if visibility == VisibilityState.VISIBLE and sample_value is None:
            raise ValueError(f"Visible checkpoint {checkpoint_id!r} has no sample.")
        sample = float(sample_value) if sample_value is not None else None
        if sample is not None and (not math.isfinite(sample) or sample < 0):
            raise ValueError(f"Checkpoint {checkpoint_id!r} has an invalid sample.")
        canonical_value = raw.get("canonical_sample_index")
        canonical = float(canonical_value) if canonical_value is not None else sample
        if canonical is not None and (not math.isfinite(canonical) or canonical < 0):
            raise ValueError(f"Checkpoint {checkpoint_id!r} has an invalid canonical sample.")
        layer_order = int(raw["layer_order"])
        if layer_order not in {1, 2, 3}:
            raise ValueError(f"Checkpoint {checkpoint_id!r} has an invalid layer order.")
        pulse_width = float(raw.get("pulse_width_samples", 7.0))
        if not math.isfinite(pulse_width) or pulse_width <= 0:
            raise ValueError(f"Checkpoint {checkpoint_id!r} has an invalid pulse width.")
        output.append(
            ValidationCheckpoint(
                checkpoint_id=checkpoint_id,
                layer_order=layer_order,
                chainage_m=chainage,
                sample_index=sample,
                canonical_sample_index=canonical,
                visibility=visibility,
                user_confirmed=True,
                selected_lobe=raw.get("selected_lobe"),
                event_family_id=raw.get("event_family_id"),
                regime_id=str(raw.get("regime_id") or "default"),
                pulse_width_samples=pulse_width,
                source=str(raw.get("source") or "radar_only"),
            )
        )
    return survey_id, output


def evaluate_retention_audit(
    result: AnalysisResult, checkpoints: list[ValidationCheckpoint]
) -> list[RetentionAuditRecord]:
    """Audit event retention without exposing checkpoints to tracking code."""
    output: list[RetentionAuditRecord] = []
    picks_by_order: dict[int, list] = {}
    events_by_order: dict[int, list] = {}
    for pick in result.picks:
        picks_by_order.setdefault(pick.layer_order, []).append(pick)
    for event in result.candidate_events:
        events_by_order.setdefault(event.layer_order, []).append(event)
    for checkpoint in checkpoints:
        row = int(np.argmin(np.abs(result.chainage_m - checkpoint.chainage_m)))
        chainage = float(result.chainage_m[row])
        expected = checkpoint.sample_index
        canonical = checkpoint.canonical_sample_index
        tolerance = max(2.0, checkpoint.pulse_width_samples)
        corridor = result.search_corridors.get(checkpoint.layer_order)
        corridor_includes = True
        if expected is not None and corridor is not None:
            corridor_includes = bool(
                corridor.lower_sample[row] <= expected <= corridor.upper_sample[row]
            )
        events = [
            item
            for item in events_by_order.get(checkpoint.layer_order, [])
            if abs(item.chainage_m - chainage) <= 1e-6
        ]
        matching = (
            [item for item in events if abs(item.sample_index - expected) <= tolerance]
            if expected is not None
            else []
        )
        matched_event = min(matching, key=lambda item: item.rank) if matching else None
        layer_picks = picks_by_order.get(checkpoint.layer_order, [])
        pick = (
            min(layer_picks, key=lambda item: abs(item.chainage_m - chainage))
            if layer_picks
            else None
        )
        selected_sample = (
            float(
                pick.selected_lobe_sample
                if pick.selected_lobe_sample is not None else pick.sample_index
            )
            if pick is not None and pick.sample_index >= 0
            else None
        )
        selected_canonical = (
            float(pick.sample_index)
            if pick is not None and pick.sample_index >= 0
            else None
        )
        graph_sample = pick.evidence.graph_selected_sample if pick is not None else -1.0
        graph_selected = bool(
            expected is not None
            and graph_sample >= 0
            and abs(graph_sample - expected) <= tolerance
        )
        visible = bool(
            pick is not None and pick.sample_index >= 0
            and pick.visibility not in {VisibilityState.ABSENT, VisibilityState.NOT_VISIBLE}
        )
        if checkpoint.visibility != VisibilityState.VISIBLE:
            loss_stage = "retained_absence" if not visible else "false_visible_pick"
        elif not corridor_includes:
            loss_stage = "corridor_exclusion"
        elif matched_event is None:
            loss_stage = "candidate_generation_or_pruning"
        elif not graph_selected and matched_event.seed_reachable is False:
            loss_stage = "seed_lineage_exclusion"
        elif not graph_selected:
            loss_stage = "graph_or_ranker_selection"
        elif not visible:
            loss_stage = "confidence_or_visibility_gate"
        else:
            loss_stage = "retained"
        output.append(
            RetentionAuditRecord(
                checkpoint_id=checkpoint.checkpoint_id,
                layer_order=checkpoint.layer_order,
                chainage_m=chainage,
                expected_sample=expected,
                expected_canonical_sample=canonical,
                expected_visibility=checkpoint.visibility,
                corridor_includes_expected=corridor_includes,
                candidate_generated=matched_event is not None,
                candidate_rank=matched_event.rank if matched_event else None,
                graph_selected=graph_selected,
                selected_sample=selected_sample,
                selected_canonical_sample=selected_canonical,
                selected_family_id=(pick.event_family_id if pick is not None else None),
                confidence=float(pick.confidence if pick is not None else 0.0),
                visible=visible,
                accepted=bool(
                    pick is not None
                    and pick.status == PickStatus.HIGH_CONFIDENCE
                ),
                loss_stage=loss_stage,
                sample_error=(
                    abs(selected_sample - expected)
                    if expected is not None and selected_sample is not None
                    else None
                ),
                canonical_sample_error=(
                    abs(selected_canonical - canonical)
                    if canonical is not None and selected_canonical is not None
                    else None
                ),
                alternative_cycle_margin=(
                    float(pick.evidence.alternative_cycle_margin) if pick is not None else 0.0
                ),
                branch_agreement=(
                    float(pick.evidence.preprocessing_agreement) if pick is not None else 0.0
                ),
            )
        )
    result.retention_audit = output
    return output
