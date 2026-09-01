from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

from gpr_layer_audit.models import CandidateEvent, SeedStation, VisibilityState

SEED_SCHEMA_VERSION = 4
MAX_SEED_STATIONS = 5


def model_seed_stations(stations: list[SeedStation]) -> list[SeedStation]:
    """Return only road-scale reflector-identity observations."""

    return [item for item in stations if item.role != "correction"]


def correction_seed_stations(stations: list[SeedStation]) -> list[SeedStation]:
    """Return analyst corrections that may only guide bounded retracking."""

    return [item for item in stations if item.role == "correction"]


def _model_station_count(stations: list[SeedStation]) -> int:
    """Count road-scale training stations; local corrections are unlimited."""

    return len(model_seed_stations(stations))


def _station_picks(item: SeedStation) -> dict[str, dict]:
    orders = sorted(
        set(item.samples)
        | set(item.visibility)
        | set(item.user_confirmed)
        | set(item.phase_class)
    )
    if not orders:
        raise ValueError(f"Seed station {item.station_id!r} contains no layer decisions.")
    output: dict[str, dict] = {}
    for order in orders:
        visibility = item.visibility.get(order, VisibilityState.VISIBLE)
        confirmed = bool(item.user_confirmed.get(order, False))
        sample = item.samples.get(order)
        if not confirmed:
            raise ValueError(
                f"Seed station {item.station_id!r}, layer {order} is not user-confirmed."
            )
        if visibility == VisibilityState.VISIBLE and sample is None:
            raise ValueError(
                f"Visible seed station {item.station_id!r}, layer {order} has no sample."
            )
        if sample is not None and (not math.isfinite(sample) or sample < 0):
            raise ValueError(
                f"Seed station {item.station_id!r}, layer {order} has an invalid sample."
            )
        if visibility == VisibilityState.VISIBLE:
            required = {
                "phase_class": item.phase_class.get(order),
                "analytic_phase_rad": item.analytic_phase_rad.get(order),
                "polarity": item.polarity.get(order),
                "selected_lobe": item.selected_lobe.get(order),
                "canonical_sample_index": item.canonical_samples.get(order),
                "pulse_width_samples": item.pulse_width_samples.get(order),
                "event_id": item.event_ids.get(order),
                "regime_id": item.regime_ids.get(order),
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    f"Visible seed station {item.station_id!r}, layer {order} is missing "
                    f"event metadata: {', '.join(missing)}. Reconfirm it in the radar preview."
                )
        output[str(order)] = {
            "sample_index": float(sample) if sample is not None else None,
            "visibility": str(visibility),
            "user_confirmed": True,
            "phase_class": item.phase_class.get(order),
            "analytic_phase_rad": item.analytic_phase_rad.get(order),
            "polarity": item.polarity.get(order),
            "selected_lobe": item.selected_lobe.get(order),
            "canonical_sample_index": item.canonical_samples.get(order),
            "pulse_width_samples": item.pulse_width_samples.get(order),
            "event_id": item.event_ids.get(order),
            "family_id": item.family_ids.get(order),
            "competing_family_id": item.competing_family_ids.get(order),
            "regime_id": item.regime_ids.get(order),
            "competing_samples": item.competing_samples.get(order, []),
            "preview_paths": item.preview_paths.get(order, {}),
            "preview": {
                "start_chainage_m": item.preview_start_chainage_m.get(order),
                "end_chainage_m": item.preview_end_chainage_m.get(order),
                "status": item.preview_status.get(order, "not_run"),
                "warning": item.warnings.get(order),
            },
        }
    return output


def seed_document(
    survey_id: str,
    stations: list[SeedStation],
    *,
    layer_names: dict[int, str] | None = None,
) -> dict:
    if _model_station_count(stations) > MAX_SEED_STATIONS:
        raise ValueError(
            f"At most {MAX_SEED_STATIONS} model-training seed stations are allowed; "
            "localized review corrections are unlimited."
        )
    identities: dict[tuple[int, str], tuple[str, int, int]] = {}
    for station in model_seed_stations(stations):
        for order in station.samples:
            if station.visible_sample(order) is None:
                continue
            regime = station.regime_ids.get(order, "")
            identity = (
                station.selected_lobe.get(order, ""),
                station.polarity.get(order, 0),
                station.phase_class.get(order, -1),
            )
            prior = identities.setdefault((order, regime), identity)
            phase_distance = abs(prior[2] - identity[2])
            phase_distance = min(phase_distance, 8 - phase_distance)
            if prior[0] != identity[0] or prior[1] != identity[1] or phase_distance >= 3:
                raise ValueError(
                    f"Layer {order} seeds in regime {regime!r} use conflicting wavelet lobes. "
                    "Reconfirm the seed or assign an explicit construction regime change."
                )
    return {
        "schema_version": SEED_SCHEMA_VERSION,
        "survey_id": survey_id,
        "layers": {str(order): name for order, name in (layer_names or {}).items()},
        "stations": [
            {
                "station_id": item.station_id,
                "chainage_m": item.chainage_m,
                "role": item.role,
                "picks": _station_picks(item),
            }
            for item in stations
        ],
    }


def save_seed_file(
    path: str | Path,
    survey_id: str,
    stations: list[SeedStation],
    *,
    layer_names: dict[int, str] | None = None,
) -> None:
    document = seed_document(survey_id, stations, layer_names=layer_names)
    Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_seed_file(path: str | Path) -> tuple[str, list[SeedStation]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SEED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported seed schema {document.get('schema_version')}; "
            f"expected {SEED_SCHEMA_VERSION}."
        )
    survey_id = str(document.get("survey_id") or "").strip()
    if not survey_id:
        raise ValueError("Seed file is missing survey_id.")
    raw_stations = document.get("stations")
    if not isinstance(raw_stations, list):
        raise ValueError("Seed file stations must be a list.")
    model_count = sum(
        str(item.get("role") or "initial") != "correction"
        for item in raw_stations
        if isinstance(item, dict)
    )
    if model_count > MAX_SEED_STATIONS:
        raise ValueError(
            f"Seed file contains more than {MAX_SEED_STATIONS} model-training stations."
        )
    stations: list[SeedStation] = []
    identifiers: set[str] = set()
    for raw in raw_stations:
        station_id = str(raw.get("station_id") or uuid4())
        if station_id in identifiers:
            raise ValueError(f"Duplicate seed station id: {station_id}")
        identifiers.add(station_id)
        chainage = float(raw["chainage_m"])
        if not math.isfinite(chainage) or chainage < 0:
            raise ValueError(f"Invalid seed chainage: {chainage}")
        raw_picks = raw.get("picks")
        if not isinstance(raw_picks, dict) or not raw_picks:
            raise ValueError(f"Seed station {station_id} has no confirmed layer picks.")
        samples: dict[int, float] = {}
        visibility: dict[int, VisibilityState] = {}
        user_confirmed: dict[int, bool] = {}
        phase_class: dict[int, int] = {}
        analytic_phase_rad: dict[int, float] = {}
        polarity: dict[int, int] = {}
        selected_lobe: dict[int, str] = {}
        canonical_samples: dict[int, float] = {}
        pulse_width_samples: dict[int, float] = {}
        event_ids: dict[int, str] = {}
        regime_ids: dict[int, str] = {}
        competing_samples: dict[int, list[float]] = {}
        family_ids: dict[int, str] = {}
        competing_family_ids: dict[int, str] = {}
        preview_paths: dict[int, dict[str, list[float]]] = {}
        preview_status: dict[int, str] = {}
        preview_start: dict[int, float] = {}
        preview_end: dict[int, float] = {}
        warnings: dict[int, str] = {}
        for raw_order, raw_pick in raw_picks.items():
            order = int(raw_order)
            if not isinstance(raw_pick, dict):
                raise ValueError(
                    f"Seed station {station_id}, layer {order} must be an object."
                )
            state = VisibilityState(raw_pick.get("visibility", VisibilityState.VISIBLE))
            confirmed = raw_pick.get("user_confirmed") is True
            if not confirmed:
                raise ValueError(
                    f"Seed station {station_id}, layer {order} is not user-confirmed."
                )
            sample_value = raw_pick.get("sample_index")
            if state == VisibilityState.VISIBLE:
                if sample_value is None:
                    raise ValueError(
                        f"Visible seed station {station_id}, layer {order} has no sample."
                    )
                sample = float(sample_value)
                if not math.isfinite(sample) or sample < 0:
                    raise ValueError(
                        f"Seed station {station_id}, layer {order} has an invalid sample."
                    )
                samples[order] = sample
                required = (
                    "phase_class",
                    "analytic_phase_rad",
                    "polarity",
                    "selected_lobe",
                    "canonical_sample_index",
                    "pulse_width_samples",
                    "event_id",
                    "regime_id",
                )
                missing = [name for name in required if raw_pick.get(name) is None]
                if missing:
                    raise ValueError(
                        f"Visible seed station {station_id}, layer {order} is missing "
                        f"event metadata: {', '.join(missing)}. Reconfirm it in the radar preview."
                    )
                phase_class[order] = int(raw_pick["phase_class"])
                analytic_phase_rad[order] = float(raw_pick["analytic_phase_rad"])
                polarity[order] = int(raw_pick["polarity"])
                selected_lobe[order] = str(raw_pick["selected_lobe"])
                canonical_samples[order] = float(raw_pick["canonical_sample_index"])
                pulse_width_samples[order] = float(raw_pick["pulse_width_samples"])
                event_ids[order] = str(raw_pick["event_id"])
                if raw_pick.get("family_id"):
                    family_ids[order] = str(raw_pick["family_id"])
                if raw_pick.get("competing_family_id"):
                    competing_family_ids[order] = str(raw_pick["competing_family_id"])
                regime_ids[order] = str(raw_pick["regime_id"])
                competing_samples[order] = [
                    float(value) for value in raw_pick.get("competing_samples", [])
                ]
                raw_paths = raw_pick.get("preview_paths") or {}
                parsed_paths = {
                    str(name): [float(value) for value in values]
                    for name, values in raw_paths.items()
                    if isinstance(values, list)
                }
                if parsed_paths:
                    preview_paths[order] = parsed_paths
            visibility[order] = state
            user_confirmed[order] = True
            preview = raw_pick.get("preview") or {}
            preview_status[order] = str(preview.get("status") or "not_run")
            if preview.get("start_chainage_m") is not None:
                preview_start[order] = float(preview["start_chainage_m"])
            if preview.get("end_chainage_m") is not None:
                preview_end[order] = float(preview["end_chainage_m"])
            if preview.get("warning"):
                warnings[order] = str(preview["warning"])
        stations.append(
            SeedStation(
                station_id=station_id,
                chainage_m=chainage,
                samples=samples,
                visibility=visibility,
                role=str(raw.get("role") or "initial"),
                user_confirmed=user_confirmed,
                phase_class=phase_class,
                analytic_phase_rad=analytic_phase_rad,
                polarity=polarity,
                selected_lobe=selected_lobe,
                canonical_samples=canonical_samples,
                pulse_width_samples=pulse_width_samples,
                event_ids=event_ids,
                regime_ids=regime_ids,
                competing_samples=competing_samples,
                preview_status=preview_status,
                preview_start_chainage_m=preview_start,
                preview_end_chainage_m=preview_end,
                warnings=warnings,
                family_ids=family_ids,
                competing_family_ids=competing_family_ids,
                preview_paths=preview_paths,
            )
        )
    stations.sort(key=lambda item: item.chainage_m)
    return survey_id, stations


def stations_as_anchors(
    stations: list[SeedStation],
    *,
    include_corrections: bool = False,
) -> dict[int, list[tuple[float, float]]]:
    """Convert stations to anchors, excluding local corrections by default."""

    output: dict[int, list[tuple[float, float]]] = {}
    selected = stations if include_corrections else model_seed_stations(stations)
    for station in selected:
        for order, sample in station.samples.items():
            if station.visibility.get(order, VisibilityState.VISIBLE) != VisibilityState.VISIBLE:
                continue
            output.setdefault(order, []).append((station.chainage_m, sample))
    return output


def attach_candidate_event_metadata(
    stations: list[SeedStation],
    events: list[CandidateEvent],
    *,
    maximum_sample_distance: float = 4.0,
) -> None:
    """Attach phase-complete event identity to confirmed radar clicks in place."""
    for station in stations:
        for order, sample in station.samples.items():
            if station.visible_sample(order) is None:
                continue
            candidates = [
                event
                for event in events
                if event.layer_order == order
                and abs(event.chainage_m - station.chainage_m) <= 0.75
            ]
            if not candidates:
                raise ValueError(
                    f"No reflection event exists near seed {station.station_id!r}, layer {order}."
                )
            selected = min(
                candidates,
                key=lambda event: (
                    abs(event.chainage_m - station.chainage_m),
                    abs(event.sample_index - sample),
                ),
            )
            if abs(selected.sample_index - sample) > maximum_sample_distance:
                separation = abs(selected.sample_index - sample)
                raise ValueError(
                    f"Seed {station.station_id!r}, layer {order} is {separation:.1f} "
                    "samples from the nearest reflection event; reconfirm it in the radar preview."
                )
            station.phase_class[order] = selected.phase_class
            station.analytic_phase_rad[order] = selected.analytic_phase_rad
            station.polarity[order] = selected.polarity
            station.selected_lobe[order] = selected.selected_lobe
            station.canonical_samples[order] = float(
                selected.canonical_sample_index
                if selected.canonical_sample_index is not None
                else selected.sample_index
            )
            station.pulse_width_samples[order] = max(
                1.0, float(selected.pulse_width_samples or 7.0)
            )
            station.event_ids[order] = selected.event_id or (
                f"L{order}:{station.chainage_m:.3f}:{selected.sample_index}"
            )
            if selected.event_family_id:
                station.family_ids[order] = selected.event_family_id
            if selected.competing_family_id:
                station.competing_family_ids[order] = selected.competing_family_id
            station.regime_ids.setdefault(order, "default")
            competing_ids = set(selected.competing_event_ids)
            station.competing_samples[order] = [
                float(event.sample_index)
                for event in candidates
                if event.event_id in competing_ids
            ]
