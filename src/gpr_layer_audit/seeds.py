from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

from gpr_layer_audit.models import SeedStation, VisibilityState

SEED_SCHEMA_VERSION = 2
MAX_SEED_STATIONS = 5


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
        output[str(order)] = {
            "sample_index": float(sample) if sample is not None else None,
            "visibility": str(visibility),
            "user_confirmed": True,
            "phase_class": item.phase_class.get(order),
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
    if len(stations) > MAX_SEED_STATIONS:
        raise ValueError(f"At most {MAX_SEED_STATIONS} seed stations are allowed.")
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
    if len(raw_stations) > MAX_SEED_STATIONS:
        raise ValueError(f"Seed file contains more than {MAX_SEED_STATIONS} stations.")
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
            visibility[order] = state
            user_confirmed[order] = True
            if raw_pick.get("phase_class") is not None:
                phase_class[order] = int(raw_pick["phase_class"])
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
                preview_status=preview_status,
                preview_start_chainage_m=preview_start,
                preview_end_chainage_m=preview_end,
                warnings=warnings,
            )
        )
    stations.sort(key=lambda item: item.chainage_m)
    return survey_id, stations


def stations_as_anchors(
    stations: list[SeedStation],
) -> dict[int, list[tuple[float, float]]]:
    output: dict[int, list[tuple[float, float]]] = {}
    for station in stations:
        for order, sample in station.samples.items():
            if station.visibility.get(order, VisibilityState.VISIBLE) != VisibilityState.VISIBLE:
                continue
            output.setdefault(order, []).append((station.chainage_m, sample))
    return output
