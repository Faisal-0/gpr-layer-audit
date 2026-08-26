from __future__ import annotations

import json
import math
from pathlib import Path
from uuid import uuid4

from gpr_layer_audit.models import SeedStation, VisibilityState

SEED_SCHEMA_VERSION = 1
MAX_SEED_STATIONS = 5


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
                "samples": {str(order): value for order, value in item.samples.items()},
                "visibility": {
                    str(order): str(value) for order, value in item.visibility.items()
                },
                "role": item.role,
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
        samples = {int(order): float(value) for order, value in raw.get("samples", {}).items()}
        if any(not math.isfinite(value) or value < 0 for value in samples.values()):
            raise ValueError(f"Seed station {station_id} contains an invalid sample index.")
        visibility = {
            int(order): VisibilityState(value)
            for order, value in raw.get("visibility", {}).items()
        }
        stations.append(
            SeedStation(
                station_id=station_id,
                chainage_m=chainage,
                samples=samples,
                visibility=visibility,
                role=str(raw.get("role") or "initial"),
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
