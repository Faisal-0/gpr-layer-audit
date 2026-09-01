from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from gpr_layer_audit.models import (
    AnalysisResult,
    DesignSegment,
    DielectricSource,
    LayerDesign,
    LayerSpec,
    SeedStation,
    VisibilityState,
)

SCHEMA_VERSION = 3


class ProjectStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @classmethod
    def create(cls, path: str | Path, name: str) -> ProjectStore:
        store = cls(path)
        store.path.parent.mkdir(parents=True, exist_ok=True)
        if store.path.exists():
            raise FileExistsError(f"Project already exists: {store.path}")
        with store.connect() as db:
            db.executescript(
                """
                CREATE TABLE meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE files (
                    role TEXT NOT NULL,
                    path TEXT NOT NULL,
                    fingerprint TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (role, path)
                );
                CREATE TABLE layers (
                    layer_order INTEGER PRIMARY KEY,
                    definition_json TEXT NOT NULL
                );
                CREATE TABLE design_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_json TEXT NOT NULL
                );
                CREATE TABLE layer_designs (
                    layer_order INTEGER PRIMARY KEY,
                    design_json TEXT NOT NULL
                );
                CREATE TABLE seed_stations (
                    station_id TEXT PRIMARY KEY,
                    chainage_m REAL NOT NULL,
                    role TEXT NOT NULL,
                    station_json TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL,
                    seed_snapshot_json TEXT NOT NULL
                );
                CREATE TABLE picks (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    layer_order INTEGER NOT NULL,
                    chainage_m REAL NOT NULL,
                    pick_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, layer_order, chainage_m)
                );
                CREATE TABLE thickness (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    layer_order INTEGER NOT NULL,
                    chainage_m REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, layer_order, chainage_m)
                );
                CREATE TABLE review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    issue_id TEXT,
                    action TEXT NOT NULL,
                    layer_order INTEGER,
                    start_chainage_m REAL,
                    end_chainage_m REAL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                );
                """
            )
            values = {
                "schema_version": str(SCHEMA_VERSION),
                "name": name,
                "survey_id": name,
                "created_utc": datetime.now(UTC).isoformat(),
            }
            db.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", values.items())
        return store

    def validate(self) -> None:
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row["value"]) if row else 0
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Project schema {version} is not supported by this prototype; "
                f"create a new schema-{SCHEMA_VERSION} project."
            )

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_layers(self, layers: list[LayerSpec]) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM layers")
            db.executemany(
                "INSERT INTO layers(layer_order, definition_json) VALUES (?, ?)",
                [(item.order, json.dumps(asdict(item), default=str)) for item in layers],
            )

    def layer_specs(self) -> list[LayerSpec]:
        """Return the stored layer definitions in layer-order order."""

        self.validate()
        with self.connect() as db:
            rows = db.execute(
                "SELECT layer_order, definition_json FROM layers ORDER BY layer_order"
            ).fetchall()
        output: list[LayerSpec] = []
        for row in rows:
            try:
                payload = json.loads(row["definition_json"])
                if not isinstance(payload, dict):
                    raise TypeError("layer definition must be a JSON object")
                payload["order"] = int(payload.get("order", row["layer_order"]))
                if payload["order"] != int(row["layer_order"]):
                    raise ValueError("layer order does not match its stored key")
                payload["dielectric_source"] = DielectricSource(
                    payload.get("dielectric_source", DielectricSource.UNRESOLVED)
                )
                output.append(LayerSpec(**payload))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid stored layer definition for order {row['layer_order']}."
                ) from exc
        return output

    def set_file(self, role: str, path: str | Path, fingerprint: str | None = None) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO files"
                "(role, path, fingerprint, metadata_json) VALUES (?, ?, ?, ?)",
                (role, str(path), fingerprint, "{}"),
            )

    def file_paths(self) -> dict[str, Path]:
        self.validate()
        with self.connect() as db:
            rows = db.execute("SELECT role, path FROM files ORDER BY role").fetchall()
        return {str(row["role"]): Path(row["path"]) for row in rows}

    def latest_parameters(self) -> dict:
        self.validate()
        with self.connect() as db:
            row = db.execute("SELECT parameters_json FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return json.loads(row["parameters_json"]) if row else {}

    def add_design_segments(self, segments: list[DesignSegment]) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM design_segments")
            db.executemany(
                "INSERT INTO design_segments(segment_json) VALUES (?)",
                [(json.dumps(asdict(item)),) for item in segments],
            )

    def design_segments(self) -> list[DesignSegment]:
        self.validate()
        with self.connect() as db:
            rows = db.execute("SELECT segment_json FROM design_segments ORDER BY id").fetchall()
        return [DesignSegment(**json.loads(row["segment_json"])) for row in rows]

    def set_layer_designs(self, designs: list[LayerDesign]) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM layer_designs")
            db.executemany(
                "INSERT INTO layer_designs(layer_order, design_json) VALUES (?, ?)",
                [(item.layer_order, json.dumps(asdict(item), default=str)) for item in designs],
            )

    def layer_designs(self) -> list[LayerDesign]:
        self.validate()
        with self.connect() as db:
            rows = db.execute(
                "SELECT design_json FROM layer_designs ORDER BY layer_order"
            ).fetchall()
        return [LayerDesign(**json.loads(row["design_json"])) for row in rows]

    def save_seed_station(self, station: SeedStation) -> None:
        payload = {
            "station_id": station.station_id,
            "chainage_m": station.chainage_m,
            "samples": station.samples,
            "visibility": {str(order): str(value) for order, value in station.visibility.items()},
            "role": station.role,
            "user_confirmed": station.user_confirmed,
            "phase_class": station.phase_class,
            "analytic_phase_rad": station.analytic_phase_rad,
            "polarity": station.polarity,
            "selected_lobe": station.selected_lobe,
            "canonical_samples": station.canonical_samples,
            "pulse_width_samples": station.pulse_width_samples,
            "event_ids": station.event_ids,
            "regime_ids": station.regime_ids,
            "competing_samples": station.competing_samples,
            "preview_status": station.preview_status,
            "preview_start_chainage_m": station.preview_start_chainage_m,
            "preview_end_chainage_m": station.preview_end_chainage_m,
            "warnings": station.warnings,
            "family_ids": station.family_ids,
            "competing_family_ids": station.competing_family_ids,
            "preview_paths": station.preview_paths,
        }
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO seed_stations"
                "(station_id, chainage_m, role, station_json, created_utc) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    station.station_id,
                    station.chainage_m,
                    station.role,
                    json.dumps(payload),
                    datetime.now(UTC).isoformat(),
                ),
            )

    def seed_stations(self) -> list[SeedStation]:
        self.validate()
        with self.connect() as db:
            rows = db.execute(
                "SELECT station_json FROM seed_stations ORDER BY chainage_m, created_utc"
            ).fetchall()
        output: list[SeedStation] = []
        for row in rows:
            payload = json.loads(row["station_json"])
            output.append(
                SeedStation(
                    station_id=str(payload["station_id"]),
                    chainage_m=float(payload["chainage_m"]),
                    samples={
                        int(order): float(value) for order, value in payload["samples"].items()
                    },
                    visibility={
                        int(order): VisibilityState(value)
                        for order, value in payload.get("visibility", {}).items()
                    },
                    role=str(payload.get("role") or "initial"),
                    user_confirmed={
                        int(order): bool(value)
                        for order, value in payload.get("user_confirmed", {}).items()
                    },
                    phase_class={
                        int(order): int(value)
                        for order, value in payload.get("phase_class", {}).items()
                    },
                    analytic_phase_rad={
                        int(order): float(value)
                        for order, value in payload.get("analytic_phase_rad", {}).items()
                    },
                    polarity={
                        int(order): int(value)
                        for order, value in payload.get("polarity", {}).items()
                    },
                    selected_lobe={
                        int(order): str(value)
                        for order, value in payload.get("selected_lobe", {}).items()
                    },
                    canonical_samples={
                        int(order): float(value)
                        for order, value in payload.get("canonical_samples", {}).items()
                    },
                    pulse_width_samples={
                        int(order): float(value)
                        for order, value in payload.get("pulse_width_samples", {}).items()
                    },
                    event_ids={
                        int(order): str(value)
                        for order, value in payload.get("event_ids", {}).items()
                    },
                    regime_ids={
                        int(order): str(value)
                        for order, value in payload.get("regime_ids", {}).items()
                    },
                    competing_samples={
                        int(order): [float(sample) for sample in values]
                        for order, values in payload.get("competing_samples", {}).items()
                    },
                    preview_status={
                        int(order): str(value)
                        for order, value in payload.get("preview_status", {}).items()
                    },
                    preview_start_chainage_m={
                        int(order): float(value)
                        for order, value in payload.get(
                            "preview_start_chainage_m", {}
                        ).items()
                    },
                    preview_end_chainage_m={
                        int(order): float(value)
                        for order, value in payload.get(
                            "preview_end_chainage_m", {}
                        ).items()
                    },
                    warnings={
                        int(order): str(value)
                        for order, value in payload.get("warnings", {}).items()
                    },
                    family_ids={
                        int(order): str(value)
                        for order, value in payload.get("family_ids", {}).items()
                    },
                    competing_family_ids={
                        int(order): str(value)
                        for order, value in payload.get(
                            "competing_family_ids", {}
                        ).items()
                    },
                    preview_paths={
                        int(order): {
                            str(name): [float(sample) for sample in samples]
                            for name, samples in paths.items()
                        }
                        for order, paths in payload.get("preview_paths", {}).items()
                    },
                )
            )
        return output

    def remove_last_seed_station(self) -> SeedStation | None:
        stations = self.seed_stations()
        if not stations:
            return None
        station = stations[-1]
        with self.connect() as db:
            db.execute("DELETE FROM seed_stations WHERE station_id = ?", (station.station_id,))
        return station

    def save_analysis(self, result: AnalysisResult, plate_path: str | None = None) -> int:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO files"
                "(role, path, fingerprint, metadata_json) VALUES (?, ?, ?, ?)",
                ("road", str(result.source.dzt_path), result.source.fingerprint, "{}"),
            )
            if plate_path:
                db.execute(
                    "INSERT OR REPLACE INTO files"
                    "(role, path, fingerprint, metadata_json) VALUES (?, ?, ?, ?)",
                    ("plate", plate_path, None, "{}"),
                )
            cursor = db.execute(
                "INSERT INTO runs"
                "(created_utc, parameters_json, diagnostics_json, seed_snapshot_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    json.dumps(result.parameters, default=str),
                    json.dumps(asdict(result.diagnostics), default=str),
                    json.dumps([asdict(item) for item in result.seed_stations], default=str),
                ),
            )
            run_id = int(cursor.lastrowid)
            db.executemany(
                "INSERT INTO picks(run_id, layer_order, chainage_m, pick_json) VALUES (?, ?, ?, ?)",
                [
                    (
                        run_id,
                        item.layer_order,
                        item.chainage_m,
                        json.dumps(asdict(item), default=str),
                    )
                    for item in result.picks
                ],
            )
            db.executemany(
                "INSERT INTO thickness"
                "(run_id, layer_order, chainage_m, result_json) VALUES (?, ?, ?, ?)",
                [
                    (
                        run_id,
                        item.layer_order,
                        item.chainage_m,
                        json.dumps(asdict(item), default=str),
                    )
                    for item in result.thickness
                ],
            )
        return run_id

    def record_review_event(
        self,
        action: str,
        *,
        issue_id: str | None = None,
        layer_order: int | None = None,
        start_chainage_m: float | None = None,
        end_chainage_m: float | None = None,
        details: dict | None = None,
    ) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO review_events"
                "(created_utc, issue_id, action, layer_order, start_chainage_m, "
                "end_chainage_m, details_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    issue_id,
                    action,
                    layer_order,
                    start_chainage_m,
                    end_chainage_m,
                    json.dumps(details or {}, default=str),
                ),
            )

    def review_events(self) -> list[dict]:
        """Return saved analyst decisions in chronological order."""

        with self.connect() as db:
            rows = db.execute(
                "SELECT id, created_utc, issue_id, action, layer_order, "
                "start_chainage_m, end_chainage_m, details_json "
                "FROM review_events ORDER BY id"
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "created_utc": row["created_utc"],
                "issue_id": row["issue_id"],
                "action": row["action"],
                "layer_order": row["layer_order"],
                "start_chainage_m": row["start_chainage_m"],
                "end_chainage_m": row["end_chainage_m"],
                "details": json.loads(row["details_json"] or "{}"),
            }
            for row in rows
        ]

    # Small adapters retained only while the current UI is replaced in this branch.
    def add_anchor(self, layer_order: int, chainage_m: float, sample_index: float) -> None:
        station = SeedStation(
            station_id=f"correction-{datetime.now(UTC).timestamp():.6f}",
            chainage_m=chainage_m,
            samples={layer_order: sample_index},
            visibility={layer_order: VisibilityState.VISIBLE},
            role="correction",
            user_confirmed={layer_order: True},
            preview_status={layer_order: "confirmed"},
            preview_start_chainage_m={layer_order: max(0.0, chainage_m - 25.0)},
            preview_end_chainage_m={layer_order: chainage_m + 25.0},
        )
        self.save_seed_station(station)

    def anchors(self) -> dict[int, list[tuple[float, float]]]:
        output: dict[int, list[tuple[float, float]]] = {}
        for station in self.seed_stations():
            for order, sample in station.samples.items():
                output.setdefault(order, []).append((station.chainage_m, sample))
        return output

    def remove_last_anchor(self) -> tuple[int, float, float] | None:
        station = self.remove_last_seed_station()
        if station is None or not station.samples:
            return None
        order = next(iter(station.samples))
        return order, station.chainage_m, station.samples[order]

    def record_export(self, path: str | Path, manifest: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO exports(created_utc, path, manifest_json) VALUES (?, ?, ?)",
                (datetime.now(UTC).isoformat(), str(path), json.dumps(manifest, default=str)),
            )
