from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from gpr_layer_audit.models import AnalysisResult, DesignSegment, LayerSpec

SCHEMA_VERSION = 1


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
        with store.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS files (
                    role TEXT NOT NULL,
                    path TEXT NOT NULL,
                    fingerprint TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (role, path)
                );
                CREATE TABLE IF NOT EXISTS layers (
                    layer_order INTEGER PRIMARY KEY,
                    definition_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS design_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    segment_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    diagnostics_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS picks (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    layer_order INTEGER NOT NULL,
                    chainage_m REAL NOT NULL,
                    pick_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, layer_order, chainage_m)
                );
                CREATE TABLE IF NOT EXISTS thickness (
                    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    layer_order INTEGER NOT NULL,
                    chainage_m REAL NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, layer_order, chainage_m)
                );
                CREATE TABLE IF NOT EXISTS anchors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layer_order INTEGER NOT NULL,
                    chainage_m REAL NOT NULL,
                    sample_index REAL NOT NULL,
                    created_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_utc TEXT NOT NULL,
                    issue_id TEXT,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS exports (
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
                "created_utc": datetime.now(UTC).isoformat(),
            }
            db.executemany("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", values.items())
        return store

    def set_layers(self, layers: list[LayerSpec]) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM layers")
            db.executemany(
                "INSERT INTO layers(layer_order, definition_json) VALUES (?, ?)",
                [(item.order, json.dumps(asdict(item), default=str)) for item in layers],
            )

    def set_file(self, role: str, path: str | Path, fingerprint: str | None = None) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO files"
                "(role, path, fingerprint, metadata_json) VALUES (?, ?, ?, ?)",
                (role, str(path), fingerprint, "{}"),
            )

    def file_paths(self) -> dict[str, Path]:
        with self.connect() as db:
            rows = db.execute("SELECT role, path FROM files ORDER BY role").fetchall()
        return {str(row["role"]): Path(row["path"]) for row in rows}

    def latest_parameters(self) -> dict:
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
                "INSERT INTO runs(created_utc, parameters_json, diagnostics_json) VALUES (?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    json.dumps(result.parameters, default=str),
                    json.dumps(asdict(result.diagnostics), default=str),
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

    def add_anchor(self, layer_order: int, chainage_m: float, sample_index: float) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO anchors"
                "(layer_order, chainage_m, sample_index, created_utc) VALUES (?, ?, ?, ?)",
                (layer_order, chainage_m, sample_index, datetime.now(UTC).isoformat()),
            )

    def anchors(self) -> dict[int, list[tuple[float, float]]]:
        output: dict[int, list[tuple[float, float]]] = {}
        with self.connect() as db:
            for row in db.execute(
                "SELECT layer_order, chainage_m, sample_index FROM anchors ORDER BY chainage_m"
            ):
                output.setdefault(int(row["layer_order"]), []).append(
                    (float(row["chainage_m"]), float(row["sample_index"]))
                )
        return output

    def remove_last_anchor(self) -> tuple[int, float, float] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT id, layer_order, chainage_m, sample_index "
                "FROM anchors ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute("DELETE FROM anchors WHERE id = ?", (row["id"],))
        return int(row["layer_order"]), float(row["chainage_m"]), float(row["sample_index"])

    def record_export(self, path: str | Path, manifest: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO exports(created_utc, path, manifest_json) VALUES (?, ?, ?)",
                (datetime.now(UTC).isoformat(), str(path), json.dumps(manifest, default=str)),
            )
