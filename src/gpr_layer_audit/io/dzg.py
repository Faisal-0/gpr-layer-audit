from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GPSObservation:
    trace_index: int
    utc: str
    latitude: float
    longitude: float
    altitude_m: float | None
    fix_quality: int
    satellites: int


def _coordinate(value: str, hemisphere: str) -> float:
    raw = float(value)
    degrees = int(raw // 100)
    minutes = raw - degrees * 100
    decimal = degrees + minutes / 60.0
    if hemisphere.upper() in {"S", "W"}:
        decimal *= -1
    return decimal


def read_dzg(path: str | Path) -> list[GPSObservation]:
    observations: list[GPSObservation] = []
    pending_trace: int | None = None
    with Path(path).open("r", encoding="ascii", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split(",")
            record = fields[0].lstrip("$").upper()
            if record == "GSSIS" and len(fields) >= 2:
                try:
                    pending_trace = int(float(fields[1]))
                except ValueError:
                    pending_trace = None
                continue
            if record != "GPGGA" or pending_trace is None or len(fields) < 10:
                continue
            try:
                latitude = _coordinate(fields[2], fields[3])
                longitude = _coordinate(fields[4], fields[5])
                fix_quality = int(fields[6] or 0)
                satellites = int(fields[7] or 0)
                altitude = float(fields[9]) if fields[9] else None
            except (ValueError, IndexError):
                pending_trace = None
                continue
            if fix_quality > 0:
                observations.append(
                    GPSObservation(
                        trace_index=pending_trace,
                        utc=fields[1],
                        latitude=latitude,
                        longitude=longitude,
                        altitude_m=altitude,
                        fix_quality=fix_quality,
                        satellites=satellites,
                    )
                )
            pending_trace = None
    observations.sort(key=lambda item: item.trace_index)
    return observations


def interpolate_gps(
    observations: list[GPSObservation], trace_indices: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    targets = np.asarray(trace_indices, dtype=float)
    latitude = np.full(targets.shape, np.nan, dtype=float)
    longitude = np.full(targets.shape, np.nan, dtype=float)
    if not observations:
        return latitude, longitude
    traces = np.asarray([item.trace_index for item in observations], dtype=float)
    lat = np.asarray([item.latitude for item in observations], dtype=float)
    lon = np.asarray([item.longitude for item in observations], dtype=float)
    unique, indices = np.unique(traces, return_index=True)
    if len(unique) == 1:
        exact = np.isclose(targets, unique[0])
        latitude[exact] = lat[indices[0]]
        longitude[exact] = lon[indices[0]]
        return latitude, longitude
    valid = (targets >= unique[0]) & (targets <= unique[-1])
    latitude[valid] = np.interp(targets[valid], unique, lat[indices])
    longitude[valid] = np.interp(targets[valid], unique, lon[indices])
    return latitude, longitude
