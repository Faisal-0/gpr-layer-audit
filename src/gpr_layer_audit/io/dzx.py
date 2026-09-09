from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True, slots=True)
class DZXMetadata:
    vertical_unit: str | None = None
    horizontal_unit: str | None = None
    dielectric: float | None = None
    units_per_mark: float | None = None
    units_per_scan: float | None = None
    system: str | None = None
    software_version: str | None = None
    antenna_name: str | None = None
    antenna_serial: str | None = None
    samples_per_ns: float | None = None
    custom_fir: tuple[float, ...] = field(default_factory=tuple)
    source_sha256: str = ""
    source_path: str = ""
    radar_name: str | None = None
    scan_range: tuple[int, ...] = ()
    layers: tuple[DZXLayer, ...] = ()
    processing_properties: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class DZXPick:
    trace: int
    sample: int
    channel: int
    interpretation_property: int
    time_ns: float
    recorded_amplitude: float
    recorded_depth: float
    recorded_velocity: float
    coordinate_provenance: str = "RADAN scanSampChanProp; stored zero-based scan/sample"


@dataclass(frozen=True, slots=True)
class DZXLayer:
    number: int
    name: str
    properties: tuple[tuple[str, str], ...]
    picks: tuple[DZXPick, ...]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(root: ElementTree.Element, name: str) -> str | None:
    for element in root.iter():
        if _local(element.tag) == name and element.text:
            return element.text.strip()
    return None


def _float(root: ElementTree.Element, name: str) -> float | None:
    value = _text(root, name)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def read_dzx(path: str | Path) -> DZXMetadata:
    try:
        root = ElementTree.parse(Path(path)).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"Invalid DZX XML: {path}") from exc
    if _local(root.tag) != "DZX":
        raise ValueError(
            "Reviewed layer references require a DZX document; candidate formats are separate"
        )
    fir_values: tuple[float, ...] = ()
    for element in root.iter():
        if _local(element.tag) == "doubleArray" and element.text:
            try:
                fir_values = tuple(float(item) for item in element.text.split(",") if item)
            except ValueError:
                fir_values = ()
            break
    layers = []
    for group in root.iter():
        if _local(group.tag) != "LayerGroup":
            continue
        picks = []
        for point in group:
            if _local(point.tag) != "LayerWayPt":
                continue
            try:
                coordinates = [int(x) for x in (_text(point, "scanSampChanProp") or "").split(",")]
                recorded = [float(x) for x in (_text(point, "timeAmpDepVel") or "").split(",")]
                if len(coordinates) != 4 or len(recorded) != 4:
                    raise ValueError("expected four coordinates and four recorded values")
                picks.append(DZXPick(*coordinates, *recorded))
            except ValueError as exc:
                raise ValueError(f"Malformed LayerWayPt in {path}: {exc}") from exc
        layers.append(
            DZXLayer(
                int(_text(group, "layerNum") or len(layers)),
                _text(group, "groupName") or "Unnamed",
                tuple((_local(x.tag), (x.text or "").strip()) for x in group if not len(x)),
                tuple(picks),
            )
        )
    file_node = next((x for x in root.iter() if _local(x.tag) == "File"), root)
    return DZXMetadata(
        vertical_unit=_text(root, "verticalUnit"),
        horizontal_unit=_text(root, "horizontalUnit"),
        dielectric=_float(root, "dielectric"),
        units_per_mark=_float(root, "unitsPerMark"),
        units_per_scan=_float(root, "unitsPerScan"),
        system=_text(root, "system"),
        software_version=_text(root, "softwareVersion"),
        antenna_name=_text(root, "antennaName"),
        antenna_serial=_text(root, "serialNumber"),
        samples_per_ns=_float(root, "sampsPerNs"),
        custom_fir=fir_values,
        source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        source_path=str(Path(path).resolve()),
        radar_name=_text(file_node, "name"),
        scan_range=tuple(int(x) for x in (_text(file_node, "scanRange") or "").split(",") if x),
        layers=tuple(layers),
        processing_properties=tuple(
            (_local(x.tag), (x.text or "").strip())
            for x in root.iter()
            if not len(x)
            and any(k in _local(x.tag).lower() for k in ("proc", "filter", "timezero", "tzrel"))
        ),
    )
