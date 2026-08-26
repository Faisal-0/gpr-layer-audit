from __future__ import annotations

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
    fir_values: tuple[float, ...] = ()
    for element in root.iter():
        if _local(element.tag) == "doubleArray" and element.text:
            try:
                fir_values = tuple(float(item) for item in element.text.split(",") if item)
            except ValueError:
                fir_values = ()
            break
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
    )
