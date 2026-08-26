from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from gpr_layer_audit import __version__
from gpr_layer_audit.models import AnalysisResult

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _serialise(value):
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _rows(items: list) -> tuple[list[str], list[list]]:
    if not items:
        return [], []
    dictionaries = [asdict(item) for item in items]
    headers = list(dictionaries[0])

    def cell(value):
        if hasattr(value, "value"):
            return value.value
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value, default=_serialise)
        return value

    return headers, [[cell(value) for value in row.values()] for row in dictionaries]


def _sheet(workbook: Workbook, title: str, headers: list[str], rows: list[list]) -> None:
    sheet = workbook.create_sheet(title)
    if headers:
        sheet.append(headers)
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="203344")
            cell.alignment = Alignment(vertical="center")
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{max(1, len(rows) + 1)}"
    for row in rows:
        sheet.append(row)
    for column, header in enumerate(headers, 1):
        values = [str(header), *(str(row[column - 1] or "") for row in rows[:300])]
        sheet.column_dimensions[get_column_letter(column)].width = min(
            34, max(10, max(map(len, values)) + 2)
        )


def _workbook(result: AnalysisResult, path: Path) -> None:
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["GPR Layer Audit", "Physics-first pavement thickness analysis"])
    summary.append(["Source", str(result.source.dzt_path)])
    summary.append(["Source SHA-256", result.source.fingerprint])
    summary.append(["Antenna", result.header.antenna])
    summary.append(["Traces", result.header.trace_count])
    summary.append(["Samples per trace", result.header.samples_per_trace])
    summary.append(["Time range (ns)", result.header.range_ns])
    summary.append(["Stack size", result.stack_size])
    summary.append(["Plate valid for dielectric", result.diagnostics.valid_for_dielectric])
    summary.append(["Calibration notes", " | ".join(result.diagnostics.messages)])
    summary.column_dimensions["A"].width = 30
    summary.column_dimensions["B"].width = 90
    summary["A1"].font = Font(bold=True, size=16, color="FFFFFF")
    summary["B1"].font = Font(bold=True, color="FFFFFF")
    for cell in summary[1]:
        cell.fill = PatternFill("solid", fgColor="203344")

    thickness_headers, thickness_rows = _rows(result.thickness)
    _sheet(workbook, "Thickness Results", thickness_headers, thickness_rows)
    design_rows = (
        [
            row
            for row in thickness_rows
            if row[thickness_headers.index("design_thickness_mm")] is not None
        ]
        if thickness_headers
        else []
    )
    _sheet(workbook, "Design Comparison", thickness_headers, design_rows)
    issue_headers, issue_rows = _rows(result.review_issues)
    _sheet(workbook, "Exceptions", issue_headers, issue_rows)
    pick_headers, pick_rows = _rows(result.picks)
    _sheet(workbook, "Interface Picks", pick_headers, pick_rows)
    reference_headers, reference_rows = _rows(result.reference_diagnostics)
    _sheet(workbook, "Manual Reference Diagnostic", reference_headers, reference_rows)
    method = workbook.create_sheet("Method & Provenance")
    method.append(["Software version", __version__])
    method.append(["Generated UTC", datetime.now(UTC).isoformat()])
    method.append(["Parameters", json.dumps(result.parameters, default=_serialise, indent=2)])
    method.append(
        [
            "Interpretation notice",
            "Thickness is estimated from TWTT and the recorded dielectric source; "
            "it is not independently verified core truth.",
        ]
    )
    method.column_dimensions["A"].width = 28
    method.column_dimensions["B"].width = 110
    if "Thickness Results" in workbook.sheetnames and thickness_headers:
        sheet = workbook["Thickness Results"]
        status_col = thickness_headers.index("status") + 1
        col = get_column_letter(status_col)
        sheet.conditional_formatting.add(
            f"{col}2:{col}{sheet.max_row}",
            CellIsRule(
                operator="equal", formula=['"review"'], fill=PatternFill("solid", fgColor="FFF2CC")
            ),
        )
    workbook.save(path)


def _csv(result: AnalysisResult, path: Path) -> None:
    headers, rows = _rows(result.thickness)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def _geojson(result: AnalysisResult, path: Path) -> None:
    features = []
    for item in result.thickness:
        if item.latitude is None or item.longitude is None:
            continue
        properties = asdict(item)
        properties.pop("latitude", None)
        properties.pop("longitude", None)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [item.longitude, item.latitude]},
                "properties": properties,
            }
        )
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features}, default=_serialise, indent=2
        ),
        encoding="utf-8",
    )


def _radargram(
    result: AnalysisResult,
    path: Path,
    start_chainage_m: float | None = None,
    end_chainage_m: float | None = None,
    title: str = "Calibrated radargram and interpreted interfaces",
) -> None:
    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    start = float(result.chainage_m[0]) if start_chainage_m is None else start_chainage_m
    end = float(result.chainage_m[-1]) if end_chainage_m is None else end_chainage_m
    selected = (result.chainage_m >= start) & (result.chainage_m <= end)
    indices = np.flatnonzero(selected)
    if not len(indices):
        plt.close(figure)
        return
    data = result.calibrated_radargram[indices].T
    limit = float(np.percentile(np.abs(data), 98.5)) or 1.0
    extent = [
        float(result.chainage_m[indices[0]]),
        float(result.chainage_m[indices[-1]]),
        result.header.range_ns,
        0,
    ]
    axis.imshow(data, cmap="gray", aspect="auto", vmin=-limit, vmax=limit, extent=extent)
    colours = {1: "#28d7e5", 2: "#ffc857", 3: "#ff6b6b"}
    styles = {1: "-", 2: "--", 3: ":"}
    for order in sorted({item.layer_order for item in result.picks}):
        items = [
            item
            for item in result.picks
            if item.layer_order == order and start <= item.chainage_m <= end
        ]
        if not items:
            continue
        axis.plot(
            [item.chainage_m for item in items],
            [item.sample_index * result.header.sample_interval_ns for item in items],
            color=colours.get(order, "#ffffff"),
            linewidth=1.1,
            linestyle=styles.get(order, "-"),
            label=items[0].layer_name,
        )
    axis.set_xlabel("Chainage (m)")
    axis.set_ylabel("Time (ns)")
    axis.set_title(title)
    axis.legend(loc="upper right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def export_audit_package(result: AnalysisResult, output_directory: str | Path) -> Path:
    root = Path(output_directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package = root / f"gpr_audit_{timestamp}"
    package.mkdir(parents=True, exist_ok=False)
    _workbook(result, package / "gpr_layer_audit.xlsx")
    _csv(result, package / "thickness_results.csv")
    _geojson(result, package / "thickness_results.geojson")
    _radargram(result, package / "annotated_radargram.png")
    radargrams = package / "radargrams"
    radargrams.mkdir()
    _radargram(
        result,
        radargrams / "accepted_overview.png",
        title="Accepted measurements and review overlays",
    )
    exception_directory = radargrams / "exceptions"
    exception_directory.mkdir()
    for index, issue in enumerate(result.review_issues, 1):
        margin = max(5.0, (issue.end_chainage_m - issue.start_chainage_m) * 0.1)
        _radargram(
            result,
            exception_directory / f"exception_{index:03d}_layer_{issue.layer_order}.png",
            max(0.0, issue.start_chainage_m - margin),
            issue.end_chainage_m + margin,
            f"Exception {index}: {issue.layer_name} "
            f"{issue.start_chainage_m:.1f}–{issue.end_chainage_m:.1f} m",
        )
    manifest = result.manifest()
    manifest.update(
        {"software_version": __version__, "generated_utc": datetime.now(UTC).isoformat()}
    )
    (package / "manifest.json").write_text(
        json.dumps(manifest, default=_serialise, indent=2), encoding="utf-8"
    )
    shutil.make_archive(str(package), "zip", root_dir=package)
    return package
