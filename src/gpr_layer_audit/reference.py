from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from openpyxl import load_workbook

from gpr_layer_audit.models import (
    AnalysisResult,
    DielectricSource,
    LabelOrigin,
    PickStatus,
    ReferenceDiagnostic,
    ReferencePoint,
)
from gpr_layer_audit.processing.dielectric import thickness_from_twtt_mm


@dataclass(slots=True)
class ManualReferencePoint:
    layer_order: int
    chainage_m: float
    interface_depth_mm: float
    individual_thickness_mm: float | None = None
    interpolated: bool = False


def _header(value) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _is_interpolated(cell) -> bool:
    colour = cell.fill.fgColor
    return cell.fill.fill_type == "solid" and str(colour.rgb).upper().endswith("FFF2CC")


def _header_row(sheet) -> int | None:
    for row in range(1, min(sheet.max_row, 30) + 1):
        values = [
            _header(sheet.cell(row, column).value)
            for column in range(1, sheet.max_column + 1)
        ]
        has_distance = any(
            value.startswith("dist") or "chainage" in value for value in values
        )
        has_layer = any(
            ("layer" in value and "depth" in value)
            or "asphalt" in value
            or value.startswith("ac depth")
            for value in values
        )
        if has_distance and has_layer:
            return row
    return None


def _layer_order(name: str) -> int | None:
    match = re.search(r"layer\s*(\d+)", name)
    if match and "depth" in name:
        return int(match.group(1))
    if "asphalt" in name or name.startswith("ac depth"):
        return 1
    if "base course" in name and "depth" in name:
        return 2
    if ("subbase" in name or "sub-base" in name) and "depth" in name:
        return 3
    return None


def _depth_scale(name: str) -> float:
    if "(in" in name or "inch" in name:
        return 25.4
    if "(cm" in name or name.endswith(" cm"):
        return 10.0
    return 1.0


def _label_origin(value_cell, formula_cell) -> LabelOrigin:
    if _is_interpolated(value_cell) or _is_interpolated(formula_cell):
        return LabelOrigin.INTERPOLATED
    if formula_cell.data_type == "f":
        formula = str(formula_cell.value or "").casefold()
        if "forecast" in formula or "trend" in formula:
            return LabelOrigin.EXTRAPOLATED
        return LabelOrigin.FORMULA
    return LabelOrigin.MANUAL


def normalize_reference_workbook(path: str | Path) -> list[ReferencePoint]:
    """Normalize the supplied RADAN-style workbook variants into one evidence schema."""
    file_path = Path(path)
    if file_path.suffix.casefold() == ".csv":
        return _normalize_reference_csv(file_path)
    values_book = load_workbook(file_path, data_only=True, read_only=False)
    formula_book = load_workbook(file_path, data_only=False, read_only=False)
    output: list[ReferencePoint] = []
    for sheet_name in values_book.sheetnames:
        values_sheet = values_book[sheet_name]
        formula_sheet = formula_book[sheet_name]
        header_row = _header_row(values_sheet)
        if header_row is None:
            continue
        headers = {
            column: _header(values_sheet.cell(header_row, column).value)
            for column in range(1, values_sheet.max_column + 1)
        }
        distance_column = next(
            (
                column
                for column, name in headers.items()
                if name.startswith("dist") or "chainage" in name
            ),
            None,
        )
        if distance_column is None:
            continue
        filename_column = next(
            (column for column, name in headers.items() if name.startswith("filename")), None
        )
        layer_columns: dict[int, tuple[int, float]] = {}
        for column, name in headers.items():
            order = _layer_order(name)
            if order is None:
                continue
            # Prefer the raw RADAN columns near the left over chart/helper duplicates.
            if order not in layer_columns or column < layer_columns[order][0]:
                layer_columns[order] = (column, _depth_scale(name))
        if not layer_columns:
            continue
        for row in range(header_row + 1, values_sheet.max_row + 1):
            distance = values_sheet.cell(row, distance_column).value
            try:
                chainage = float(distance)
            except (TypeError, ValueError):
                continue
            filename = (
                str(values_sheet.cell(row, filename_column).value or file_path.stem)
                if filename_column
                else file_path.stem
            )
            road_id = Path(filename).stem
            cumulative: dict[int, float] = {}
            cells: dict[int, tuple[object, object]] = {}
            for order, (column, scale) in layer_columns.items():
                value_cell = values_sheet.cell(row, column)
                formula_cell = formula_sheet.cell(row, column)
                try:
                    cumulative[order] = float(value_cell.value) * scale
                except (TypeError, ValueError):
                    continue
                cells[order] = (value_cell, formula_cell)
            previous_depth = 0.0
            for order in sorted(cumulative):
                depth = cumulative[order]
                value_cell, formula_cell = cells[order]
                thickness = depth - previous_depth if depth >= previous_depth else None
                output.append(
                    ReferencePoint(
                        road_id=road_id,
                        line_id=road_id,
                        chainage_m=chainage,
                        layer_order=order,
                        cumulative_depth_mm=depth,
                        individual_thickness_mm=thickness,
                        label_origin=_label_origin(value_cell, formula_cell),
                        source_path=file_path,
                        source_sheet=sheet_name,
                        source_cell=value_cell.coordinate,
                    )
                )
                previous_depth = depth
    return output


def _normalize_reference_csv(file_path: Path) -> list[ReferencePoint]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        normalized = {_header(name): name for name in headers}
        distance_name = next(
            (
                original
                for name, original in normalized.items()
                if name.startswith("dist") or "chainage" in name
            ),
            None,
        )
        if distance_name is None:
            return []
        filename_name = next(
            (
                original
                for name, original in normalized.items()
                if name.startswith("filename")
            ),
            None,
        )
        layer_columns = {
            order: (original, _depth_scale(name))
            for name, original in normalized.items()
            if (order := _layer_order(name)) is not None
        }
        output: list[ReferencePoint] = []
        for row_number, row in enumerate(reader, 2):
            try:
                chainage = float(row[distance_name])
            except (KeyError, TypeError, ValueError):
                continue
            filename = str(row.get(filename_name, "") or file_path.stem)
            road_id = Path(filename).stem
            previous_depth = 0.0
            for order in sorted(layer_columns):
                column, scale = layer_columns[order]
                try:
                    depth = float(row[column]) * scale
                except (KeyError, TypeError, ValueError):
                    continue
                output.append(
                    ReferencePoint(
                        road_id=road_id,
                        line_id=road_id,
                        chainage_m=chainage,
                        layer_order=order,
                        cumulative_depth_mm=depth,
                        individual_thickness_mm=(
                            depth - previous_depth if depth >= previous_depth else None
                        ),
                        label_origin=LabelOrigin.MANUAL,
                        source_path=file_path,
                        source_sheet="CSV",
                        source_cell=f"row {row_number}, {column}",
                    )
                )
                previous_depth = depth
        return output


def read_manual_reference(path: str | Path) -> list[ManualReferencePoint]:
    canonical = normalize_reference_workbook(path)
    if not canonical:
        raise ValueError("Manual reference workbook has no recognizable layer-depth table.")
    return [
        ManualReferencePoint(
            layer_order=item.layer_order,
            chainage_m=item.chainage_m,
            interface_depth_mm=item.cumulative_depth_mm,
            individual_thickness_mm=item.individual_thickness_mm,
            interpolated=item.label_origin != LabelOrigin.MANUAL,
        )
        for item in canonical
    ]


def evaluate_manual_reference(
    result: AnalysisResult,
    points: list[ManualReferencePoint],
) -> list[ReferenceDiagnostic]:
    picks_by_layer = {
        order: [item for item in result.picks if item.layer_order == order]
        for order in {item.layer_order for item in result.picks}
    }
    dielectric_parameters = result.parameters.get("dielectric_by_layer", {})
    output: list[ReferenceDiagnostic] = []
    for point in points:
        cumulative_depth = 0.0
        previous_sample = float(result.reference_surface_sample)
        selected_status = PickStatus.UNRESOLVED
        measurable = True
        measured_individual: float | None = None
        for order in range(1, point.layer_order + 1):
            candidates = picks_by_layer.get(order, [])
            if not candidates:
                measurable = False
                break
            selected = min(candidates, key=lambda item: abs(item.chainage_m - point.chainage_m))
            selected_status = selected.status
            if selected.status == PickStatus.UNRESOLVED or selected.sample_index < 0:
                measurable = False
                break
            dielectric_item = dielectric_parameters.get(order) or dielectric_parameters.get(
                str(order)
            )
            dielectric = dielectric_item.get("value") if dielectric_item else None
            source = dielectric_item.get("source") if dielectric_item else None
            if dielectric is None or source == DielectricSource.UNRESOLVED:
                measurable = False
                break
            if selected.sample_index <= previous_sample:
                measurable = False
                break
            twtt = (selected.sample_index - previous_sample) * result.header.sample_interval_ns
            layer_thickness = thickness_from_twtt_mm(twtt, float(dielectric))
            cumulative_depth += layer_thickness
            if order == point.layer_order:
                measured_individual = layer_thickness
            previous_sample = selected.sample_index
        measured = cumulative_depth if measurable else None
        if not measurable:
            measured_individual = None
        error = abs(measured - point.interface_depth_mm) if measured is not None else None
        individual_error = (
            abs(measured_individual - point.individual_thickness_mm)
            if measured_individual is not None and point.individual_thickness_mm is not None
            else None
        )
        target = 12.7 if point.layer_order == 1 else 25.4
        output.append(
            ReferenceDiagnostic(
                layer_order=point.layer_order,
                chainage_m=point.chainage_m,
                reference_interface_depth_mm=point.interface_depth_mm,
                measured_interface_depth_mm=measured,
                absolute_error_mm=error,
                interpolated_reference=point.interpolated,
                pick_status=selected_status,
                within_release_target=error <= target if error is not None else None,
                reference_individual_thickness_mm=point.individual_thickness_mm,
                measured_individual_thickness_mm=measured_individual,
                individual_absolute_error_mm=individual_error,
                individual_within_release_target=(
                    individual_error <= target if individual_error is not None else None
                ),
            )
        )
    result.reference_diagnostics = output
    return output
