from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path

from gpr_layer_audit.io import read_dzt_header
from gpr_layer_audit.models import CalibrationCandidate, SurveyCatalog, SurveyLine
from gpr_layer_audit.processing.calibration import headers_compatible_for_processing

_CALIBRATION_WORDS = {
    "plate",
    "calibration",
    "calibrate",
    "configuration",
    "configration",
}
_DESIGN_WORDS = {"design", "schedule", "boq", "tolerance", "specification"}
CALIBRATION_PAIRING_MARGIN = 0.015


def _tokens(value: str) -> set[str]:
    return {item for item in re.split(r"[^a-z0-9]+", value.casefold()) if item}


def _survey_id(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    return relative.as_posix()


def _is_calibration(path: Path, scans_per_meter: float) -> bool:
    words = _tokens(str(path.parent)) | _tokens(path.stem)
    return bool(words & _CALIBRATION_WORDS) or scans_per_meter <= 0


def _survey(root: Path, path: Path) -> SurveyLine:
    header = read_dzt_header(path)
    dzg = path.with_suffix(".DZG")
    dzx = path.with_suffix(".DZX")
    warnings: list[str] = []
    if not dzg.exists() and header.scans_per_meter > 0:
        warnings.append("No matching DZG GPS file")
    if not dzx.exists():
        warnings.append("No matching DZX metadata file")
    return SurveyLine(
        survey_id=_survey_id(root, path),
        dzt_path=path,
        dzg_path=dzg if dzg.exists() else None,
        dzx_path=dzx if dzx.exists() else None,
        is_calibration=_is_calibration(path, header.scans_per_meter),
        trace_count=header.trace_count,
        samples_per_trace=header.samples_per_trace,
        range_ns=header.range_ns,
        scans_per_meter=header.scans_per_meter,
        antenna=header.antenna,
        gain_signature=header.gain_signature,
        warnings=warnings,
        project_path=path.parent if path.parent.suffix.casefold() == ".prj" else None,
    )


def _top_group(root: Path, path: Path) -> str:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return ""
    return parts[0].casefold() if len(parts) > 1 else ""


def _candidate(root: Path, road: SurveyLine, plate: SurveyLine) -> CalibrationCandidate:
    road_header = read_dzt_header(road.dzt_path)
    plate_header = read_dzt_header(plate.dzt_path)
    problems = headers_compatible_for_processing(road_header, plate_header)
    waveform_compatible = not problems
    gain_compatible = road_header.gain_signature == plate_header.gain_signature
    if not gain_compatible:
        problems = [*problems, "range gain"]
    score = 1.0
    score -= 0.22 * len(set(problems) - {"range gain"})
    if _top_group(root, road.dzt_path) == _top_group(root, plate.dzt_path):
        score += 0.12
    # Compare acquisition names only. Absolute parent paths share zone names,
    # project suffixes, and workspace tokens that are not evidence that a
    # particular plate belongs to a road.
    road_words = _tokens(road.dzt_path.stem)
    plate_words = _tokens(plate.dzt_path.stem) - _CALIBRATION_WORDS
    overlap = road_words & plate_words
    score += min(0.18, 0.04 * len(overlap))
    if not gain_compatible:
        # Preserve geographic/name evidence for choosing the most plausible
        # waveform reference, but keep every gain-mismatched option below the
        # amplitude-calibration suitability boundary. A hard 0.49 cap made all
        # otherwise compatible plates tie and silently discarded pairing
        # information.
        score = 0.49 * min(max(score, 0.0), 1.30) / 1.30
    return CalibrationCandidate(
        road_survey_id=road.survey_id,
        calibration_survey_id=plate.survey_id,
        compatibility_score=float(max(0.0, min(1.0, score))),
        gain_compatible=gain_compatible,
        waveform_compatible=waveform_compatible,
        problems=sorted(set(problems)),
    )


def discover_survey_catalog(root: str | Path) -> SurveyCatalog:
    directory = Path(root).resolve()
    if not directory.is_dir():
        raise ValueError(f"Survey catalog root is not a directory: {directory}")
    dzt_files = sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.casefold() == ".dzt"
    )
    surveys: list[SurveyLine] = []
    for path in dzt_files:
        try:
            surveys.append(_survey(directory, path))
        except (OSError, ValueError) as exc:
            surveys.append(
                SurveyLine(
                    survey_id=_survey_id(directory, path),
                    dzt_path=path,
                    dzg_path=None,
                    dzx_path=None,
                    is_calibration=False,
                    trace_count=0,
                    samples_per_trace=0,
                    range_ns=0.0,
                    scans_per_meter=0.0,
                    antenna="",
                    gain_signature="",
                    warnings=[f"Could not read DZT header: {exc}"],
                    project_path=(path.parent if path.parent.suffix.casefold() == ".prj" else None),
                )
            )
    roads = [item for item in surveys if not item.is_calibration and item.trace_count > 0]
    plates = [item for item in surveys if item.is_calibration and item.trace_count > 0]
    candidates = [_candidate(directory, road, plate) for road in roads for plate in plates]
    candidates.sort(
        key=lambda item: (
            item.road_survey_id,
            -item.compatibility_score,
            item.calibration_survey_id,
        )
    )
    reference_files: list[Path] = []
    design_files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in {".xlsx", ".xlsm", ".csv"}:
            continue
        if _tokens(path.stem) & _DESIGN_WORDS:
            design_files.append(path)
        else:
            reference_files.append(path)
    files_by_type = {
        ".PRJ": sorted(
            path
            for path in directory.rglob("*")
            if path.is_dir() and path.suffix.casefold() == ".prj"
        ),
        ".DZT": dzt_files,
        ".DZG": sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".dzg"
        ),
        ".DZX": sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".dzx"
        ),
    }
    attached = {
        path.resolve()
        for survey in surveys
        for path in (survey.dzg_path, survey.dzx_path)
        if path is not None
    }
    orphan_files = sorted(
        path
        for extension in (".DZG", ".DZX")
        for path in files_by_type[extension]
        if path.resolve() not in attached
    )
    return SurveyCatalog(
        root=directory,
        surveys=surveys,
        calibration_candidates=candidates,
        reference_files=reference_files,
        design_files=design_files,
        files_by_type=files_by_type,
        orphan_files=orphan_files,
    )


def calibration_candidates_for(
    catalog: SurveyCatalog, road_survey_id: str
) -> list[CalibrationCandidate]:
    return [
        item for item in catalog.calibration_candidates if item.road_survey_id == road_survey_id
    ]


def calibration_pairing_is_ambiguous(
    candidates: list[CalibrationCandidate],
) -> bool:
    return bool(
        len(candidates) > 1
        and candidates[0].compatibility_score - candidates[1].compatibility_score
        < CALIBRATION_PAIRING_MARGIN
    )


def catalog_as_dict(catalog: SurveyCatalog) -> dict:
    def serialise(item):
        values = asdict(item)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in values.items()
        }

    return {
        "root": str(catalog.root),
        "surveys": [serialise(item) for item in catalog.surveys],
        "calibration_candidates": [asdict(item) for item in catalog.calibration_candidates],
        "reference_files": [str(path) for path in catalog.reference_files],
        "design_files": [str(path) for path in catalog.design_files],
        "files_by_type": {
            extension: [str(path) for path in paths]
            for extension, paths in catalog.files_by_type.items()
        },
        "orphan_files": [str(path) for path in catalog.orphan_files],
    }
