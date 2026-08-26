# GPR Layer Audit

GPR Layer Audit is an offline Windows workbench for GSSI pavement surveys. It reads raw GSSI files without RADAN, tracks pavement interfaces, keeps dielectric assumptions explicit, routes uncertain segments to an analyst, and exports a reproducible design-comparison package.

## Run during development

Double-click `Run-GPR-Layer-Audit.cmd`. It runs directly from the source environment and keeps a console open so startup or processing errors remain visible. If `.venv` is absent, the launcher uses `uv` to create it first.

From PowerShell, the equivalent command is:

```powershell
uv run gpr-layer-audit-gui
```

## Current capabilities

- Memory-mapped DZT reader with DZG GPS and DZX metadata parsing.
- Metal-plate compatibility checks, surface alignment, coherent-waveform subtraction, and calibrated/assumed dielectric handling.
- Automatic interpretation preprocessing: dewow, zero-phase data-driven band-pass, bounded time gain, robust trace normalisation, light anisotropic denoising, and plate-wavelet matched filtering.
- Deterministic multi-interface dynamic-programming tracker with confidence and review issues.
- Versioned SQLite project store with anchors and audit history.
- Re-openable projects, persisted anchors, bounded local re-tracking, and anchor undo.
- Excel, CSV, GeoJSON, PNG, JSON manifest, and ZIP export.
- PySide6/PyQtGraph radar-first review workbench and command-line analysis.

## Development

```powershell
uv sync --extra dev
uv run pytest
uv run gpr-layer-audit-gui
```

Analyze Talagang from the command line:

```powershell
uv run gpr-layer-audit analyze `
  "GPR Data\talagang\TALAGANG.PRJ\TALAGANG_001.DZT" `
  --plate "GPR Data\talagang\TALAGANG METAL PLATE.PRJ\TALAGANG METAL PLATE_001.DZT" `
  --accept-scan-dielectric `
  --output exports\talagang
```

The `--accept-scan-dielectric` flag explicitly labels the DZT/DZX value as **assumed**. It does not turn that value into measured ground truth.

## GSSI files versus audit projects

A GSSI `NAME.PRJ` is normally a folder. Select that folder with **Import GSSI survey → PRJ folder…**, or select the `.DZT` waveform inside it. Matching `.DZG` GPS and `.DZX` metadata are attached automatically. Do the same for the metal-plate calibration folder.

The application-created `.gprproj` is different: it is the resumable SQLite audit record containing parameters, anchors, runs, review history, and exports. Use **Open audit project** only for those `.gprproj` files.

## Scientific interpretation

Interface sample/time picks are direct signal interpretations. Thickness is derived from travel time and a dielectric source. A failed plate/gain compatibility check disables reflection-amplitude dielectric estimation; the software never silently substitutes a value.

Automatic visibility enhancement runs on an interpretation-only branch. Time gain and normalisation are never used for reflection-amplitude dielectric inversion. Every applied step and its numeric parameters are written to the project and export manifest.

## Packaging

Run `powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1`. The script produces a PyInstaller application folder. `installer\GPRLayerAudit.iss` can then be compiled with Inno Setup; code-signing hooks are documented in that file.

For direct use or copying to another folder, use `dist\GPRLayerAudit-Portable.exe`. It is a true single-file build. The non-portable `dist\GPRLayerAudit\GPRLayerAudit.exe` must remain beside its complete `_internal` directory.

Run `uv run python scripts\benchmark_talagang.py` to reproduce the Talagang engine timing and peak-working-set check.
