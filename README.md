# GPR Layer Audit

GPR Layer Audit is a research prototype for seed-driven pavement-layer tracking in GSSI surveys. It catalogs a survey directory, proposes representative stations, learns the intended interfaces from up to five user-seeded stations, tracks the road bidirectionally, and groups only uncertain or conflicting spans for review.

## Run during development

Double-click `Run-GPR-Layer-Audit.cmd`. It runs directly from the source environment and keeps a console open so startup or processing errors remain visible. If `.venv` is absent, the launcher uses `uv` to create it first.

From PowerShell, the equivalent command is:

```powershell
uv run gpr-layer-audit-gui
```

## Prototype workflow

1. Click **Catalog directory**, choose the folder containing the road data, and confirm the road, optional calibration, reference, and design files.
2. Build the preview and inspect the **Raw**, **Clean**, **Phase**, **Gradient**, and **Candidates** views.
3. At each of the three suggested stations, select an interface and Ctrl+click it. Mark an interface **Not visible** or **Absent** instead of inventing a click. Up to two later correction stations are allowed.
4. Click **Track from completed seeds**. Seeded paths are tracked jointly and bidirectionally; missing evidence is emitted as a gap rather than a boundary-hugging line.
5. Work through the prioritized review regions with Accept, Correct point, Not visible, Layer absent, or Add structural break.
6. Export the evidence package. Interface sample/TWTT remains the primary observation; millimetres are derived separately from the recorded dielectric source.

The implementation includes memory-mapped DZT input, DZG/DZX attachment, calibration compatibility checks, seed-adaptive waveform/phase/correlation features, ordered optional-state Viterbi paths, a soft design-assisted second pass, and versioned seed/project formats. After three complete seed stations, the highest-priority uncertain regions are automatically reread from the raw acquisition at one-quarter of the global stack size; later corrections use the same bounded fine retracker with fixed constraints at both ends.

## Development

```powershell
uv sync --extra dev
uv run pytest
uv run gpr-layer-audit-gui
```

Catalog Talagang first:

```powershell
uv run gpr-layer-audit catalog "GPR Data\talagang"
```

The supplied Talagang plate scan has incompatible range gain, so it is not auto-selected. A signal-only folder run is:

```powershell
uv run gpr-layer-audit analyze-folder "GPR Data\talagang" `
  --survey-id "TALAGANG.PRJ/TALAGANG_001" `
  --accept-scan-dielectric `
  --output exports\talagang
```

The `--accept-scan-dielectric` flag explicitly labels the DZT/DZX value as **assumed**. It does not turn that value into measured ground truth.

Replay the exact same UI seeds with `--seeds seeds.json`. Run deterministic blocked-span and leave-one-road-out evaluation with:

```powershell
uv run gpr-layer-audit benchmark benchmarks\talagang-baseline.json `
  --output benchmarks\talagang-baseline-result.json
```

A benchmark exits with code 2 when the acceptance gates are not met. That is an intentional research result, not a crash.
Use `--method` to reproduce the primary joint seed-adaptive tracker, the current/enhanced baselines, or the deconvolution and phase/coherence ablations.

## GSSI files versus audit projects

A GSSI `NAME.PRJ` is normally a folder. Select that folder with **Import GSSI survey → PRJ folder…**, or select the `.DZT` waveform inside it. Matching `.DZG` GPS and `.DZX` metadata are attached automatically. Do the same for the metal-plate calibration folder.

The application-created `.gprproj` is different: it is the schema-2 SQLite prototype record containing parameters, seeds, runs, review history, and exports. Prototype schemas are intentionally not migrated; create a new project after incompatible changes.

## Scientific interpretation

Interface sample/time picks are direct signal interpretations. Thickness is derived from travel time and a dielectric source. A failed plate/gain compatibility check disables reflection-amplitude dielectric estimation; the software never silently substitutes a value.

Automatic visibility enhancement runs on an interpretation-only branch. Time gain and normalisation are never used for reflection-amplitude dielectric inversion. A signal-only pass is always preserved. If a design schedule is supplied, a separate pass applies a soft expected-gap prior capped at 20%; disagreement greater than one pulse width is routed to review and never replaces the signal-only observation.

## Prototype status

Executable and installer work is deliberately deferred. Use the BAT/CMD launcher or `uv run` until the tracking and benchmark gates are satisfactory.

Run `uv run python scripts\benchmark_talagang.py` to reproduce the Talagang engine timing and peak-working-set check.
