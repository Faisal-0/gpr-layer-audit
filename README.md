# GPR Layer Audit

GPR Layer Audit is a research prototype for design-corridor pavement-layer tracking in GSSI surveys. Enter tentative individual layer thicknesses, run the radar-driven automatic pass, seed only unknown or ambiguous interfaces, and review grouped exceptions instead of tracing the road manually.

Current reliability findings and limitations are recorded in
[Tracking reliability status](docs/TRACKING_RELIABILITY_STATUS.md). Talagang
still fails seed-withholding stability; automatic coverage is not field accuracy.
The accuracy-first defaults refine all uncertain spans and independently refit
without each training station, routing unstable picks to review. There are no
runtime/memory release gates or road-length-based coarsening budgets.

## Run during development

Double-click `Run-GPR-Layer-Audit.cmd`. It runs directly from the source environment and keeps a console open so startup or processing errors remain visible. If `.venv` is absent, the launcher uses `uv` to create it first.

From PowerShell, the equivalent command is:

```powershell
uv run gpr-layer-audit-gui
```

## Prototype workflow

1. Click **Catalog directory**, choose the folder containing the road data, and confirm the road, optional calibration, reference, and design files.
2. Confirm the quick design form (defaults: 2 in asphalt, 4 in base, subbase unknown, assumed εr 7) and click **Build preview**.
3. Inspect **Raw**, **Clean**, **Phase**, **Gradient**, and **Candidates**. Design-known layers run without seeds; an unknown subbase requests two stations. Mark **Not visible** or **Absent** rather than inventing a click.
4. Re-run from completed requested seeds. The joint tracker follows radar evidence inside recursive physical corridors; missing evidence and structural anomalies remain gaps.
5. Inspect the linked **Depth profiles** and work through prioritized review regions with Accept, Correct point, Not visible, Layer absent, or Add structural break.
6. Export interface sample/TWTT, dielectric-derived depths, profiles, confidence, candidates, anomalies, seed history, retention audits, and provenance to Excel/CSV/GeoJSON/PNG.

For tracker development, enable **Capture validation checkpoints** after the
preview and Ctrl+click radar-only events. The app writes a separate
`.checkpoints.json` file; these points are never supplied to candidate
generation, ranking, thresholds, or retracking. A benchmark manifest may name
that file with `"checkpoints": "road.checkpoints.json"` to report exactly where
the expected packet was retained or lost.

The implementation includes memory-mapped DZT input, DZG/DZX attachment, waveform-compatible gain-mismatched plate use, stationary-wavelet denoising, matched correlation, phase/coherence/deconvolution/DTW candidate features, phase-locked event packets, three stripping hypotheses, ordered optional-state graph paths, anomaly gaps, schema-4 seed files, and schema-3 project storage. Tracking runs globally near 0.4 m resolution and rereads selected uncertain spans near 0.1 m resolution.

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

The supplied Talagang plate scan has incompatible range gain. It is still useful for normalized waveform timing, ringing suppression, and matched filtering, but amplitude dielectric inversion remains disabled. Run the design corridor directly with:

```powershell
uv run gpr-layer-audit analyze-folder "GPR Data\talagang" `
  --survey-id "TALAGANG.PRJ/TALAGANG_001" `
  --asphalt-thickness 2in `
  --base-thickness 4in `
  --output exports\talagang
```

The `--accept-scan-dielectric` flag explicitly labels the DZT/DZX value as **assumed**. It does not turn that value into measured ground truth.

Replay the exact same UI seeds with `--seeds seeds.json`. Run deterministic blocked-span and leave-one-road-out evaluation with:

```powershell
uv run gpr-layer-audit benchmark benchmarks\talagang-design.json `
  --output benchmarks\talagang-design-result.json
```

A benchmark exits with code 2 when the acceptance gates are not met. That is an intentional research result, not a crash.
Use `--method` to reproduce the primary joint seed-adaptive tracker, the current/enhanced baselines, or the deconvolution and phase/coherence ablations.

## GSSI files versus audit projects

A GSSI `NAME.PRJ` is normally a folder. Select that folder with **Import GSSI survey → PRJ folder…**, or select the `.DZT` waveform inside it. Matching `.DZG` GPS and `.DZX` metadata are attached automatically. Do the same for the metal-plate calibration folder.

The application-created `.gprproj` is different: it is the schema-3 SQLite prototype record containing quick designs, schedules, seeds, runs, review history, and exports. Prototype schemas are intentionally not migrated; create a new project after incompatible changes.

## Scientific interpretation

Interface sample/time picks are direct signal interpretations. Thickness is derived from travel time and a dielectric source. A failed plate/gain compatibility check disables reflection-amplitude dielectric estimation; the software never silently substitutes a value.

Automatic visibility enhancement runs on an interpretation-only branch. Time gain and normalization never enter reflection-amplitude dielectric inversion. A signal-only path is always preserved as diagnostics. Design recursively bounds interface identity and adds at most a 10% tie-break; radar SNR, correlation, ensemble agreement, candidate margin, and forward/backward agreement determine visibility and confidence.

## Prototype status

Executable and installer work is deliberately deferred. Use the BAT/CMD launcher or `uv run` until the tracking and benchmark gates are satisfactory.

Run `uv run python scripts\benchmark_talagang.py` to reproduce the Talagang engine timing and peak-working-set check.
