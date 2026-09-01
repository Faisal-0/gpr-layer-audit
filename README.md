# GPR Layer Audit

GPR Layer Audit is a research prototype for design-corridor pavement-layer tracking in GSSI surveys. Enter tentative individual layer thicknesses, run the radar-driven automatic pass, seed only unknown or ambiguous interfaces, and review grouped exceptions instead of tracing the road manually.

Current reliability findings and limitations are recorded in
[Tracking reliability status](docs/TRACKING_RELIABILITY_STATUS.md). Manual seeds
are valid operating inputs; automatic coverage is not field accuracy.
The interactive default refines at most one local 10 m high-information window;
remaining ambiguity stays in the review queue. Exhaustive uncertain-span
refinement is an explicit research mode. Leave-one-station-out seed validation
runs automatically once three model stations exist. There are no road-length-
based coarsening budgets.

## Run during development

Double-click `Run-GPR-Layer-Audit.cmd`. It runs directly from the source environment and keeps a console open so startup or processing errors remain visible. If `.venv` is absent, the launcher uses `uv` to create it first.

From PowerShell, the equivalent command is:

```powershell
uv run gpr-layer-audit-gui
```

## Prototype workflow

1. Click **Catalog directory**, choose the folder containing the road data, and confirm the road, optional calibration, reference, and design files.
2. Optionally enter tentative design thicknesses or dielectric assumptions, then click **Build preview**. Empty design fields remain unknown; no thickness or dielectric is invented.
3. Inspect **Raw**, **Clean**, **Phase**, **Gradient**, and **Candidates**. Design values are corridor/scale aids only and never replace manual reflector identity. Mark **Not visible** or **Absent** rather than inventing a click.
4. At requested stations, select the layer and Ctrl+click the intended reflector. Asphalt requires at least two distributed observations; base and subbase require three so one can be withheld while two still define identity. Click **Re-run with new model seed** to propagate those observations. You need not pick an unrelated layer at each station. Stronger competing reflectors remain review candidates rather than silently replacing the seeded event.
5. Inspect the linked **Depth profiles** and work through prioritized review
   regions. A solid path is an accepted measurement. A dashed path is a graph
   proposal only and never produces TWTT or thickness. Inspect the radargram
   and A-scan, then use **Confirm proposed reflector** only when its identity is
   correct; otherwise use **Correct point**, **Not visible**, **Layer absent**,
   or **Add structural break**.
6. Export interface sample/TWTT, dielectric-derived depths, profiles, confidence, candidates, anomalies, seed history, retention audits, and provenance to Excel/CSV/GeoJSON/PNG.

Up to five road-scale model stations are supported. Local review corrections
are unlimited: **Correct point** followed by Ctrl+click retracks a ±25 m section
and preserves outside picks. On a full rerun or project reopen, corrections are
excluded from the global fit, seed guides, dropout audit, design calibration,
and required model-station count; they are then replayed only through their
saved ±25 m windows. Confirmed proposals and correction seeds persist in the
project. A saved proposal is reaccepted only when the source fingerprint,
reflector-family identity, lobe, and frozen display/canonical coordinates all
match the new run. Enabled layer choices also round-trip through the project,
with required upper interfaces restored automatically.
Suggestions navigate to useful windows but never move a click to another trace.
Leave subbase disabled unless there is evidence to identify that interface.
Explicit **Not visible** and **Absent** decisions remain accepted analyst
decisions, but are never counted as accepted measurements or automatic coverage.

For tracker development, enable **Capture validation checkpoints** after the
preview and Ctrl+click radar-only events. The app writes a separate
`.checkpoints.json` file; these points are never supplied to candidate
generation, ranking, thresholds, or retracking. A benchmark manifest may name
that file with `"checkpoints": "road.checkpoints.json"` to report exactly where
the expected packet was retained or lost.

The implementation includes memory-mapped DZT input, DZG/DZX attachment,
gain-compatibility checks, stationary-wavelet denoising, matched correlation,
phase/coherence/deconvolution/DTW candidate features, phase-locked event packets,
three stripping hypotheses, ordered optional-state graph paths, anomaly gaps,
schema-4 seed files, and schema-3 project storage. Gain-incompatible plate
subtraction and amplitude dielectric calibration are disabled. Tracking runs
globally near 0.4 m resolution. The interactive automatic pass rereads at most
one 10 m review core near 0.1 m resolution; explicit analyst corrections
retrack their local section, while exhaustive refinement remains opt-in.

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

The supplied Talagang plate scan has incompatible range gain. The software
therefore disables both plate subtraction and amplitude dielectric inversion
for that pairing. Run the design corridor directly with:

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
  --output exports\talagang-design-result.json
```

A benchmark exits with code 2 when the acceptance gates are not met. That is an intentional research result, not a crash. The sole tracker is `joint_seed_adaptive`; superseded baseline implementations have been removed.

## GSSI files versus audit projects

A GSSI `NAME.PRJ` is normally a folder. Select that folder with **Import GSSI survey → PRJ folder…**, or select the `.DZT` waveform inside it. Matching `.DZG` GPS and `.DZX` metadata are attached automatically. Do the same for the metal-plate calibration folder.

The application-created `.gprproj` is different: it is the schema-3 SQLite prototype record containing quick designs, schedules, seeds, runs, review history, and exports. Prototype schemas are intentionally not migrated; create a new project after incompatible changes.

## Scientific interpretation

Interface sample/time picks are direct signal interpretations. Thickness is derived from travel time and a dielectric source. A failed plate/gain compatibility check disables reflection-amplitude dielectric estimation; the software never silently substitutes a value.

Automatic visibility enhancement runs on an interpretation-only branch. Time gain and normalization never enter reflection-amplitude dielectric inversion. A signal-only path is always preserved as diagnostics. Design recursively bounds interface identity and adds at most a 10% tie-break; radar SNR, correlation, ensemble agreement, candidate margin, and forward/backward agreement determine visibility and confidence.

## Prototype status

Executable and installer work is deliberately deferred. Use the CMD launcher or `uv run` until the tracking and benchmark gates are satisfactory.
