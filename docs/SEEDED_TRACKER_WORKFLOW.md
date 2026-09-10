# Native processed seed and correction workflow

The application can run the validated processed DZT coordinates, request an
observation, apply a local correction, and replay saved corrections. This is an
experimental interpretation workflow. The available road experiments do **not**
establish useful base/subbase coverage at 95% accepted agreement. The normal raw
application default remains `joint_seed_adaptive`; `seed_hybrid` remains experimental.

## Run the real three-seed case

From the repository root, with the supplied local survey directory present:

```powershell
$case = (Get-Content benchmarks/seeded-evaluation-inputs.json -Raw | ConvertFrom-Json).cases.'mandiali-short'
.venv/Scripts/python.exe -m gpr_layer_audit.cli analyze $case.dzt `
  --input-mode processed --processed-stride 1 --method seed_hybrid `
  --native-seeds $case.seed_source --query-layers 2 3 `
  --conventional-config benchmarks/conventional-motion-calibrated-development.json `
  --output exports/processed-mandiali
```

For Gujrat, select `gujrat-second` from the same manifest and use stride 4.
These files contain three native observations per interface. Loading Layer 3
observations enables subbase. Incompatible trace grids fail; seeds are never
silently moved onto a retained trace. The input hash is checked before loading
observations. The application does not open the evaluation reference for tracking.

In the GUI:

1. Open the processed DZT in a fresh project. Select **Hybrid tracing ·
   experimental**, **Processed DZT · preserve coordinates**, and the case's
   trace stride.
2. Use **Load tracing configuration…** with the configuration above, then
   **Load native seed observations…** with the case's seed file. The import
   starts tracking. Existing project observations are preserved; import into a
   fresh project when replaying a supplied seed file.
3. Keep **Request observations: Base and subbase**, or select the interface to
   review. At **Resolve ambiguity**, inspect the radar and Ctrl+click the
   intended reflector. A requested observation is a local correction. Use
   **Not visible** or **Absent** if appropriate.
4. The correction replaces only that interface within 25 m of the answer.
   Outside picks and competing hypotheses remain preserved. The worker is
   cancellable. Its computation still uses the full original seed context;
   the tested 11.45 m Mandiali corrections took 66.2 and 60.8 seconds after
   [exact alignment acceleration](EXACT_DTW_ACCELERATION.md). Longer-road
   interactive latency remains a limitation.
5. Save and reopen the project, then run tracking to reconstruct it. Local
   correction order, processed coordinates, selected query layers and explicit
   dielectric assumptions are saved. Export the audit package to inspect
   accepted timing, unresolved picks, seeds and provenance.

**Model seed at clicked chainage** is the explicit operation for adding a
road-wide model observation; it requires a full rerun. It has different scope
from a correction. Additional layers need not be clicked at the same station.
Original model observations determine the pulse/context scale; local answers
do not silently recut every initial seed packet.

Each successful processed correction now saves an ordered snapshot of the
observations and exact layer/window used for that fit. Repeated edits at the
same station replay in their original order. Cancellation restores the prior
stations; undo records a scoped removal. See [correction history](PROCESSED_CORRECTION_HISTORY.md)
for the legacy-project fallback and historical model-context limitation.

Unresolved proposals and negative observations have no accepted TWTT or
thickness. A preserved accepted pick keeps its path at the same sample.
Confirmations are tied to the saved proposal identity and coordinates. An
explicit dielectric assumption permits a derived thickness; it does not make
that thickness an independently calibrated measurement. Processed annotation
exports identify their coordinate system and cannot enter the raw training
loader as raw labels.

Processed reporting bins also require accepted, visible observations throughout
the bin. Individual layer timing uses paired observations of the immediate
upper interface at the same native coordinates. A base-only interpretation can
retain accepted interface picks, but cannot produce a base thickness without
an accepted asphalt interface. Missing upper measurements withhold derived
timing and thickness; review proposals never become numerical measurements
during aggregation.
These same measurement rules remain in force after an analyst review action,
including when a different layer is marked absent, not visible or anomalous.

## Replay a measured interaction budget

```powershell
.venv/Scripts/python.exe scripts/replay_seeded_interaction.py `
  --case mandiali-short --policy active --actions 4 --layers 2 3 `
  --config benchmarks/conventional-motion-calibrated-development.json `
  --output exports/interaction-mandiali-active.json
```

Use `--policy midpoint` in a separate output for the spacing comparator;
`--seeding endpoints` compares two of the fixed initial observations;
`--scope interval` explicitly replaces the original seed bracket. The latter
is an evaluation operation, not the default local GUI correction. `--refit-pulse`
reproduces the old pulse-recalculation mechanism as a comparator.

Only the answer at the radar-selected request coordinate becomes a new input.
If the reviewed answer is unavailable there, the request is logged and consumes
an action. Scoring excludes initial and subsequently supplied observations,
keeps the initial signed-lobe/time tolerance, and also reports coverage using
the fixed initial scoring pool. The request policy never receives references.

Every run snapshots its source and script. Full-path checkpoints preserve
alternatives, supplied answers and visited requests. To resume an interrupted
run, repeat its arguments with `--resume`; source/configuration/input mismatches
are rejected. Resume only checkpoints generated locally by this evaluator.

The actual GUI/CLI/save/reopen/export verification is executable with:

```powershell
.venv/Scripts/python.exe scripts/verify_processed_workflow.py `
  --output exports/processed-workflow-verification
```

See [frozen evaluation and results](seeded-evaluation.md) for hashes, baseline
denominators and road grouping. Mandiali and Gujrat layer identities were
confirmed by the user. There is no unused annotated road in the supplied data,
so these results are development evidence, not untouched-road validation.

On a new checkout, if `exports/seeded-tracker/baseline-source` is absent, restore
the frozen numerical source from the recorded Git commit before running the
evaluation scripts. Do not overwrite an existing snapshot:

```powershell
New-Item -ItemType Directory -Force exports/seeded-tracker | Out-Null
git archive --format=zip --output=exports/seeded-tracker/baseline-source.zip c774cf97572e889fa6cbff690a490e9ae4d2e951 src
Expand-Archive exports/seeded-tracker/baseline-source.zip exports/seeded-tracker/baseline-source
```

The original data directory is intentionally excluded from Git. Supply the
same hash-matched local files listed in the manifest; no unregistered raw to
processed transform is inferred by these commands.
