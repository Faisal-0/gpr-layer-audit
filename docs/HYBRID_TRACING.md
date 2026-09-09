# Seed-guided hybrid tracing

The `seed_hybrid` backend is implemented as an **experimental option**. The
established tracker remains the default. Asphalt uses its established engine;
base and subbase use sparse waveform correspondence and a reflector-segment
graph. A validated model can contribute bounded evidence to that same graph.
Current work prioritizes the conventional tracing core; see
[real-road development findings](HYBRID_REAL_ROAD_DEVELOPMENT.md) for the
remaining coverage and seed-recovery failures. ML fitting is deferred.

## Tracing behaviour

- Mexican-hat wavelet responses propose candidates at pulse-relative scales.
  Candidate retention reserves space for seed identity and optional learned
  evidence, alongside strong conventional reflectors.
- Reciprocal waveform matching uses adjacent traces and approximately 1, 2, 5,
  10 and 25 m offsets. Banded DTW, polarity, phase, and local slope constrain
  correspondence. Exact waveform alignments have a bounded reuse cache.
- Segment paths prioritize participation in manual seeds before brightness.
  Up to three distinct hypotheses are retained. Disconnected seeded regions
  are solved separately; conflicting overlaps remain unresolved.
  Segment contraction preserves the entry/exit junctions of non-local routes,
  including routes between horizontally overlapping reflector segments.
- Correspondence support is computed at individual observations. Non-local
  routes can support reacquisition, but missing observations remain gaps.
  Structural breaks disconnect propagation.
- A click near waveform cancellation is retained exactly, but cannot seed
  propagation when its amplitude is below one quarter of the nearby lobe's
  amplitude. Analysis requests recapture of that specific layer and station.
  This is a local observability safeguard, not an accuracy calibration.
- Accepted output requires observable measurement support, seed correspondence,
  and a competing-candidate margin. Default thresholds are research settings,
  not a calibrated probability of correctness. A provisional path cannot
  produce TWTT or thickness. Physical thickness still needs dielectric provenance.
  Exact complete-path ambiguity additionally vetoes tied or infeasible seeded
  routes, including competitive missing states. The path margin is exported
  separately from the local candidate margin. Competing observable routes
  inform layer-specific requests for further analyst observations.

Global analysis, refinement, dropout checks and bounded local corrections use
the shared dispatch. Additional seeds are permitted beyond the suggested initial
budget. Run metadata records method, model availability/hash, preprocessing,
acceptance settings and path hypotheses.

## Dataset and model commands

Use the project environment's `gpr-layer-audit` command:

```powershell
uv run gpr-layer-audit dataset audit "GPR Data" --seeds benchmarks --annotations confirmed.csv --output exports/audit.json
uv run gpr-layer-audit dataset build exports/audit.json --output exports/dataset
uv run --extra ml gpr-layer-audit ml train exports/dataset/dataset.json --output exports/model --epochs 50 --device auto
uv run --extra ml gpr-layer-audit hybrid evaluate exports/audit.json --model exports/model --split validation --output exports/validation.json
uv run gpr-layer-audit ml calibrate exports/validation.json --output exports/calibration.json
uv run --extra ml gpr-layer-audit hybrid evaluate exports/audit.json --model exports/model --calibration exports/calibration.json --split test --output exports/evaluation.json
uv run gpr-layer-audit ml promote exports/model exports/evaluation.json
```

Ordinary startup and conventional tracing do not import PyTorch. The `ml` extra
is optional; use an official CUDA-compatible PyTorch installation for GPU
training. The current development environment has CPU PyTorch. The training
loop supports CUDA mixed precision, at most 50 epochs, and early stopping.

Annotations use this CSV header:

```text
source_sha256,trace_index,layer_order,sample_raw,visibility,verified,origin,training_use,selected_lobe,pulse_width_samples
```

Coordinates are original acquisition trace/sample coordinates, before surface
alignment. A visible observation requires a verified `positive_peak` or
`negative_trough`. Use `manual` or `manual_correction` provenance and
`training_use=allowed` for learning. `not_visible` supervises visibility only;
`absent` supplies known background. Unknown traces are ignored.

Evaluation-only observations use `training_use=evaluation_only` (or
`prohibited`) and can use `origin=checkpoint`. Their entire physical-road group
is reserved for test evaluation. They never become model inputs or training
crops. Evaluation seeds come only from the separate allowed manual observations.
Imported historical display-space seeds are inventoried but need verified
source/coordinate mapping. Workbook depths are not silently converted to labels.

The user confirmed that processed Talagang radar files and manual tracings are
available on their work PC, with access expected Monday. These are pending
transfer to this laptop and are intended for conventional validation and later
ML labels after raw/processed alignment is verified. Preserve their processing,
time-zero, dielectric and pick metadata; group every Talagang version together.
See `TALAGANG_REFERENCE_DEVELOPMENT.md` for the transfer note and validation steps.

The audit collapses duplicate fingerprints, groups known repeat/adjacent roads,
and assigns groups before making numeric caches. Review the groups; `--groups`
accepts a survey-ID-to-physical-road JSON mapping. Grouping unknown acquisitions
from filenames cannot replace acquisition provenance.

The independent four-level, width-16 U-Net uses signed amplitude, envelope,
seed-template similarity, positive/competing click maps and layer identity.
Boundary and visibility outputs are tiled on a recorded physical grid.
Training support clicks are excluded from supervised target rows. Test roads
are excluded from training and calibration. A model remains experimental until
held-out evaluation of its exact weights passes the per-layer gates.

## Verification and present limits

Development artifacts are under `exports/hybrid-development/` (ignored by Git).
The supplied-data audit found 16 unique acquisitions but no eligible verified
raw-coordinate training observations. The resulting dataset has zero chunks;
the real-data experiment correctly refuses to train. No pavement model has
been trained or promoted. A separate one-epoch synthetic smoke test verifies
training, weight packaging and inference plumbing only.

Mechanism tests cover stronger ringing, waveform/gain drift, noise, missing
signal, reacquisition, structural breaks, contradictory seeds, ordering, exact
seed preservation and model fallback. Workflow checks include correction replay
outside the saved ±25 m window and project storage. None establishes field accuracy.

The controlled dropout comparison now demonstrates a specific conventional
benefit. With the same three seeds before a gap and a stronger competing
reflector, the established tracker accepted no post-gap observations. The
combined tracker correctly reacquired 110, 100 and 54 observations after gaps
of 2, 6 and 14 m respectively, without accepting measurements inside the gaps
or selecting the wrong reflector. Permanent-disappearance controls with nearby
ringing accepted no post-gap observations. The closest overlapping-waveform
case initially failed and exposed the cancellation-seed propagation defect;
the safeguard above fixes that case. These are development regressions with
known synthetic truth, not independent field validation. Numeric results are
in `exports/hybrid-development/dropout-comparison.json`.

A bounded 50 m Talagang probe is a runtime/development check using historical
seed coordinates whose alignment has not been independently verified. Its low
acceptance does **not** demonstrate improved real-road coverage. Use verified
held-out observations before enabling either the conventional hybrid or ML
contribution as a production default.

Remaining validation includes real-road correctness/coverage and switch audits,
ablation measurements, three held-out groups with 30 observations per group and
layer, full-road memory/runtime, GPU training and warm correction latency.
The existing review-interval priority is still a heuristic, not a calibrated
expected reduction in corrections per kilometre. Cached training chunks are
incremental, but cross-chunk crop context and broader evidence-cache reuse remain
performance/data-pipeline follow-up work.

## Algorithm references

These are independent implementations; no upstream source was vendored:

- [HorizonTracker](https://github.com/ajbugge/HorizonTracker), MIT: sparse
  non-local correspondence and repeated matching.
- [ARESELP](https://github.com/xiongsiting/ARESELP), MIT: multiscale peaks and
  local reflector geometry.
- [seismiQB](https://github.com/GeoscienceML/seismiqb), Apache-2.0: horizon masks,
  patches and ignored sparse labels.
- [IRHMapNet](https://zenodo.org/records/15111993), CC BY 4.0 record: compact
  radar-horizon U-Net architecture reference. Its conflicted local loaders are
  not application dependencies; no pretrained pavement weights were identified.
