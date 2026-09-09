# Dense identity experiments

The experiments below are rejected for application promotion. The tested processed
seed/correction workflow remains available; these changes do not establish the
product's required deep-layer reliability or calibrated physical thickness.
All roads have prior development use. Jamshoro's interface names retain their
historical layer numbering; user-confirmed pavement semantics cover Mandiali
and Gujrat.

## Native patch matcher

A 36,602-parameter shared CNN learns separate timing and signed-lobe scores
from both Mandiali acquisitions. Training uses 55,290 reviewed nonseed candidate
pairs and 12 immutable seed patches, with nearby timing and neighboring-lobe
negatives. Unknown rows are not targets. The fixed 12-epoch experiment uses
native amplitude/validity patches over ±0.9375 ns and ±0.5 m. No checkpoint or
threshold is selected on Jamshoro outcomes.

Both inference arms use the same exact dense graph, extrema/shoulder observation
mask, gap positions, native seeds, tensor geometry and acceptance settings.
The learned arm averages signed seed NCC with the CNN timing score mapped to
[-1, 1], then blends the bracketing immutable seed templates. Prediction opens
no withheld reference. Independent scoring and exact seed checks pass.

| Jamshoro | Reviewed nonseed denominator | NCC agreeing proposals | Learned agreeing proposals | Accepted, both arms |
|---|---:|---:|---:|---:|
| Layer 2 | 4,115 | 422 | 255 | 0 |
| Layer 3 | 2,242 | 6 | 76 | 0 |

Correct and wrong accepted coverage are both zero; unresolved coverage is 100%.
Accepted precision is undefined. This is a failure, not evidence of reliability.
The model is not transferred to Gujrat or used by the application.

The post-inference exact oracle separates local candidate retention from route
feasibility. At the original maximum displacement of seven samples per 0.1 m,
the graph can recover at most 3,782/4,115 and 2,034/2,242 reviewed observations.
Doubling that bound recovers at most 3,990 and 2,172. These are label-only upper
bounds; they never supply candidates, scores or acceptance to an actual tracker.
The discrepancy with actual proposals identifies a major objective failure.

The independent audit also reproduces two numerical limitations:

- Masked neighboring amplitudes could affect valid peak retention. Peak finding
  now operates inside finite, valid runs. A regression covers this counterexample.
  Candidate arrays are unchanged at every sample on both Mandiali inputs and
  Jamshoro. The original execution sources remain preserved separately.
- The dense comparator divides a local alternative's exact cost by the full
  seed interval length. The same cost of 0.23 gives margins 0.02277 and 0.002298
  for 101 and 1,001 rows, crossing the fixed 0.02 gate solely through interval
  length. This score is uncalibrated. Lowering the gate would expose many wrong
  proposals and is not a repair.

Legacy `correspondence_survived` and per-row correspondence/support zeros in the
raw evaluation are unavailable instrumentation defaults, not observed graph
losses. The separate diagnosis uses saved candidate, proposal and visibility
arrays and labels these fields explicitly. Proposal timing misses are separated
from wrong signed-lobe observations by the independent review.

## Adjacent packet objective

One independent change adds signed NCC costs between every retained adjacent
packet pair. It preserves seed unaries, candidates, gaps, tensor geometry,
displacement bounds and the existing gate. Costs use physical metres and are
checked against direct packet arithmetic, including invalid samples. No top-k
or reciprocal filter discards competing edges.

| Short Mandiali | Original correct / accepted | Pairwise correct / accepted | Correct coverage, original → pairwise |
|---|---:|---:|---:|
| Base, 443 reviewed nonseed rows | 38/54 | 34/59 | 8.58% → 7.67% |
| Subbase, 404 reviewed nonseed rows | 36/40 | 28/55 | 8.91% → 6.93% |

Wrong accepted observations increase from 20 to 52 across both layers. The branch
is rejected without tuning or a Gujrat run. Full-road overlays color agreeing
acceptances green, wrong acceptances red and unresolved proposals amber. Unknown
reference rows are unscored; reported metres are observation footprints, never
interpolation through missing labels. Signed-lobe error counts are not independently
verified semantic switch events. No pointwise independence assumption is made.

## Reproduce

Use the same local files and hashes listed in `benchmarks/seeded-evaluation-inputs.json`.
PyTorch is an optional dependency, needed only for the patch experiment. Current
commands use the corrected candidate mask and support a fresh output directory:

```powershell
.venv/Scripts/python.exe scripts/experiment_patchnet_dense.py prepare --output exports/patchnet-reproduction
.venv/Scripts/python.exe scripts/experiment_patchnet_dense.py train --output exports/patchnet-reproduction
.venv/Scripts/python.exe scripts/experiment_patchnet_dense.py predict --output exports/patchnet-reproduction
.venv/Scripts/python.exe scripts/experiment_patchnet_dense.py evaluate --output exports/patchnet-reproduction
.venv/Scripts/python.exe scripts/diagnose_patchnet_dense.py --input exports/patchnet-reproduction --output exports/patchnet-diagnosis
.venv/Scripts/python.exe scripts/experiment_dense_pairwise.py --case mandiali-short --output exports/pairwise-reproduction
.venv/Scripts/python.exe scripts/summarize_dense_pairwise.py exports/pairwise-reproduction
.venv/Scripts/python.exe -m pytest tests/test_patchnet_experiment.py tests/test_dense_pairwise_experiment.py tests/test_seeded_challenger.py
```

Stages refuse existing outputs. `--workspace-root` selects the repository when
executing a preserved patch experiment source. Exact executed sources and hashes
remain in each original export directory. A separate compact evidence supplement
is indexed at `benchmarks/seeded-dense-results/index.json`; the existing seeded
tracker evidence package is unchanged. The trained model SHA-256 is
`4250ff586283c8d74ed54d32514c7ad656c5fcd68e812e2136dd65042f47cb47`.

The complete corrected-mask reproduction preserves every training array, model
tensor, candidate array, path, alternate, visibility flag, margin, CNN evidence
value and evaluation field exactly. Its provenance differs because it records
the new source/contract and output directory. The focused verification passes
20 tests. Verify the supplement without original data using
`python scripts/package_dense_experiments.py --verify`; add `--originals` to
also compare the local export sources.
Large per-observation JSON records are stored as lossless gzip files; verification
checks both the packaged bytes and exact decompressed source hash.

The useful missing capability remains persistent reflector selection through
waveform changes at an economical intervention budget. These negative comparisons
do not establish that learning or local correspondence in general cannot help.

## Completed focused followups

Four additional comparisons retain package source `c442b60`, the frozen scorer,
native operating seeds and original acceptance thresholds. None is promoted.
The working native-seed/correction application remains as previously verified.

**Distant training negatives.** Replace three of six nearby timing negatives
with distant candidates most similar to the same nearest immutable seed.
Architecture, positive targets, 12 epochs and inference are unchanged. Independent
audit verifies all 18,366 positive patches, targets and seed partners, all 12 seed
patches, the complete frozen inputs, classical predictions and candidate arrays.
One remote negative fails the unchanged patch-validity mask, leaving 55,289 pairs
instead of 55,290. The model hash is
`f1b447ae369148f29af3cef5be4b81df027133fca41aa924a6a183dbc5fd67be`.

| Jamshoro | Classical correct proposals | Original CNN | Distant-negative CNN | Accepted, all |
|---|---:|---:|---:|---:|
| Layer 2, 4,115 reviewed nonseed rows | 422 | 255 | 148 | 0 |
| Layer 3, 2,242 reviewed nonseed rows | 6 | 76 | 81 | 0 |

Reject training change and Gujrat transfer. Jamshoro diagnosis informed this
change, so it remains development evidence. A source-dependency contract now
checks model, package, script and numerical-library versions between stages.
The prepared input-metadata file is not independently hash-guarded by every
stage; its exact equality with the frozen input is verified for this run.

**Actual dense correction replay.** Three initial observations per enabled
deep interface, then four requested answers, use the existing query boundary,
local merger, ordering guard and frozen pulse. Compare existing active and
midpoint policies under same-lobe versus same-lobe-and-timing alternatives.
Timing ambiguity changes margins and acceptance; initial selected paths match.

| Mandiali after four total requests | Base correct/accepted | Subbase correct/accepted |
|---|---:|---:|
| Same-lobe alternatives, active | 42/47 | 117/138 |
| Same-lobe alternatives, midpoint | 83/95 | 99/113 |
| Timing alternatives, active | 37/41 | 23/33 |
| Timing alternatives, midpoint | 77/90 | 9/9 |

Initial same-lobe results are 42/47 and 46/50; initial timing results are 37/41
and 9/9. Revealed answers are excluded from automatic credit. The packaged curves
use fixed initial pools of 443 base and 404 subbase observations; complete
per-step dynamic denominators, actions and losses are also preserved. This road
is only 11.45 m, so four requests mean **349.34 requests/km**, and each +/-25 m
correction covers the whole short road. Its 0.21-0.23 s fit latency does not
establish long-road latency or an economical action budget. Pointwise alternate
events are not coherent full routes; the active ranking remains a heuristic.
Reject: added errors or unusably low coverage, despite more accepted samples.

**Measured peak states.** Retain the same dense latent positions but permit
observations only at exact measured extrema. All initial seeds already satisfy
this condition. Base correct/accepted falls 42/47 to 21/22; subbase falls 46/50
to 20/24. Correct proposals fall 220 to 71 and 240 to 162. Reject.

**Relative gap cost.** An exact counterexample shows that a common offset to
all observation costs changes the original preference for gaps. One isolated
repair makes gaps pay the unchanged penalty above each row's best eligible
event, with the original cost retained for entirely missing rows. Exact tests
verify row-potential invariance, surface exclusion and explicit missing data.
Mandiali base correct proposals improve 220 to 276, but correct/accepted changes
42/47 to 46/57: wrong acceptances rise 5 to 11. Subbase stays 46/50. Reject
without a penalty sweep or long-road transfer. The numerical invariance repair
does not establish a better physical identity objective.

The separate package at `benchmarks/seeded-followup-results/index.json` contains
89 byte-verified artifacts, including action curves, exact execution sources,
model, predictions, independent scoring, and radar overlays. Large JSON files
are losslessly compressed; data and dense CNN evidence maps remain local with
hashes recorded. Original evidence packages are preserved. Current scripts
contain formatting-only wraps of some executed sources; AST parity is recorded.
The complete suite passes **507 tests, with one optional upstream test skipped**.

```powershell
.venv/Scripts/python.exe scripts/experiment_patchnet_far_negatives.py prepare --output exports/far-negatives-reproduction
.venv/Scripts/python.exe scripts/experiment_patchnet_far_negatives.py train --output exports/far-negatives-reproduction
.venv/Scripts/python.exe scripts/experiment_patchnet_far_negatives.py predict --output exports/far-negatives-reproduction
.venv/Scripts/python.exe scripts/experiment_patchnet_far_negatives.py evaluate --output exports/far-negatives-reproduction
.venv/Scripts/python.exe scripts/experiment_dense_replay.py --case mandiali-short --actions 4 --output exports/dense-replay-reproduction
.venv/Scripts/python.exe scripts/experiment_dense_peak_modes.py --case mandiali-short --output exports/dense-peaks-reproduction
.venv/Scripts/python.exe scripts/experiment_dense_relative_gap.py --case mandiali-short --output exports/relative-gap-reproduction
.venv/Scripts/python.exe scripts/package_dense_followups.py --verify --originals
```

Omit `--originals` to verify packaged bytes without the local exports. This
checks integrity against the index, not the independent truth of the labels.
The common coordinate, source-snapshot and frozen-scorer setup remains documented
in `SEEDED_TRACKER_WORKFLOW.md` and `seeded-evaluation.md`.

The subsequent [basin-state experiment](BASIN_TRACKER_EXPERIMENT.md) reproduces
the longer-road guide control and evaluates explicit measured-peak emissions
through actual corrections on both roads. Its short-road gain fails transfer;
the separate geometry-confidence correction also loses useful coverage.
