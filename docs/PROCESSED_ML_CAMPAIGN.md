# Processed sparse-seed learning campaign — 10 September 2026

**Decision: retain the experimental tooling and trained models; do not promote a
model into the production tracker.** Learned evidence materially improves many
base/subbase proposals. It has not demonstrated useful automatic coverage at the
95% analyst-agreement operating objective. Every completed model has null
training-selected deep-layer gates, and all 22 trained-model correction replays
retain zero automatically accepted deep observations through four extra requests.
The proposed milestone of at least 50% correct coverage at 95% accepted agreement
is unmet.

This is **grouped development generalization**: each evaluated physical road was
excluded from its model's fitting and calibration, but all annotated roads have
historical development use. Agreement with stored analyst interpretations is not
physical thickness accuracy. Thousands of adjacent observations are correlated;
they do not create thousands of independent human judgments or test surveys.

The research branch is `codex/seed-conditioned-campaign-20260910`, starting from
`ee1e6dad4ff6bf3906b48c0f20da1072ed0f976b`. The original checkout, source data and
untracked survey/IRHMapNet work were preserved. New models, caches and evidence
reside outside Git in
[the artifact directory](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910).
Below, `A` denotes that directory and `W` denotes
`C:/Users/faisa/Desktop/gpr_next_stage_20260910`.

## What was actually built and trained

The new processed-coordinate path consists of an audited memory-mapped dataset,
153,536-parameter multiscale ResUNet, randomized sparse-support training, native
all-depth decoding, training-only calibration, frozen evaluation and real
correction replay. The existing production dispatch and asphalt route remain
unchanged. Integration is an explicit research CLI/backend, with lazy ML imports,
processed/raw and fold guards, exact clicks, cancellation and bounded history;
there is no new default or UI dispatcher.

The model preserves signed native amplitudes and all 512 temporal samples. Fine
context covers 65 trace centers, 1.6 m; coarse context covers 6.4 m. Each remote
support has a 65-sample by 9-trace patch, 1.875 ns by 0.2 m. Individual support
embeddings/attention and relative depth/lateral coordinates reach distant queries,
including windows with no internal click. Supports can be hundreds of metres
away. Deterministic per-scan RMS normalization uses radar alone; the depth-RMS
challenger rescales depth separately. Neither fits target-road weights.

Masked depth-wise cross-entropy supervises the selected interface at reviewed
traces. Unknown traces are ignored; no absence/visibility labels or heuristic
negative clicks are manufactured. Episodes sample layer first, then eligible
roads/blocks, with 1–5 randomized acquisition-wide support observations. Supports
are excluded from query loss. All eligible training road groups contribute;
subbase uses the two available training groups in each outer fold. No pretrained
weights, target-road pretraining or few-click weight adaptation was used.

The direct control is native depth argmax. The dense decoder admits every valid
native depth, with exact seed constraints, soft time-domain continuity and an
explicit gap state. It has no inherited old-graph candidate pruning or seed
corridor. Depth probability is evidence, not a calibrated identity probability.
The no-radar control linearly interpolates only the authorized seeds. The
correction-aware challenger learns from its own previous prediction and simulated
training-reference corrections, inspired by RITM; test request selection never
consults hidden reference error. The verified old bottlenecks and selective
paper/license review are in
[PROCESSED_ML_RESEARCH_REVIEW.md](C:/Users/faisa/Desktop/gpr_next_stage_20260910/docs/PROCESSED_ML_RESEARCH_REVIEW.md).

All ten runs completed 2,400 optimization steps with validation every 200 steps,
36 validation episodes, patience 8, learning rate 0.0003 and weight decay 0.0001.
Each selected checkpoint is the best inner-validation step, not the best outer
road result. All stopped at the configured optimization budget.

| Completed run | Selected step | Training seconds | Weights SHA256 prefix |
|---|---:|---:|---|
| primary-gujrat-bf16-s42 | 2,200 | 201.1 | `c6ac473f4b87d2bc` |
| primary-mandiali-bf16-s42 | 2,000 | 204.4 | `dc4aa0d1133932c6` |
| primary-jamshoro-bf16-s42 | 2,200 | 203.6 | `e7bb31e702308c8c` |
| layer-only-gujrat-bf16-s42 | 2,400 | 328.3 | `342c51ffaa3748a1` |
| layer-only-mandiali-bf16-s42 | 2,200 | 39.1 | `1f5cfb31a83f5f88` |
| layer-only-jamshoro-bf16-s42 | 2,200 | 41.2 | `2c3267ba97051a83` |
| correction-gujrat-bf16-s42 | 2,200 | 66.2 | `574cc6bbf0bfe2e4` |
| depth-rms-gujrat-bf16-s42 | 2,400 | 53.7 | `2c33d88d2389ff75` |
| no-context-gujrat-bf16-s42 | 2,400 | 47.0 | `05b4e22f09428f49` |
| primary-gujrat-bf16-s84 | 2,400 | 55.3 | `53b0d6bd5471512d` |

Every `A/models/<run>/model.json` retains the full configuration, full weight and
checkpoint hashes, training history, excluded records and physical context. The
[numerical appendix](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/report/completed-results.md)
and [machine-readable appendix](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/report/completed-results.json)
index model evidence and full per-acquisition/layer counts. The initial legacy
`primary-gujrat-s42` FP16 attempt failed before an update with a nonfinite gradient;
it is preserved and excluded from the ten completed models. Hardware-supported
BF16 without gradient scaling resolved that failure. A separate 200-step CPU
overfit achieved 0.385-sample Daska base MAE and 3.077-sample Jamshoro subbase MAE;
these are plumbing/learning checks, not grouped results.

## Exact data and split support

The re-audit found 61 DZT files: 35 processed, 26 nonprocessed and 59 distinct
DZT hashes; 12 reviewed processed DZT/DZX pairs contain 555,439 stored observations
across eight road groups. All paired scans use 0.025 m trace spacing,
0.029296875 ns/sample and a −3 ns origin. Cached float32 amplitudes match every
stored amplitude exactly. Temporal coordinates, polarity and picks are retained;
no offsets were fitted to reference agreement. Full provenance is in
[the dataset document](C:/Users/faisa/Desktop/gpr_next_stage_20260910/docs/PROCESSED_ML_DATASET.md)
and [manifest](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/processed_dataset/manifest.json).

Whether each stored observation was manually drawn, interpolated or assisted
remains unknown: `interpretation_property=2` has no verified per-pick creation
meaning. Gujrat/Mandiali layer semantics are user-confirmed; other groups retain
the recorded historical RADAN convention. No explicit prohibited/evaluation-only
paired DZX label file was found. Historical tracker predictions, seed JSONs and
workbook thickness values are not training labels.

Physical-road grouping precedes crop creation. The longest eligible acquisition
per training road is selected by physical extent/hash, excluding unregistered
other passes/portions to protect inner separation. The canonical record IDs are
`bahawalpur-910ba4e15e23`, `burewala-69828a1d412e`, `daska-561b5acd7b8c`,
`gujrat-e5d307871c45`, `jamshoro-e5f804e88e0d`, `jhang-51b3ceaa247a`,
`mandiali-006668ce9135` and `pattoki-6211deb3c237`. Each fold fits seven of these,
omitting its entire held-out road including all processing variants. Inner
validation uses 64 m spatial blocks with a 3.225 m full-context buffer. Support
patches and fitting queries do not cross that buffer into validation blocks.

| Held-out road | Asphalt fitting / inner validation | Base fitting / inner validation | Subbase fitting / inner validation | Subbase training roads |
|---|---:|---:|---:|---|
| Gujrat | 190,831 / 41,993 | 91,188 / 22,194 | 15,035 / 4,403 | Mandiali, Jamshoro |
| Mandiali | 189,424 / 41,993 | 90,573 / 20,994 | 19,012 / 3,611 | Gujrat, Jamshoro |
| Jamshoro | 196,856 / 44,505 | 89,845 / 21,957 | 21,971 / 5,802 | Gujrat, Mandiali |

Noncanonical exclusions, when the corresponding road is eligible for training:

| Excluded record | Stored asphalt / base / subbase observations |
|---|---:|
| bahawalpur-c4aac5338098 | 15,820 / 8,263 / 0 |
| burewala-1b8af7fb9255 | 22,300 / 7,941 / 0 |
| gujrat-087182ae3b6f | 10,226 / 6,350 / 4,270 |
| mandiali-650f2aa665a3 | 450 / 446 / 407 |

Thus Gujrat excludes the two alternate Bahawalpur/Burewala records and Mandiali
short; Mandiali excludes the two alternates and Gujrat second; Jamshoro excludes
all four. Held-out road exclusions apply separately. Exact per-record support and
exclusion counts are authoritative in the three primary model JSONs:
[Gujrat](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/models/primary-gujrat-bf16-s42/model.json),
[Mandiali](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/models/primary-mandiali-bf16-s42/model.json),
[Jamshoro](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/models/primary-jamshoro-bf16-s42/model.json).

## Frozen scoring and initial coverage

All 42 initial observations in the [case registry](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/evaluation/cases.json)
were verified at exact native trace/sample coordinates. Each listed interface has
three frozen seeds. Gujrat and Mandiali enable all three interfaces; Jamshoro has
only its six existing base/subbase observations. Initial seed work is distinct
from additional requests. Mandiali short uses every native trace; other cases use
every fourth trace (0.1 m), with all temporal samples retained. Full-road inference
includes Gujrat first (1,036.05 m), Mandiali long (1,066.65 m) and Jamshoro
(1,005.2 m), plus Gujrat second (287.425 m) and Mandiali short (11.45 m).

The unchanged scorer requires the frozen native sample/time tolerance and the
same signed lobe, with no intervening zero/sign crossing. `N` is the original
reviewed non-seed cohort. Correct/wrong automatic coverage divides automatic
counts by this fixed `N`; accepted agreement divides correct by all accepted and
is undefined when nothing is accepted. Revealed answers get no automatic credit.
Unknown labels break observed error spans. Wrong signed lobes are not automatically
verified semantic reflector-switch events.

The frozen conventional Gujrat-second predictions were scored again with the
unchanged native helper. This is **fresh scoring of cached immutable predictions**,
not a fresh run of the historical graph. Counts reproduce the reported control:

| Interface | N | Correct / wrong accepted | Unresolved | Accepted agreement | Correct automatic coverage | Correct proposals |
|---|---:|---:|---:|---:|---:|---:|
| Asphalt | 2,550 | 2,036 / 92 | 422 | 95.68% | 79.84% | 2,234 |
| Base | 1,586 | 369 / 178 | 1,039 | 67.46% | 23.27% | 392 |
| Subbase | 1,067 | 4 / 59 | 1,004 | 6.35% | 0.37% | 99 |

The primary model changes Gujrat-second correct base proposals from 392/1,586
(24.72%) to 801/1,586 (50.50%), and subbase from 99/1,067 (9.28%) to 526/1,067
(49.30%). Those gains precede acceptance; comparing them with the old accepted
coverage would conflate two stages.

All entries below are **correct proposal coverage**, with three initial seeds and
no extra requests. Every primary deep cell has correct accepted = 0, wrong
accepted = 0, unresolved = `N`, and undefined accepted agreement.

| Case / interface | N | Layer-only dense | Primary direct | Primary dense | No-radar interpolation |
|---|---:|---:|---:|---:|---:|
| Gujrat second / base | 1,586 | 58.45% | 50.38% | 50.50% | 11.35% |
| Gujrat second / subbase | 1,067 | 41.80% | 47.05% | 49.30% | 8.81% |
| Gujrat first / base | 3,427 | 30.55% | 53.08% | 56.52% | 10.88% |
| Gujrat first / subbase | 4,229 | 38.07% | 56.63% | 57.79% | 6.41% |
| Mandiali short / base | 443 | 0.90% | 65.46% | 66.59% | 45.37% |
| Mandiali short / subbase | 404 | 35.15% | 85.15% | 88.61% | 44.80% |
| Mandiali long / base | 4,010 | 5.81% | 66.91% | 67.66% | 13.69% |
| Mandiali long / subbase | 3,309 | 54.43% | 58.33% | 57.30% | 13.45% |
| Jamshoro / base | 4,115 | 15.12% | 71.66% | 77.64% | 20.34% |
| Jamshoro / subbase | 2,242 | 39.16% | 54.06% | 58.79% | 11.86% |

Acquisitions are pooled within each physical road before averaging roads:

| Held-out road | Base correct dense proposals / N | Subbase correct dense proposals / N |
|---|---:|---:|
| Gujrat | 2,738 / 5,013 (54.62%) | 2,970 / 5,296 (56.08%) |
| Mandiali | 3,008 / 4,453 (67.55%) | 2,254 / 3,713 (60.71%) |
| Jamshoro | 3,195 / 4,115 (77.64%) | 1,318 / 2,242 (58.79%) |
| **Road macro** | **66.60%** | **58.52%** |
| Pooled observations, descriptive only | 8,941 / 13,581 (65.84%) | 6,542 / 11,251 (58.15%) |

For the pooled primary deep cohorts, correct/wrong accepted are **0 / 0**, and
unresolved are **13,581 base / 11,251 subbase**. Zero wrong accepted spans here
reflect complete abstention, not demonstrated identity safety. Dense decoding
provides mostly modest gains over direct evidence and reduces Mandiali-long
subbase proposals slightly. It is not the sole cause of the learning gains.

Calibration uses 192 buffered training-road episodes per model, separately for
direct/dense outputs, seeking 95% agreement with at least 20 accepted observations
per eligible training road. All ten models' deep thresholds are null. Original
stride-four calibration is incompatible with Mandiali short's stride one, so its
original gate is correctly withheld. Separate training-only stride-one
[primary](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/evaluation/primary-mandiali-bf16-s42-stride1-supplement/summary.json)
and [layer-only](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/evaluation/layer-only-mandiali-bf16-s42-stride1-supplement/summary.json)
supplements were completed without overwriting the registered evidence; the deep
gate remains null for the primary model.

Independent review identified a numerical mismatch: original calibration used
CUDA FP16 forward inference, while full-road prediction uses FP32. The calibrator
was corrected to FP32 and **all 12 original model/stride calibrations were run
again on identical training-only episode cohorts**. All 72 layer/stage thresholds
are unchanged; all 48 deep entries remain null. No old artifact was overwritten,
and the recorded road acceptance decisions remain identical. The
[precision comparison](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/report/calibration-precision-comparison.md)
and [full comparison JSON](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/report/calibration-precision-comparison.json)
record this correction. A separate limitation remains: calibration uses short
1.6 m query windows without local decoder anchors, whereas evaluation decodes
complete roads with exact seeds. Failure of this confidence/gating procedure
does not prove no useful operating point can exist, but it supplies no basis for
accepting these deep predictions now.

Asphalt is not a reason to replace the default: on Gujrat second the primary
accepts 1,140 correct and 5 wrong (99.56% agreement, 44.71% correct coverage),
versus the frozen comparator's 2,036 correct (79.84% coverage). Mandiali-long
primary asphalt accepts 10,562 correct and 99 wrong (99.07%). These differences
remain experimental; the established asphalt route is preserved.

## Conditioning, context and challengers

On the same Gujrat-second radar/cohort, removed seeds reduce dense correct
proposals from 801/526 (base/subbase) to 0/0; disabling conditioning also gives
0/0. Seed permutation gives exactly the original 801/526. Changing to another
authorized target's seeds while holding the requested layer token fixed gives
6/0 agreement with the original target; these deliberately inconsistent
layer-token/seed semantics were not scored against the alternate reference.
Disabling only coarse context at inference gives 794/511. These controls prove
material seed dependence. Real-road removal/swap controls do not alone isolate
waveform identity from a seed-depth guide or establish correct arbitrary target
selection. A separate synthetic learning check holds radar/layer/geometry fixed
and changes support waveform identity to switch selected reflectors; that is
mechanism evidence, not a new real-road accuracy claim. Full controls are in
[the frozen control report](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/evaluation/primary-gujrat-seed-controls/summary.json).

The following independently trained variants use the same Gujrat holdout, budget
and initial observations. Percentages are correct dense proposals, before gating:

| Configuration | Second base | Second subbase | First base | First subbase |
|---|---:|---:|---:|---:|
| Primary, seed 42 | 50.50% | 49.30% | 56.52% | 57.79% |
| Correction-aware, seed 42 | 45.90% | 45.74% | 59.47% | 57.32% |
| Depth-RMS, seed 42 | 54.41% | 50.70% | 62.30% | 66.66% |
| No coarse context, seed 42 | 48.87% | 48.17% | 59.76% | 61.01% |
| Primary, seed 84 | 52.90% | 42.55% | 65.07% | 65.95% |

Depth-RMS improves all four Gujrat proposal cells, supporting further work on
weak-depth scaling, but it has only one outer fold/initialization and no usable
deep gate. The no-context model improves Gujrat first and worsens Gujrat second:
this campaign does not establish a clean causal gain from broader context. Seed
84 improves three cells but loses 6.75 percentage points on Gujrat-second subbase;
proposal improvements cannot yet be called stable. Correction-aware training is
a completed challenger with mixed initial performance, not an assumed repair
success. No further pretraining, synthetic-pretraining or transductive arm was
run; the bounded correction and normalization/context comparisons were prioritized.

## Real interventions and evidence

**22 trained-model replays completed:** 18 preserve the real inclusive ±25 m
local correction contract; four use a separately named global-model-seed action.
Gujrat second compares primary and correction-aware models for both deep layers,
with uncertainty, fixed spacing and largest-interval midpoint request policies.
The global action uses uncertainty at the same request budgets. Mandiali long
adds six primary/local scenarios over its complete kilometre acquisition.
There is no weight fitting during either action.

Every request is chosen from model/radar/visited state before an exact native
reference lookup. Missing answers are charged and never moved to a nearby known
label. These are stored-reference answer simulations, not measured human time or
usability trials. Across alternative scenarios there were 88 charged requests,
31 exact answers and 57 unavailable answers; repeating the roads under different
policies does not create independent analyst workloads. An unavailable stored
answer does not establish reflector absence or human inability to interpret it.

The table reports correct proposal **counts at 0 / 1 / 2 / 4 extra requests**.
`U` is the unavailable count out of four. In every row and at every budget,
correct accepted = wrong accepted = 0, automatic unresolved = `N`, correct/wrong
automatic coverage = 0%, and accepted agreement is undefined. Both automatic and
proposal numerators exclude newly revealed answers without shrinking `N`.

| Model / case / interface | Action / request policy | N | Correct proposals at 0 / 1 / 2 / 4 | U |
|---|---|---:|---:|---:|
| Primary / Gujrat second / base | Local / fixed spacing | 1,586 | 801 / 828 / 828 / 948 | 1 |
| Primary / Gujrat second / base | Local / midpoint | 1,586 | 801 / 884 / 890 / 892 | 0 |
| Primary / Gujrat second / base | Local / uncertainty | 1,586 | 801 / 801 / 801 / 891 | 3 |
| Primary / Gujrat second / base | Global / uncertainty | 1,586 | 801 / 801 / 801 / 1,045 | 3 |
| Primary / Gujrat second / subbase | Local / fixed spacing | 1,067 | 526 / 546 / 554 / 554 | 2 |
| Primary / Gujrat second / subbase | Local / midpoint | 1,067 | 526 / 526 / 526 / 526 | 4 |
| Primary / Gujrat second / subbase | Local / uncertainty | 1,067 | 526 / 526 / 532 / 531 | 2 |
| Primary / Gujrat second / subbase | Global / uncertainty | 1,067 | 526 / 526 / 595 / 595 | 3 |
| Correction / Gujrat second / base | Local / fixed spacing | 1,586 | 728 / 746 / 746 / 874 | 1 |
| Correction / Gujrat second / base | Local / midpoint | 1,586 | 728 / 783 / 813 / 820 | 0 |
| Correction / Gujrat second / base | Local / uncertainty | 1,586 | 728 / 854 / 854 / 854 | 3 |
| Correction / Gujrat second / base | Global / uncertainty | 1,586 | 728 / 970 / 970 / 970 | 3 |
| Correction / Gujrat second / subbase | Local / fixed spacing | 1,067 | 488 / 504 / 512 / 512 | 2 |
| Correction / Gujrat second / subbase | Local / midpoint | 1,067 | 488 / 488 / 488 / 488 | 4 |
| Correction / Gujrat second / subbase | Local / uncertainty | 1,067 | 488 / 488 / 488 / 511 | 3 |
| Correction / Gujrat second / subbase | Global / uncertainty | 1,067 | 488 / 488 / 488 / 583 | 3 |
| Primary / Mandiali long / base | Local / fixed spacing | 4,010 | 2,713 / 2,713 / 2,731 / 2,749 | 2 |
| Primary / Mandiali long / base | Local / midpoint | 4,010 | 2,713 / 2,713 / 2,713 / 2,742 | 3 |
| Primary / Mandiali long / base | Local / uncertainty | 4,010 | 2,713 / 2,713 / 2,713 / 2,713 | 4 |
| Primary / Mandiali long / subbase | Local / fixed spacing | 3,309 | 1,896 / 1,896 / 1,896 / 1,896 | 4 |
| Primary / Mandiali long / subbase | Local / midpoint | 3,309 | 1,896 / 1,896 / 1,896 / 1,896 | 4 |
| Primary / Mandiali long / subbase | Local / uncertainty | 3,309 | 1,896 / 1,896 / 1,897 / 1,897 | 3 |

Three initial observations per interface accompany each sequence. Costs use the
retained evaluated span (287.4 m Gujrat second; 1,066.6 m Mandiali long):

| Case | Additional requests/km at budgets 0 / 1 / 2 / 4 | Initial observations/km | Total initial observations + requests/km at budget 4 |
|---|---:|---:|---:|
| Gujrat second | 0 / 3.48 / 6.96 / 13.92 | 10.44 | 24.36 |
| Mandiali long | 0 / 0.94 / 1.88 / 3.75 | 2.81 | 6.56 |

The largest primary base gain is global Gujrat second: 801→1,045 correct
proposals (65.89%), comprising **355 gained and 111 lost** common nonmanual rows.
The correction-aware counterpart is 728→970 (61.16%), with 339 gained and 97 lost.
Local primary fixed-spacing base gives 801→948 (59.77%): 151 gained, two lost and
two initially correct proposals removed from automatic credit after being answered.
Five individual request steps reduce total correct proposals. Global propagation
therefore repairs more proposals in these cases while also disturbing correct
work. Equal budgets can yield different answered locations/counts after adaptive
updates, so this is a descriptive operation comparison, not an isolated causal
comparison at identical new seeds.

Uncertainty requests are not consistently better than simple spacing: Mandiali
long base gains 36/29/0 proposals for spacing/midpoint/uncertainty, and subbase
gains 0/0/1. Correction-aware training improves some response gains but remains
below primary in the final Gujrat proposal counts, and never obtains accepted
deep coverage. No policy is established as a product winner.

The [complete interaction table](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/interactions-analysis/interaction-summary.md)
and [interaction JSON](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/interactions-analysis/interaction-summary.json)
retain per-action gains/losses/manual removals, cost, native answers, calibration
hashes, strata, checkpoint checks and 12 full-acquisition before/after and policy
curve figures. Examples are the
[primary Gujrat base curves](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/interactions-analysis/primary-gujrat-bf16-s42-gujrat-second-L2-curves.png)
and [complete-road before/after](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/interactions-analysis/primary-gujrat-bf16-s42-gujrat-second-L2-before-after.png).

All 22 initial proposal/confidence/acceptance arrays match batch evaluation
bitwise. Every saved local array is unchanged outside its declared window; all
22 checkpoints passed real serializer save/reopen checks, including model
history. Completed primary and correction-aware replays were resumed on copies
through the frozen CLI on CPU with prediction methods replaced by failing
sentinels: zero forward calls, and JSON/checkpoint bytes unchanged. Historical
nested replay reports express rows in retained-grid coordinates and working
spacing; the interaction analysis explicitly converts to native trace numbers
before confirming cohort parity with batch evaluation. Counts and physical
distances agree; raw nested native-coordinate metadata/hashes should not be
compared without that conversion.

Independent review also found two recovery edge cases absent from these completed
uninterrupted single-layer runs: a requested answer was not durably recorded until
retracking returned, and a neighboring-interface rejection restored paths/history
but still activated the rejected answer as a seed. Replay v2 repairs both: pending
requests and exact answers are durable before retracking, and rejected answers
remain separately counted without entering active model seeds. Resume processes
the saved pending action without selecting or revealing it again.

Fresh [trained-model recovery probes](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/persistence-v2-real-probe/README.md)
injected an interruption after the first real correction forward pass for both
primary and correction-aware Gujrat models. Both retained the charged answer,
passed policy/answer-lookup sentinels on resume, and matched every uninterrupted
scoring metric. All 15 saved arrays per model at budgets 0/1/2 were bitwise equal
to uninterrupted v2 and the original v1 campaign. These integration probes are
separate from the 22 scientific replay scenarios and do not inflate their budget.

## Error context and fixed visual evidence

Small median errors conceal substantial wrong proposals. Primary dense error
summaries retain the unchanged picked sample; these values do not replace the
signed-lobe correctness criterion:

| Case | Base median / p95 sample error | Subbase median / p95 sample error |
|---|---:|---:|
| Gujrat second | 2 / 47 | 5 / 35 |
| Gujrat first | 1 / 45 | 1 / 43 |
| Mandiali short | 1 / 32 | 0 / 11.85 |
| Mandiali long | 0 / 40 | 1 / 26 |
| Jamshoro | 0 / 25 | 1 / 31 |

Multiply samples by 0.029296875 for nanoseconds: Gujrat-second base has median
0.0586 ns and p95 1.3770 ns; subbase has median 0.1465 ns and p95 1.0254 ns.
Nearer seeds help some regions without guaranteeing selected identity. For
Gujrat-second base, correct proposal coverage is 75.56% within 5 m (N=225),
49.01% at 5–25 m (N=710) and 43.47% at 25–100 m (N=651). Full per-road sample/time
errors, bracketed/left-tail/right-tail counts, seed-distance strata and longest
wrong accepted observed spans are retained in `per_case_details` in the
[numerical appendix JSON](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/report/completed-results.json).
They are descriptive strata, with no invented labels in unknown regions.

The [comparison figure index](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/comparisons/index.md)
contains five PNG/SVG figures separating model, decoder, interpolation, gate,
conditioning controls and initialization. The
[training-versus-outer precision/coverage curves](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/comparisons/primary-train-vs-outer-precision-coverage.png)
show diagnostic threshold sweeps; operating markers use training-selected gates.
Outer-road curves are not a license to choose a threshold on that road.

The [initial atlas manifest](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/evaluation/initial-atlas/manifest.json)
froze 48 windows before challenger results (SHA256
`d90ee78341e28bafb7ac684d39496207aa381ef78a7767fafc412a5d82f6f5f3`).
Its categories use signal/reference proxies for weak returns, strong neighbors,
transitions, sparse coverage, distant brackets and tails; they are not new absence
or physical-break annotations. Full context and close-ups use these same windows
for [primary Gujrat](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/primary-gujrat-v2/figures.json),
[primary Mandiali](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/primary-mandiali-v2/figures.json),
[primary Jamshoro](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/primary-jamshoro-v2/figures.json),
[correction-aware](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/correction-gujrat-v2/figures.json)
and [depth-RMS](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/depth-rms-gujrat-v2/figures.json).
The v2 rendering corrects image-cell plotting extent; numerical predictions and
scoring are unchanged. The
[Gujrat-second subbase atlas](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/figures/primary-gujrat-v2/gujrat-second-layer3-fixed-atlas.png)
shows useful selected-depth regions alongside neighboring-lobe and transition
errors. Attractive proposal segments are not shown as accepted measurements.

## Reproduction and provenance

All training used immutable
[source-v3](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/source-v3/executable-snapshot.json).
SHA256 identities are:

| Artifact | SHA256 |
|---|---|
| Dataset manifest | `c81944434af55c8f2f231ccf6cacf42ccc2a33fb6e1ad29cb44370c6b772667a` |
| Case registry | `282c7b43340f854f85ecadedab046534ca2e2489f9f3ab7d96a02b892b37ccd2` |
| Executable source-v3 manifest | `7031920bd465ed501e66c505b590ec397deb66cc8d61eccdaf34e1d388eb40be` |
| FP32 calibration source-v4 manifest | `8c3fad236c39d378c136d537b00c827f8dc94e7dcecfee5c1d78f1fb11dc5ca1` |
| Executed trainer | `24f7e709556c74de6f002adc24abca7162574672aa1c06f4faffab9e7d89ad27` |
| Model module | `2537976b21fb12a7c9a341a54407e66e19137e32397bb5617faf39c47c7c10d5` |
| Data module | `3a3e565362c152b9a312a3daeb3970dfdc873d1a65d8383d479cdffd473de491` |

Each evaluation and replay stores its own immutable source snapshot, source/model/
input/seed/scorer identities and command. Numeric model predictions are saved
before target labels are opened for batch scoring. The independent provenance
[audit](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/review/provenance-final.json)
found no invalidating discrepancy in its bounded metadata/source/weight scope;
it did not independently rehash raw arrays or every probability file.

The consolidated production/coordinate/measurement and research verification
passed **189 tests** after the replay repair, with no failures or skips and no
source changes during the run;
[logs and exact commands](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/verification-delivery/consolidated-tests.json)
retain that snapshot. The earlier 181-test result is preserved separately.
Source-bound earlier scorer/decoder probes are reported
with their own hashes, not generalized to every later revision. The three
checkpoint-bound data/model/training files have scoped LF Git attributes so
Windows checkout does not change strict byte fingerprints. One cosmetic Ruff
E501 remains on the trainer's 101-character BF16 context-manager line, retained
to preserve the executed training-source identity.

The machine has an i9-14900HX, 34.05 GB RAM and an RTX 4060 laptop with 8,188 MiB
reported VRAM. The isolated environment is Python 3.13.12 with PyTorch
2.8.0+cu128; no system driver change was needed. Training used a single serialized
GPU queue. Allocated CUDA peaks were 81.4–117.8 MB (116.0 MB primary), distinct
from reserved/framework/device memory. Observed training runtimes range 39.1–328.3
seconds; load/pauses differ and their cause was not established, so they are not a
controlled architecture-speed benchmark. A resumed trainer's elapsed field covers
its latest invocation; use the ledger for cumulative timing. Dependencies and
resource samples are in
[environment-requirements.txt](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/environment-requirements.txt)
and [gpu-resource-samples.csv](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/gpu-resource-samples.csv).

Primary batch prediction took 46.2 s for both Gujrat acquisitions/all three
interfaces, 41.2 s for both Mandiali acquisitions/all three interfaces, and
26.7 s for Jamshoro's two interfaces; full evaluate-and-score runs took 65.7,
57.3 and 34.1 s respectively. These batch runs perform separate layer jobs and
include their recorded overhead. In the real replay runner, one Gujrat-interface
initial inference took approximately 1.25–1.36 s; Mandiali long took 3.83–3.93 s.
An unavailable-answer lookup is not a meaningful inference-latency measurement.

The independent [scientific review](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/review/final-scientific-review.md)
supports the no-promotion decision. Remaining scientific limits are three subbase
road groups, unknown per-pick creation history and partly conventional semantics,
correlated/repeated inner observations, early-stop/calibration validation reuse,
short-window versus full-road calibration, mixed initialization/context effects,
and no accurately scored alternate-target real-road control. A fresh prospective
road remains necessary for an independent final reliability claim. The next
defensible research target is stronger long-range selected-interface evidence
with training-road full-sequence acceptance calibration; depth normalization is
a candidate, not a demonstrated deployable solution.

The following PowerShell reproduces the primary Gujrat training/calibration and
evaluation into new directories. Exact executed commands for all completed arms,
controls, supplements and replays are retained in
[ledger.jsonl](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/ledger.jsonl).
Run jobs sequentially on the one GPU; do not overwrite historical evidence.

```powershell
$artifactRoot = 'C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910'
$workspace = 'C:/Users/faisa/Desktop/gpr_next_stage_20260910'
$python = "$workspace/.venv/Scripts/python.exe"
$modelOutput = "$artifactRoot/models/reproduce-primary-gujrat-s42"
$evalOutput = "$artifactRoot/evaluation/reproduce-primary-gujrat-s42"
& $python "$artifactRoot/source-v3/scripts/train_processed_ml.py" `
  --dataset "$artifactRoot/processed_dataset/manifest.json" --output $modelOutput `
  --held-out gujrat --layers 1 2 3 --steps 2400 --validation-interval 200 `
  --validation-episodes 36 --patience 8 --fine-width 65 --seed 42 --device cuda
& $python "$artifactRoot/source-v4-calibration/scripts/calibrate_processed_ml.py" `
  --model $modelOutput --dataset-manifest "$artifactRoot/processed_dataset/manifest.json" `
  --output "$modelOutput/calibration.json" --episodes 192 --trace-stride 4 --device cuda
& $python "$workspace/scripts/evaluate_processed_campaign.py" --model $modelOutput `
  --dataset-manifest "$artifactRoot/processed_dataset/manifest.json" `
  --cases "$artifactRoot/evaluation/cases.json" --calibration "$modelOutput/calibration.json" `
  --layers 1 2 3 --device cuda --output $evalOutput
```

Use `--held-out mandiali` or `jamshoro` for the other registered folds. The
independent training flags are `--layer-only`, `--correction-aware`,
`--normalization depth_rms`, `--no-context`, or `--seed 84`, as recorded per run.
For a genuinely interrupted optimization, repeat its **original configuration,
output and source snapshot** with `--resume <model-directory>/checkpoint.pt`.
The completed original queue can be inspected/skipped with
`& $python "$artifactRoot/run_queue.py" --jobs gujrat:primary:42 mandiali:primary:42 jamshoro:primary:42`.

To reproduce one explicit local action sequence on the completed primary model:

```powershell
& $python "$workspace/scripts/replay_processed_ml.py" `
  --manifest "$artifactRoot/processed_dataset/manifest.json" `
  --record-id gujrat-087182ae3b6f `
  --checkpoint "$artifactRoot/models/primary-gujrat-bf16-s42" `
  --case gujrat-second --layer 2 --stride 4 --policy uncertainty `
  --operation local_correction --acceptance-threshold inf --device cuda `
  --output "$artifactRoot/interactions/reproduce-primary-gujrat-base-local.json"
```

The default recorded prefixes are 0/1/2/4 requests. Replace only the operation
with `global_model_seed` for the separately costed global action. For an interrupted
replay, repeat its exact command with `--resume`; the CLI follows the original
checkpoint's source snapshot. Use only trusted local replay checkpoints. The
[continuation note](C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/continuation.json)
records durable campaign state. Weights and large evidence remain local, outside
Git; no remote push is part of this campaign.
