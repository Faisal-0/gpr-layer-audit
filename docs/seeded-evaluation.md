# Seeded tracker evaluation, 2026-09-09

This evaluation freezes the executable source at
`c774cf97572e889fa6cbff690a490e9ae4d2e951`. The original comparator under
`exports/conventional/frozen` is preserved. The new complete source snapshot is
`exports/seeded-tracker/baseline-source/src`; source, input, seed and scoring
hashes are recorded in `benchmarks/seeded-evaluation-inputs.json`.

The immediate test cases are the previously used short Mandiali processed
acquisition (459 native traces, 11.475 m sampled footprint) and the second
Gujrat portion (11,498 native traces, 287.45 m sampled footprint). Both use
512 samples, 0.029296875 ns/sample, 0.025 m/native trace and a -3 ns header
time origin. The origin is resolved from the header, without fitting to picks.
Working strides are 1 and 4 respectively. Processing order and polarity are
unchanged; geographic driving direction is not independently verified.

The new input audit verifies exact recorded amplitude correspondence for all
reviewed observations on these two processed files. Header-relative time
residuals are at most 0.00000005 ns, consistent with stored decimal rounding.
There are no duplicate/out-of-bounds observations or ordering conflicts on
these cases. DZX layer numbers 0/1/2 use the historical asphalt/base/subbase
convention; the stored names are only "Layer 1", "Layer 2", "Layer 3".
This remains an interpretation-reference comparison, not independent semantic
or thickness ground truth.

The user subsequently confirmed the layer mapping for Mandiali and Gujrat and
confirmed that no unused annotations are available. This evidence is persisted
separately in `benchmarks/seeded-label-confirmation.json`; the frozen manifest
and all original input hashes remain unchanged. Interface semantics are now
user-confirmed for those roads. Physical thickness and independent-road
correctness remain separate unestablished claims.

`benchmarks/seeded-mandiali-short-seeds.json` preserves the calibrated short
Mandiali run's three native observations per layer. The matching Gujrat seed
file preserves the earlier 0.4 m experiment's native clicks at both 0.4 and
0.1 m. Files contain only operating observations, never withheld reference
samples. Incompatible working grids fail rather than snapping a click.

The signed-lobe/time contract is unchanged: agreement requires error at most
`max(2 samples, initial-seed lobe width / 4)` and no intervening zero or sign
crossing. Initial seeds and any additionally revealed observation must be
excluded from automatic scoring. Unknown labels remain unknown. Automatic
coverage is divided by eligible reviewed non-seed observations. Converted
metres are labelled sample footprints (count times working spacing), not
interpolation over missing reference labels. Contiguous wrong spans require
adjacent scored working rows. The existing `reflector_switches` count means
wrong signed-lobe observations; it does not count independently verified
semantic switch events.

The fresh inventory contains 12 reviewed processed DZX files, with 555,439
stored layer observations across eight physical-road groups. Subbase labels
occur on Mandiali, Gujrat and Jamshoro. Repeated acquisitions are grouped by
road; both Gujrat portions and both Mandiali acquisitions belong together.
These are available interpretation labels. Historical "zero eligible raw
labels" reports do not describe this processed-label inventory. The existing
conditional U-Net uses six input channels and masked boundary/visibility
losses, but its annotation loader requires original raw sample coordinates
and explicit per-observation provenance. Processed observations must not be
silently fed to that raw-coordinate loader. This evaluation does not train a
new model or transfer evaluation references into inference.

Neither tested road is held out. Daska/Pattoki have extensive documented
development use. Burewala appears in `HYBRID_REAL_ROAD_DEVELOPMENT.md`, and
Jhang/Jamshoro also occur in the August 29 radar-only seed manifest. Therefore
the existing phase-reserved road names are not evidence of untouched physical
roads. With two previously used road groups and strong within-road correlation,
sample-level binomial intervals would overstate independence. No such interval
or independent-road correctness claim accompanies these results.

Reproduce an immutable baseline from the repository root:

```powershell
.venv/Scripts/python.exe scripts/seeded_eval.py run --case gujrat-second --config benchmarks/conventional-motion-calibrated-development.json --output exports/seeded-tracker/baseline/gujrat-control.json
.venv/Scripts/python.exe scripts/seeded_eval.py run --case mandiali-short --config benchmarks/conventional-interval-template-development.json --output exports/seeded-tracker/baseline/mandiali-interval.json
```

`run` refuses an existing output and verifies source data and seed hashes.
`--source` selects another immutable package root and requires unchanged
evaluation, reference-coordinate and seed-loading helpers. `summarize` writes
per-layer counts, bracket/tail coverage, proposal/gate losses, timing errors
and longest wrong spans. `overlay` draws agreement and error colors only after
inference, on the actual processed radar. Neither command modifies labels.

The null-config `joint_seed_adaptive` run records the exact existing helper's
default. The helper applies numerical padding extension only with an explicit
nonempty config. A separately labelled `default-shared-input` comparator uses
the same validity/extension as the conventional control so input handling is
not confused with an algorithm change. Experimental control, interval scoring,
distinct inference and their combination use identical masks, amplitudes,
native seeds and initial scoring widths. Simultaneous benchmark runs are for
correctness comparison; their wall times are not isolated latency claims.

Results and interaction curves are generated artifacts under
`exports/seeded-tracker`; compact result manifests accompany completed runs.
An accepted-agreement threshold alone is insufficient: useful correct coverage
and a measured intervention budget remain required product gates.

The working checkout and archived source can differ only in Git's CRLF/LF
line endings. Helper validation normalizes CRLF to LF for this comparison,
records both raw hashes and the normalized hash, and rejects every other
scorer/coordinate/seed-helper edit. The signed-lobe/time contract is unchanged.

Completed frozen-source comparisons (agreeing accepted / accepted; automatic
correct coverage in parentheses):

| Road/configuration | Base | Subbase |
|---|---:|---:|
| Mandiali, application default | 0/0 (0%) | 0/0 (0%) |
| Mandiali, conventional control | 4/4 (0.90%) | 42/44 (10.40%) |
| Mandiali, interval scoring | 4/4 (0.90%) | 42/44 (10.40%) |
| Mandiali, distinct paths | 4/4 (0.90%) | 42/44 (10.40%) |
| Mandiali, interval plus distinct | 4/4 (0.90%) | 42/44 (10.40%) |
| Gujrat, application default, 0.1 m | 6/6 (0.38%) | 0/0 (0%) |
| Gujrat, conventional control, 0.1 m | 369/547 (23.27%) | 4/63 (0.37%) |
| Gujrat, interval scoring, 0.1 m | 382/559 (24.09%) | 6/69 (0.56%) |
| Gujrat, distinct paths, 0.1 m | 369/547 (23.27%) | 4/63 (0.37%) |
| Gujrat, same control/seeds, 0.4 m | 29/29 (7.34%) | 0/0 (0%) |

The controlled Gujrat failure reproduces exactly. Its base has 157 wrong
signed-lobe observations, subbase 56. Longest contiguous incorrectly accepted
observed spans are 1.6 m base and 1.2 m subbase; unknown reference rows break
those spans rather than silently extending them. Bracketed base is 303/473
agreeing; subbase accepts none of 808 bracketed non-seed observations. Its
subbase tail agreement is 1/28 on the left and 3/35 on the right. The control's
178 incorrect accepted base observations constitute 11.22% of all 1,586
eligible base observations, alongside 23.27% correct and 65.51% unresolved.

Null-config and identical-input default comparators are numerically identical
on both tested roads. The default's deep-layer result is near-total abstention.
The conventional control and interval variant fail accepted correctness badly
on Gujrat. Distinct-path inference alone changes no reported counts. Neither
abstention nor these experimental variants pass the product gate, so these
comparisons do not justify promoting a tracker or relaxing its thresholds.

`benchmarks/seeded-mandiali-comparisons.json` and
`benchmarks/seeded-gujrat-comparisons.json` contain all layer/bracket/tail metrics.
`benchmarks/seeded-gujrat-resolution-comparison.json` separately restricts the
resolution comparison to common physical observations and checks the exact
same seeds and scoring widths. The compact post-inference failure atlas is
`exports/seeded-tracker/baseline/mandiali-failure-atlas.json` plus its PNG. It
finds 438/443 matching base candidates and 352/404 subbase candidates. At the
first observed loss stage, 190 base and 103 subbase checkpoints have zero
matching segment support; another 138/124 have a wrong or missing selected
route. Correct proposals rejected at the final gate number 106/83. These are
different failure mechanisms; treating every unresolved output as absent or
merely lowering the final gate would misdiagnose them.

Run `python scripts/seeded_eval_contract_test.py` to verify the reproduced
control counts, exact native seeds, seed exclusion, all eight original frozen
comparator hashes, input-fingerprint rejection, and fixed denominators after a
tracker exception. The existing reference/seed suite also passes 29 tests.

The supplemental input manifest is
`exports/seeded-tracker/additional-inputs/supplemental-evaluation-inputs-v1.json`
(SHA256 `e4442c20a4144b36b1800216899ad07fb7705086e57d1e8473594ef541377004`).
It adds first-portion Gujrat (1,036.05 m), long Mandiali (1,066.65 m), and
both Bahawalpur acquisitions (414.60 m and 693.80 m), retaining every fourth
native trace at 0.1 m spacing. All four paired processed coordinate audits
pass: recorded amplitudes match exactly, stored times agree with the header
within decimal rounding, and no duplicate, bounds, or ordering issues occur.
Three native observations per available interface were prospectively frozen
near 10/50/90% of the annotated trace extent, using trace distance alone.
Bahawalpur has no subbase reference and its layer semantics remain historical
convention; user confirmation covers Mandiali and Gujrat. All four acquisitions
already occur in historical development results. Their physical-road grouping
is retained, and none is described as untouched validation.

The original manifest and seeds are unchanged. A supplemental run uses the
same worker and frozen scorer, with an explicit manifest:

```powershell
.venv/Scripts/python.exe scripts/seeded_eval.py run --manifest exports/seeded-tracker/additional-inputs/supplemental-evaluation-inputs-v1.json --case gujrat-first --config benchmarks/conventional-motion-calibrated-development.json --output exports/seeded-tracker/additional-baseline/gujrat-first-control.json
```

New runs fingerprint the whole Python package as well as the processing
backend, data, seeds, configuration, and manifest. The worker verifies that
the package did not change during inference. The four initial supplemental
jobs were launched before the wrapper acquired its before-inference script
fingerprint. For the three completed jobs, the wrapper-file hash is observed
after inference and may reflect subsequent summary-only edits; their frozen
numerical package and scorer hashes are still checked before and after
inference. The original scalar Gujrat-first launch never produced a completed
result. This distinction
must be preserved when auditing those runs.

`benchmarks/seeded-mechanism-ledger.json` registers completed baseline,
identity/geometry, and independent packet/tensor/pyseistr challenger artifacts
with their array and result hashes. The completed geometry comparison reduces
Gujrat base agreement from 369/547 to 346/518 and gives only 5/62 agreeing
subbase observations. It is rejected as an application default. The matching
Gujrat geometry failure atlas and full-road before/after overlay are under
`exports/seeded-tracker/identity`. The atlas verifies graph hashes, native
anchors, grid, pulse width, and input fingerprint; it is specifically a
diagnostic of that rejected geometry run, not an unverified substitute for
a baseline graph capture.

The fixed-graph correct-only oracles recover at most 629/1,586 Gujrat base
and 19/1,067 subbase reviewed observations without visiting a known wrong
candidate. Candidate retention alone is much higher: 1,581/1,586 and
971/1,067. The oracle results disable production correspondence/geometry
costs and exclude known wrong nodes; they diagnose retained-route limitations
and provide no deployable accuracy claim. A captured Gujrat persistent base
graph without a completed evaluation is excluded from the result ledger.

Actual replay metrics can be summarized with:

```powershell
.venv/Scripts/python.exe scripts/seeded_eval.py interaction exports/seeded-tracker/interaction-prospective/mandiali-fixed-active.json exports/seeded-tracker/interaction-prospective/mandiali-fixed-midpoint.json --output exports/seeded-tracker/interaction-prospective/fixed-policy-curves.json
```

This produces metrics and coverage-versus-request PNGs for the observed steps.
It keeps each layer's initial reviewed nonseed cohort as the denominator.
An additionally revealed observation is separately analyst supplied, receives
no automatic accuracy credit, and stays in that denominator. The summary
independently checks accepted samples, visibility, signed lobes, fixed scoring
tolerance, and exact initial seeds against saved per-step inference arrays.
It records new correct observations, lost correct observations, errors,
observed wrong spans, bracket/tail results, and requests per road kilometre.
File presence alone is not treated as proof that a replay finished its budget.

The original four-action Mandiali replays are preserved as failed comparators
under `exports/seeded-tracker/interaction`; the first active correction loses
substantial correct subbase coverage. Their fixed-cohort summary is
`prior-observed-steps-fixed-denominator.json`. Prospective compatible-source
fixed/refitted-pulse comparisons have separate artifacts. Merged paths in
those first prospective logs retain initial global graph diagnostics, so
step-specific graph sizes must not be inferred from the copied provenance.
Per-row saved predictions and frozen scoring remain directly checkable.

The real-road reference inventory does not establish reviewed not-visible,
absent, or structural-break labels. Missing DZX picks remain unknown.
Recorded replay answers support positive confirmations/corrections at exact
reviewed coordinates; tests of other GUI action types are separate from
real-road accuracy evidence.

The processed learning experiment is also rejected for production acceptance.
Its verified model hash is
`224dcd67bcb7825839f3ded19d37c64cef515f4e3ab6856446460c07b6c49b5c`.
Training groups both Mandiali acquisitions together and evaluates Gujrat as
development transfer. Improved neighboring-lobe ranking does not improve the
fixed retained routes meaningfully: base correct/wrong/missing proposals change
from 397/556/633 to 396/557/633; subbase from 97/269/701 to 98/266/703.
These are proposal-selection counts, not accepted measurements. Model and
prediction file hashes were checked against the recorded result and added to
`benchmarks/seeded-mechanism-ledger.json`. The full contract and reproducible
negative result are in `exports/seeded-tracker/learning/DECISION.md` and
`exports/seeded-tracker/learning/verified/result.json`.

The four compatible-source Mandiali replays have now completed four requests
each. Their exact native anchors, initial scoring widths, backend, script, and
configuration hashes match. Fixed-pulse active replay ends at base 4/4 and
subbase 45/48 agreeing accepted observations, versus 14/16 and 12/13 with pulse
refitting. Both midpoint runs finish at base 4/4 and subbase 47/58, despite
different intermediate losses. Fixed-pulse active subbase correct coverage is
45/404 = 11.14% of the fixed initial cohort, with three incorrect accepted
observations. Its first correction gains 19 correct automatic observations
and loses 16 others; the refitted version gains 10 and loses 40. These counts
exclude the newly revealed answer, so they describe actual neighboring-route
changes rather than denominator effects.

The four additional requests over 11.45 m equal 349.34 requests/km; the
initial three observations per enabled interface are reported separately.
The active fixed-pulse subbase result is only 93.75% accepted agreement, and
base retains 0.90% correct coverage. This demonstrates a correction-scale
mechanism and a measured loop, but does not pass the useful 95% deep-layer
engineering gate. Complete per-step, per-layer, bracket/tail metrics and the
shared-viewport radar overlays are
`exports/seeded-tracker/interaction-prospective/mandiali-four-action-comparison.json`
and `mandiali-fixed-policy-before-after.png`. The plots show every scored
accepted error in the common viewport and distinguish unresolved locations.

The one-answer scope comparison uses the same requested native trace 13,
sample 327. Explicit interval reseeding changes rows 0–53 and yields subbase
44/46 agreeing observations, versus 45/48 when the declared 25 m local radius
covers this entire 11.45 m file. The interval operation gains eight correct
automatic observations and loses six; the larger scope gains 19 and loses 16.
The wrong accepted counts are two and three respectively, all same-lobe timing
errors under the frozen tolerance. Neither should be called a count of
independent physical reflector switches. The common-input scope metrics and
radar plots are in `interaction-prospective/scope-comparison`. This small
scope result does not establish useful deep-layer transfer.

The completed endpoint ablation is scored against a common original-three-seed
cohort and its frozen tolerance. The inference arms consume two versus three
observations per interface (six versus nine total, including asphalt). On
Mandiali, endpoint base gives 9/9 versus 4/4 agreeing accepted observations,
and subbase gives 11/13 versus 42/44; endpoint acceptance occurs only in the
tails, with no deep bracket acceptance. On Gujrat, endpoint base deteriorates
to 174/758 versus 369/547, and subbase to 4/285 versus 4/63. The extra middle
observation helps constrain deep identity but does not solve it. The exact
common-cohort results, operational scores, array hashes, and executable
postscorer are under `exports/seeded-tracker/endpoint-common` and registered
in the mechanism ledger. Their coverage denominators are 443/404 for Mandiali
base/subbase and 1,586/1,067 for Gujrat; revealed seeds receive no accuracy credit.

Two supplemental Bahawalpur frozen-control runs have completed. Layer 2
agreement is 1,087/1,255 on the first acquisition and 551/1,694 on the second;
correct coverage is 52.64% of 2,065 reviewed nonseed observations and 18.65% of
2,955 respectively. Wrong acceptance is 168 and 1,143 observations. These
acquisitions share a historically used physical road group and must not be
pooled to hide their large difference. Their Layer 1/2 names retain the
historical asphalt/base convention; the user's explicit layer confirmation
covered Mandiali and Gujrat. No reviewed subbase exists for Bahawalpur. Exact
metrics and full-view radar overlays are in `additional-baseline`; the
supplemental manifest records the processed input hashes and native anchors.
The first Bahawalpur run finished before Windows sleep; the second crossed
the recorded suspension, so its elapsed runtime is not compute latency.

The durable endpoint postscorer and figure builder reproduce the original
comparison without rerunning inference:

```powershell
.venv/Scripts/python.exe scripts/seeded_eval_endpoints.py --endpoint exports/seeded-tracker/interaction-prospective/gujrat-second-fixed-endpoints.json --three exports/seeded-tracker/interaction-prospective/gujrat-fixed-active.json --output exports/seeded-tracker/endpoint-common/gujrat-reproduced.json
.venv/Scripts/python.exe scripts/seeded_eval_endpoint_plot.py --output exports/seeded-tracker/endpoint-common/seed-budget-reproduced.png
```

All 30 original whole/bracket/tail comparison cells and both arms' operational
scores were reproduced exactly on the two roads; the figure is byte-identical.
The parity artifact records the original and durable script hashes. The
commands require the documented local inputs, saved replay arrays, and frozen
source snapshot; neither script exposes withheld references to inference.

The result tables contain descriptive counts. A separate post hoc paired-block
analysis now retains all reviewed rows in each physical block and uses 10,000
shared before/after resamples. The primary block size is 25 m, inherited from
the local correction radius; 10/50 m sensitivity is retained. Mandiali short
has one primary block and no interval is reported. Gujrat has 12 primary
blocks: active subbase gains 2.1 m of correct reviewed-observation footprint
with a descriptive 95% interval of [-0.3, 6.6] m, while its wrong footprint
changes by -2.4 m with interval [-7.9, 0.8] m. Midpoint base gains 8.8 m of
correct footprint with interval [1.3, 20.0] m, alongside a 0.6 m increase in
wrong footprint with interval [-0.8, 2.3] m. Footprints count labelled retained
rows; they do not interpolate classified road length through unknown labels.

`exports/seeded-tracker/block-uncertainty/paired-block-report.json` retains all
block sizes, layer/policy results, undefined zero-acceptance precision draws,
source hashes, and frozen-array scoring checks. `verification.json` records
six passed checks and a byte-identical 10,000-draw rebuild. These intervals
describe resampling within a historically used road; 25 m is not a proven
independence length, and correction effects can cross block boundaries.
All available roads have development or calibration use, and the user has no
unused annotations. Generalization uncertainty and independently annotated
physical switch-event rates remain unestablished; the minimum 95% agreement
gate is an engineering target only.

The canonical reported block analysis is `scripts/seeded_block_uncertainty.py`
with 10,000 draws. `scripts/seeded_eval_blocks.py` preserves the earlier
4,000-draw postscorer and is a provenance dependency pinned by the canonical
report. Its bytes are retained unchanged; it is not the reported 10,000-draw
analysis.

`wrong_signed_lobe_observations` counts accepted reviewed rows that violate
the frozen signed-lobe test. `wrong_signed_lobe_observed_runs` groups consecutive
retained rows with that error; an unlabelled, unresolved, or correctly accepted
row breaks a run. The longest wrong accepted footprint includes same-lobe
timing errors and equals consecutive wrongly accepted scored rows times
working spacing. It never bridges unknown observations. These are reproducible
interpretation-error diagnostics, not independently established counts of
physical reflector switches. The legacy `reflector_switches` key in full
verifier/scorer artifacts must be read with that same limitation.

The 1,066.65 m Mandiali acquisition completed the same frozen control. Asphalt
agrees at 10,339/10,396 accepted observations, with 96.98% correct coverage of
10,661 reviewed nonseed observations. Base agrees at only 2/312 (0.050% correct
coverage of 4,010), and subbase at 24/180 (0.725% of 3,309). Neither deep layer
accepts any bracketed reviewed observations: all automatic acceptance is in
the extrapolated tails. Base retains candidates at 98.90% of reviewed rows,
but only 52 proposals agree and 50 of those are gated out. Subbase retains
74.10%, proposes 47 agreeing observations, and gates out 23. Candidate presence
does not establish a correct connected route. The full-view radar plot is
`additional-baseline/mandiali-long-radar.png`; metrics are in
`mandiali-long-metrics.json`. Its 6,435.8 s elapsed runtime includes the recorded
Windows suspension; peak process memory is 8,897,282,048 bytes.

The original first-portion Gujrat supplemental control was interrupted. Its
process is gone and no result JSON, inference arrays, scored metrics, or radar
result was produced. The launch record and logs are retained, but this run is
excluded from completed accuracy and runtime results. Its registered inputs
and frozen seeds do not constitute a completed evaluation. This interruption
is separate from the completed Gujrat second-portion compiled parity check
and the new first-portion compiled control described below.

A fresh first-portion Gujrat control has now completed using frozen `c774`
source plus the exact compiled DTW change. It is recorded separately under
`additional-compiled-control-v1`; `additional-compiled-source-v1/provenance.json`
identifies the two changed waveform-matching files, the 63 unchanged Python
files, the frozen wrapper, and the numerical parity evidence. This fresh run
does not complete or replace the interrupted scalar launch. Asphalt agrees at
9,592/9,713 accepted observations, with 93.37% correct coverage of 10,273
reviewed nonseed observations. Base agrees at 37/975, with 938 wrong and only
1.080% correct coverage of 3,427; subbase agrees at 109/616, with 507 wrong
and 2.577% correct coverage of 4,229. Runtime is 1,100.724 s and peak process
memory is 11,573,133,312 bytes. The full metrics, radar overlay, job record,
result/array hashes, and source provenance are retained. This previously used
acquisition supplies another unfavorable development result, not useful deep
tracking or independent-road validation.

Both fixed-pulse Gujrat second-portion replays completed four additional
requests, or 13.92 requests/km over 287.45 m. Active selection ends at base
369/547 and subbase 25/60 agreeing accepted observations; subbase correct
coverage improves from 4/1,067 to 25/1,067 (2.34%), while 35 accepted observations
remain wrong. Midpoint selection ends at base 457/641 (28.81% correct coverage
of 1,586) and subbase 4/63. Midpoint base accepts 184 wrong observations,
compared with 178 initially. Initial seeds and newly supplied answers receive
no automatic credit. All steps, losses, bracket/tail results, and the shared
radar viewport are retained in `interaction-prospective/gujrat-four-action-comparison.json`
and `gujrat-fixed-policy-before-after.png`. Neither policy meets the useful
95% deep-interface gate.

The compiled DTW workflow verification completed successfully on Mandiali
short. Initial GUI inference took 57.51 s, the two corrections took 66.24 s
and 60.79 s, and save/reopen reconstruction took 168.94 s. GUI/CLI output,
initial anchors, correction order and coordinates, deep query scope, unresolved
TWTT withholding, and reconstruction checks all passed. The application
comparison preserves every saved path array bit for bit and every pick
identity/status, query, and answer. Pick TWTT values include the explicitly
declared timing-origin correction of -0.01171875 ns; they are not falsely
described as byte-identical across that correction. The separate Gujrat
control comparison has identical arrays and all layer metrics. The compact
records are in `dtw-acceleration/application-parity.json` and
`gujrat-control-parity.json`, with the full verification under
`workflow-verification-compiled-v2`. The benchmark, implementation decision,
and Numba license are retained. Runs occurred under concurrent workloads and
some historical wall times include suspension, so before/after wall-time
ratios do not isolate the kernel's speedup. Current correction latency remains
about a minute on this 11.45 m acquisition, and accuracy is unchanged.

The isolated reciprocal signed-lobe-group experiment is rejected for
production. Mandiali base changes from 4/4 to 19/20 agreeing accepted
observations, but correct coverage is still only 4.29%. Subbase changes from
42/44 to 78/91: wrong acceptance increases from two to 13 and accepted
agreement falls from 95.45% to 85.71%. Its prospective Mandiali gate failed,
so the Gujrat challenger was not launched. `edge-gate-loss/decision.json`,
`comparison.json`, and `DECISION.md` preserve the unfavorable result and
diagnostics. No lobe-mutual production promotion is supported.

The completed scoped-query experiment is also rejected. It ranks observable
disagreement within the declared 25 m correction radius and collapses routes
that differ only outside that window, with original inputs, seeds, scoring,
tracker thresholds, and correction behavior unchanged. After four Gujrat
requests, base remains 369/547 agreeing accepted and subbase ends at 6/92
(86 wrong), compared with 25/60 (35 wrong) for the existing active policy.
Only the first scoped request received a reviewed answer; the other three
requested coordinates had no reviewed answer and remain unknown. The
comparison does not assume that an unreviewed location is absent. Initial
arrays match all three compared runs exactly, and every action preserves
other layers and rows outside its declared scope. Verification covers the
exact 25 m configuration; other radii are unverified. The readout, verdict,
complete comparison, request outcomes, regression checks, and array contract
are retained under `scoped-queries`. No performance promotion follows.

The timing diagnosis of the rejected lobe-mutual run retains all 14 accepted
errors: one base and 13 subbase errors pass the signed-lobe identity test but
fail its frozen timing tolerance. Six are unambiguous shoulder errors within
one extremum basin, seven select a different extremum in the same signed
lobe, and one lies at a shared valley. All six initial deep seeds lie exactly
at their strongest local signed-lobe extrema. Three missing reviewed extrema
at rows 247, 263, and 348 were removed by the 12-packet cap, despite valid
bounds and waveform context. Their zero-based packet ranks are 13, 13, and
14. Nine error rows have a feasible exact reviewed-peak route, but only six
such peaks pass the unchanged correspondence gate. The diagnosis therefore
does not justify output snapping or transferring confidence between peaks.
`timing-identity/FINDINGS.md`, `summary.json`, `extremum-basins.json`, and
`missing-peak-cap-loss.json` preserve the exact limitations. These are
diagnostic findings, not performance evidence for a subsequent extremum-state
experiment.

The subsequent extremum-member experiment was implemented and evaluated on
both Mandiali graph controls. On the original graph, base remains 4/4 agreeing
accepted while subbase decreases from 42/44 to 35/36. On the rejected lobe-mutual
graph, base changes from 19/20 to 20/20 and subbase from 78/91 to 77/82. Combined
deep correct acceptance stays at 97; wrong acceptance decreases from 14 to five.
Subbase precision is still only 93.90%, two new errors appear, and the original
graph loses useful coverage. It is rejected without further tuning or Gujrat
promotion. Ten numerical contracts pass; all four captured graphs reconstruct,
16 interval objectives match independent optima within 2.3e-15, and all recorded
emitted-state margins replay exactly. These are correctness checks on the
implemented objective, not evidence of physical identity. The emitted base
timing also changes sequential subbase bounds; the experiment includes that
effect and is not a fixed-candidate-table comparison. Full results are retained
under `extremum-states` and in the compact package.

The paired-block v3 provenance refresh preserves the original report and exact
legacy helper. It changes only the hash of three formatting-only string wraps
in the current helper; all numerical results and design fields remain identical.
`block-uncertainty/verification-v3.json` records AST equality, independent review
of the original numerical accounting, and a byte-identical v3 rebuild.

Final integration review reproduced two defects beyond the earlier workflow
test's cases. Final station values could merge interleaved corrections during
reopening, and legacy aggregation could regenerate timing from review-only
proposals. Processed corrections now persist immutable ordered action snapshots;
processed aggregation now requires accepted, visible, matched interface
observations and withholds derived quantities when the immediate upper
interface is unavailable. The default raw inference and frozen comparators
remain unchanged. See `PROCESSED_CORRECTION_HISTORY.md` and the focused action
history/accepted-aggregation regression tests. These repairs establish workflow
behavior, not an improvement in automatic deep-layer precision.

The final source at `e035b4a` passes 494 tests with one optional upstream test
skipped. Its fresh actual GUI/CLI run passes all 11 workflow checks. Initial
Mandiali inference took 52.41 s; two corrections took 54.25/55.07 s and saved
reconstruction 151.03 s. Source hashes match the working source exactly.
All four path stages, pick identities/statuses, queries, answers and timing
match the earlier compiled control; strict derived-measurement aggregation and
ordered action persistence are separately regression-tested. The final records
are `final-integration-v1.json`, `workflow-action-history-parity-v2.json`, and
`workflow-verification-action-history-v2` under `exports/seeded-tracker`, with
compact copies and current radar/action figures in the review package. Station
undo was also fixed to use creation order rather than road distance. These
tests do not certify the unobserved workflows or the deep-layer product gate.

`benchmarks/seeded-results/index.json` indexes the compact review package,
including completed supplemental controls, both roads' interaction curves,
endpoint scoring, unsuccessful prior replay/geometry/learning/lobe-mutual/
scoped-query experiments, timing diagnosis, paired-block uncertainty, compiled
workflow verification, and the original interruption record. The fresh
compiled Gujrat-first control is identified separately. Each entry records the
original and packaged SHA256 separately.
Compacted result tables retain every cell and hoist only identical repeated
provenance; workflow summaries omit repeated observation rows while retaining
all per-step metrics. Hashed full exports and inference arrays remain the
source of detailed evidence. Package generation does not run inference or
alter an existing export:

```powershell
.venv/Scripts/python.exe scripts/package_seeded_results.py --check
.venv/Scripts/python.exe scripts/package_seeded_results.py --output benchmarks/seeded-results
.venv/Scripts/python.exe scripts/package_seeded_results.py --verify
.venv/Scripts/python.exe scripts/package_seeded_results.py --verify-package-only
.venv/Scripts/python.exe scripts/package_seeded_results.py --self-test-package-only
```

Generation refuses to replace an existing package. Full `--verify` checks every
packaged file, every corresponding original source hash, and the exact
packaging-script bytes. It requires the original ignored exports. A fresh
checkout can instead use the explicit `--verify-package-only` mode, which
checks all packaged sizes, hashes, and Git byte-preservation attributes against
the committed index without reading original exports. Its report marks original
sources, inference, and independent provenance validation as unverified. Full
verification remains strict; it never silently falls back to package-only mode.

The index records both the exact packaging-script hash and a companion hash
that normalizes CRLF to LF only. Package-only verification permits either
recorded script hash and reports which matched. No other whitespace, encoding,
or code edits are accepted. Packaged evidence itself always requires exact
bytes; `.gitattributes` disables its newline conversion. These checks establish
agreement with the index, not independent truth of the recorded metrics.

`--self-test-package-only` copies the package and verifier into a fresh
directory containing no original exports. It checks package-only success and
full-verification failure there, rejects a same-size corrupted metric and
changed Git attributes, accepts LF/CRLF verifier copies, and rejects an
arbitrary verifier edit. The isolated copy is retained for inspection. None
of these checks reruns inference. These artifacts remain development
interpretation evidence on historically used roads; no physical-thickness
accuracy or untouched-road result is claimed.

The subsequent native CNN/common-dense-graph and adjacent-packet objective
experiments are completed and rejected. Their controls, numerical counterexamples,
mask repair, oracle bounds, reproduction commands and before/after radar overlays
are described in [dense identity experiments](DENSE_TRACKER_EXPERIMENTS.md).
