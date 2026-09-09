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
fingerprint: their wrapper-file hash is observed after inference and may
reflect subsequent summary-only edits. Their frozen numerical package and
scorer hashes are still checked before and after inference. This distinction
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
