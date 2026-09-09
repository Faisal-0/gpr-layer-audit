# Conventional correspondence development, 2026-09-09

Work continues on base and subbase. ML remains disabled; no interface has been
promoted. These experiments follow `CONVENTIONAL_TRACING_VALIDATION.md`.

## Diagnosed failures

The short Mandiali processed input has 459 traces at 0.025 m spacing. Capturing
its observation graph, without passing evaluation picks into inference, found:

- Long links extrapolated one adjacent-trace displacement over the whole gap.
  A one-sample local displacement becomes 40 samples over a 1 m link, even when
  neighbouring displacements subsequently reverse. Forward and backward local
  slopes can also disagree despite compatible integrated displacement.
- The configured graph distances jumped from one trace directly to 1 m. A few
  noisy traces could therefore isolate an otherwise observable seed neighbourhood.
- Raw single-trace matching is noisy on these deep reflections. For example,
  the first subbase seed's best adjacent incoming waveform agreement was around
  0.73, below the uncalibrated 0.80 acceptance setting. The third seed had no
  outgoing retained link. This is evidence about correspondence, not an ML limit.

## Experimental changes

`integrated_motion` composes measured adjacent displacements through intermediate
waveform coordinates. Forward and reverse fields are composed independently.
Invalid signal and boundaries invalidate the motion prediction. Composition does
not produce intermediate measurements.

The near-link configuration retains the established 1/2/5/10/25 m distances and
adds 0.05/0.125/0.25/0.5 m. Missing graph observations remain gaps.

`waveform_context_radius_m` optionally supplies aligned neighbouring snippets to
waveform matching. Contributions require observable snippets, reciprocal motion,
and matching central polarity. Structural breaks prohibit context sharing.
Original amplitudes, candidate coordinates and seed reference templates remain
unchanged. This is local matching context, not adaptive template-bank promotion.

Both new options default to inactive. Experimental configuration files are
`benchmarks/conventional-motion-near-experimental.json` and
`benchmarks/conventional-motion-context-experimental.json`.

## Graph replay evidence

These counts include seeds and any supported candidate, including wrong-lobe
candidates. They **are not accuracy or accepted-coverage measurements**.

| Experiment | Base rows with support ≥0.80 | Subbase rows with support ≥0.80 |
|---|---:|---:|
| Existing graph | 5 | 3 |
| Integrated displacement | 5 | 3 |
| Integrated displacement + near links | 14 | 7 |
| Plus 0.1 m waveform context | 22 | 40 |

At any nonzero support, base connectivity rose from 20 to 357 rows in the combined
experiment; subbase rose from 4 to 45. Complete-path feasibility, path margins and
reference agreement still determine whether these connections are useful.

Captures and replays are under `exports/conventional/continuation-diagnostics`.
`scripts/diagnose_conventional_correspondence.py` captures seed-only graph inputs
and compressed links; `--replay` changes correspondence without regenerating
candidates. Complete tracker benchmarks preserve separate source snapshots.

## Verification

New tests check composed motion against stepwise advection in both directions,
invalid intermediate samples, cancellation, noise reduction, immutable original
waveforms, zero signal and structural isolation. An integrated tracker test
preserves exact clicks and selected lobes against a brighter competing reflector,
while retaining a signal gap. The targeted integration suite passed 69 tests
before the last added tracker test; that test and the four motion tests passed.

Full tracker comparisons are required before adopting any option. Template-bank
adaptation, wider physical-road validation and promotion remain incomplete.

## Complete short-road comparisons and observability repair

The original graph, near-link and contextual comparisons used the same backend
fingerprint, stored in `benchmarks/conventional-continuation-results.json`.
Near links recovered feasible base paths between both pairs of seeds, but the
existing observability gate still rejected every non-seed base proposal.

Inspection found a definite width-contract bug: `measurement_packet_support`
already derives its background and packet windows from **lobe width**, but the
hybrid caller supplied **packet width**. For the 29-sample base lobe, that made
the background window 435 samples rather than 145 and included the much stronger
shallow reflection. All three base seeds received zero observability. Using lobe
width restored seed support to 0.2225, 0.0660 and 0.0854, respectively, without
changing any measurement amplitude. The caller is repaired and a synthetic
bright-shallow/weak-wide-base regression test verifies accepted deeper picks.

| Short-road experiment | Base accepted / agreeing | Subbase accepted / agreeing |
|---|---:|---:|
| Control | 0 / 0 | 0 / 0 |
| Integrated motion + near links | 0 / 0 | 0 / 0 |
| Plus 0.1 m matching context | 0 / 0 | 7 / 6 |
| Plus corrected observability width | 4 / 4 | 7 / 6 |

Final short-road correct coverage is only 0.90% for base and 1.49% for subbase.
The subbase discrepancy is an 11-sample timing error within a split positive
waveform, despite high reciprocal correspondence. Its accepted agreement is
85.7%, below the required 95%. The contextual configuration therefore remains
opt-in; neither these small counts nor the graph improvements justify promotion.

The observability run also removes a redundant quadratic scan from path-margin
calculation: component bounds are computed once rather than once per segment.
This does not change the path objective. Regression tests for geometry, signal
gaps, exact clicks, endpoint consistency and path margins passed: **71 tests**.

Wider development runs of the observability repair are under
`exports/conventional/continuation-width-fix`. The existing frozen comparator
remains immutable, and held-out road scoring has not been used for these changes.

The full non-ML regression suite now passes **275 tests in 167.16 s**; Ruff and
the whitespace check pass. The diagnostic overlay is
`exports/conventional/continuation-subbase-disagreement.png`.

The next path-selection investigation should address temporary split waveforms:
high reciprocal similarity alone can prefer an early peak and return to the
original reflector on the next trace. This is a concrete failure of local
correspondence to express timing ambiguity. Acceptance thresholds should not
simply be lowered to recover the many remaining weakly connected base proposals.
Template adaptation must retain original seed references and exclude such
ambiguous segments from its updates.

## Motion ties and missing competing paths

A symmetric synthetic split pulse reproduced false certainty: replacing one
90-sample pulse by equal pulses at 84 and 96 caused the tracker to accept sample
83 with path margin 1.0. The motion estimator's equal-score tie chose the earlier
shift by array order. The graph used high correlation as evidence that this
shift was unique, eliminating the competing later route before path comparison.

The `minimum_motion_margin` experiment requires a separated shift match before
using motion as a hard displacement constraint. Composition carries this
ambiguity through intermediate traces, and contextual waveform averaging cannot
borrow a neighbour through an ambiguous shift. Both forward and reverse estimates
remain independent. The two hypotheses then survive to path comparison: the
synthetic split sample stays missing with margin 0.0072, while at least 45 of its
55 other/seeded observations remain accepted. This is a correction to evidence
handling rather than merely lowering acceptance thresholds.

The additional configuration is
`benchmarks/conventional-motion-unique-experimental.json`; its 0.02 shift-margin
setting remains uncalibrated and opt-in. The targeted suite now passes **73 tests**,
including symmetric splitting and ambiguous intermediate-flow composition.
The split-waveform regression also passes with integrated motion and waveform
context independently enabled or disabled (11 motion tests in total).

Across the six coarse development/calibration acquisitions, the observability
width repair alone retained only four accepted non-seed base reference points
(one short Mandiali, three Gujrat second portion), all agreeing; it accepted no
subbase reference points. It repairs a real blocker but does not solve coverage.
The full results are in `benchmarks/conventional-continuation-results.json`.

On short Mandiali, the motion-uniqueness experiment recovered both complete
subbase seed intervals: 106 non-seed reference observations were accepted, 88
agreed, and correct coverage rose to 21.78%. Accepted agreement was only 83.02%,
so that acceptance setting is insufficient. Development-only calibration selected
correspondence ≥0.85 and path margin ≥0.01: 44 retained observations, 42 agreeing
(95.45%), representing 10.40% correct coverage. This is fitted on the same
development acquisition, **not** an independent validation result. Base still
lacks adequate accepted observations for calibration, and settings remain
unfrozen. The calibrated configuration is
`benchmarks/conventional-motion-calibrated-development.json`.

Complete-tracker verification of that fitted selection is under
`exports/conventional/continuation-calibrated` and reproduced **42/44 agreement
and 10.40% correct coverage**, with no selected-lobe switches among those accepted
points. Base retained 4/4 agreement and 0.90% correct coverage. Separate 0.4 m working-resolution
comparisons on both Gujrat portions are under `continuation-gujrat-control` and
`continuation-gujrat-experimental`. At that resolution, a 0.1 m waveform-context
radius rounds to zero; those runs primarily test displacement/motion ambiguity
handling and the fitted acceptance settings, not native-grid contextual averaging.

## Gujrat transfer and finer-resolution comparison

The short-Mandiali thresholds do **not** yet generalize across these inputs.
Both Gujrat portions belong to one physical-road group. These remain development
experiments, not held-out tests.

| Gujrat second portion | Interface | Accepted agreement | Correct coverage |
|---|---|---:|---:|
| Prior repaired hybrid, 0.4 m | Base | 0/2 | 0.0% |
| Motion-uniqueness configuration, 0.4 m | Base | 29/29 | 7.34% |
| Prior repaired hybrid, 0.4 m | Subbase | 0/1 | 0.0% |
| Motion-uniqueness configuration, 0.4 m | Subbase | No accepted points | 0.0% |
| Frozen original hybrid, 0.1 m | Base | 66/85 (77.65%) | 4.16% |
| Motion-uniqueness configuration, 0.1 m | Base | 409/606 (67.49%) | 25.79% |
| Frozen original hybrid, 0.1 m | Subbase | 17/43 (39.53%) | 1.59% |
| Motion-uniqueness configuration, 0.1 m | Subbase | 6/44 (13.64%) | 0.56% |

The first Gujrat portion accepted no non-seed base/subbase reference points in
either 0.4 m configuration. At 0.1 m on the second portion, the candidate graph
recovers complete seed intervals but also admits switches to other events. All
44 accepted subbase reference points lie beyond the last seed; most incorrect
ones follow a shallower event. The improved connectivity is therefore not enough
to establish identity. No configuration is promoted from these results.

## Guarded template adaptation and numerical performance

`adaptive_templates.py` now implements one opt-in scoring update from supported
segments. Each update requires at least three observations, correspondence ≥0.90,
path margin ≥0.10, observable signal, compatible ordering and sufficient similarity
to its original seed. Separate seed/structural regions retain immutable reference
waveforms. Ambiguous samples cannot enter the update. Its influence is bounded in
physical distance and weight; it cannot create graph links or candidates. Both
selection and competing-path margins are recomputed, and an update that changes
its own supporting observations is discarded. There is no feedback iteration.

On short Mandiali, one guarded update per deep interface was applied and accepted
results remained 4/4 base and 42/44 subbase. This is an implemented and tested
mechanism, **not a demonstrated field-accuracy improvement**. The opt-in settings
are in `benchmarks/conventional-adaptive-development.json`.

Numerical DTW now batches sufficiently large collections of short waveforms too,
while retaining scalar matching for small collections. Tests compare independent
forward/reverse scalar results with batches at lengths 17, 31, 43 and 87, including
exact ties and opposite lobes. This changes execution, not the match objective.

The three-layer 120 m synthetic runtime benchmark completed full-road inference
in 6.95 s and a warm seed-edit correction in **8.66 s**, meeting the 10 s target.
Every pick outside [35, 85] m stayed unchanged and project settings reopened
equivalently. These figures do not establish real-road latency. The source/config
fingerprints and memory measurement are in
`exports/conventional/local-runtime-three-layers/metrics.json`.

## Ordered profile context experiment

The optional whole-profile registration path now obtains reverse registration
from a separate optimization; an inverse of the forward warp is not treated as
independent evidence. Invalid samples are excluded from balancing statistics and
cannot supply fit confidence. The ordered profiles constrain otherwise similar
waveform snippets across neighbouring reflections. This mode remains opt-in.

On short Mandiali it retained 24/25 agreeing subbase observations (96%) with 5.94%
correct coverage, compared with 42/44 and 10.40% without that constraint. It removes
the earlier large split-waveform timing error but sacrifices correct coverage.
The remaining error is three samples on the same signed lobe. The full Gujrat
0.1 m comparison is under `exports/conventional/continuation-gujrat-ordered-0p1`;
the completed experiment retained 389/565 agreeing base observations (68.85%,
24.53% correct coverage) and 5/10 subbase observations (50%, 0.47% correct coverage).
Base had 166 signed-lobe switches and subbase five. These results do not justify
adopting the added constraint: accepted correctness still fails badly, and correct
coverage falls relative to the motion-uniqueness configuration at the same seeds.
This experiment took 569.45 s for the full processed acquisition.

## Controlled support across working resolutions

Earlier 0.4 m and 0.1 m experiments selected seeds after retaining their working
trace grid. Consequently they used different native observations. For example,
Gujrat second-portion subbase seeds were native rows 1104/5616/9936 at 0.4 m,
but 1104/5628/9940 at 0.1 m, with different sample positions and waveforms. The
comparisons between trackers at each resolution remain valid; differences across
resolutions cannot be attributed to sampling alone.

`conventional evaluate --seed-source` and the development script's matching option
now reuse exact native observations from a processed evaluation JSON or a
`conventional-native-seeds-v1` manifest. Both DZX and DZT fingerprints and the
reference-label mapping must match. Every evaluated interface needs support.
Incompatible trace grids are rejected rather than snapping the seeds. Raw-mode
reuse is rejected until the exact transformed support can be verified.

New evaluations export the native observations, their original interpretation
properties and coordinate provenance, plus the reused source fingerprint. Older
evaluations recover sample coordinates from the reviewed DZX at their recorded
native seed rows; tracker paths and accuracy scores are never used for this.
Withheld samples cannot change the reused support or seed-only pulse estimates,
and all reused seeds remain excluded from metrics. Existing frozen comparator
modules and prior source snapshots remain unchanged.

The completed controlled Gujrat second-portion reruns are under
`exports/conventional/continuation-gujrat-fixed-seeds-0p1` and
`exports/conventional/continuation-gujrat-fixed-seeds-0p4`, using the 0.4 m
experiment's support at both working resolutions. The comparator helper
`scripts/compare_conventional_resolution.py` verifies matching backend,
configuration, origin, native seeds and pulse tolerance, then scores only common
annotated native rows. Positive stored samples with a false acceptance/visibility
flag are excluded from accepted counts.

| Interface | Shared non-seed observations | 0.4 m accepted agreement | 0.1 m accepted agreement | Correct coverage, 0.4 m → 0.1 m |
|---|---:|---:|---:|---:|
| Asphalt (established shallow engine) | 640 | 596/613 (97.23%) | 505/529 (95.46%) | 93.13% → 78.91% |
| Base | 395 | 29/29 (100%) | 88/134 (65.67%) | 7.34% → 22.28% |
| Subbase | 261 | No accepted observations | 0/14 (0%) | 0% → 0% |

Base signed-lobe switches increase from zero to 42 on those common observations;
subbase has twelve at 0.1 m. The fine run's full reference set also fails: base
369/547 (67.46%) and subbase 4/63 (6.35%). Thus the resolution-sensitive identity
and acceptance failures persist after the seed-selection confound is removed.
These results are development diagnostics, not promotion evidence. The coarse
rerun reproduces its earlier deep-layer results exactly. The two new runs partly
overlapped in wall time; their timings are not used for performance claims.

The reproducible shared-observation metrics are in
`exports/conventional/continuation-gujrat-fixed-seeds-comparison.json`. A useful
next experiment must improve path identity while retaining valid competitors and
calibrating acceptance across working resolutions; merely increasing connectivity
or lowering thresholds has already failed. The existing objective penalizes
match disagreement and physical gaps but has no explicit reflector-slope term.
Separate seed templates survive candidate construction, while path scoring still
uses the strongest seed correlation at each candidate. Those are concrete
conventional mechanisms to investigate, not evidence that ML is necessary.

The last complete non-ML suite passed **292 tests in 176.34 s** before this benchmark
helper addition. The updated reference/seed benchmark suite passes **29 tests**,
including twelve new cases for source mismatch, exact coordinates, incompatible
grids, withheld-label isolation and seed exclusion. These are correctness checks
for the software; they do not establish field-accuracy promotion.
