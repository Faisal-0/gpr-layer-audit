# Cross-road tracking progress — 2026-08-29

## Current decision

The program is materially safer and more auditable. Frozen blind evidence now
**establishes semi-automated asphalt accuracy** under the declared release
protocol: three qualifying roads, 356 held-out manual checkpoints, 97.75%
automatic coverage, and 93.68% of visible checkpoints within ±12.7 mm. Reliable
base and subbase thickness remain unproven, so the full requested layer set is
not yet released.

## Changes made

- Preserved a dewow-only, surface-flattened road measurement before metal-plate
  subtraction. `interpretation_input_radargram` and the UI Raw view now expose
  that actual acquisition branch rather than a plate-subtracted copy.
- Added packet-scale, dip-steered lateral support computed only from that
  pre-subtraction branch. It has zero weight in candidate ranking and therefore
  cannot select a reflector.
- Added a low fail-closed automatic-pick veto for sustained absence of the raw
  packet. Manual seeds remain authoritative and are restored regardless of the
  automatic gate.
- Fixed coarse segment retracking so it no longer reapplies gain, filtering,
  and denoising to the already enhanced output. The cropped matched template
  is also no longer passed where a full-length plate waveform is required.
- Fixed metal-plate candidate ranking. Gain mismatch still keeps all plates
  below amplitude-calibration suitability, while road/plate name and zone
  evidence now survives for waveform pairing.
- Ambiguous plate pairings are no longer auto-selected in either the UI or CLI.

## Bounded seed-identity checks

The production graph family did not change because raw measurement support is
an acceptance gate, not a selector.

| Section | Retained before raw gate | Retained after raw gate | Explicit gaps after | Seed samples |
|---|---:|---:|---:|---|
| Talagang 950–1250 m, exploratory deeper seeds | 30.13% | 29.33% | 70.67% | exact |
| Pattoki-Jhoru 300–750 m, confirmed base seeds | 62.22% | 60.80% | 39.20% | exact |

The Talagang exploratory seeds are analyst hypotheses, not field truth. The
Pattoki check confirms that a user-seeded persistent horizontal event is not
globally blacklisted.

## Cross-road acquisition readiness

`scripts/audit_cross_road_readiness.py` inspected a bounded centre window from
all 17 road acquisitions under `GPR Data`.

- Pre-subtraction support was finite on 17/17 roads.
- The low raw gate left 87.7–92.9% of below-surface locations eligible before
  the tracker's other identity, continuity, and confidence checks.
- 11/17 plate pairings had a distinct name/zone winner; six remain ambiguous
  and now require analyst selection.
- 0/17 winning plate candidates are gain compatible. Plate waveform timing may
  be used, but amplitude-derived dielectric must remain disabled.

This proves acquisition-path robustness, not pavement-layer accuracy.

## Sparse-seed recovery diagnostic

`scripts/evaluate_sparse_seed_recovery.py` withheld each provisional Talagang
or Pattoki development station and ran the full radar-only graph with the other
stations.

- All remaining training seeds were preserved exactly.
- 9/10 comparable withheld interfaces were graph-selected within one declared
  pulse width.
- 6/10 were automatically visible; the others failed closed.
- The sole event-identity miss was Pattoki base at 325.8 m: training seeds were
  both more than 400 m away, candidates near the withheld 278-sample lobe were
  present, but none was seed-reachable. The graph selected the adjacent
  296-sample cycle and correctly hid it.

This demonstrates useful sparse-seed transfer and a remaining long-range
extrapolation limit. Because the seeds are development data, the result is not
an independent accuracy rate.

## Opposite-direction repeatability and seed consistency

Daska-Pasrur has two independent 1.19 km acquisitions of the same road in
opposite directions. The tracker used radar-only development clicks; neither
the legacy workbook nor design thickness entered either fit. This is a
repeatability check, not physical layer truth.

With three distributed stations at about 100, 600, and 1100 m:

- Asphalt graph selections agreed within one seven-sample pulse on 77.9% of
  comparable rows; jointly visible agreement was 80.8%, with a median
  difference of three samples.
- Base graph selections agreed within one pulse on 69.4% of rows; jointly
  visible agreement was 76.1%, with a median difference of one sample.

Adding development clicks at 300 and 900 m without first auditing every click
reduced base graph agreement to 46.2% and raised the median difference to ten
samples. The cause was isolated by withholding the 600 m station:

- The remaining four forward-pass stations independently selected base sample
  256 at 600 m, exactly matching the withheld click, and passed visibility.
- The remaining four backward-pass stations also independently selected sample
  256 and passed visibility, contradicting that pass's manual sample 243 by 13
  samples (nearly two pulses).
- With the suspect station withheld, whole-road base agreement recovered to
  63.8% and the median difference returned to one sample.

The production workflow now runs a leave-one-station-out identity audit by
default when at least three training stations exist. A contradicted manual
click is preserved at its trace, demoted to review, and requested again at the
same station and layer. The tracker never silently replaces it with the
independent prediction. One- and two-station runs retain the ordinary
fail-closed gates because withholding from two stations leaves only one
prototype and cannot establish consensus. Runtime is approximately one full
non-recursive refit per audited station, bounded by the five-station limit.

## Blinded multi-road evaluation and strict physical scale

Radar-only seed sheets were frozen before per-station workbook values were
revealed for Jhang, Rawalpindi, Burewala–Vehari 001, Bahawalpur, Jamshoro, and
Mandiali. The first three-road comparison
incorrectly paired each seed to the nearest manual workbook row regardless of
distance. At Jhang, one asphalt seed at 298 m was therefore calibrated from a
manual row at 545 m. Its physical-thickness percentages are invalid and must
not be cited.

The corrected evaluator reuses the already frozen paths and seeds. It never
reselects a radar event after seeing workbook thickness. Event identity and
time-to-depth scale are now separate gates:

- Every scale reference must be a manual workbook observation within 10 m of
  its seed.
- At least two local references are required per road/layer.
- Each local mm/sample ratio must be within 15% of the median. This limits the
  conversion uncertainty to below the layer-specific release tolerance for
  ordinary asphalt and base thicknesses.
- The ratio must imply a physical relative permittivity in `(1, 40]` using the
  acquisition sample interval. A consistent conversion faster than propagation
  in air is rejected rather than fitted.
- Physical-thickness metrics are not emitted unless all leave-one-station-out
  seed checks for that validation layer also remain within one seven-sample
  pulse.

Strict results from `exports/blinded-multiroad-strict-scale-20260829`:

| Road/layer | Seed identity | Scale | Auto coverage | Visible within target | Median visible error |
|---|---|---|---:|---:|---:|
| Rawalpindi asphalt | pass | pass, 6.0% max ratio deviation | 96.0% | 86.6% within ±12.7 mm | 3.47 mm |
| Burewala–Vehari asphalt | pass | pass, 2.0% max ratio deviation | 99.4% | 95.7% within ±12.7 mm | 3.30 mm |
| Bahawalpur asphalt | pass | pass, 5.2% max ratio deviation | 96.2% | 96.0% within ±12.7 mm | 3.96 mm |
| Bahawalpur base | pass | pass, 14.2% max ratio deviation | 96.2% | 76.0% within ±25.4 mm | 13.19 mm |
| Mandiali asphalt | pass | pass, 4.9% max ratio deviation | 96.7% | 97.7% within ±12.7 mm | 3.40 mm |
| Mandiali base | pass | fail | withheld | withheld | withheld |
| Jamshoro asphalt | fail | pass | withheld | withheld | withheld |
| Jamshoro base | pass | fail | withheld | withheld | withheld |
| Gujrat asphalt | fail | pass | withheld | withheld | withheld |
| Gujrat base | pass | pass | withheld: upstream asphalt failed | withheld | withheld |
| Bahawalpur 002 asphalt | pass | insufficient local references | withheld | withheld | withheld |
| Bahawalpur 002 base | pass | insufficient local references | withheld | withheld | withheld |
| Jhang asphalt | fail | fail | withheld | withheld | withheld |
| Jhang base | pass | fail | withheld | withheld | withheld |
| Burewala–Vehari base | fail | fail | withheld | withheld | withheld |

Rawalpindi base was deliberately excluded because its workbook validates
asphalt only. The qualifying asphalt release set excludes Bahawalpur because
only 26 checkpoints remain after the seed-neighborhood protection; it is kept
as supplementary evidence. Rawalpindi, Burewala–Vehari, and Mandiali each have
at least 30 held-out checkpoints and jointly establish asphalt accuracy.

The failed blind cases demonstrate the production guardrails rather than being
silently tuned away. Jhang asphalt differs from independently fitted events by
26 and 20 samples at two stations. Jamshoro's first two asphalt clicks reproduce
within one sample, but its third click at sample 203 is independently predicted
at sample 178, a 25-sample adjacent-cycle conflict. Burewala base differs by 12
and 11 samples at two stations. Those are event-family failures, not errors that
a dielectric or confidence adjustment can repair.

An untouched 1.04 km Gujrat road added another important dependency check. Its
base seeds reproduce within one sample and its local base scale ratios pass,
but all three asphalt seeds fail dropout by 20–26 samples. Base thickness is
therefore withheld even though the base boundary alone is stable, because an
individual base-course thickness requires both its top and bottom interfaces.
The evaluator and production design-calibration gate now propagate an unstable
upper boundary through every dependent deeper layer. Before that correction,
the diagnostic would have misleadingly reported 28.8% base coverage and 64.7%
visible accuracy from a contradicted top interface; those numbers are invalid
and must not be cited as base performance. Its 6.83–8.03 mm/sample base ratios
also imply εr=0.30–0.41, independently failing the new physical-scale gate.

Bahawalpur is the only base configuration to pass both prerequisite gates. It
does not meet the 30-checkpoint-per-road minimum and only 76% of visible base
checkpoints meet ±25.4 mm. The largest misses select a shallow, nearly fixed
sample-247 family while the reference thickness requires a deeper event by
roughly 10–15 samples. Jhang, Jamshoro, and Mandiali base seeds are mutually
stable in sample space, but their local mm/sample ratios exceed the 15%
consistency limit. This isolates the current base blocker: semantic selection
of the correct deeper boundary and defensible layer velocity, not generic path
continuity.

Bahawalpur `_002` supplies independent event-timing evidence: all six frozen
asphalt/base leave-one-out checks reproduce within one sample. The fixed
15/50/85% seed rule placed its first two stations before the workbook's local
manual observations, leaving only one scale reference within 10 m. Neither
layer can be thickness-validated, and the stations were not moved after that
coverage was revealed. Across the blind cases, base timing is therefore often
repeatable even where physical thickness is not defensible. Production may
present those interface samples for review, but must require analyst εr,
accepted design calibration, or another valid velocity source before emitting
millimetres.

The production application also no longer manufactures physical thickness
from εr=7 when scan dielectric acceptance is disabled. Unresolved dielectric
now preserves interface timing and returns no millimetre thickness. The new
project dialog starts with design thickness and dielectric blank; recorded
DZX/header εr is only used after the analyst explicitly opts in. Design values
remain a bounded tracking prior and comparison target, not proof of actual
as-built depth.

Asphalt, base, and subbase now accept independent εr values in both the UI and
CLI. When no measured/analyst εr is supplied, explicit design thickness plus at
least three consistent manual seed gaps can infer a layer-specific effective εr.
The seed samples/mm ratios must all remain within 15% of their median; otherwise
the conversion is rejected. Accepted values are labeled `design_calibrated`,
carry a 20% dielectric uncertainty allowance, and are described as an
assumption calibration in the audit—not amplitude-measured as-built truth.
Any leave-one-station-out contradiction at an interface invalidates this
design-derived conversion for that layer and all deeper dependent layers.

A disclosed Bahawalpur development run with 50.8 mm asphalt and 177.8 mm base
tested the existing bounded design-guided branch. The calibrated corridors were
26±14 and 63±14 samples, but the joint selector retained the signal-only family
at every row; base accuracy remained 76%. This negative result is retained as
evidence that providing design cannot repair a coherent wrong/ambiguous radar
family by itself. It can calibrate time to depth after a family is confirmed,
but the base semantic-selection blocker remains.

### Daska base development result

Daska's forward pass was added only as disclosed development evidence because
both its seeds and workbook had already been used in the earlier opposite-pass
diagnostics. Its three asphalt and three base clicks all survived leave-one-out
identity checks within two samples. The base scale was also physical and
consistent: 1.539 mm/sample, 4.0% maximum ratio deviation, with implied εr
7.53–8.63. The asphalt scale missed the strict ratio gate by 1.2 percentage
points and was not evaluated.

The base result used 154 held-out manual checkpoints outside the protected seed
neighborhoods:

- 93.51% automatic coverage;
- 76.39% of visible checkpoints within ±25.4 mm;
- 14.18 mm median visible absolute error.

Adding two correct midpoint development stations at 350 and 850 m raised base
coverage to 95.68% and visible accuracy to 80.45%, still below release. A
50.8/121.92 mm design-assisted run reached 80.15% at the normal bounded design
weight; increasing the experimental guided weight to 0.35 reached only 81.16%
and was reverted rather than shipped.

The bottom-interface candidate set is not the main loss on this road. A
development candidate audit found a packet within one seven-sample pulse of the
disclosed base timing at 96.8% of checkpoints, while the graph selected such a
packet at only 45.2%. The retained reference-like candidates were typically
low-ranked (median rank 14) because the much stronger nearly horizontal packet
around sample 256 dominates the available radar features. No radar-only feature
combination tested on Daska separated those packets well enough out of sample.
This road therefore remains a review-heavy ambiguity case, not evidence that a
more aggressive prior is reliable.

Authoritative released-reference result:
`exports/blinded-multiroad-strict-scale-20260829/daska-pasrur-forward-development-summary.json`.
The five-seed and candidate-ranking experiments remain development-only under
`exports/base-semantic-development-20260829`.

### Second-acquisition blind base/subbase cohort

Two previously unopened road/reference pairs were frozen in a separate seed
manifest before their workbook values were inspected:

- Burewala–Vehari 002: asphalt 184/182/184 and base 247/247/248 at the fixed
  15/50/85% stations.
- Gujrat second portion: asphalt 181/178/169, base 258/242/241, and subbase
  307/291/302 at the same fixed fractions.

The acquisitions reset radar chainage near zero while their workbooks retain
cumulative project chainage. Evaluation-only offsets of 975 m and 1040 m were
recorded after path/dropout freezing in a separately hashed alignment file; the
offsets cannot influence tracking.

Neither road qualifies for thickness accuracy:

| Road/layer | Seed identity | Scale result |
|---|---|---|
| Burewala 002 asphalt | fail: middle click independently selects a different packet by 40 samples | fail: 18.9% ratio deviation |
| Burewala 002 base | pass: all three within one sample | fail: 18.9% ratio deviation |
| Gujrat second asphalt | pass: all three within one sample | fail: 30.7% ratio deviation |
| Gujrat second base | fail: first click differs by 17 samples | fail: 22.5% ratio deviation |
| Gujrat second subbase | pass: all three within five samples | fail: 29.3% ratio deviation |

The base contradictions are not resolved by repeatedly accepting the same
trace. Production now requests both reconfirmation and, when the five-station
capacity permits, a nearby layer-specific companion observation to determine
whether the click begins a local event/velocity regime. The contradicted layer
and all dependent deeper thicknesses remain withheld until rerun.

### Active manual-observation placement

Candidate retention on Daska shows that another scoring weight is not a safe
substitute for human event identity: a reference-like base packet is retained
within one pulse at 96.8% of disclosed checkpoints, yet its typical radar rank
is 14 and a stronger persistent horizontal packet is also physically
plausible. The operational workflow now treats observation placement as part
of the tracker rather than silently resolving this ambiguity:

- asphalt requires at least two distributed manual observations;
- base and subbase require at least three, so leave-one-station-out identity
  validation still has two independent observations after withholding one;
- design thickness does not reduce those counts or replace reflector identity;
- follow-up locations maximize uncovered road span first and use radar quality
  only as a tie-breaker, without design or workbook values.
- until those observations are complete, automatic interfaces remain visible
  only as review candidates for manual picking; TWTT, thickness, uncertainty,
  and cumulative profile values for the affected layer and its dependants are
  withheld from results and exports.
- the zero-seed preview solves each interface independently for candidate/A-scan
  display and is explicitly marked provisional; only the seeded rerun invokes
  the full joint ordered tracker. On the three-layer synthetic regression this
  reduced preview validation to 2.67 seconds, while the preview-to-seeded-joint
  handoff test passed in 67.75 seconds with dropout disabled only for that
  unrelated test.
- independent leave-one-station-out fits now use bounded adaptive parallelism
  (at most three workers on the current machine) and apply results in the
  original station order. A serial-versus-three-worker synthetic audit produced
  identical dropout records, selected samples, statuses, and stability values.
- a disclosed Burewala–Vehari 002 real-road benchmark then measured the coarse
  two-interface production fit plus three dropout fits at 156.38 seconds with
  one worker and 138.88 seconds with three workers (1.13× overall speedup).
  Complete result snapshots are byte-equivalent after canonical JSON
  normalization and have the same SHA-256. Automatic fine-region retracking was
  explicitly excluded so this measures bounded dropout parallelism rather than
  the separate fine-retracking cost. No reference file was opened.
- the joint history beam now uses an exact stable top-k partition instead of
  fully sorting every finite transition score at every radar bin. It retains
  the same 128-state budget, predecessor histories, destination diversity, and
  stable tie order. Repeating the Burewala benchmark produced the same complete
  snapshot SHA-256 as the pre-optimization run, while its single-run timings
  were 129.29 seconds serial and 120.80 seconds parallel. These observed times
  are useful development evidence, not a hardware-independent speed guarantee.
- on the production 128-history three-interface Gujrat state space, one
  radar-only joint fit completed in 299.88 seconds and exactly reproduced all
  frozen accepted, alternate, graph, and canonical arrays for all three
  interfaces (identical combined path SHA-256). This verifies path fidelity;
  it does not establish physical base/subbase accuracy or make the remaining
  five-minute seeded fit interactive.
- automatic fine retracking no longer tiles every broad ambiguous span by
  default. The interactive policy selects at most one 10 m high-information
  core with 5 m context, records the plan, and leaves all other uncertainty in
  the review queue. Exhaustive tiling remains an explicit `None` research
  setting. Every dropout refit replays the main run's exact window rather than
  independently choosing a different location. On the disclosed 287 m Gujrat
  three-layer case, preview plus the bounded seeded run took 10.78 + 513.98
  seconds, selected one 10 m core/203-bin fine segment, retained three review
  regions, and still requested another subbase observation. No reference file
  was opened and this is runtime/workflow evidence, not physical accuracy.
- review issue identifiers are deterministic across identical reruns. Random
  UUIDs were the only mismatch in the first real-road parity attempt and made
  saved issue/request provenance unstable even though numerical results agreed.
- provisional automatic samples remain visible for A-scan review, but their
  TWTT is now withheld alongside physical thickness and profile values until
  the required manual reflector-identity observations are complete.

A radar-only development audit on the two required roads produced:

- Talagang, with base observations at 0.2 and 1689.0 m: proposed third station
  866.6 m, 822.4 m from the nearest existing base observation.
- Pattoki, retaining base observations at 325.7 and 989.4 m: proposed follow-up
  626.2 m, 300.5 m from the nearest retained observation and 102.5 m from the
  withheld 728.6 m development station.

This validates useful station placement, not physical layer accuracy. The
audit opens no reference workbook and is reproducible with
`scripts/evaluate_active_seed_placement.py`; its output is
`exports/active-seed-placement-development-20260829/summary.json`.

Evidence and immutable inputs:

- `benchmarks/blinded-base-subbase-seeds-20260829.json`
- `benchmarks/blinded-base-subbase-reference-alignment-20260829.json`
- `exports/blinded-base-subbase-strict-scale-20260829/summary.json`

## What still blocks a reliability claim

1. Develop base-boundary semantic selection using only the now-revealed
   development roads, then freeze it before another blind evaluation. The
   second Gujrat portion has now been evaluated and its frozen validation seeds
   must not be modified after seeing the outcome. The local inventory contains
   no third unopened acquisition/reference pair with base or subbase labels:
   remaining second acquisitions share already disclosed workbooks, Rawalpindi
   has asphalt-only reference data, and other acquisitions have no matching
   layer-thickness reference. New independent field/reference data are required
   for the three-road release gate.
2. Keep checkpoint identities completely out of seeds, design corridors,
   parameter tuning, and confidence calibration until the configuration is
   frozen.
3. Require candidate retention, correct graph family, accepted accuracy, and
   review coverage to pass separately. A hidden wrong cycle is safer than a
   false thickness but still fails automation coverage.
4. Add another seed when the requested layer must be extrapolated hundreds of
   metres beyond the nearest confirmed event or when a new construction/phase
   regime appears.
5. Reconfirm every station flagged by the leave-one-out audit before treating
   the seeded model as stable; do not simply add more clicks, because an
   adjacent-cycle click can reduce repeatability.
6. Do not release base or subbase tracking until each independently passes the
   same three-road, 30-checkpoint, 85%-coverage, and 85%-accuracy gates now met
   by asphalt.
7. Continue reducing seeded three-interface joint-search runtime without
   pruning a retained seed-family state. Zero-seed preview latency, unbounded
   automatic fine tiling, and serial dropout are addressed. Reducing the
   predecessor-history budget from 128 to 64 or 96 changed the retained
   subbase result, so both experiments remain rejected and production stays at
   128. The exact top-k and bounded-fine improvements preserve frozen paths,
   but the roughly five-minute coarse three-layer joint fit remains serial;
   runtime is improved, not resolved.

## Reproducible evidence

- `exports/cross-road-readiness-20260829.json`
- `exports/sparse-seed-recovery-20260829.json`
- `exports/daska-repeatability-seeded-20260829/summary.json`
- `exports/daska-repeatability-five-seeds-20260829/summary.json`
- `exports/daska-repeatability-withhold-600m-20260829/summary.json`
- `exports/blinded-multiroad-20260829/summary.json` (superseded physical scale)
- `exports/blinded-multiroad-strict-scale-20260829/summary.json`
- `exports/design-assisted-development-20260829/bahawalpur-local-road-sub-engr-001-summary.json`
- `exports/blinded-multiroad-strict-scale-20260829/daska-pasrur-forward-development-summary.json`
- `exports/blinded-base-subbase-strict-scale-20260829/summary.json`
- `exports/pattoki-holdout-a-diagnostic-v2-20260829.json`
- `exports/measurement-support-20260829/seed-identity-sections-gated-v2/summary.json`
- `exports/seed-dropout-parallelism-development-20260830-v5/summary.json`
- `exports/three-layer-joint-runtime-development-20260830-v3/summary.json`
- `exports/bounded-auto-fine-development-20260830-v3/summary.json`

Generated exports are intentionally unversioned. The scripts, tests, and this
assessment are versioned so the evidence can be recreated from the source
acquisitions.
