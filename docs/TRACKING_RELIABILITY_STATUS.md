# Tracking reliability checkpoints — 27 August 2026

## Latest: practical seed-assisted workflow (v17)

The operating target is a few analyst picks followed by automatic tracing,
not seed-independent autonomy. Leave-one-seed-out validation is optional and
disabled by default. Independent physical accuracy is still unestablished.

Latest replay: asphalt/base only, 0.4 m coarse bins, existing development
seeds, assumed dielectric 7, no workbook input, no automatic fine retracking.
These are accepted-coverage measurements, not accuracy measurements.

| Road | Asphalt automatic | Base automatic | Base review | Base no-pick |
|---|---:|---:|---:|---:|
| Talagang | 3,614/4,715 (76.6%) | 2,955/4,715 (62.7%) | 1,298 (27.5%) | 460 (9.8%) |
| Pattoki-Jhoru | 2,271/2,762 (82.2%) | 1,310/2,762 (47.4%) | 402 (14.6%) | 1,047 (37.9%) |

Manual accepted seeds are separate: three asphalt seeds per road, two base
seeds on Talagang, and three on Pattoki. Rounded fractions therefore need not
sum to 100%. No-pick rows are never automatically accepted.

Changes retained:

- Adaptive upper-reflector subtraction tolerates small timing errors; an
  unstripped deep branch remains available.
- Fixed detection/identity scoring replaces the per-run learned ranker;
  experimental lineage reachability is advisory rather than a hard exclusion.
- A soft, bounded position penalty derived from the bracketing manual seeds
  helps the graph distinguish parallel reflectors. It neither creates nor
  removes candidates and leaves the missing state untouched. It is an explicit
  seed prior, not additional independent radar evidence or a design target.
- A materially stronger competing reflector outside the seed-position envelope
  requires review. At Talagang 672.6 m, sample 249 is a review hypothesis, while
  the stronger candidate near 301 remains available; neither is confirmed as
  the physical base interface.
- New thickness-regime seeds cannot be vetoed by the majority seed gap; their
  local bracketing envelopes can widen without widening unrelated road sections.
- Suggested or arbitrary **model seeds** enable road-scale retraining. Explicit
  **corrections** remain local. Click positions and event metadata use the same
  trace; edits enable rerunning, and unrelated subbase picks are not mandatory.
- Explicit not-visible/absent seed decisions survive processing at their trace.
  Aggregation leaves dependent thickness unresolved across those gaps.

The fixed-radius-to-contiguous-growth experiment (v15) did not improve either
road and was removed. Position guidance without the competing-reflector review
check (v16) produced higher apparent coverage (77.9% Talagang, 54.8% Pattoki),
but falsely implied resolution at the disputed Talagang location; it was not
retained as the final acceptance rule. Relative to the earlier v14 checkpoint,
Pattoki automatic base coverage increases from 36.8% to 47.4%; Talagang drops
from 69.0% to 62.7% as ambiguity is exposed instead of concealed.

Run artifacts, input/source hashes and candidate snapshots:

- `exports/seed-assisted-v17-20260827/talagang/`
- `exports/seed-assisted-v17-20260827/pattoki/`

The subsequent explicit-visibility and seed-entry fixes were checked separately
with focused tests; they do not alter these all-visible-seed solver runs.
Synthetic checks cover varying interface paths, ordering, anomalies, gaps,
imperfect stripping, seed edits and negative decisions. They do not establish
field accuracy. The full default fine-refinement workflow on these roads and
an analyst-confirmed additional-base-seed replay remain outstanding. No new
manual field labels or workbook agreement claims were manufactured.

## Historical checkpoint below (superseded defaults and status)

### Earlier status

Not ready to claim reliable field tracking. No new blinded user checkpoints
were captured in this work, and neither workbook was used to choose or tune
the changes below. Existing development seeds are not independent validation.
Pattoki transfer testing has not been started for this revision.

Computational cost is not an acceptance criterion. Default horizontal spacing
no longer gets coarser on long roads to meet a 5,000-bin budget. Automatic fine
retracking covers the full uncertain spans, rather than three windows. Actual
leave-one-station-out refits are enabled by default when at least two training
stations exist. Runtime and memory remain diagnostics only.
Seed-withholding runs now retain the parent's fine-refinement policy instead
of comparing a fine result with a cheaper coarse refit. Duplicate copies of a
withheld pick in the anchor list are also removed. A full fine replay of this
latest policy is still outstanding; the completed comparisons below are coarse
on both sides and therefore are not affected by that policy repair.

## Verified defects repaired

- The active “joint” selector combined independently completed layer paths.
  A forward/backward candidate-combination beam now searches joint layer
  states, enforces display/canonical ordering and hard seeds, and carries
  slope and latent gap history. Its hypotheses drive joint support.
- Base generation depended on the accepted asphalt path. Candidate bounds
  now include the preceding layer's candidate envelope, and stripping uses
  provisional hypotheses rather than confidence-filtered observations.
- Original and residual branches could contribute identity features from
  different winning prototypes. Phase/polarity now refer to the prototype
  whose correlation and canonical offset were selected.
- Tracklet correlation already ranged from zero to one but was remapped
  again, giving zero similarity a spurious 0.5 floor. That floor is removed.
- Fine retracking projected distant seeds onto local endpoints and sometimes
  used canonical time where a display lobe was required. Seeds are restricted
  to their actual window and nearby accepted boundary conditions carry both
  observations. Fine candidates replace stale coarse review candidates.
- Candidate export re-ranked and re-pruned the solver table. It now preserves
  solver candidates and ranks, including raw graph selection/hypothesis counts.
  Exported radar scores now also come from the actual ranker, not the generic
  display feature at that pixel. Older frozen candidate scores predate this fix.
- Retention auditing confused final visibility with the graph decision, making
  final-gate losses impossible to distinguish. Raw graph samples are retained.
- Radargram export removed missing rows before plotting, connecting lines
  through gaps. Missing/anomalous rows are now NaN breaks; review candidates
  use a separate dotted series from accepted observations.
- A full radar array was allocated inside every packet-member loop because of
  an eagerly evaluated `setdefault` default. Those arrays are allocated once.
- A confirmed click on a packet edge could disappear because ownership was
  tested against the canonical peak instead of packet membership. The full
  Talagang refinement exposed this at window 39/47; membership now preserves
  that seed exactly once. The failed run is retained under
  `exports/talagang-accuracy-first-20260827/failure.json`.
- Benchmark selection can no longer pass without checkpoints through the
  vacuous truth of `all([])`: each target road/layer needs at least 30 records.
- Packets collapsed competing seed-selected lobes before the graph. They now
  retain a representative for each seed prototype, with one shared packet ID.
  The 12-packet limit no longer means 12 lobes. A synthetic positive-centre /
  negative-trough case verifies that both modes remain available at every bin.
- Template selection could borrow phase/polarity identity from the opposite
  seed regime. Its winning signed correlation must now match original-radar
  polarity, including when a stripping residual supplies additional evidence.
- Long tracklets used to become training labels on support alone. Expansion
  is restricted to local (10 m), mutually consistent continuation with branch
  support and a separated alternative; unsupported tracklets remain hypotheses.
- Final signal/design merging was per-layer and could undo joint ordering.
  It now reconciles joint states, preserves hard confirmed samples and requires
  a gap/break to switch distant or opposite-lobe hypotheses. Selection uses
  pre-gate graph observations: hiding an event cannot select a different family.
- The joint beam kept alternatives only when spare destination slots remained.
  Competing histories disappeared after converging at an anchor. Independent
  destination/history budgets now retain distinct prefixes and slope history;
  a converging-ridge regression verifies that uncertainty survives the anchor.
  This remains an approximate beam search, not an exhaustive probability model.

## Earlier frozen development diagnostic

Command (from repository root):

```powershell
uv run python scripts/diagnose_tracking.py --output exports/talagang-joint-dropout-20260827 --drop-one-seed
```

This is an asphalt/base-only, 0.4 m coarse diagnostic, **not** the final
fine-refined workflow. It uses the existing Talagang development seed file,
2 in asphalt, 4 in base, and assumed dielectric 7. Inputs, source hashes,
configuration, raw graph choices and candidates are saved beside each run.
The script does not load a workbook. Its explicit dropout diagnostic uses a
fixed 7-sample display-lobe tolerance; it is not a calibrated accuracy estimate.

| Layer | Automatically accepted before dropout | Unstable after withholding A | Withholding B | Withholding C |
| --- | ---: | ---: | ---: | ---: |
| Asphalt | 1,846 / 4,715 | 240 | 383 | 1,605 |
| Base | 2,917 / 4,715 | 1,372 | 761 | 866 |

Only 33 initially automatic asphalt picks and 750 base picks survive **all**
three fixed-tolerance display comparisons. The production check additionally
checks canonical samples, visibility and polarity, and sends unstable picks
to review without changing their selected samples. Similarity to two seed
templates is no longer treated as a substitute for actual seed-withholding
evidence. These failures must not be hidden by lowering a confidence threshold.

## Event-mode diagnostic (before history-preservation repair)

`exports/talagang-event-modes-20260827` repeats the same coarse, two-layer,
radar-only protocol after retaining lobe alternatives and fixing polarity.

| Layer | Automatically accepted before dropout | Stable in every fixed-tolerance dropout comparison |
| --- | ---: | ---: |
| Asphalt | 3,722 / 4,715 | 1,428 / 4,715 (30.3% of road bins) |
| Base | 2,880 / 4,715 | 702 / 4,715 (14.9% of road bins) |

Asphalt stability improved markedly, but base stability did not. Neither
automatic coverage nor these stability percentages measure tracking accuracy.
The 672.6 m base interpretation remains unresolved. A radar-only decision image
with a labelled A-scan is saved as `base-seed-window.png` in this run directory;
no workbook or predicted path is drawn on it, and no new confirmation is assumed.

The earlier full 47-window retry completed in 584.9 s, including three actual
station-withholding refits. It accepted only 10/4,715 asphalt and 762/4,715 base
bins after reliability checks. Its frozen result is retained in
`exports/talagang-accuracy-first-packetfix-20260827`. It predates the event-mode
and history-preservation repairs above and compared fine baseline picks with
coarse withheld runs. It must not be reported as a clean seed-only sensitivity
experiment or as a result for the current implementation.

## Remaining accuracy work

1. Separate spatial reflector lineage from seed-template compatibility.
   Similar phase/polarity on persistent ringing is not proof of physical family
   continuity. Reject unexplained cycle changes or expose them as alternatives.
2. Replace peak-neighbourhood preprocessing support with actual independent
   packet-track agreement. Multiple subtraction hypotheses are correlated,
   and must not be presented as independent repeated measurements.
3. Finish packet-level pruning/merging retention diagnostics, actual processing
   perturbation runs, measured pulse-width stability tolerances and complete
   ragged lobe sequences (the current dense evidence exports eight members).
4. Obtain genuinely blinded seed/checkpoint confirmations. Talagang's disputed
   672.6 m base pick remains removed; it must not be replaced by a tracker
   prediction labeled “manual.” The opposite-polarity asphalt end regime also
   needs careful confirmation. Workbook disagreement alone cannot adjudicate
   those identities.

The release gate remains failed. More accepted pixels, a smoother path, passing
synthetic tests, or a lower runtime is not evidence of field accuracy.

Verification at this checkpoint: 81 tests pass; Ruff and `git diff --check`
pass. Regression
coverage includes packet-edge ownership, competing lobes, polarity-correct
templates, conservative training expansion, retained joint histories, hard seeds,
joint signal/design reconciliation, gate-independent family selection, bounded
refinement, seed dropout and missing-overlay breaks.

The completed coarse dropout run including history-preservation and
gate-independent selection is `exports/talagang-history-preservation-20260827`.
Its source hashes were fixed before execution:

| Layer | Automatically accepted before dropout | Stable in every fixed-tolerance dropout comparison |
| --- | ---: | ---: |
| Asphalt | 3,697 / 4,715 | 1,378 / 4,715 (29.2% of road bins) |
| Base | 2,792 / 4,715 | 818 / 4,715 (17.3% of road bins) |

Base stability improved modestly, but the reliability gate still fails by a
large margin. All four runs are retained, including their failures of stability.
The final audit-score export and like-for-like fine-dropout policy repairs were
made after this code freeze and do not change this coarse path comparison.

At 672.6 m in this newer run, the negative event at sample 301 remains ranked
first but the selected graph event is sample 211, with a design/family conflict.
This isolates that particular loss to path selection rather than missing
candidates. It does **not** establish which reflector is the physical base;
that requires the pending radar-only interpretation decision.
