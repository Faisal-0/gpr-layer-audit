# Independent tracker assessment — 27 August 2026

## Decision

Retain the ingestion, calibration, candidate-packet, joint-ordering, gap,
review, and export machinery. Add one narrow seed-identity constraint: between
two bracketing manual base seeds, a strong persistent packet that is separated
from both seeds is kept as a review alternative but is inadmissible to the
tracking graph. Do not apply that exclusion outside the bracketing seeds or
across a structural break. A seed placed on the persistent packet protects it.

This is a seed-adherence improvement, not evidence that the retained path is
the physical base-course bottom. The system still needs another user seed when
the seed-selected packet is fragmented or ambiguous.

## Independent findings

### The main failure is event selection, not simply missing candidates

On the bounded Talagang 950–1250 m experiment, the candidate table contains
events near all three deeper seed hypotheses. Without an identity exclusion,
the graph nevertheless returns to the strong packet near sample 249 on 53.6%
of rows for the deeper negative-trough hypothesis. The final visibility gate
hides many of those graph choices, so an apparently conservative displayed gap
can conceal an underlying family switch.

Waveform correlation, phase class, and polarity do not uniquely identify one
physical packet when parallel wavelet cycles have similar character. The
continuous provisional graph also makes no-pick deliberately expensive; that
helps it carry a latent event through a short fade, but it encourages selection
of some available packet when the seeded packet is not separable. A global
change making no-pick competitive was tested and rejected: at a null emission
of -0.35, the deliberately seeded Pattoki path fell from 62.2% retained to
6.2%. That is too fragmented for the intended workflow.

### The original-signal check is not yet a production gate

The production candidate amplitude, polarity, and evidence gates use the
processed interpretation radargram. The pre-enhancement radar is stored for
display/export but does not currently veto a processed-only pick. Also, the
stored `interpretation_input_radargram` is already plate-subtracted even though
both development roads and their plate scans have different range-gain
signatures.

For this assessment, retained paths were overlaid separately on a
surface-flattened, dewow-only road branch before plate subtraction and on the
processed branch. Lateral packets are visible in the road signal at many
retained sections, but 47.3% of retained Talagang samples and 48.0% of retained
Pattoki samples are below the median within-trace packet rank in their plotted
base windows. That rank does not prove absence—the laterally coherent boundary
is often visible—but it demonstrates that processed visibility is not an
independent confirmation.

### Preprocessing branches are correlated evidence

Background removal, denoising, phase, envelope, gradient, deconvolution,
coherence, and stripping are transformations of the same acquisition. Their
agreement is useful robustness evidence but is not repeated independent
measurement. Base candidates also depend on uncertain asphalt subtraction,
although the current tracker correctly retains an unstripped branch and several
upper-path stripping hypotheses.

### Workbooks cannot adjudicate reflector identity

Both solver variants finish before a workbook is loaded by the evidence script.
Workbook marks are plotted afterward. On retained manual workbook stations,
the median absolute canonical-depth disagreement is 16.6 mm for the Talagang
deeper hypothesis (16 stations) and 13.4 mm for the Pattoki path (30 stations),
assuming relative permittivity 7. Agreement or disagreement is secondary
comparison evidence only; the workbooks contain interpolation and uncertain
picking conventions.

## Focused implementation

`processing/seed_identity.py` detects a persistent, strong packet within the
ordinary radar search bounds. It excludes that packet only when all of the
following are true:

- the interval is bounded by two manual seeds for the same layer;
- both seeds are separated from the packet by more than one packet clearance;
- the packet is supported on at least 80% of the interval;
- its median relative peak strength is at least 0.35; and
- no structural break separates the seeds.

Candidates are not deleted from review/export evidence. The exclusion is
applied before forward, backward, and joint graph search, and gap interpolation
cannot reinsert an excluded packet. Four regressions cover bracketing, a
user-seeded horizontal packet, a single seed, and a structural break.

The initial experimental implementation projected a hard exclusion before the
first seed and after the last seed. That extrapolation was removed: one seed is
not authority for a road-length hard decision.

## Bounded field-data results

These figures measure seed adherence and explicit gaps, not physical accuracy.

| Section and interpretation | Variant | Retained | Explicit gaps | Graph near persistent packet | Changed rows |
|---|---|---:|---:|---:|---:|
| Talagang 950–1250 m, deeper negative-trough candidate seeds | Before | 17.7% | 82.3% | 53.6% | — |
| Talagang 950–1250 m, deeper negative-trough candidate seeds | After | 30.1% | 69.9% | 0.0% | 103 |
| Pattoki 300–750 m, supplied positive-lobe base seeds | Before | 62.2% | 37.8% | 42.6% | — |
| Pattoki 300–750 m, supplied positive-lobe base seeds | After | 62.2% | 37.8% | 42.6% | 0 |

All seeds are preserved exactly. The Talagang seed set in this experiment was
assistant-selected to reproduce the behavior and is not a confirmed interface.
The increased retained fraction is not itself a success criterion; the useful
result is that the graph no longer silently substitutes the sample-249 packet.
Remaining unsupported areas stay as gaps. The Pattoki result demonstrates that
the same mechanism does not blacklist a persistent reflector when the user
intentionally seeds it.

![Before/after seed-identity overlays](../exports/independent-takeover-20260827/seed-identity-sections/before-after-overlays.png)

![Retained paths on road measurement and processed branches](../exports/independent-takeover-20260827/seed-identity-sections/raw-processed-evidence.png)

## Full-road regression check

Coarse 0.4 m runs used the supplied development seeds, asphalt/base only,
assumed relative permittivity 7, and no automatic fine retracking.

| Road | Layer | Automatic | Review | Unresolved | Accepted seeds |
|---|---|---:|---:|---:|---:|
| Talagang | Asphalt | 3,614 / 4,715 (76.6%) | 1,036 | 62 | 3 |
| Talagang | Base | 2,955 / 4,715 (62.7%) | 1,298 | 460 | 2 |
| Pattoki-Jhoru | Asphalt | 2,271 / 2,762 (82.2%) | 430 | 58 | 3 |
| Pattoki-Jhoru | Base | 1,310 / 2,762 (47.4%) | 404 | 1,045 | 3 |

Talagang asphalt/base display paths, raw graph paths, and automatic decisions
are identical to the frozen v17 snapshot. Pattoki raw graph paths and automatic
decisions are also identical; two base rows at 863.8–864.2 m that were hidden
in v17 now pass an existing visibility gate. This is a regression observation,
not an accuracy gain.

## What is validated

- The failure is reproducible on a bounded Talagang section.
- A bracketing seed-identity constraint prevents substitution of the identified
  persistent competitor in that section.
- The same constraint leaves the supplied Pattoki seeded persistent path
  unchanged in a representative section.
- Single seeds and structural breaks do not create extrapolated hard exclusions.
- Candidate alternatives, workbooks, and raw/processed radar views remain
  distinguishable.
- All 100 collected tests pass in grouped runs; the 29 focused identity,
  reflection-event, and joint-graph tests also pass together. Changed files pass
  Ruff and `git diff --check`.

## What is not validated

- No retained path has been independently confirmed as asphalt bottom, base
  bottom, or subbase by cores, test pits, as-built records, or a blinded field
  checkpoint set.
- The Talagang deeper seed hypotheses are not user-confirmed ground truth.
- Coverage, workbook agreement, smoothness, and synthetic tests do not establish
  physical accuracy.
- A production raw-road-signal evidence gate has not been implemented.
- The full automatic fine-retracking workflow was not rerun on both roads for
  this change.
- Subbase remains unsupported and should stay disabled.

## Recommended next work

1. Preserve a surface-flattened, dewow-only, pre-plate-subtraction road branch
   in the analysis result. Add packet-scale lateral support from that branch as
   a conservative acceptance gate, never as a selector that can pull a path to
   another event.
2. Export the seed-mismatched persistent packet explicitly as a competing
   review candidate and propose a new seed near the longest exclusion/gap
   boundary.
3. Collect blinded analyst checkpoints in the Talagang 950–1250 m window and
   at Pattoki weak/gap regions without showing tracker or workbook overlays.
4. Only after asphalt/base checkpoint retention and identity are acceptable,
   evaluate subbase on sections with visibly separate raw-signal evidence.
