# Combined recovery plan: reliable GPR pavement-interface tracking

**Status:** consolidated planning artifact only. No tracker code, data, seeds, or outputs are changed by this document.  
**Supersedes for future work:** `PROPOSED_TRACKER_RECOVERY_PLAN.md` and the pasted “Fix Reliable Base-Course Tracking” proposal.  
**Priority order:** physical event identity, calibrated uncertainty, conservative coverage, then runtime/UI/packaging.  
**Development data:** Talagang (2 in asphalt, 4 in base, assumed εr = 7) and Pattoki-Jhoru (2 in asphalt, 6 in base).

## Combined decision

Use a **repair-and-replace** strategy:

- **Preserve** the reliable acquisition readers, geometry, surface alignment, candidate packets and alternative lobes, diagnostic exports, manual seeds, joint ordering constraints, explicit gaps, and reproducible manifests.
- **Repair and experimentally validate** the upper-layer subtraction/stripping pipeline, including plate use, asphalt-path uncertainty, and stripping-quality measurement.
- **Simplify** candidate evidence into an auditable separation between reflector detection and interface identity. Remove the in-run logistic ranker as an authority for deep layers.
- **Replace** hard lineage reachability, the monolithic trace-by-trace selection objective, and heuristic confidence with local reflector segments, multiple hypotheses, a missing-state-aware segment graph, and calibrated selective acceptance.

This is not a full rewrite. It changes the scientific decision core while keeping the parts of the prototype that already provide useful infrastructure.

## What is adopted from each proposal

| Idea | Combined decision | Reason |
|---|---|---|
| Adaptive asphalt subtraction | Keep as a controlled candidate method | Upper-layer leakage is a credible base failure source, but is not yet proven to be the single bottleneck |
| Separate “reflector exists” from “this is the seeded interface” | Keep | This makes scores interpretable and exposes wrong-reflector failures |
| Remove the seed-reachable `-inf` filter | Keep | The reproduced lineage run proves that the hard rule fragments real data |
| Keep lineage as soft evidence | Keep, without a preselected `-0.3` penalty | Its weight must be learned/calibrated on held-out evidence, not guessed |
| Include an unstripped branch | Keep | It protects base evidence when subtraction is unnecessary or damaging |
| Track which asphalt/stripping hypothesis produced a base candidate | Keep | Necessary for auditing upper-to-lower-layer coupling |
| Search all asphalt candidates and take the maximum base score | Replace with weighted top-k marginalization | A maximum over many branches creates a multiple-testing bias and favors artifacts |
| Three-feature multiplicative score with fixed thresholds | Keep the two-stage concept, not the fixed formula | Correlated features and arbitrary zero/threshold behavior would recreate brittle gating |
| Coverage ≥60% and stability ≥40% as success gates | Demote to later operational objectives | Coverage targets before precision calibration reward manufactured continuity |
| Segment-first multi-hypothesis inference with explicit missing states | Keep | It matches the local nature of reflector evidence and preserves honest gaps |
| Independent checkpoints and frozen transfer testing | Keep as mandatory | Neither workbook agreement nor synthetic performance establishes field accuracy |

## Evidence-backed diagnosis

### Confirmed failure 1: hard lineage causes the current collapse

The current Talagang baseline and all three leave-one-seed-out runs were independently reproduced:

| Result | Reproduced value |
|---|---:|
| Asphalt automatic bins | 646 / 4,715 (13.7%) |
| Base automatic bins | 83 / 4,715 (1.76%) |
| Base graph coverage | 3.12% |
| Base automatic bins destabilized by dropping seed A | 77 / 83 |
| Base automatic bins destabilized by dropping seed C | 8 / 83 |

Candidates that are not in a seed-connected union-find component receive `-inf` emission. The rule converts a broken local phase-motion link into a road-scale assertion that the candidate cannot be the layer. This experiment should be removed from the production baseline and retained only as diagnostic evidence.

### Confirmed failure 2: base event selection remains unreliable without hard lineage

The older broadly usable run automatically accepted 2,792/4,715 base bins (59.2%), but only 818/4,715 (17.3%) were stable in every seed-withholding comparison. At about 672.6 m, the base candidate near sample 301 survived generation and ranked first, while the graph selected a path near sample 211. The physical reflector at that location is not independently confirmed.

This proves that candidate loss is not the only problem. The objective can choose a different coherent wavelet event from a retained, locally strong alternative.

### Credible but unproven failure 3: upper-layer subtraction can alter base evidence

The base candidates are drawn from original and asphalt-stripped branches. The current subtractor already uses a local median template from approximately ±25 traces and fits real, quadrature, and DC components. Therefore, merely adding a sliding local template would duplicate existing behavior.

The following weaknesses still require controlled testing:

- the selected asphalt path can be displaced by a lobe or a few samples;
- one local waveform basis may not represent dispersive/regime-dependent pulse changes;
- the fit is not explicitly regularized against removing overlapping deeper energy;
- the top stripping paths are correlated outputs of the same graph;
- base selection is coupled to a provisional upper-layer decision;
- plate and road acquisitions have different range-gain signatures, yet a full post-surface plate subtraction is still applied.

There is also a concrete instrumentation defect to resolve before judging alternative subtractors: stripping improvement is stored only in a narrow fitted sample window, but stripping hypotheses are ranked using a median over the selected full traces. Because most entries are zero, the median will normally be zero, so the ranking does not distinguish hypotheses meaningfully.

Stripping is therefore a high-priority hypothesis, not an established single root cause.

### Confirmed limitation 4: present scoring and confidence are not independent evidence

The fixed feature score, locally propagated positives, fitted logistic ranker, design contribution, transition costs, visibility rules, and post-hoc confidence all act on correlated transforms of the same radar measurement. Talagang has only two base seeds about 1.69 km apart. This is insufficient support for treating an in-run learned ranker as calibrated interface identity.

Agreement among filtered, phase, gradient, envelope, deconvolution, and stripped products is useful perturbation evidence, but those products are not independent measurements. They must not be counted as independent votes.

### Validation boundary

The full test suite passes (81 tests). This confirms implementation integrity against the existing tests; it does not validate field accuracy. No distributed set of independently confirmed field checkpoints is supplied.

| Evidence | Valid conclusion | Invalid conclusion |
|---|---|---|
| Synthetic tests | Known-event mechanism and regression behavior | Field accuracy |
| Supplied workbooks | Disagreement targeting and secondary comparison | Ground truth or physical accuracy |
| Raw radar and reproducible runs | Candidate visibility, stability, and algorithm behavior | Which ambiguous event is the physical base without confirmation |
| Blinded independent radar interpretations | Agreement/repeatability of radar interpretation | Physical layer depth unless independently measured |
| Cores, test pits, tied as-builts, or validated independent measurement | Physical event identity and depth error | Road-wide performance unless spatially representative |

Several workbook sheets/files describe interpolated data even though hard-coded cells are classified as manual by the current importer. Workbook origin labels must be audited rather than treated as trust labels.

## Target scientific pipeline

```text
raw road data
   │
   ├── conservative preprocessing ensemble
   │      ├── road-only reference
   │      ├── plate/no-plate ablation
   │      └── limited filter/stack/surface perturbations
   │
   ├── asphalt hypothesis set
   │      ├── unstripped evidence
   │      └── regularized adaptive subtraction branches
   │
   ├── local candidate evidence
   │      ├── detection: a coherent reflector exists
   │      └── identity: compatibility with the interface family
   │
   ├── short reflector segments with alternative lobes/events
   │
   └── joint segment inference with explicit missing states
          ├── accepted where one event is well separated
          ├── review where alternatives remain plausible
          └── unresolved gap where evidence is insufficient
```

## Phased work plan

### Phase 0 — Freeze inputs and establish evaluation truth

1. Hash and freeze the Talagang and Pattoki road/plate files, seeds, options, and existing diagnostic outputs.
2. Deduplicate byte-identical survey and workbook copies in the evaluation index without deleting source files.
3. Create a provenance-complete checkpoint schema: road, chainage, layer, sample/depth, visibility, interpreter, evidence class, uncertainty, and immutable train/validation/test role.
4. Collect at least 30 blinded checkpoints per layer per road before viewing results from the new tracker. Stratify ordinary, weak, discontinuous, anomalous, and regime-change sections.
5. Store physical measurements separately from radar interpretations. If physical confirmation is unavailable, label metrics “radar-interpretation agreement,” never “field accuracy.”
6. Develop on Talagang. Keep Pattoki frozen as the first transfer test; use a small, separately declared Pattoki development subset only if a polarity/acquisition compatibility defect prevents the frozen run.

**Gate 0:** automatic-acceptance thresholds cannot be called calibrated until the checkpoint set exists. Prototyping may report candidate recall, signal stability, and diagnostics only.

### Phase 1 — Build the stripping and acquisition ablation harness

Implement diagnostics before changing the subtraction algorithm:

1. Correct stripping-quality measurement so it is evaluated within declared windows and by purpose:
   - asphalt-fit error inside the fitted upper-layer window;
   - base-window preservation outside the fit support;
   - sideband/noise behavior;
   - held-trace or neighboring-trace generalization.
2. Record the exact asphalt hypothesis, shift, template regime, coefficients, removed energy, and residual branch for every base candidate.
3. Produce before/after radargrams, A-scan overlays, phase/envelope plots, and complete candidate tables at seeds, 672.6 m, and stratified strong/weak/anomalous windows.
4. Compare:
   - no stripping;
   - current stripping;
   - current stripping with deliberately perturbed asphalt paths;
   - regularized adaptive/elastic subtraction;
   - road-only, current plate subtraction, and narrowly normalized plate-wavelet use;
   - surface and filter/stack perturbations.

The adaptive method may use local template dictionaries, sub-sample shifts, real/quadrature bases, and ridge/Tikhonov regularization. It must fit using the asphalt window only. **Do not select a shift by minimizing energy in the expected base window**; that can reward erasing the reflector the tracker is meant to recover. Any energy-removal cap must be validated rather than fixed at 95% by assumption.

**Gate 1:** retain a subtraction method only if it improves visible-base candidate separation without reducing candidate recall, changing the event family under modest perturbations, or degrading no-strip cases. Disable plate subtraction by default if plate/no-plate changes base identity or stability because of the gain mismatch.

### Phase 2 — Create an auditable detection/identity evidence model

Replace the 17-feature/in-run-ranker authority with two named evidence groups:

**Detection evidence — “a reflector is present here”**

- locally normalized amplitude/envelope or generative impulse strength;
- oriented 2-D coherence and persistence;
- phase-consistent ridge support;
- stability under declared preprocessing perturbations;
- waveform-model residual improvement on held samples/traces.

**Identity evidence — “this reflector is compatible with the requested interface”**

- signed seed-waveform compatibility by local regime;
- lobe/phase-family consistency;
- compatibility with adjacent reflector segments;
- hard agreement at manual seed locations;
- separation from competing interface-family hypotheses.

Remove the in-run logistic ranker for base. If a learned calibration model is later used, train it only on a sufficient development checkpoint set, freeze it, and test it out of sample.

Do not multiply two arbitrary 0–1 scores and threshold the product. Combine interpretable likelihood/log-evidence factors with recorded provenance, or calibrate the two-dimensional detection/identity surface on held-out data. A strong detection with uncertain identity is a review candidate, not a hidden failure.

**Gate 2:** on radar-visible development checkpoints, the correct candidate family must remain in the retained top-k set at least 98% of the time. Report top-1 separately. Correlated preprocessing products must be grouped as one evidence source in confidence accounting.

### Phase 3 — Replace hard lineage with local reflector segments

1. Build 5–20 m multi-scale segments using phase motion, waveform similarity, oriented coherence, curvature, and perturbation stability.
2. Retain parallel segments and their packet/lobe relationship.
3. Permit segment birth, death, and explicit gaps at discontinuities.
4. Attach nearby seeds to compatible segments; never invalidate every distant unconnected segment.
5. Store seed reachability and reciprocal phase motion as diagnostic features.
6. Use soft lineage evidence only after calibration. Until then, disconnected candidates are review-prone but admissible.

**Gate 3:** the correct visible event must be retained in the segment hypotheses at least 98% of development checkpoints. Synthetic gaps and breaks must terminate segments instead of producing forced bridges.

### Phase 4 — Marginalize asphalt uncertainty when evaluating base

For each local window:

1. retain a bounded top-k set of genuinely different asphalt event families, not merely adjacent nearly identical graph paths;
2. include the unstripped branch;
3. generate base evidence under each upper-layer hypothesis;
4. retain branch provenance and branch probability/evidence;
5. combine base evidence by posterior-weighted marginalization or joint local model evidence, not a maximum across every candidate;
6. flag windows where the best base explanation requires an asphalt family inconsistent with the selected asphalt profile.

Where feasible, fit a local sparse/generative waveform model with surface, asphalt, base, and missing-interface alternatives. A deeper interface is supported only if it materially improves held-trace/local fit and survives conservative perturbations.

Design thickness may bound a broad physically plausible search. A design-blind result must always be retained. Design cannot create a candidate, fill a gap, or convert review/unresolved to accepted.

**Gate 4:** base candidate identity must no longer change materially when an asphalt-only seed is withheld outside the local influence region. Windows with material cross-layer dependence are automatically review/unresolved.

### Phase 5 — Replace monolithic path selection with conservative segment inference

Use a semi-Markov/factor graph or equivalent k-best dynamic program over segment hypotheses plus an explicit missing state:

- hard constraints: manual seeds and interface ordering;
- soft, layer-specific constraints: curvature, gap duration, waveform regime, and design support;
- independent regime-break state per layer;
- multiple global hypotheses retained when evidence is close;
- no penalty structure that makes a fabricated bridge cheaper than an honest gap;
- forward/reverse agreement assessed before acceptance.

The output profile contains only the portions common to sufficiently supported hypotheses. Ambiguous alternatives form a review region; absent evidence forms a gap.

**Gate 5:** no ordering violations, no forced path through synthetic or radar-visible gaps, and no accepted cycle/lobe switch under forward/reverse or reasonable preprocessing perturbations.

### Phase 6 — Calibrate uncertainty and review requests

Calibrate acceptance using held-out checkpoints and failure-specific features:

- probability/evidence separation between reflector families;
- detection and identity calibration;
- forward/reverse stability;
- leave-one-seed-out stability outside a declared local exclusion radius;
- preprocessing and strip-branch stability;
- segment support and distance from the last supported segment;
- upper-layer uncertainty inherited by base;
- local waveform residual and anomaly/change-point proximity.

Output exactly three decision states:

- `accepted`: one event family meets calibrated selective-risk requirements;
- `review`: radar evidence exists but competing identities remain plausible;
- `unresolved`: insufficient evidence; no depth is emitted.

Thickness is emitted only where both bounding interfaces are resolved. Interpolation across an unresolved interface is prohibited.

**Gate 6:** automatically accepted picks must achieve at least 95% precision against frozen blinded radar checkpoints for each layer, with Wilson confidence intervals. This is radar-interpretation agreement unless checkpoints are physically confirmed. Coverage is reported but is not allowed to lower the precision gate.

### Phase 7 — Expand realistic synthetic tests

Keep all existing tests and add known-truth cases for:

- base overlapping asphalt ringing;
- asphalt-path offsets of ±1 and ±2 samples;
- gradual and abrupt wavelet-shape/phase changes;
- dispersive attenuation and time-varying range gain;
- plate/road gain mismatch;
- parallel lobes and cycle ambiguity;
- surface-pick jitter;
- local disappearance/reappearance of an interface;
- misleading design thickness;
- polarity regimes and structural breaks;
- a strong unseeded reflector from the wrong interface family.

Synthetic tests validate mechanisms and known-truth failure handling. They remain separate from field metrics.

### Phase 8 — Frozen transfer to Pattoki and operational assessment

1. Freeze all parameters after Talagang development.
2. Run Pattoki once as the primary transfer test.
3. Report candidate recall, top-1/top-k identity, accepted precision, review fraction, unresolved fraction, and failure classes separately for each road and layer.
4. Compare workbooks only after predictions are frozen and only as disagreement evidence.
5. Report physical depth/thickness error only where independently confirmed measurements exist.

**Gate 8:** promote the base tracker only if the precision and stability gates pass on both roads. If not, ship only the portions that pass as a conservative interpretation assistant.

After the precision gate passes, pursue an **operational coverage objective** of at least 60% automatic base coverage on radar-visible, non-anomalous sections. This is an optimization objective, not permission to fill gaps. Its denominator and visibility definition must be frozen before measurement.

### Phase 9 — Subbase feasibility

Begin only after base promotion. Repeat candidate-recall, event-identity, uncertainty, and transfer gates for subbase. Never infer subbase from design thickness or from whichever residual remains after base subtraction. If no validated, perturbation-stable event family exists, produce no subbase profile.

## Diagnostic deliverables

Every experiment must produce a versioned report containing:

- input hashes, code revision, settings, seeds, and checkpoint split;
- acquisition/gain compatibility and plate-use decision;
- asphalt-fit, base-preservation, and sideband stripping metrics;
- base candidates grouped by unstripped/stripped branch and asphalt provenance;
- detection and identity distributions;
- candidate recall before segment/global selection;
- top-1 and top-k event identity;
- accepted/review/unresolved fractions;
- forward/reverse, perturbation, and leave-one-seed-out stability;
- failure classes: candidate missing, wrong lobe, wrong reflector family, stripping artifact, corridor/design bias, seed overreach, surface error, discontinuity, and no evidence;
- workbook disagreement in a separate appendix;
- synthetic performance in a separate appendix;
- a clear statement of whether physical field accuracy has been established.

The 672.6 m region remains a required ambiguity case: all plausible candidates must remain visible with their provenance and relative evidence. It must be `review` unless independent confirmation and calibrated evidence separate the physical interface.

## Implementation order after approval

1. Correct diagnostic measurement and build the stripping/plate/no-strip ablation harness.
2. Remove hard seed-reachability from emissions while retaining lineage metadata.
3. Add candidate and stripping provenance plus candidate-recall reports.
4. Prototype and compare regularized adaptive subtraction against current/no-strip branches.
5. Introduce auditable detection/identity evidence and retire the base in-run ranker.
6. Prototype local competing reflector segments on fixed Talagang windows.
7. Add upper-layer marginalization and the missing-state segment graph.
8. Calibrate review/acceptance on frozen checkpoints.
9. Freeze and run Pattoki transfer validation.

Each step must be evaluated independently before the next is accepted. No UI polish, packaging, runtime optimization, or subbase work begins before the base precision/stability gates pass.

## Stop conditions

Stop tuning and revisit the signal/measurement model if:

- a confirmed visible base event is absent from retained candidates in more than 2% of checkpoints;
- modest plate, filter, surface, or strip-path perturbations repeatedly switch event family;
- accepted precision cannot reach 95% without collapsing to a trivial set;
- base identity changes nonlocally when an asphalt-only seed is removed;
- the tracker meets coverage mainly by following design thickness;
- two independent interpreters cannot agree and there is no physical confirmation;
- increasing graph complexity improves coverage but not candidate identity or held-out precision.

In ambiguous or information-poor sections, the correct engineering output is a review interval or explicit gap.
