# Proposed recovery plan: reliable pavement-interface tracking

**Status:** planning artifact only; no tracker implementation is changed by this document.  
**Priority:** establish event identity and calibrated uncertainty before coverage, runtime, UI, or packaging.  
**Development roads:** Talagang (2 in asphalt, 4 in base, assumed εr = 7) and Pattoki-Jhoru (2 in asphalt, 6 in base).

## Decision

Do not continue tuning the current end-to-end graph. Preserve the proven ingestion, alignment, candidate-packet, export, gap, and diagnostic machinery, but **replace the event-selection and confidence core with a conservative, segment-first, multi-hypothesis inference layer**.

The experimental local-lineage rule should be removed from the production baseline. It is useful as a diagnostic feature, but its present hard `seed_reachable` requirement turns a local continuity failure into global infeasibility. Reverting that rule alone would recover coverage, not reliable base identity; the older graph still changed base paths under seed withdrawal and sometimes chose a lower-ranked parallel lobe.

The proposed replacement will:

1. extract short, locally supported reflector segments without requiring a seed-connected component;
2. retain several plausible event/lobe hypotheses per segment;
3. score them with local generative waveform evidence and genuine processing perturbations;
4. link only well-supported segments with a missing-state-aware semi-Markov/factor graph;
5. marginalize over plausible asphalt paths when evaluating base rather than stripping one assumed asphalt path;
6. emit review regions or explicit gaps whenever event identity is not sufficiently separated.

Subbase is deferred until the asphalt and base gates in this plan pass.

## Independent findings

### Reproduced behavior

The current Talagang diagnostic and all three leave-one-seed-out runs were reproduced from source. The full repository test suite also passes: **81 tests passed**. This demonstrates reproducibility and code integrity, not field accuracy.

| Observation | Reproduced or inspected result | Meaning |
|---|---:|---|
| Current lineage run, asphalt automatic | 646 / 4,715 bins (13.7%) | Severe fragmentation |
| Current lineage run, base automatic | 83 / 4,715 bins (1.76%) | Not a usable automatic tracker |
| Current lineage run, base graph coverage | 3.12% | Most generated candidates are inadmissible to the graph |
| Drop Talagang seed A, base instability | 77 / 83 automatic bins | The small accepted set is highly seed-dependent |
| Drop Talagang seed C, base instability | 8 / 83 automatic bins | Instability also occurs at the opposite seeded end |
| Earlier coarse run, base automatic | 2,792 / 4,715 bins (59.2%) | Coverage existed before hard lineage gating |
| Earlier coarse run, base stable under every seed withdrawal | 818 / 4,715 bins (17.3%) | Most apparent coverage was not robust |

At approximately 672.6 m, the earlier run retained the deep base candidate near sample 301 and ranked it first (score about 0.660), but the joint graph selected the shallower path near sample 211. In the current lineage run, the candidate near 301 and many alternatives still exist, but they are not seed-reachable and therefore cannot be selected. This isolates two different problems:

- **Immediate regression:** a hard lineage feasibility rule destroys coverage.
- **Underlying problem:** among retained parallel events, the global objective does not reliably identify the physical base reflector.

The second problem is the main scientific weakness. Candidate generation is no longer the dominant bottleneck at the disputed location.

### Why the base is vulnerable

The base decision is assembled from several correlated, heuristic signals rather than an independently identifiable physical event:

- The preprocessing branches are transformations of the same measurement; agreement among them is not independent corroboration.
- Base candidates depend on surface/asphalt subtraction and on the provisional upper-layer hypothesis. A wrong asphalt event or waveform fit can create, suppress, or re-rank base alternatives.
- Seed-trained waveform/ranker behavior is extrapolated over long distances. Talagang has only two base seeds, about 1.69 km apart, and no confirmed mid-road base seed.
- Seed regime breaks from different layers are currently pooled into a shared set, allowing an asphalt regime change to relax base transitions.
- Design corridors are adjusted from seed residuals and candidate envelopes. They can influence selection before radar evidence has established event identity.
- All inspected road acquisitions and all inspected plate scans use different range-gain signatures. Plate subtraction is still applied even though amplitude-based dielectric calibration is rejected. Because range gain changes waveform shape as well as amplitude, this is a credible source of deep-event distortion, but it has not yet been isolated by ablation.
- A monolithic path objective prefers a coherent answer even when two parallel wavelet cycles are locally indistinguishable. The confidence calculation does not yet measure the posterior separation between those physical alternatives.

### Evidence boundaries

The following categories must never be merged in reports or metrics:

| Evidence class | What it can establish | Current status |
|---|---|---|
| Independent physical confirmation: cores, test pits, construction/as-built records tied to chainage, or another validated measurement | Physical interface identity and field accuracy | Not supplied |
| Blinded, independently interpreted radar checkpoints with adjudication | Agreement with expert radar interpretation and event-repeatability | Not yet collected |
| Direct radar observations and controlled software replays | Candidate presence, signal visibility, path stability, and algorithm behavior | Available; does **not** identify the true reflector at 672.6 m |
| Supplied interpretation workbooks | Secondary disagreement and review targeting | Available; origin labels are unreliable in places and values may be interpolated despite appearing as manual cells |
| Synthetic data | Mechanism integrity, regression detection, and known-event stress tests | Available; current signals are substantially cleaner and simpler than field data |

The workbook parser presently infers trust largely from cell representation. Several sheets or filenames describe interpolated data while their hard-coded cells are classified as manual. Workbook agreement therefore must not be reported as field accuracy.

## Target architecture

```text
raw road signal
    │
    ├── conservative preprocessing ensemble
    │      (road-only, plate/no-plate where valid, filter/stack perturbations)
    │
    ├── local reflector segments + lobe/event alternatives
    │
    ├── local joint waveform evidence
    │      (surface/asphalt/base hypotheses; asphalt uncertainty marginalized)
    │
    └── missing-state-aware segment graph
           ├── accepted path where one event is well separated
           ├── review region where alternatives remain plausible
           └── explicit gap where evidence is absent
```

### Components to preserve

- GSSI DZT/DZG/DZX readers, scan geometry, and plate-scan ingestion.
- Surface flattening and the diagnostic preprocessing products, subject to ablation.
- Multi-candidate reflection packets and explicit alternative lobes.
- Hard manual seed constraints at the seed location.
- Signal-only versus design-assisted reporting.
- Explicit gaps, anomaly regions, exports, profile views, and reproducible manifests.
- Synthetic regression tests, expanded as specified below.

### Components to retire or demote

- Hard seed-reachable lineage as a candidate admissibility condition.
- A seed-trained global logistic score as the primary authority for event identity.
- A single stripped asphalt path as the input to base detection.
- Cross-layer sharing of regime-break rows.
- Design thickness as an emission reward or a mechanism for filling weak evidence.
- Confidence based mainly on correlated processing views and heuristic score margins.

## Work plan and gates

### Phase 0 — Freeze evidence and create an evaluation protocol

1. Freeze the current Talagang and Pattoki raw files, seeds, options, and diagnostic outputs with hashes.
2. Deduplicate byte-identical survey and workbook copies so repeated files cannot enter metrics twice.
3. Build a chainage-aligned checkpoint table with immutable train/validation/test roles and recorded provenance.
4. Obtain at least 30 blinded checkpoints per layer per road, sampled before seeing new tracker output. Stratify them across ordinary, weak, discontinuous, and anomalous radar sections.
5. Where physical measurements exist, store them separately from radar interpretations. If no physical confirmation is available, label every accuracy result as **radar-interpretation agreement**, not field accuracy.
6. Hold Pattoki out as the frozen transfer test while developing on Talagang. Do not tune to Pattoki test results.

**Gate 0:** no field-accuracy claim and no automatic-acceptance calibration until provenance-complete checkpoints exist. Algorithm prototyping may proceed, but only candidate recall, stability, and diagnostic results may be reported.

### Phase 1 — Determine which signal operations preserve base evidence

Run a controlled factorial ablation on fixed chainage windows, including the 672.6 m ambiguity and representative strong, weak, and anomalous sections:

- road-only versus existing plate subtraction;
- plate use restricted to a short normalized wavelet window versus full post-surface subtraction;
- no layer stripping, single asphalt strip, and an ensemble of plausible asphalt strips;
- signal-only wide search versus design corridor;
- hard lineage, soft lineage diagnostic, and no lineage;
- alternative stack lengths, filter bands, background removal scales, and deconvolution strengths;
- surface-pick perturbation and plausible dielectric ranges;
- forward/reverse processing and seed-free/leave-one-seed-out runs.

For every configuration, measure candidate survival and event identity at checkpoints, not only selected-path coverage. Plot the complete candidate energy/rank distribution around each checkpoint.

**Gate 1:** choose a conservative preprocessing set only if checkpoint candidate recall is at least 98% for visible asphalt and base events and the same physical candidate survives reasonable perturbations. Disable plate subtraction by default if the gain-mismatch ablation changes base identity or materially reduces stability.

### Phase 2 — Build local reflector segments

Replace trace-to-trace seed reachability with short-range, multi-scale 2-D ridge/phase tracking:

- form segments over approximately 5–20 m windows using local phase motion, waveform similarity, amplitude persistence, and curvature;
- retain competing parallel segments and their lobe-family relationship;
- allow segment birth, death, crossing in score space, and explicit gaps;
- store segment support, perturbation stability, waveform residuals, and alternative-event separation;
- use seeds to label compatible nearby segments, never to invalidate all unconnected distant segments.

Local-lineage code may be reused as a feature generator after removing hard feasibility semantics.

**Gate 2:** on synthetic stress cases and blinded radar checkpoints, the correct visible event must remain in the retained segment set at least 98% of the time. Discontinuities must terminate a segment rather than create a forced bridge.

### Phase 3 — Add local generative interface evidence

For each local window, compare a small set of joint waveform models containing surface, asphalt, base, and optional missing interfaces. Estimate a local shared wavelet and interface coefficients with sparse/regularized deconvolution, while keeping arrival alternatives discrete. Evaluate model evidence from waveform residuals and perturbation consistency rather than peak strength alone.

Base evaluation must integrate over the plausible asphalt hypotheses. It must not inherit one asphalt stripping decision. A second or third interface is admitted only when it materially improves out-of-sample/local held-trace fit and remains stable across conservative preprocessing perturbations.

Design thickness may define a broad physically possible search window. Run and retain a design-blind result; design must not create an interface or convert an unresolved result to accepted.

**Gate 3:** at blinded checkpoints, top-1 event identity should reach at least 95% for asphalt and 90% for base among radar-visible cases, with at least 98% top-k candidate recall. When the top alternatives are not separated, the result must be review/unresolved rather than an incorrect automatic pick.

### Phase 4 — Link segments with conservative global inference

Use a semi-Markov/factor graph or equivalent k-best dynamic program whose states are local segments plus an explicit missing state:

- enforce interface ordering and hard seeds;
- apply layer-specific continuity and layer-specific regime breaks;
- preserve multiple global hypotheses when their evidence is close;
- permit long gaps without a path-closure penalty that encourages invention;
- treat structural breaks as supported change points, not shared relaxations from another layer;
- compare forward, reverse, seed-withholding, and preprocessing-ensemble solutions before acceptance.

The selected profile is the locally supported portion common to sufficiently probable hypotheses. Disputed portions become review regions; unsupported portions remain gaps.

**Gate 4:** at least 95% precision for automatically accepted picks against the frozen blinded radar checkpoints for each layer, with Wilson confidence intervals reported. Seed withholding must keep every accepted nonlocal pick on the same event family within one measured pulse width; failures are demoted to review. Coverage has no minimum gate during this phase.

### Phase 5 — Calibrate uncertainty and review behavior

Calibrate acceptance on held-out checkpoints using features that correspond to distinct failure mechanisms:

- posterior/margin between event families, not adjacent lobes of the same packet;
- processing-perturbation stability;
- forward/reverse and seed-withholding stability;
- segment support length and gap distance;
- local waveform fit/residual whiteness;
- upper-layer uncertainty inherited by base;
- proximity to anomalies or change points.

Outputs must distinguish `accepted`, `review`, and `unresolved`. Review regions should include the competing hypotheses and the reason for ambiguity. Thickness is emitted only where both bounding interfaces are resolved; interpolation across an unresolved interface is prohibited.

**Gate 5:** reliability diagrams and selective-risk curves must show that the accepted set meets the precision target on held-out data. Every false automatic pick is assigned a documented failure class before expanding coverage.

### Phase 6 — Frozen transfer and physical validation

1. Freeze all parameters after Talagang development.
2. Run Pattoki once as the transfer test using its 2 in asphalt/6 in base design only as a broad support prior.
3. Report candidate recall, accepted precision, review rate, gap rate, and failure classes separately by road and layer.
4. Compare with workbooks only after predictions are frozen, and report agreement/disagreement without treating either side as truth.
5. Where physically confirmed depths exist, report signed depth/thickness error separately from radar-interpretation agreement.

**Gate 6:** promote a base tracker only if it satisfies the accepted-precision and stability gates on both roads. Claim field accuracy only if a sufficiently distributed independent physical set supports it. Otherwise ship it as a conservative radar-interpretation assistant with explicit limits.

### Phase 7 — Subbase feasibility, only after base promotion

Repeat Phases 0–6 for subbase. Do not infer subbase from design thickness or from the residual left by a fitted base. If a distinct, perturbation-stable segment family is not present and validated, output no subbase profile.

## Required experiment report

Every candidate implementation should produce one versioned report containing:

- exact input hashes, settings, seeds, and code revision;
- road/layer/checkpoint provenance and visibility labels;
- candidate recall before graph selection;
- top-1 and top-k event identity;
- accepted/review/unresolved fractions;
- leave-one-seed-out, forward/reverse, and perturbation stability;
- failure counts for cycle/lobe jump, wrong reflector family, stripping artifact, corridor bias, seed overreach, surface error, and missing evidence;
- workbook disagreement as a separate appendix;
- synthetic results as a separate regression appendix;
- an explicit statement of whether physical field accuracy has or has not been established.

## Stop conditions

Stop and reconsider the signal model rather than tuning the graph if any of these occurs:

- the confirmed base event is absent from candidates in more than 2% of visible checkpoints;
- plate/no-plate or modest filter changes repeatedly switch event family;
- accepted precision cannot reach 95% without collapsing to a trivial set;
- base identity changes materially when an asphalt-only seed is withheld;
- the model can satisfy its score mainly by following design thickness;
- independent interpreters cannot agree on the radar event and no physical confirmation exists.

In the final case, the correct output is an ambiguous interval or gap. More optimization cannot create physical information that the acquisition did not record.

## First implementation slice after approval

The first implementation should be deliberately narrow:

1. remove hard lineage admissibility while retaining lineage diagnostics;
2. add the fixed-window preprocessing/plate/stripping ablation harness;
3. create checkpoint-provenance and candidate-recall reports;
4. prototype local competing reflector segments on Talagang windows only;
5. evaluate segment retention and ambiguity at 672.6 m and the frozen checkpoint set before writing a new global graph.

No UI, packaging, runtime optimization, or subbase work should begin until these steps pass Gates 0–2.
