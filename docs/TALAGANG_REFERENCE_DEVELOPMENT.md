# Talagang workbook-guided accuracy development

The user supplied `GPR Data/talagang` as their manually traced reference. Internal
radar sample indices are the application's responsibility, not required user input.
This supersedes the pending request to confirm Gujrat sample indices.

## Processed files available from the work PC

The user confirmed that the processed radar files **and manual tracings exist**
on their work PC, and they expect access on **Monday**. They are not currently on
this laptop. The workbook's processed `TALAGANG_001 P_21.DZT`/RADAN project is
therefore pending transfer, not permanently unavailable. Do not ask the user to
transcribe internal sample numbers in the meantime.

When available, preserve and inspect the processed DZT, associated RADAN project/
pick files and processing/depth settings alongside the raw DZT and workbook.
Recover the time-zero/depth origin, dielectric and selected-lobe convention,
establish raw-to-processed trace/sample transforms, and compare the exported picks
against the actual manual tracings. These are intended both for conventional
accuracy validation and **later supervised ML training**. Verified original picks
may become labels after alignment/provenance checks; derived/interpolated values
and evaluation-only checkpoints remain excluded from training. All Talagang
duplicates/processed versions must retain the same physical-road group to avoid
train/evaluation leakage. ML remains deferred during current conventional work.

## What the files establish

- Raw acquisition: `TALAGANG.PRJ/TALAGANG_001.DZT`, 75,436 traces,
  512 samples/trace, 15 ns, 0.025 m/trace; approximately 1.886 km.
- Workbook: `Talagang_layer_thickness_.xlsx`, 378 stations at 5 m intervals,
  from 0 to 1,885 m. Columns are Layer 1/2 **Depth(in)**, interpreted as
  cumulative depths to asphalt/base interfaces, not individual base thickness.
- 367 original asphalt values and 284 original base values. The workbook's
  `Interpolation Summary` explicitly documents 11 filled asphalt values and
  94 filled base values, including edge extrapolation. Yellow cells remain
  excluded from original-observation comparisons and seed selection.
- No subbase labels are present.
- Workbook scan bounds 0–75,435 match the raw acquisition. GPS interpolated at
  the stated chainages agrees with 377 available workbook positions: median
  0.0496 m, maximum 1.2242 m. No chainage offset was fitted.
- Both acquisition DZT and DZX record dielectric 7. The working sample interval
  is 15/512 ns; the full-road detected surface reference is 156. Under a uniform
  dielectric-7 conversion, depth per sample is approximately 1.66 mm.

The workbook names `TALAGANG_001 P_21.DZT`, a processed RADAN file absent from
the supplied folder. Its depth origin, processing history, dielectric used during
interpretation, and lobe/timing convention are not independently established.
The acquisition settings support a useful **provisional projection**, not verified
raw-coordinate training labels. No workbook offset is optimized against tracking.
Original workbook values remain legitimate analyst interpretation references;
this uncertainty concerns their conversion, not whether the user traced the road.

The metal-plate and road gains differ. Current processing correctly retains the
measured road signal without plate subtraction. The historical v17 snapshot used
plate subtraction and is therefore not a controlled comparator for current code.

## Reproducible development check

Run from the repository root:

```powershell
.venv/Scripts/python.exe scripts/evaluate_talagang_workbook.py --output exports/hybrid-development/talagang-workbook-new
```

The command prepares the full acquisition, compares its first 150 m, and uses
three original workbook observations per layer near 15, 75 and 140 m as rounded
depth-projected **development guides**. Other original observations are comparison
checkpoints. Derived cells supply neither guides nor the reported checkpoint
metrics. No remaining checkpoint enters tracker input, ML conditioning, or
training. This is development on one physical road, not held-out evaluation.

`--method joint_seed_adaptive` runs the established comparator with identical
guides. Each output directory must be new. JSON records input/code fingerprints,
guide cells, conversion assumptions, label counts, GPS checks, all comparison
errors, and runtime. NPZ stores measured radar, accepted and provisional samples.
Neither a missing nor a provisional path is counted as an accepted checkpoint.

Completed hybrid output: `exports/hybrid-development/talagang-workbook-integrated-v2`.
The first 150 m has 375 stacked observations. Hybrid asphalt accepted 375;
28 non-guide original checkpoints were covered, with median projection error
2.263 samples (13 within two samples). Hybrid base accepted 60 observations,
but only two of 19 non-guide original checkpoints, with median projection error
9.258 samples; neither was within two samples. **This does not satisfy promotion.**
Tracking took 6.30 s excluding preparation, with concurrent development activity;
it is not a controlled local-correction timing benchmark.

The established comparator (`talagang-workbook-established-v2`) accepted 51 base
observations and one non-guide base checkpoint, with a 3.542-sample projection
error. Its asphalt covered all 28 checkpoints with median error 1.571 samples
(17 within two samples). Its tracking took 7.22 s. Thus the hybrid has **not**
demonstrated superior accuracy: its additional base acceptance includes larger
projection errors. Although it calls the established shallow engine, hybrid
evidence preparation also changes shallow results; asphalt parity is not proven.
Keep both methods explicit and retain the existing default.

## Conventional mechanism changes

Both changes affect only the experimental hybrid correspondence graph:

1. Same-lobe candidate timings within the existing pulse-derived tolerance no
   longer destroy reciprocal-match margin merely because multiple feature
   branches proposed them. Sign/zero crossings keep neighbouring lobes distinct.
   Candidates and exact manual observations are retained.
2. Repeated locally admissible jumps cannot accumulate unlimited displacement
   within 25 m of the nearest seed in the same structural region. The cumulative
   allowance equals the existing direct-match allowance,
   `pulse_width * (1 + sqrt(distance_m)/2)`. Contradictory seeds retain independent
   neighbourhoods and the existing barrier constraints. This limits propagation;
   it does not interpolate depth or relax acceptance thresholds.

In a base-only diagnostic, the original hybrid accepted a point about 50 samples
shallower than the workbook projection at 20 m. The same-lobe correction alone
increased continuation but retained that excursion. The seed-neighbourhood rule
removes that measurement and other extreme nearby excursions. Residual accepted
errors at other checkpoints remain, as the integrated result above shows.

Regression coverage includes duplicate timing versus a genuinely separate lobe,
cumulative small jumps, contradictory seeds, structural breaks, exact seed
preservation, waveform drift, missing reflectors, zero-signal gaps, ML-only false
support, no crossing, and complete-path ambiguity. Focused processing/workflow/
reference regressions passed. Acceptance remains uncalibrated and the existing
default backend remains unchanged.

## Remaining accuracy work

Use this workbook and its verified road alignment to investigate reflector
identity over broader neighbourhoods, particularly where the base band is weak
and competing ringing is stronger. The current sparse graph still fragments and
can change lobe identity through individually plausible matches. Assess proposed
and accepted paths separately; increased accepted coverage is not itself success.
The depth projection alone cannot certify matching lobe identity. Resolve the
processed-file depth convention when that information becomes available, without
asking the user to transcribe internal sample coordinates. ML stays inactive;
this one road cannot meet the three-held-out-road promotion requirement.

Visual comparison of original workbook values against the measured radar at
0–150, 950–1100 and 1550–1700 m:
`exports/hybrid-development/talagang-workbook-development/workbook-measurement-comparison.png`.
The display uses depth-wise RMS gain only to show weak bands; measurement and
tracking arrays are unchanged. No interpolated workbook values are plotted.

## Follow-up: alternative correspondences and full-road test

The next conventional change retains up to three mutually ranked, near-best
distinct-lobe correspondences. Previously, local ambiguity removed the connection
before the seeded path solver could assess it. Duplicate feature timings on one
signed lobe do not consume the three alternatives. Competing matches receive a
transition-quality penalty; their presence is not acceptance evidence. Adjacent
nodes are contracted only when the source has one outgoing and the destination
one incoming adjacent connection, preserving every ambiguous branch. The
correspondence-rule version is recorded in run provenance.

The 150 m rerun (`talagang-workbook-multiroute-v3`) accepted 43 base observations
and three original non-guide checkpoints; one was within two samples of its
provisional workbook projection. Median accepted checkpoint error was 5.628
samples. This is a small development change, not a reliable-accuracy result.
Two diagnostic smoothing experiments were not incorporated: lateral averaging
increased proposal coverage but worsened checkpoint errors; vertical smoothing
did not produce a consistent coverage gain. Raw surface variation within the
first 150 m's 16-trace stacks was mostly 0–2 samples (only 1.07% of stacks had a
range of at least four samples), so pre-stack bounce alignment is not established
as the main cause of failure here.

Full-road command with just three distributed workbook-derived guides:

```powershell
.venv/Scripts/python.exe scripts/evaluate_talagang_workbook.py --stop-m 1885 --seed-chainages 15 945 1860 --output exports/hybrid-development/talagang-full-road-multiroute-v3
```

This **failed** the intended road-length base-tracing goal: 192 of 4,713 base
observations were accepted (4.07%), covering 12 of 281 original non-guide base
checkpoints. None was within two samples of the workbook projection; the median
accepted projection error was 27.943 samples. Asphalt accepted 4,637 observations,
covering 360 of 364 original non-guide checkpoints; median projection error was
2.406 samples. Tracking took 173.3 s, excluding preparation. The settings remain
experimental; neither the short nor full-road results justify promotion.

The initial 150 m workbook-derived base guides at 15, 75 and 140 m have measured
polarities negative, positive and negative. Directly projecting depths therefore
does not establish consistent selected-lobe identity. The runner now also supports
`--radar-seeds benchmarks/talagang-development-seeds.json`: these saved radar clicks
alone guide tracking, their recorded pulse width is used, and the workbook is
comparison-only. The uncertain depth/lobe conversion is explicitly retained in
the output. Saved seed files and workbook values are not modified.

`scripts/plot_hybrid_development.py` plots frozen runs, for example:

```powershell
.venv/Scripts/python.exe scripts/plot_hybrid_development.py exports/hybrid-development/talagang-workbook-integrated-v2 exports/hybrid-development/talagang-workbook-multiroute-v3 --output exports/hybrid-development/talagang-before-after-multiroute.png
```

Solid lines mean the backend accepted the observation, **not that it is verified
correct**. Provisional paths are dotted, missing observations remain gaps, guide
points are circles, and original workbook checkpoints have separate colours.
The plotting command supports `--start-m` and `--stop-m` for readable road sections.
The full-road 900–1100 m plot visibly exposes the incorrect near-stationary base
proposal in `exports/hybrid-development/talagang-full-road-900-1100.png`.

The mechanism/processing/workflow run passed 92 tests after the correspondence
change, including preservation of an initially second-best route to a later seed,
three distinct alternatives, zero-signal gaps, disappearance and crossing checks.
This verifies those mechanisms and contracts, not field accuracy.

The completed `talagang-full-road-radar-seeds-v3` run uses the existing three
asphalt clicks and two base clicks at 0.1875 and 1688.9875 m, with the recorded
14-sample pulse width. It accepted 390/4,713 base observations (8.28%), covering
23 of 284 original base checkpoints; one was within two samples of the provisional
workbook projection and median accepted projection error was 14.468 samples.
Asphalt accepted 4,593 observations and 356/366 non-guide original checkpoints,
with median projection error 3.809 samples. Tracking took 232.4 s. These results
also **fail** to establish reliable tracing, even when no workbook depths enter
the tracker. Seed count, coordinates and pulse width differ from the workbook
guide experiment, so the two runs are not a controlled algorithm ablation.

Saved radar-click overlays:
`exports/hybrid-development/talagang-radar-seeds-full-road.png` and
`exports/hybrid-development/talagang-radar-seeds-1550-1750.png`.
Next work should investigate broader reflector-neighbourhood correspondence and
adaptive continuation; isolated short-waveform similarities still admit ringing
switches and fragmentation. Keep the failed baselines for comparison.

## Whole-trace correspondence experiment

Independently centred, four-pulse gain-balanced neighbourhood snippets were
tested and rejected: they reduced continuation without a consistent checkpoint
benefit. Their code and results are frozen only in the ignored development
directory (`hybrid-context-experiment.py`, `context_correspondence.py`,
`experiment-context.npz`, `experiment-context-bridge.npz`). They are not application
dependencies.

The next implementation, `processing/trace_registration.py`, registers the
ordered waveform section across each sparse trace pair, rather than separately
recentering a window for every candidate pair. Signed local RMS balancing keeps
one strong neighbouring packet from dominating the cost. A trace-relative floor
limits amplification of quiet tails. Banded dynamic programming permits only
(1,1), (1,2), (2,1) steps, so arbitrary flat runs cannot collapse many cycles onto
one sample. The per-pair warp band equals the existing displacement allowance.
Forward/inverse maps are derived from the same monotonic alignment; local fit
is explicitly not a visibility or correctness probability.

Where both local fits are at least .75, existing candidate correspondences must
agree with this order-preserving mapping within the existing pulse-derived
tolerance. Otherwise the previous local matching remains available. The original
waveform, polarity, DTW, seed and signal-visibility gates still apply. The measured
radargram and candidate coordinates are unchanged. Registration is batched in
groups of 64; structural-region boundaries are never paired. The new rule is
recorded as `whole-trace-registration-v4` in hybrid provenance.

Eight focused registration tests verify shifted weak reflections between similar
packets, gain/polarity behaviour, zero-signal preservation, identity alignment,
batch equivalence, cancellation, structural isolation, and agreement with an
independent exhaustive tiny-path calculation. The 36 hybrid, 10 path-ambiguity,
29 processing and 17 seed-workflow tests also passed during this change. These
100 checks verify mechanisms and workflow, not field correctness.

The full-road v4 comparison uses the exact v3 radar clicks, pulse width, stack
size, source fingerprints and conversion settings. Its output directory is
`exports/hybrid-development/talagang-full-road-radar-seeds-v4-registration`.

The full-road result did **not** improve accuracy: base accepted 364/4,713
observations versus 390 previously, covered 22 original checkpoints versus 23,
and had a median provisional-depth projection error of 15.173 samples versus
14.468. One accepted checkpoint was within two samples in each run. Asphalt's
arrays and aggregate metrics were unchanged. Tracking took 244.9 s versus
232.4 s in the prior run; concurrent test activity makes this an indicative
comparison only. A further short-window variant that allowed whole-trace fit
to bridge weak local waveform matches also failed to produce consistent
checkpoint improvement and was rejected.

**Disposition:** registration is inactive by default. Its tested implementation
is retained for explicit conventional ablations, enabled by
`--whole-trace-registration` in `scripts/evaluate_talagang_workbook.py`. The tracker
uses it only when `HybridEvidence.provenance["whole_trace_registration"]` is true;
the flag and correspondence version are recorded. Ordinary hybrid runs retain
the previous v3 behaviour. No acceptance threshold was loosened and no method
or layer was promoted. The bridge variant exists only in the ignored development
archive (`hybrid-registration-bridge.py`).

The inspected side-by-side image is
`exports/hybrid-development/talagang-registration-comparison.png`, covering
1550–1750 m with identical saved radar clicks. It exposes the remaining wrong
ringing proposals and gaps. This result strengthens the need to inspect the
user's processed files and manual tracings on Monday, while continuing to keep
the conventional pipeline and reproducible experiment tools operational.

The image reveals a useful **local proposal** improvement despite the failed
full-road acceptance result. In the inspected 1550–1750 m section, median base
proposal error against the provisional workbook conversion falls from 21.621 to
9.536 samples; proposals within two samples increase from zero to four. Available
original checkpoints increase from 14 to 17. This window was inspected after
the runs, so it is exploratory development evidence, not held-out validation.
Across the complete road, proposal median error is essentially unchanged
(25.167 versus 25.333 samples), with six proposals within two samples in each
run. Registration therefore has local potential but is not a consistent road-wide
improvement. Retaining it as an opt-in experiment permits further investigation
without claiming it has solved accepted-pick accuracy.
