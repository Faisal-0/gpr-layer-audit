# Timing identity diagnosis before another mechanism

The exact retained graph, provisional samples and error-row max-marginals replay
unchanged. Every reported route decomposition sums to the exact forward/backward
max-marginal objective. No production code, thresholds or output coordinates changed.

All14 current accepted errors satisfy the frozen **same signed lobe** identity test:
one base timing error and13 subbase timing errors. Twelve subbase errors are new
(25,236,247,249,260,263,269,341,348,349,359,362);47 was already wrong. The old error351
is no longer accepted wrong. The new base error is18. No signed-lobe switch is hidden
by this classification, but a signed lobe can contain distinct waveform extrema.

All six initial deep-layer operating seeds are exactly at the strongest local
amplitude extremum within their original signed lobe. Their offsets are0 samples.
This supports testing an extremum timing fiducial derived from the operating seeds.
It does not justify posthoc output snapping.

## The missing candidates are cap losses

The three missing reference extrema are inside the original radar candidate union,
inside the search bounds, and have valid sample and complete waveform context.
Replaying the original packet construction/reservation exactly reproduces every
retained candidate sample on these rows. All three reference extrema are removed
by the12-packet cap:

| Subbase row | Peak | Zero-based packet rank | Packet selected/canonical sample |
|---|---:|---:|---:|
|247|320|13|324/320|
|263|322|13|318/322|
|348|328|14|328/328|

These are losses before correspondence, graph admissibility or context rejection.
In row247 the retained shoulder325 belongs to the same extremum basin as missing
peak320. At263, retained peak325 is a distinct basin from missing peak322. At348,
retained shoulder337 belongs to peak336, a distinct basin from missing peak328.
Thus filling in missing extrema must preserve separate modes; a selected timing
cannot simply donate support to another peak in the same signed lobe.

## Objective and gate failures

Base18 selects259 versus reviewed250, within one extremum basin bounded by the
valley at260. Peak250 is seed-feasible, but correspondence0.748 fails the unchanged
0.80 gate. Correct timing256 has support0.8083 and route objective1.05638509;
selected259 has1.05964232. The difference0.00325723 consists of seed evidence
+0.00081331, wavelet−0.00002570, generic+0.00012017, directed support+0.00028930,
external correspondence cost advantage+0.00206016, and negligible rounding.
Because256 lies within the7.25-sample bucket centred at259, it is not treated as a
competitor. The winning margin competitor is a missing observation, and margin clips1.

Subbase25 selects secondary extremum326 rather than reviewed peak322. The correct
candidate has **higher** local score0.934899 versus0.839191, support0.8693, and a
feasible route. The wrong full route still wins0.66579802 versus0.65556238: its
external correspondence cost advantage is0.024298, exceeding the other evidence
differences. This cannot be explained solely by locally brighter shoulders.

Subbase341 selects subsidiary peak331. Correct timing329 is feasible and locally
stronger, but is pooled into the selected timing bucket. Its full objective is only
0.00382529 lower, predominantly a0.004383 external correspondence cost disadvantage.
Exact reviewed peak328 is feasible but support0.7502 fails0.85; timing329 also fails
at0.8336. A score-only peak preference would abstain here.

Best reference-consistent route deficits at other rows are:249=0.02992734,
260=0.13123211,269=0.01287891,349/359/362=0.00949722 each. Row236 has the correct
candidate320 but no complete seed-feasible route. At47, reviewed peak330 has no
complete route; correct timing331 has support0.6997 and a route deficit0.50807218.
The full per-term values and exact route sequences are in the layer JSON files.

There are nine rows with a feasible exact reviewed-peak candidate, but only six
have that peak's correspondence above the unchanged layer gate. Therefore merely
reweighting existing peak nodes is insufficient evidence for useful acceptance.

## What the current margins mean

With `distinct_path_inference=false`, the margin function pools candidates whose
polarity agrees and whose timing is within tolerance of the selected sample.
Same-lobe timings farther away are already competitors. Thus alternatives are not
summed as independent evidence or double-counted into the objective; each path has
one observation per row. The problem is that a broad, selected-centred timing bucket
can hide a near-optimal correct timing, while its reported margin describes a
different event or a missing-observation competitor. The margin is not a calibrated
probability of timing within the reviewed tolerance.

Raw extremum basins divide these14 errors into six unambiguous same-basin shoulder
errors, one sample exactly at a valley shared by two basins (359), and seven errors
in a different extremum basin of the same signed lobe. These are local waveform
modes, not independently established physical reflector identities.

## Bounded candidate design

Test a **seed-derived extremum-mode state** on the isolated lobe-mutual branch:

1. Infer the timing fiducial solely from the initial operating seeds' local extrema
   and offsets. Preserve each exact seed and retain explicit incompatible/uncertain
   fiducials when seeds disagree.
2. Before the packet cap, represent local extremum basins within every retained
   signed-lobe hypothesis. Retain split/secondary extrema as separate modes. Packet
   boundaries must not preserve a basin's shoulder while dropping its actual extremum.
   Any added secondary basin requires its own computed edges and support.
3. Preserve waveform-member correspondence inside one basin as evidence about that
   basin, with member-to-extremum offsets explicitly represented. Recompute motion,
   objective geometry and graph support in the declared fiducial coordinate. Never
   transfer quality between distinct extrema or reuse the old selected confidence.
4. Score and select mode plus fiducial jointly; distinguish within-mode timing
   ambiguity from competing extrema and explicit gaps. Recompute competing-route
   margins under exactly that same objective and apply the original gates.
5. Run one prospective Mandiali comparison across **all** scored accepted/proposed
   observations, including losses and newly accepted errors. Continue to Gujrat only
   if precision and useful correct coverage improve together. No posthoc snapping,
   tolerance changes or adaptive weight sweep.

This is a proposed experiment, not an implemented result. The support failures and
different-basin cases remain material risks; they preclude promising that peak timing
alone repairs tracking.

## Reproduce

```powershell
.venv/Scripts/python.exe scripts/experiment_timing_identity.py
.venv/Scripts/python.exe exports/seeded-tracker/timing-identity/assess_basins.py
.venv/Scripts/python.exe exports/seeded-tracker/timing-identity/locate_missing_peaks.py
```

`layer-2-objective-errors.json` and `layer-3-objective-errors.json` contain exact
scoring terms and per-candidate feasibility/support. `extremum-basins.json` records
the separate within-sign waveform modes. `missing-peak-cap-loss.json` records the
exact candidate-cap replay. The first objective report was generated before adding
the descriptive basin fields; those fields are a separate derived artifact and
do not change any objective calculation.
