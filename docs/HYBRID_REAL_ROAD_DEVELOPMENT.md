# Combined-tracker real-road development check — 2026-09-05

The combined conventional tracker is **not ready to replace the established
default**. Controlled dropout recovery improves, but the first complete Gujrat
run still has poor base/subbase coverage and unreliable seed withholding.
ML and UI development are deferred while these tracing defects are addressed.

## Reproducible protocol

`scripts/evaluate_hybrid_seed_recovery.py` loads a frozen radar-only seed
manifest, reconstructs its full-acquisition calibration/chainage grid, freezes
the full path, and withholds each station in turn. It never opens a workbook,
uses design depths, or converts tracker output into training labels. Output
contains acquisition/code fingerprints, numeric radar and paths, seed signal
polarity, runtime, and held-out sample errors. These previously disclosed roads
are development evidence; three seed stations cannot establish field accuracy.

```powershell
.venv\Scripts\python.exe scripts/evaluate_hybrid_seed_recovery.py --manifest benchmarks/blinded-base-subbase-seeds-20260829.json --case sohal-kalan-gujrat-second-portion-001-validation --output exports/new-gujrat-run
```

Use `--method joint_seed_adaptive` for the established comparator. Output
directories must be new so earlier results are not silently overwritten.

## Findings and retained fixes

1. Deep waveform/polarity identity used processed data while local motion used
   the unstripped measurement. Filtering can reverse the sign at a selected
   sample. Deep correspondence, waveform templates and reported lobe identity
   now use the measurement consistently. Processed data remains candidate
   evidence. Derived phase/polarity metadata is recomputed for this signal;
   clicked and canonical sample coordinates remain authoritative. A regression
   verifies continuation through a processed sign reversal.
2. Positive connection bonuses rewarded fragmentation: a weaker path with many
   segments could outrank a stronger coherent path. Connections now incur
   uncertainty costs, and contracted chain interiors retain the same costs.
   Tests check the reproduced wrong-path failure and equivalent contracted and
   uncontracted scoring.
3. Correspondence support could bypass an authoritative seed through a long
   edge, even when path selection obeyed that seed. Support and path selection
   now share the observation constraints. A contradictory-lobe test verifies
   that support cannot leak around the intervening seed.

These changes repair demonstrated mechanisms. They do not establish improved
field accuracy or justify loosening acceptance.

## Gujrat second portion

719 stacked observations cover approximately 287 m; three seeds are supplied
per interface. The full established comparator accepted 713 asphalt, 81 base
and 35 subbase observations in 46.9 s. The latest combined development run
accepted 712 asphalt, 7 base and 9 subbase observations in 24.8 s. Runtime
includes concurrent development activity and is not a controlled speed benchmark.
Accepted counts are coverage indicators, not correctness measurements.

Withholding each station in the latest combined run:

| Interface | First withheld station | Middle withheld station | Last withheld station |
|---|---|---|---|
| Base | 11-sample error, rejected | Missing | Missing |
| Subbase | Missing | 1-sample error, rejected | 1-sample error, accepted |

The existing asphalt engine also fails strict sample/lobe recovery on this
case (4, 8 and 8 samples at the three withheld observations). In addition,
the saved last asphalt seed declares a negative trough while the current
measurement and processed values at its coordinate are positive. The saved
manifest has not been altered. This inconsistency must be resolved before
treating that record as an accuracy label. Base's middle seed is negative in
measurement data but positive after processing, directly demonstrating the
signal-identity inconsistency addressed above.

An experiment restricting graph candidates to measured amplitude extrema
raised full subbase acceptance to 41 observations, but a withheld first station
then produced a 45-sample error. The error remained rejected. That experiment
was **not incorporated**; increasing accepted coverage alone is not a valid
selection criterion.

Frozen outputs:

- `exports/hybrid-development/gujrat-seed-recovery-v1`
- `exports/hybrid-development/gujrat-established-v1`
- `exports/hybrid-development/gujrat-measurement-identity-v2`
- `exports/hybrid-development/gujrat-transition-cost-v3`
- `exports/hybrid-development/gujrat-extrema-experiment-v4` (rejected experiment)
- `exports/hybrid-development/gujrat-seed-barriers-v5`

## Next accuracy work

A second full-road combined check on Burewala acquisition 002 accepted 1,270
asphalt and 64 base observations out of 1,463, in 30.2 s. It used the same frozen
manifest and settings; no withholding or correctness assessment was performed
for that run. Its output is
`exports/hybrid-development/burewala-seed-barriers-v5`. This corroborates that
limited base coverage is not isolated to the shorter Gujrat acquisition.

Improve local seed-connected continuation and ambiguity representation inside
the same graph. The present graph can give several similar ringing candidates
high correspondence support, leaving no useful margin at the intended lobe.
Candidate-source pruning alone fails this case. Next experiments must measure
seed recovery, wrong-lobe switches and supported continuation separately, while
retaining disappearance/gap and contradictory-seed regressions. Full-road
promotion remains unproven; it needs independent verified sample/lobe picks,
not workbook projection or the tracker judging its own paths.

## Complete-path ambiguity and analyst observations

The next change adds exact forward/backward max-marginals on each seeded
component of the segment DAG. It compares complete route scores, with the same
transition and missing-state costs as selection. A route must satisfy every
seed in that component. Opposite lobes remain competitors even when separated
by only two samples; explicit edges skipping an observation also compete.
The numerical margin is normalized by component span and is **not** an error
probability. It is exported separately as `hybrid_path_margin`.

The local acceptance thresholds have not been lowered. An additional feasibility
check rejects automatic measurements on tied or infeasible seeded paths.
Manual observations remain exact. On Gujrat's full run, this preserves the seven
accepted base observations and reduces accepted subbase observations from nine
to the three manual seeds. This reveals unsupported acceptance; it is not a
coverage improvement. Frozen diagnostics are in `gujrat-path-marginal-v6` and
`gujrat-path-feasibility-v7` under `exports/hybrid-development`.

Additional-click selection now uses separated, observable competing paths and
their contested span. It excludes zero-signal intervals and observations within
25 m of an existing seed for that layer. The span is a ranking heuristic, not
a promise that a click will resolve that distance. Generic existing review
requests remain available where no two explicit alternatives survive. Weak-seed
recapture requests take precedence.

An unoverlaid measurement radar sheet was prepared at the three original
Gujrat seed stations:
`exports/hybrid-development/gujrat-measured-pick-sheet.png`. Analyst confirmation
of base/subbase samples at A=43.3875 m, B=143.7875 m and C=244.1875 m has been
requested. No new labels or seed changes are inferred from the assistant's
inspection. The user subsequently supplied the Talagang raw acquisition and
manual-depth workbook instead; that supersedes the pending sample-index request.
See `TALAGANG_REFERENCE_DEVELOPMENT.md` for the coordinate audit, conventional
matching changes, and reproducible workbook-assisted comparison. Gujrat's
historical seed-definition uncertainty remains unresolved.
