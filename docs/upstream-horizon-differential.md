# HorizonTracker differential experiment

The optional research adapter in `processing/upstream_horizon.py` calls the
same `tslearn.metrics.dtw_path` primitive as pinned HorizonTracker commit
`bf550594bd55d7741a0fcbfdef0e4ede2567d48b`. The experiment also imports and
executes the actual upstream `DynamicTimeWarping` function and asserts exact
path equality. This is a numerical comparison, not an assertion that either
method establishes physical interface identity.

Upstream source is MIT, copyright (c) 2019 Aina Juell Bugge. The unmodified
checkout, including `licence.txt`, is under
`exports/seeded-tracker/upstream/HorizonTracker`. The adapter calls upstream
dependencies and contains no copied upstream function. `tslearn` 0.7.0 declares
BSD-2-Clause; its dependency versions are recorded in each output. No upstream
interpolation, morphology or horizon interpolation function is used. Please
retain the upstream attribution and cite Bugge et al., *Automatic extraction of
dislocated horizons from 3D seismic data using non-local trace matching*,
Geophysics 84(6), 2019, when using the research comparison.

HorizonTracker does not pin its `tslearn` dependency. This experiment pins
`tslearn` 0.7.0 and verifies execution of the pinned upstream Python function
under that dependency; it does not claim to recreate an unspecified 2019
software environment.

## Numerical contracts and differences

- Repository radar is `(trace, sample)`; upstream is `(sample, inline,
  crossline)`. The two-trace differential constructs `(sample, 2, 1)` explicitly.
- Both map integer native sample indices. `dt_ns` and retained trace distance
  are metadata; no coordinate shift, resampling or interpolation is applied.
- Upstream aligns complete input A-scans with unconstrained dynamic time
  warping and keeps all event mappings, including many-to-many alternatives.
  Its `DynamicTimeWarping` function does not apply the similarity filter whose
  commented-out line remains in its source.
- Repository candidate correspondence uses normalized centered wave packets,
  a warp band, a raw cosine prefilter, polarity/phase gates, motion gates and
  independent reverse alignment. Its separate whole-profile registration also
  balances amplitude and constrains stretching. These are different algorithms.
- The adapter uses only common contiguous valid observation runs. No path may
  bridge an internal invalid sample range. It retains every mapped center
  sample instead of replacing a split mapping by an interpolated observation.
- Valid all-zero traces yield no correspondence. Numeric warp scores and
  reciprocal agreement are not calibrated correctness probabilities.

## Reproduction

The experiment uses a separate interpreter; application dependencies are
unchanged. From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m venv exports/seeded-tracker/upstream/venv
exports/seeded-tracker/upstream/venv/Scripts/python.exe -m pip install tslearn==0.7.0 ipython==9.17.1 scikit-image==0.25.2 scipy==1.18.1 numpy==2.5.3 matplotlib==3.11.1 PyWavelets==1.10.0 pytest==8.4.2
git clone https://github.com/ajbugge/HorizonTracker.git exports/seeded-tracker/upstream/HorizonTracker
git -C exports/seeded-tracker/upstream/HorizonTracker checkout bf550594bd55d7741a0fcbfdef0e4ede2567d48b
exports/seeded-tracker/upstream/venv/Scripts/python.exe scripts/experiment_upstream_horizon.py --output exports/seeded-tracker/upstream/new-differential.json
exports/seeded-tracker/upstream/venv/Scripts/python.exe scripts/experiment_upstream_horizon.py --run --control --case mandiali-short --output exports/seeded-tracker/upstream/new-control.json
exports/seeded-tracker/upstream/venv/Scripts/python.exe scripts/experiment_upstream_horizon.py --run --case mandiali-short --output exports/seeded-tracker/upstream/new-warped.json
exports/seeded-tracker/upstream/venv/Scripts/python.exe -m pytest tests/test_upstream_horizon.py
```

Outputs cannot overwrite previous experiments. Each invocation snapshots the
frozen baseline source, adapter and runner and then launches the snapshot in a
fresh interpreter. Inputs and native operating seeds are checked against
`benchmarks/seeded-evaluation-inputs.json`. The full comparison uses the same
calibrated conventional configuration, seeds, scorer and signed-lobe/time
tolerance as the frozen control. Source, input, configuration and dependency
identifiers accompany outputs.

## Initial differential result

`exports/seeded-tracker/upstream/differential-v3.json` contains 18 seed-selected
trace pairs per road. Crops retain four operating-seed context radii on each
side in time. Offsets are one retained trace, 0.5 m and 2 m; no scoring reference
chooses a crop or target. All 36 paths exactly match the actual pinned upstream
function. On the same normalized packets, the repository's actual scalar DP
with a nonrestrictive band also matches upstream's accumulated error and
center displacement to `1e-10` on all 36 pairs. Thus the substantive difference
is the waveform context, constraints and prefilters; this comparison found no
defect in the scalar DTW recurrence. Four observation/mapping/provenance tests pass in the optional
interpreter (the DTW test skips when `tslearn` is not installed).

Reviewed answers are read only after the numerical matching. At targets with
an exact eligible answer, upstream mapped alternatives include an agreeing
signed-lobe/time observation for 9/18 Mandiali pairs and 6/13 Gujrat pairs. Five
Gujrat targets have no reviewed answer. Some mappings span 14 samples. These
are **alternative-retention diagnostic counts**, not accepted-track precision
or independent road trials. Both roads influenced historical development.

The one full-tracker mechanism experiment uses reciprocal full-profile warped
context cosine where it exceeds the raw packet cosine. Candidate generation,
seed scoring, phase, motion, packet-DTW, path objective and final gates remain
frozen. This mechanism is confined to the research runner; application dispatch
does not expose it.

The `cosine_rescues` counter counts candidate pairs moved across the raw cosine
threshold before the remaining gates. It is not a count of retained edges,
reference-consistent routes or correct added observations.

The paired optional-environment Mandiali control reproduces the frozen result:
base 4/4 agreeing accepted observations and subbase 42/44, corresponding to
4/443 and 42/404 correct accepted coverage. Its wall time was 475 seconds under
concurrent experiments; this is not an isolated speed benchmark.

## Full-tracker decision: reject promotion

The paired Mandiali experiment finished with identical initial native seeds
and frozen scoring tolerance. It adds correct accepted observations, but adds
wrong acceptance and fails the 95% minimum agreement gate on both deep layers.
No larger Gujrat run or compensating threshold sweep is justified by this
result. The Gujrat evidence in this experiment remains the 18-pair numerical
differential described above, not a whole-road tracker evaluation.

| Layer and method | Agreeing / accepted | Agreement | Correct coverage | Incorrect coverage | Unresolved coverage |
|---|---:|---:|---:|---:|---:|
| Base control | 4 / 4 | 100% | 0.90% | 0% | 99.10% |
| Base warped context | 87 / 104 | 83.65% | 19.64% | 3.84% | 76.52% |
| Subbase control | 42 / 44 | 95.45% | 10.40% | 0.50% | 89.11% |
| Subbase warped context | 64 / 70 | 91.43% | 15.84% | 1.49% | 82.67% |

Coverage denominators are 443 base and 404 subbase eligible reviewed
observations, excluding operating seeds. Base wrong accepted observations rise
from 0 to 17, of which 7 are on a different signed lobe (4 observed runs); the
remaining errors exceed the frozen timing tolerance. The longest consecutive
wrong accepted base span is 0.075 m. Subbase wrong observations rise from 2 to 6,
all within-lobe timing errors; its longest wrong observed span remains 0.025 m.
These runs are diagnostic counts, not independently confirmed physical
reflector switches.

Base candidate retention remains 438/443. Correct proposals rise 110 to 148,
and correct-proposal losses at final acceptance fall 106 to 61. Subbase correct
proposals rise 125 to 127. Its retained candidate count changes 352 to 364
because the changed base output enters the existing upper-interface handling;
this experiment does not isolate subbase evidence from that interaction.
Asphalt output is exactly unchanged. Three initial observations per layer and
zero additional analyst actions were used.

The warped run recorded about 2,652 seconds including evaluation under concurrent
host load, compared with about 478 seconds for the paired control. The extra
runtime buys a material coverage increase at an unacceptable error rate.

Artifacts in `exports/seeded-tracker/upstream/`:

- `mandiali-control.json`, `mandiali-warped.json`: complete results, source/data
  hashes, unchanged configuration, and optional dependency versions.
- `mandiali-comparison.json`: whole road, seed-bracketed interval, left tail
  and right tail denominators, timing/lobe errors, coverage and gate losses.
- `mandiali-control-vs-warped.png`: real radar with agreeing acceptance,
  wrong acceptance, unresolved observations, proposals and exact supplied seeds.
- The corresponding `*-source/` directories preserve executed source and runner.

The independent contract review found no mask/coordinate leakage in these
results. Two runner hardening fixes followed the frozen runs: staged upstream
changes are rejected against `HEAD`, and configuration bytes are parsed and
hashed before inference. The original run snapshots remain untouched; their
recorded configuration hashes match the unchanged executed configuration, and
their upstream checkout was verified clean against the pinned commit.
