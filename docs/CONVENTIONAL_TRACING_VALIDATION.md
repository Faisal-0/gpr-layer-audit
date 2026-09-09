# Conventional tracing: implementation and validation

Date: 2026-09-09. Status: **experimental; no production promotion**.

Subsequent correspondence repairs, guarded template updates, transfer failures
and runtime measurements are recorded in `CONVENTIONAL_CONTINUATION.md`. The
measurements below retain their original evaluated source snapshots.

The reference benchmark and experimental conventional backend repairs are
implemented; the full plan is not yet complete. The measured
deep-layer coverage does **not** meet the promotion requirement. Calibration
cannot yet freeze acceptance for the enabled deep interfaces. Held-out accuracy
evaluation has therefore not been run. Subbase remains experimental.

## Reproducible inputs and comparators

- `io/dzx.py` imports `LayerGroup` identity/properties and every `LayerWayPt`'s
  zero-based scan, stored sample, channel, interpretation-property code, recorded
  time, amplitude, depth and velocity. It retains DZX fingerprints, filenames,
  declared scan range and processing properties. RFP/target detections are not
  imported as reviewed layers. Missing observations remain unknown.
- `conventional_reference.py` audits paired DZT dimensions, bounds, duplicate
  observations, recorded amplitude, header-relative time and ordering conflicts.
  Analyst labels and coordinates are retained when disagreements occur.
- The supplied-data audit inventories 68 DZX files. Reviewed development traces
  match their paired processed DZT amplitudes exactly; their time residuals are
  consistent with decimal rounding. These checks establish stored-coordinate
  correspondence, not independent confirmation of every analyst interpretation.
- Initial support observations are selected near 10%, 50% and 90% of each
  annotated extent. Seed traces are excluded from scoring. Evaluation observations
  are passed only to the scorer, never to templates, corridors or correspondence.
- Mandiali acquisitions share one physical group, as do both Gujrat portions.
  Bahawalpur is the separate asphalt/base calibration group. Jhang, Pattoki,
  Daska, Burewala and Jamshoro are reserved from accuracy scoring in this phase.
  Earlier repository Daska/Pattoki development artifacts mean that an eventual
  claim of untouched physical roads needs an additional history/lineage audit.
- The pre-edit working tree, including the existing uncommitted hybrid work, was
  captured in `exports/conventional/frozen`. The frozen tracker, graph,
  preprocessing and calibration hashes are recorded in
  `benchmarks/conventional-frozen-comparators.json`. Benchmark input/scoring
  helpers are separate from those immutable comparator modules.

## Benchmark contract

The processed-input benchmark uses the paired processed amplitudes and the DZT
header time origin. For a header origin of -3 ns, sample zero is not the surface
reference: the caller resolves the corresponding stored time-zero sample. It
does not perform raw acquisition calibration on processed input.

Every comparator uses the same tolerance, `max(2 samples, seed-estimated selected
lobe width / 4)`. Agreement additionally requires the same signed lobe with no
intervening zero/sign crossing. Candidate retention, correspondence survival,
proposed agreement, accepted agreement, correct coverage, wrong-lobe picks and
correction-observation burden are separate metrics. Correction burden is a
disagreement proxy, not a measured count of analyst actions.

Long-road development runs use every 64th trace (1.6 m where native spacing is
0.025 m); the short Mandiali acquisition uses all traces. This is a coarse
full-extent development experiment, not a full native-resolution field release
test. Seed selection and scoring use the same retained grid for both comparators.
Runtime is tracker/feature inference time. Memory is the operating system's
process-lifetime peak resident set, including native array allocations; when
both comparators share a process, it is not isolated incremental memory.

`exports/conventional/verified-input` contains the corrected-input runs and
overlays. Earlier runs under the top-level `baseline`, `repaired`, and `final`
directories are exploratory/superseded because their input origin or scoring
tolerance differed. Do not combine them with the corrected-input scores.

## Backend changes

- Recognized 32-bit SIR-30 acquisition layouts mask the first two stored values
  before filtering. Recognition requires acquisition metadata, an acquisition
  filename and an unshifted header; unknown layouts remain unresolved. Processed
  padding is handled separately. Numerical extensions retain the original grid
  and cannot become measurements. Source files remain read-only.
- Boolean validity propagates through stacking, integer alignment, interpolation,
  feature scaling and candidate/matching context. Stack statistics exclude invalid
  contributors. Exported signal evidence includes masks and coordinate provenance.
- A shared physical pulse description separates lobe width, packet width,
  template context and displacement. Verified metadata is supported; otherwise
  only support-seed waveforms estimate widths, with recorded fallback use.
  Packet width currently uses three times the seeded lobe width as a proxy;
  it is not an independently measured packet boundary.
  Horizontal graph distances and costs use metres; pulse settings use nanoseconds.
- Seed-conditioned forward/backward DAG sweeps replace undirected identity
  propagation. Bracketed observations require both directions. Reciprocal DTW
  and geometry are independently computed, and available intermediate routes
  constrain direct links. Batched DTW is checked against the scalar reference.
- Each seed interval must honour both endpoints. Infeasible intervals preserve
  exact observations and remain unresolved. Up to three distinct hypotheses are
  retained, with missing observations kept as gaps. Selection and max-marginal
  acceptance use one distance-scaled objective. Competing routes are compared over
  their own disagreement intervals, including missing prefixes/suffixes and skips.
- Local candidate margins remain diagnostic. Acceptance uses observable signal,
  seed correspondence and feasible-path separation. Scores remain uncalibrated.
- Accepted upper picks impose hard ordering. Provisional upper hypotheses are
  retained as compatible alternatives instead of forcing a single provisional
  boundary. Accepted combinations cannot cross.
- Requests can occur close to existing seeds. They identify observable competing
  routes or the supported frontier of an incompatible interval. Priority records
  supported ambiguous length rather than claiming a calibrated correction saving.
- Seed/regime reference templates remain immutable. Neighbouring observed
  waveforms provide local drift correspondence. Mutating/adapting the reference
  template bank itself remains disabled while route acceptance is uncalibrated;
  ambiguous proposals never become reference templates.
- A bounded 96 MiB feature cache holds only seed-independent numerical features.
  Seed edits rebuild correspondence and path evidence. Global, fine, withheld
  and local dispatch receive the same conventional configuration. Project replay
  preserves that configuration. ML defaults to off, and `off` also suppresses
  explicitly supplied learned evidence.

The established tracker remains the application default. The usual experimental
hybrid configuration retains established asphalt tracing; the all-layer research
configuration exercises conventional hybrid asphalt without promotion.

## Corrected-input development findings

These compare the frozen hybrid with the repaired deep-layer configuration.
Percentages below exclude support seeds; a dash means no accepted observations.

| Acquisition | Interface | Candidate retention before → after | Accepted agreement before → after | Correct coverage before → after |
|---|---|---:|---:|---:|
| Mandiali 001 | Base | 54.5% → 98.0% | 20.0% → — | 0.4% → 0.0% |
| Mandiali 002, short | Base | 66.1% → 98.9% | 100% → — | 0.9% → 0.0% |
| Gujrat second portion | Base | 81.2% → 99.0% | — → 100% (3 picks) | 0.0% → 3.1% |
| Gujrat first portion | Base | 75.8% → 98.6% | 50.0% → — | 0.5% → 0.0% |
| Bahawalpur 001 | Base | 89.4% → 100% | 75.0% → — | 9.8% → 0.0% |
| Bahawalpur 002 | Base | 86.8% → 99.5% | 100% → — | 7.1% → 0.0% |

Subbase candidate retention improved on most development acquisitions, but the
strict endpoint-consistent configuration accepted no non-seed subbase observations
on these runs. This is a coverage failure, not evidence that subbase is absent.

Established asphalt output is unchanged between these two hybrid configurations.
Its accepted agreement varies considerably by acquisition and resolution. A
separate short-Mandiali all-layer experiment accepted 67 asphalt observations,
with 98.5% agreement and 14.8% correct coverage. The frozen hybrid accepted 254,
with 64.6% agreement and 36.7% correct coverage. Higher selective correctness alone
does not satisfy the required useful-coverage/correction improvement.

Across the six all-layer development/calibration acquisitions, the acceptance
calibration retained 753 asphalt observations at 99.867% agreement. This is
development calibration, not held-out correctness or a probability estimate.
Base and subbase still had insufficient correctly supported observations, so
the complete configuration remains unfrozen and no interface is promoted.

Short-Mandiali staged experiments are saved in
`exports/conventional/verified-ablation`. They use the corrected processed origin
and the same seed-only scoring tolerance:

| Enabled repairs | Base candidate retention | Base correspondence survival | Subbase candidate retention | Subbase accepted observations |
|---|---:|---:|---:|---:|
| Signal validity | 62.8% | 7.4% | 73.3% | 9 |
| Plus physical pulse/context | 98.9% | 32.3% | 87.1% | 2 |
| Plus directed correspondence | 98.9% | 2.7% | 87.1% | 0 |
| Plus complete paths/acceptance | 98.9% | 2.7% | 87.1% | 0 |

Base accepted no non-seed observations in these four runs. The accepted subbase
observations in the first two runs agreed with their references, but represented
only 2.2% and 0.5% correct coverage. These experiments expose the survival loss;
they do not establish that stricter correspondence improved total accuracy.
Signal, pulse and directed stages captured different evolving backend snapshots,
so those comparisons are exploratory rather than isolated causal ablations.
Directed and complete-path stages share the same backend fingerprint. Each
configuration and backend hash is preserved in the compact results summary.

The current bottleneck is correspondence/complete-path survival after candidate
retention, rather than a demonstrated need for more candidate generators or ML.
The benchmark exports failed intervals and observable recapture requests so that
additional analyst observations can test that diagnosis. The supplied references
are not relabelled to fit the tracker.

## Validation and unresolved gates

- The final non-ML regression suite passed **269 tests in 168.14 s** after the
  evidence and ordering fixes. Ruff passed on all new conventional modules and
  modified integration modules. All eight frozen-comparator hashes still match.
  New tests cover DZX interpretation properties, boundary contamination, masked
  stacking/resampling, coordinate round trips, physical pulse resolution,
  withholding isolation, directed-junction leakage, incompatible endpoints,
  uncertain upper hypotheses, scalar/batched DTW equivalence, processed header
  origin, calibration split protection and promotion gates. Existing tests cover
  waveform drift, adjacent lobes, gaps, breaks, clicks, cancellation and exports.
- A real `retrack_segment` benchmark on a deterministic 120 m synthetic road
  preserved every pick outside [35, 85] m. Project settings survived reopening.
  Full-road inference took 5.21 s; warm correction with a seed edit took **10.24 s**,
  narrowly missing the 10 s target. Peak process memory was about 254 MiB. This ran
  alongside development jobs; it is not an isolated real-road latency claim.
- Corrected full-extent coarse development tracker runs took roughly 12–105 s,
  depending on acquisition and comparator. The highest process peak was about
  831 MiB. Per-run values and scope are retained in the JSON summary.
- Radar-only raw/processed registration of short Mandiali was rejected: the
  proposed global sample shift had 19 samples of variation and unresolved trace
  offset. No layer picks fitted that transform, and no raw accuracy score is
  reported from it. Nonuniform processing-stage mappings remain necessary.
- Calibration on the five consistent coarse development/calibration runs found
  insufficient correctly supported observations for freezing the enabled deep
  interfaces. No held-out settings or acceptance probability are claimed.
- The 95% accepted/lobe agreement, three independent adequately observed road
  groups, and coverage/correction improvement gates remain unmet. Subbase has the
  additional independent-evidence restriction. No layer was promoted.

Remaining work includes calibrated template-bank adaptation if supported by new
evidence, verified nonuniform raw mappings, native/0.4 m field evaluation of the
final frozen backend, and held-out evaluation after calibration succeeds. No ML
training or field ML inference was used for these experiments.

The final evidence-provenance and authoritative-ordering integration fixes were
unit/integration tested after the recorded road snapshots. Road metrics identify
those snapshots; they must not be treated as validation of a later source hash.

## Commands and artifacts

```powershell
uv run gpr-layer-audit conventional audit "gpr including proc files" --output exports/conventional/reference-audit.json
uv run gpr-layer-audit conventional evaluate "path/to/reviewed.DZX" --mode processed --config benchmarks/conventional-v2.json --output exports/conventional/evaluation.json
uv run gpr-layer-audit conventional register "path/to/processed.DZT" "path/to/raw.DZT" --output exports/conventional/registration.json
uv run gpr-layer-audit conventional evaluate "path/to/reviewed.DZX" --mode raw --raw "path/to/raw.DZT" --mapping exports/conventional/registration-transform.json --config benchmarks/conventional-v2.json --output exports/conventional/raw-evaluation.json
uv run python scripts/evaluate_conventional_development.py --stage repaired --stride 64 --output-root exports/conventional/verified-input
uv run python scripts/benchmark_conventional_local.py
uv run pytest --ignore=tests/test_hybrid_ml.py
```

`conventional calibrate` accepts development/calibration evaluation JSON files,
`--config`, `--output`, and optional `--freeze`. It rejects held-out input, mixed
backend/settings provenance and unsupported freeze attempts. Held-out evaluation
requires the matching frozen configuration and backend fingerprints. An explicit
`--layer-map` JSON can map source layer numbers to interface orders without
changing the imported analyst labels.

The compact, versioned measurements are in
`benchmarks/conventional-results-2026-09-09.json`. Full reference audits, per-layer
observations, arrays and before/after radar overlays are under
`exports/conventional/verified-input`; snapshot packages preserve the evaluated
backend versions. `benchmarks/conventional-protocol.json` records the gates and
keeps `settings_frozen` false.
