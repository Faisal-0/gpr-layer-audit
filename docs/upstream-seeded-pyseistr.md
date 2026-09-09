# Seeded packet challenger and pyseistr contract

Date: 2026-09-09. Decision: **reject the dense packet comparator for application
dispatch; retain its executable experiment and optional geometry adapter**.

The experiment is an independent all-sample min-sum chain. Immutable packets
from the two operating endpoint observations contribute spatially weighted
correlation, without a pointwise maximum over the template bank. The chain
retains every time sample, has a physical local slope penalty, preserves exact
seed samples, and uses finite-cost latent missing positions. Missing signal
never produces an accepted or provisional measurement. Exact forward/backward
min-marginals use the same objective as selection. The alternative at each
position must belong to a different signed lobe. These are uncalibrated objective
differences, not correctness probabilities.

The implementation is in `processing/seeded_challenger.py`; it is **not connected
to application dispatch**. `pick_seeded_packet(...)` accepts only radar, validity,
operating anchors, physical sampling, seed-only pulse/context, optional breaks
and explicit research settings. It returns `SeedConditionedPath`. No asphalt
behavior, tracker configuration, project format or runtime dependency was changed.

## Processed-road results

The frozen native observations in `benchmarks/seeded-evaluation-inputs.json` were
used. Mandiali short has 0.025 m sampling; Gujrat second uses native stride 4,
0.1 m. Both are historically used development roads. Three seeds per layer are
excluded. The unchanged scoring contract is the initial seed-only
`max(2 samples, selected signed-lobe width / 4)` tolerance **and** no intervening
zero/sign crossing. A numerical gap does not remove a reviewed observation from
the denominator. There is no claim of independent thickness accuracy.

| Road | Layer | Scored | Tensor accepted correct/accepted | Correct coverage | Correct proposals/scored | PWD accepted correct/accepted |
|---|---|---:|---:|---:|---:|---:|
| Mandiali short | Asphalt | 447 | 146/184 | 32.66% | 331/447 | 148/185 |
| Mandiali short | Base | 443 | 38/54 | 8.58% | 222/443 | 35/62 |
| Mandiali short | Subbase | 404 | 36/40 | 8.91% | 238/404 | 32/38 |
| Gujrat second | Asphalt | 2550 | 0/0 | 0% | 1421/2550 | not rerun |
| Gujrat second | Base | 1586 | 0/140 | 0% | 20/1586 | 0/140 |
| Gujrat second | Subbase | 1067 | 0/216 | 0% | 8/1067 | 0/216 |

The tensor Gujrat base and subbase median accepted timing errors are respectively
214 and 135.5 samples; all 356 accepted deep observations are on the wrong lobe.
The baseline has selected remote bright packets. More retained sample states and
physically expressed local geometry did not establish semantic identity. PWD
also fails, with only 22 correct base proposals and 3 correct subbase proposals.
Changing thresholds cannot repair this proposal failure. The branch was stopped
after the isolated geometry comparison; it was not given compensating
road-specific bounds or acceptance thresholds.

One final requested ablation added an explicit soft endpoint-time prior, with
Huber width equal to one initial selected-lobe width and weight 1.0. This is a
prior from the operating seeds; it does not manufacture observations. The same
acceptance settings yielded Mandiali base 42/47 and subbase 46/50 accepted
agreement. Gujrat base accepted nothing and subbase accepted 0/2 correctly.
Correct proposals increased from 20 to 412/1586 for Gujrat base, and from 8 to
112/1067 for subbase. This is still a proposal and useful-coverage failure, so
the soft guide does not rescue the comparator. Those artifacts use suffix
`-guide`; the gap/seed tests also run with this prior enabled.

Tensor inference for all three layers took approximately 0.4 s on Mandiali and
6.7 s on Gujrat under concurrent workstation load. PWD deep-layer Gujrat inference
took approximately 12.5 s. These are secondary diagnostics, not an improvement
claim. Detailed per-layer bracketed/tail counts, runtime, memory, input/config
hashes, accepted/provisional arrays and radar overlays are under
`exports/seeded-tracker/challenger-final`. Each experiment includes its source
snapshot. The existing overlay uses cyan references, magenta excluded seeds,
orange provisional traces and green accepted traces; green denotes the model's
decision and does **not** imply agreement. JSON contains per-observation errors.

## Pinned upstream inspection

- Repository: <https://github.com/aaspip/pyseistr>
- Revision: `a411c11c6ba0d10d9d939763c5941321b0b13fc9`.
- Declared package version: `0.0.4.4.2`; imported `__version__` is `0.1.0`.
- Inspected `LICENSE`, `setup.py`, `pyseistr/dip2d.py`, and the mask/dip wrapper in
  `pyseistr/src/dip_cfuns.c` at that revision.
- License: MIT, copyright (c) 2025 pyseistr. No upstream implementation was
  copied. The optional adapter imports an independently installed package.
- Context7 searches for pyseistr and HorizonTracker returned no matching
  libraries. The pinned source was used to establish behavior.

`dip2dc` expects **[time sample, trace]**, flattens arrays and masks in Fortran
order, and returns dip in **samples per trace**. The adapter transposes both
radar and masks, then converts slope by `dt_ns / dx_m` to ns/m. It validates
shapes and positive units. Invalid neighborhoods are withheld after estimation
using a conservative stencil mask. Upstream PWD interpolation/denoising routines
are not called, and estimated geometry cannot fill a missing measurement.

The C extension compiled successfully on the workstation with Python 3.13,
NumPy 2.5.2 and SciPy 1.18.1. The local wheel SHA-256 was
`820ba276eb11d0e15af30d019027e003da2829b06bba5a9179103b705ec42662`.
An analytic dipping wavelet with slope 0.2 samples/trace verifies positive sign,
axis orientation and physical conversion to 0.06 ns/m at dt=0.03 ns and
dx=0.1 m. Masked rows and their stencil neighborhood have zero support. These
are numerical contract tests, not road accuracy evidence.

The pure Python `dip2d` explicitly documents that masks are not implemented.
The adapter therefore rejects masked input to that mode. An unmasked C/Python
differential check was attempted on a 64-by-20 synthetic packet. **Pure Python
failed under NumPy 2.5.2** in `B5d`, where a singleton array from `dip[i1,i2]` is
assigned to a scalar array element (`ValueError: setting an array element with
a sequence`). C/Python numerical equivalence is consequently not established.
No local patch was made to upstream or silently substituted for its algorithm.

To reproduce the optional dependency in the ignored research directory:

```powershell
git clone https://github.com/aaspip/pyseistr.git exports/seeded-tracker/upstream/pyseistr
git -C exports/seeded-tracker/upstream/pyseistr checkout a411c11c6ba0d10d9d939763c5941321b0b13fc9
.venv/Scripts/python -m pip install --no-build-isolation --no-deps exports/seeded-tracker/upstream/pyseistr
.venv/Scripts/python -m pytest tests/test_seeded_challenger.py
.venv/Scripts/python scripts/experiment_seeded_challenger.py --case mandiali-short --output exports/seeded-tracker/challenger/mandiali.json
.venv/Scripts/python scripts/experiment_seeded_challenger.py --case gujrat-second --output exports/seeded-tracker/challenger/gujrat.json
```

For the isolated PWD comparison, pass `--config` containing
`{"geometry": "pyseistr"}`. The application does not require pyseistr; its
optional contract test skips when it is not installed.

## Correspondence sensitivity

The same experiment script supports frozen-source capture, exact graph replay
and an evaluation-only retained-route oracle. The Mandiali capture at backend
hash `623782cfe31511b86677838b5508d09fac4b5dbba5727f55842bd17aa79c00b7`
uses the default conventional config, retaining established asphalt behavior.
It has 438/443 correct base candidates and 352/404 correct subbase candidates.
Neither bracket of either layer has a fully reference-consistent endpoint route
after wrong known-row nodes are removed, even allowing existing non-local
links to skip observations as explicit gaps. This is stricter than a local-score
oracle that may retain forced wrong sections. Unknown rows remain unrestricted.

An exact graph replay reproduces all 24,238 base links and 21,376 subbase links.
Increasing retained distinct lobes from three to six and the near-best window
from 0.15 to 0.30 yields 24,271 and 21,477 links respectively. It restores none
of the four entirely reference-consistent endpoint routes. This narrow top-k
change is therefore not a solution to the observed bottleneck. The test showing
a fourth plausible lobe can be discarded remains a valid mechanism regression;
it does not establish that this mechanism dominates these roads.

```powershell
.venv/Scripts/python scripts/experiment_seeded_challenger.py --source-root exports/seeded-tracker/baseline-source/src --capture-graph --case mandiali-short --output exports/seeded-tracker/challenger/graph-mandiali-current-default
.venv/Scripts/python scripts/experiment_seeded_challenger.py --source-root exports/seeded-tracker/baseline-source/src --replay-graph exports/seeded-tracker/challenger/graph-mandiali-current-default --output exports/seeded-tracker/challenger/graph-mandiali-replay
.venv/Scripts/python scripts/experiment_seeded_challenger.py --source-root exports/seeded-tracker/baseline-source/src --replay-graph exports/seeded-tracker/challenger/graph-mandiali-current-default --retained-lobes 6 --near-best 0.3 --output exports/seeded-tracker/challenger/graph-mandiali-generous
.venv/Scripts/python scripts/experiment_seeded_challenger.py --diagnose-graph exports/seeded-tracker/challenger/graph-mandiali-generous --output exports/seeded-tracker/challenger/graph-mandiali-generous/diagnostic.json
```

The diagnostic oracle is explicitly separate from inference. It may only
consume reviewed observations after the captured seed-only graph exists.

A final full tracker run used a copied frozen source, six retained lobes, a
0.30 near-best window, and disabled hard phase/local-motion agreement rejection.
Polarity, physical displacement, cosine/DTW floors, seed barriers, the original
path objective and acceptance remained fixed. It used the same
`conventional-motion-calibrated-development.json` config and exact seeds as
`exports/seeded-tracker/baseline/mandiali-control.json`:

| Layer | Control accepted correct/accepted | Generous accepted correct/accepted | Correct proposals, control → generous |
|---|---:|---:|---:|
| Asphalt | 164/254 | 164/254 | 196 → 196 |
| Base | 4/4 | 4/4 | 110 → 119 |
| Subbase | 42/44 | 56/70 | 125 → 119 |

The subbase wrong accepted count rises from 2 to 14 and accepted agreement falls
from 95.45% to 80%. **Reject the combined pruning relaxation.** Added connectivity
does not justify this regression. The graph contains 90,561 base and 84,178
subbase links. Its strict reference-only oracle can recover 60/177 observations
on a first-base-bracket endpoint route; the second base bracket remains
infeasible. Subbase endpoint routes visit only 31/152 and 45/162 reviewed
observations. Skipped rows remain gaps. Those oracle results diagnose retained
graph limitations; none enter the full tracker run.

The generous full-run config differs from the default-config top-k-only
sensitivity above, so their raw link counts are not a controlled comparison of
phase relaxation alone. Full-run accuracy is compared against its matching
motion-calibrated control. Its complete artifacts and copied source are
`exports/seeded-tracker/challenger/graph-mandiali-generous-combined` and
`exports/seeded-tracker/challenger/generous-source/src`.

```powershell
.venv/Scripts/python scripts/experiment_seeded_challenger.py --source-root exports/seeded-tracker/baseline-source/src --make-generous-source --output exports/seeded-tracker/challenger/generous-source/src
.venv/Scripts/python scripts/experiment_seeded_challenger.py --source-root exports/seeded-tracker/challenger/generous-source/src --capture-graph --case mandiali-short --config benchmarks/conventional-motion-calibrated-development.json --output exports/seeded-tracker/challenger/graph-mandiali-generous-combined
```

The regression package has 13 passing tests: exhaustive min-marginal checks
including endpoint constraints and infeasible routes; exact non-extremum seed
retention; missing-signal and structural-break behavior; cancellation; analytic
geometry axes/units; optional upstream mask behavior; and a fourth-lobe pruning
counterexample. Test count is a numerical-contract check, not performance
evidence. Ruff passes on the three new Python files.

## License notice

MIT License. Copyright (c) 2025 pyseistr.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
