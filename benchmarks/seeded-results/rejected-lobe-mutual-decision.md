# Paired-edge gate loss: reject the isolated mechanism

The diagnosed ranking loss is real. It is not the only cause of wrong reflector
selection, and repairing it alone increases wrong accepted length. Keep the
executable diagnostic and the negative experiment; do not promote the mechanism.

The numerical source is frozen at `source-e4b9c239079145ea/src`. Both fresh controls
reproduce the frozen deep-layer acceptance counts exactly. The matching configuration,
processed DZT/DZX hashes, native observations and signed-lobe/time tolerance are
unchanged. All roads have previous development use. No physical thickness claim is made.

## Specific mechanism

`_ranked_lobe_matches` suppresses duplicate timings when ranking target lobes, but
forward and reverse contenders are intersected at exact candidate coordinates.
Two directions can retain the same pair of endpoint signed-lobe groups using different
timing representatives. The correct timing edge can then disappear at mutual ranking.
Post-inference inspection finds 123 base/56 subbase reviewed row pairs on Mandiali,
and 258 base/180 subbase row pairs on Gujrat, with this specific loss. These counts
include operating anchors as diagnostic context and exclude graph edges crossing
an authoritative observation. They are not coverage gains or independent trials.

The single tested change lifts reciprocal ranking through the existing `same_left`
and `same_right` relations, retaining admissible timing alternatives in an already
reciprocal signed-lobe pair. It changes no motion, polarity, phase, cosine, packet-DTW,
slip, route-consistency, objective or acceptance threshold. Four executable numerical
contracts verify timing recovery, unchanged distinct-lobe behavior, retained hard
gates and no edges from nonfinite scores.

## Actual downstream result

Counts exclude initial operating observations. Coverage denominator is the common
eligible reviewed nonseed cohort: Mandiali base443, subbase404. Unknown labels remain
unscored. The selected base can change actual downstream subbase candidate bounds;
the base candidate table remains unchanged, while retained correct subbase candidates
change352→353. This is the real application's existing upper-path coupling.

| Mandiali | Control | Lobe mutual |
|---|---:|---:|
| Base agreeing/accepted | 4/4 | 19/20 |
| Base accepted agreement | 100% | 95% |
| Base correct coverage | 0.903% | 4.289% |
| Base wrong coverage | 0% | 0.226% |
| Base correct proposals | 110 | 190 |
| Base correct-only oracle | 78 | 200 |
| Subbase agreeing/accepted | 42/44 | 78/91 |
| Subbase accepted agreement | 95.455% | 85.714% |
| Subbase correct coverage | 10.396% | 19.307% |
| Subbase wrong coverage | 0.495% | 3.218% |
| Subbase correct proposals | 125 | 171 |
| Subbase correct-only oracle | 100 | 167 |

The oracle forbids known wrong nodes and zeroes correspondence/geometric costs on
the retained contracted graph. It uses reviewed labels only after inference and is
ineligible for production. The improved oracle demonstrates restored feasibility;
it does not validate the actual objective or acceptance.

Wrong accepted observed runs increase0→1 for base and2→12 for subbase. Longest
contiguous wrong accepted observed span increases0→0.025m and0.025→0.05m respectively.
Those short spans reflect the 11.45m crop, sparse accepted output and label continuity;
they are not proof of a low semantic switch rate. All timing, lobe, bracket/tail,
gate-loss and runtime fields are retained in `comparison.json`.

The experiment fails the subbase95% minimum gate and adds accepted error. Base95%
on19 agreeing observations has only4.29% correct coverage and no independent road
evidence. A Gujrat branch run was therefore not launched. This is a rejected mechanism,
not a tuned default or a successful semi-automatic tracker.

## Gujrat's actual first losses

Each row below counts neighboring reviewed/anchor row pairs with at least one
correct admissible candidate at both ends. If several correct timing pairs exist,
the category is the last gate reached by any of them. The graph-distance1 counts
are a subset of the detailed all-distance reports.

| First family loss at adjacent retained rows | Base | Subbase |
|---|---:|---:|
| At least one correct edge retained | 1202 | 505 |
| Forward displacement budget | 73 | 127 |
| Reverse displacement budget | 30 | 36 |
| Polarity | 7 | 0 |
| Phase | 6 | 38 |
| Local motion | 50 | 83 |
| Mutual ranking | 92 | 41 |
| Cosine | 84 | 33 |
| Reciprocal packet DTW | 22 | 18 |

The displacement gates include the trusted motion prediction when available and
the documented zero-prediction fallback otherwise. Polarity and phase are not the
dominant base losses. Removing those gates cannot address the other failures.
Actual control candidate retention is1581/1586 base and973/1067 subbase; the
correct-only oracle is629 and19 respectively. The earlier geometry-objective capture
had971 correct subbase candidates because its different base winner changed bounds.

Every `layer-N-diagnostic/report.json` contains concrete failed row/sample checkpoints,
phase/polarity pairs and cosine values. `paired-edges.npz` stores every inspected
candidate pair with first-loss IDs, true DTW/slip results when that stage was reached,
and the specific same-lobe ranking counterfactual. NaN DTW means not evaluated after
an earlier rejection; it is never treated as agreement. Candidate/edge creation and
selection receive radar and operating seeds only. The observer runs afterward,
and graph signatures match exactly for all six observed graphs.

## Artifacts and runtime

- `decision.json`: concise machine-readable decision, metrics, oracles and hashes.
- `comparison.json`: frozen scorer's whole-road and bracket/tail metrics.
- `mandiali-control-vs-lobe-mutual.png`: accepted agreeing, accepted wrong,
  unresolved, reviewed observations and unchanged initial seeds.
- `*/control/evaluation.json`: fresh real-road controls.
- `mandiali-short/lobe-mutual/evaluation.json`: isolated branch.
- Each graph capture includes candidate features, radar-derived motion/context,
  initial anchors, pre/post route-filter links and exact output signature.
- `provenance-note.json`: wrapper-hash timing caveat; numerical snapshot and
  saved transformed graph source are independently recorded.

Whole instrumented evaluation wall times: Mandiali control68.73s, branch86.86s,
Gujrat control394.95s. Peak process resident memory:554.3MB,604.8MB,2620.8MB.
These include graph capture serialization and concurrent process scheduling;
they are not interactive latency measurements. Graph-only capture times on Mandiali
are16.44/15.88s control and27.03/23.59s branch for base/subbase respectively.

## Reproduce

Run from the repository root with `.venv/Scripts/python.exe`. Existing completed
controls and comparison artifacts must be preserved; select a fresh output directory
for new inference. The included snapshot uses the independently bitwise-verified
compiled DTW primitive. The frozen numerical reference remains in the package.

```powershell
$source = 'exports/seeded-tracker/edge-gate-loss/source-e4b9c239079145ea/src'
$experiment = 'scripts/experiment_edge_gate_loss.py'
.venv/Scripts/python.exe $experiment self-test --source $source
.venv/Scripts/python.exe $experiment run --source $source --case mandiali-short --output exports/seeded-tracker/edge-gate-loss/reproduce-control
.venv/Scripts/python.exe $experiment diagnose --source $source --case mandiali-short --input exports/seeded-tracker/edge-gate-loss/reproduce-control
.venv/Scripts/python.exe $experiment oracle --source $source --case mandiali-short --input exports/seeded-tracker/edge-gate-loss/reproduce-control
.venv/Scripts/python.exe $experiment run --source $source --case mandiali-short --mechanism lobe-mutual --output exports/seeded-tracker/edge-gate-loss/reproduce-lobe-mutual
.venv/Scripts/python.exe $experiment diagnose --source $source --case mandiali-short --mechanism lobe-mutual --input exports/seeded-tracker/edge-gate-loss/reproduce-lobe-mutual
.venv/Scripts/python.exe $experiment oracle --source $source --case mandiali-short --mechanism lobe-mutual --input exports/seeded-tracker/edge-gate-loss/reproduce-lobe-mutual
```

Use `--case gujrat-second` for the Gujrat control and its gate audit. No retained graph
or diagnostic pickle is accepted from an external source: these files are trusted
local artifacts created by this script. Regenerate them from the hashed radar if
their origin is unknown. Production code was not edited by this experiment.
