# Local learned edges and measured corrections

The trained local-patch model increases correct proposal counts on Jamshoro and
Gujrat, but also produces more wrong proposals. It fails the accepted-precision
and useful-coverage requirements, including after actual corrections. **Reject
application promotion.** The functioning native processed workflow and established
asphalt/default dispatch remain unchanged at application source hash
`4a7c2ef3f536f30bdb2414a2dc06b5ca53af52635484e47f6221ed332392e6a6`.

## Frozen mechanism and numerical contracts

The existing 36,602-parameter PatchMatcher trains for 12 fixed epochs using only
Mandiali: 55,695 local pairs and 6,204 source patches. A reviewed source observation
lies exactly 0.1 m before each target. Initial operating seed rows are excluded
from training targets; unknown source/target labels are excluded. Complete audits
reconstruct all source patches, target patches, coordinates and labels exactly.
Model SHA256 is
`aac5a797f98eb4676927f5f088e2684d0a4689b0a9e210a5b79ac47a5f68575f`.

Both graph arms use the same native 21x65 amplitude/validity patches, candidate
bank, displacement edges and support mask. Each lateral A-scan patch is normalized
by its temporal L2 norm. Classical NCC centers over common valid pixels in float64;
both arms require at least 1,024/1,365 common pixels, usable centers and nonzero
common variance. The network compares the following target with the preceding
source, matching training orientation. Cost per metre is `1 - NCC` or
`2 * (1 - timing_score)`; unsupported pairs retain cost 0.65. Neither classifier
scores nor objective margins are calibrated correctness probabilities.

All sample states and the original seven-sample displacement bound remain.
Original seed NCC unaries, unit endpoint guide, geometry, masks and acceptance
thresholds are unchanged. Exact forward and backward inference includes the new
edges. Both native-seed controls reproduce every original result array; candidate
arrays and exact seeds match across arms. Independent arithmetic tests cover
common-mask centering, support thresholds, asymmetric comparison direction,
physical scaling and all constrained min-marginals on a small enumerated graph.

References are opened only after encoding, edge construction and prediction.
Jamshoro has historical development use and did not train this model. Gujrat is
development transfer, also historically used. The user confirmed Mandiali/Gujrat
Layer 1/2/3 as asphalt/base/subbase and has no unused annotated road. These are
interpretation results, not physical thickness measurements.

## Full-road results

Denominators are reviewed retained-grid observations excluding the identical
three native seed rows per enabled layer. Unknown labels are unscored. All arms
use native 0.025 m data with stride four, without stacking or snapping.

| Road/layer | Denominator | Correct proposals: control / NCC / CNN | CNN correct / accepted |
|---|---:|---:|---:|
| Jamshoro Layer 2 | 4,115 | 855 / 1,128 / 2,188 | 0 / 0 |
| Jamshoro Layer 3 | 2,242 | 343 / 324 / 882 | 0 / 0 |
| Gujrat base | 1,586 | 302 / 440 / 515 | 0 / 1 |
| Gujrat subbase | 1,067 | 88 / 88 / 159 | 0 / 2 |

More correct proposals do not establish better precision. On Gujrat, correct among
emitted proposals falls from 302/585 (51.62%) to 515/1,114 (46.23%) for base, and
88/274 (32.12%) to 159/517 (30.75%) for subbase. Wrong-signed-lobe proposals increase
263 to 577 and 141 to 293. Every initial accepted CNN error is in an extrapolated
tail. The legacy interval-normalized margin still causes excessive long-bracket
abstention; its threshold has not been lowered.

The diagnostic oracle replaces local scores with reviewed-agreement rewards while
retaining the actual graph and native seed constraints. Maximum recoverable counts
are 3,782/4,115 and 2,034/2,242 on Jamshoro; 1,456/1,586 and 912/1,067 on Gujrat.
Candidate retention is 4,093, 2,212, 1,582 and 1,052 respectively. Thus displacement
restrictions lose some jointly recoverable events, and objective selection loses
substantially more. Oracle arrays are explicitly diagnostic and never feed tracking.

One isolated hypothesis removed the constant endpoint guide only in tails.
Bracketed result arrays stayed exact. Jamshoro correct proposals fell 2,188 to
1,920 and 882 to 732, and ten wrong base observations became accepted. Reject this
change without tuning or transfer.

## Actual seed–request–correct–retrack loop

Gujrat replay starts with the same three observations per deep layer. The existing
request policy, exact-answer boundary, local merger and ordering guard execute
after every supplied answer. The trained model, edge cache, pulse, candidate mask
and thresholds remain frozen. Every initial fit reproduces the frozen CNN arrays.
Off-bank answers retain their exact seed coordinate and are explicitly recorded;
the fixed edge cache does not invent new measurement evidence for them.

| Policy/step | Base correct / accepted | Subbase correct / accepted |
|---|---:|---:|
| Initial | 0 / 1 | 0 / 2 |
| Active, request 1 | 0 / 1 | 0 / 2 |
| Active, request 2 | 55 / 59 | 0 / 2 |
| Active, request 3 | 76 / 92 | 0 / 2 |
| Active, request 4 | 103 / 128 | 0 / 2 |
| Midpoint, request 4 | 0 / 1 | 0 / 2 |

The active policy requests base at 191.3, 232.4, 133.8 and 239.1 m: three
corrections and one confirmation. Midpoint requests subbase at 84.0 and 194.4 m,
where exact reviewed answers are unavailable, then a base correction at 85.8 m
and confirmation at 193.4 m. Unavailable answers consume requests without invented
labels. Every answer is excluded from automatic scoring.

Four requests over 287.4 m equal 13.92 requests/km, beyond the initial six seeds.
With fixed initial reviewed pools 1,586/1,067, final active base correct coverage
is 6.49%, wrong coverage 1.58%, and accepted agreement 80.47%. Subbase has no correct
accepted coverage. The longest contiguous wrong accepted footprint is 0.5 m for
base and 0.1 m for subbase. Base has 22 wrong-signed-lobe accepted observations and
three same-lobe timing errors. These are observation errors, **not counted semantic
reflector-switch events**. Unknown labels do not bridge the contiguous-span count.

Every saved accepted-sample, provisional-sample and visibility array is verified
unchanged outside the affected layer and inclusive +/-25 m window after each
action. Answers retain their native coordinates. Cached full-context regeneration
takes 2.24–2.68 s per answered action; the merger limits the displayed change.
These are research replays through the real correction functions, not a claim
that this rejected model has been integrated into GUI dispatch.

## Reproduction and artifacts

`benchmarks/seeded-local-edge-results/index.json` preserves the model, source
captures, all scored observations, controls, predictions, oracle readouts, radar
overlays and actual action curves. Raw radar, the large training array, embedding
banks and edge tensors remain local; hashes and native paths are recorded.
Jamshoro encoding takes 73.21 s, edges 383.18 s, and each cached solve about 5.3 s.
Gujrat encoding takes 19.36 s, edges 89.30 s, and each solve about 1.4 s. Observed
edge-stage process high-water memory is about 1.6 GB and 0.75 GB respectively.

Run with the recorded Windows environment and local frozen baseline artifacts:

```powershell
.venv/Scripts/python.exe scripts/experiment_local_patch_pairs.py prepare --output exports/local-pairs-reproduction
.venv/Scripts/python.exe scripts/experiment_local_patch_pairs.py train --output exports/local-pairs-reproduction
.venv/Scripts/python.exe scripts/experiment_local_patch_edges.py encode --output exports/local-edges-reproduction
.venv/Scripts/python.exe scripts/experiment_local_patch_edges.py edges --output exports/local-edges-reproduction
.venv/Scripts/python.exe scripts/experiment_local_patch_edges.py predict --output exports/local-edges-reproduction
.venv/Scripts/python.exe scripts/experiment_local_patch_edges.py evaluate --output exports/local-edges-reproduction
.venv/Scripts/python.exe scripts/diagnose_local_patch_edges.py exports/local-edges-reproduction
.venv/Scripts/python.exe scripts/experiment_local_edge_replay.py exports/seeded-tracker/local-patch-edges-v1/gujrat exports/local-edge-replay-reproduction
.venv/Scripts/python.exe scripts/summarize_local_edge_replay.py exports/local-edge-replay-reproduction
.venv/Scripts/python.exe scripts/package_local_edge_results.py --verify --originals
```

The edge runner deliberately uses the audited original model and frozen Jamshoro
guide controls; it does not silently select a new training output. The Gujrat
wrapper's `RESULT.md` contains its sequential control/encode/edges/predict/evaluate
commands and explains its frozen output paths. Every stage refuses overwrite.
Omit `--originals` for package-only byte verification without local exports.

The product objective remains unmet. No acceptance or physical-accuracy claim is
supported by these experiments, and all previously published comparators remain
preserved.

Validation: 522 tests pass, with one optional upstream test skipped. This includes
nine new training-pair, masked edge, exact-graph and tail-objective regressions.
The 114-file evidence package verifies against its original artifacts.
