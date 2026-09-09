# Extremum-member states: executable experiment, reject for promotion

The one declared mechanism was implemented and run on both frozen Mandiali graph
controls. It improves timing precision on the lobe-mutual graph but does not
increase correct accepted coverage. It is **not promoted** and was not extended
to Gujrat or tuned after scoring. This is a working inference experiment, not a
finished base/subbase tracker.

| Graph | Layer | Before agreeing / accepted | After agreeing / accepted | Correct coverage before → after | Proposed agree before → after |
|---|---|---:|---:|---:|---:|
| Original mutual ranking | Base | 4 / 4 | 4 / 4 | 0.90% → 0.90% | 110 → 107 |
| Original mutual ranking | Subbase | 42 / 44 | 35 / 36 | 10.40% → 8.66% | 125 → 114 |
| Lobe-mutual | Base | 19 / 20 | 20 / 20 | 4.29% → 4.51% | 190 → 191 |
| Lobe-mutual | Subbase | 78 / 91 | 77 / 82 | 19.31% → 19.06% | 171 → 177 |

Denominators remain 443 base and 404 subbase reviewed nonseed observations.
The lobe-mutual graph has 97 correct accepted deep observations before and after;
wrong accepted observations decrease from 14 to 5. Subbase precision increases
from 85.71% to 93.90%, but 24 new correct acceptances accompany 25 lost correct
acceptances. Base row 18 is corrected without another base acceptance loss.
Subbase rows 11 and 323 become new accepted errors; rows 236, 263 and 348 remain
accepted errors. These are all included in the comparison and overlays.
The frozen scorer reports zero signed-lobe switches in these accepted errors;
that does not make their timing correct.

## Implemented contract

Each graph node keeps its original waveform-member timing, features, edge
geometry and individual correspondence quality. Before inference it also gets
a measured amplitude-extremum emission from its own signed waveform basin.
Secondary extrema within one signed lobe remain separate modes. Members exactly
at an interpeak minimum have ambiguous basin assignment and emit a gap; their
two neighboring peaks remain explicit candidates when eligible. No strongest
peak is selected across a whole signed lobe.

All six operating deep seeds were measured extrema with zero offset. The
experiment checks this condition using operating seeds only and preserves those
exact seeds and their waveform contexts. Nonzero/incompatible seed fiducials are
outside this vertical slice and raise an error rather than being silently reset.

Before the original 12-packet cap, packet membership is closed over its members'
own measured peaks, provided the original candidate union and masks allow them.
The original ranking and representative **formulas** remain; changed membership
can change their inputs and thus which packets survive. This clarification is
important: the pre-inference `contract.json` wording "Original ranking and member
retention remain" described those formulas, not identical rank order or node
membership. The saved executed source unambiguously implements closure before
ranking. The later wording clarification does not change any inference code or
result. No saved execution artifact has been overwritten to hide the wording.

After the cap, every retained shoulder has its own measured peak retained, or its
fiducial remains unresolved. Added peaks have separately computed features,
waveforms, links and support. A selected member's support belongs only to that
member and its unique basin. No quality is pooled across separate peaks.

Path scores keep the frozen member-coordinate objective: local scores integrated
over horizontal step, contracted internal correspondence penalties, and external
correspondence/gap penalties. The emitted state is a fixed node attribute during
inference. Exact forward/backward route margins are recomputed in that emitted
identity: distinct peaks compete even when close, same-basin members share the
same emitted event, and gaps compete explicitly. This does not reuse confidence
from an earlier sample path. Full emitted waveform-context validity is checked
before acceptance. All original numeric acceptance thresholds, ordering rules,
anomaly masks and frozen scoring tolerances remain.

This is a deliberately limited member-geometry model. It does not recompute
motion in extremum coordinates or learn nonzero seed offsets. A waveform basin
is not proof of physical reflector identity, and state support remains an
uncalibrated correspondence statistic. Wrong subbase rows 11, 236 and 263 still
receive large route margins, demonstrating this remaining limitation.

## Executed verification

`self-test.json` records 10 contracts, including exhaustive enumeration of all
five routes on a split/shoulder/gap DAG, selected-path optimality, exact emitted
mode max-marginals, valley and plateau behavior, masked peak closure, and chain
contraction objective parity.

`mandiali-short/objective-audit.json` independently reconstructs all four captured
graphs from their original member links. Every graph signature matches. All 16
seed intervals are feasible; the selected member-route objective equals an
independent DAG optimum within 2.3e-15. Replayed emitted-state margins equal every
frozen scored observation's recorded margin exactly (maximum error 0).

For lobe-mutual base, retained member nodes change 39,890 → 39,583 and links
158,424 → 156,654. For subbase, nodes change 42,282 → 48,180 and links
131,470 → 147,513. The base's consistent emitted acceptance changes the deeper
layer's bounds as the established sequential ordering pipeline requires, so the
subbase comparison includes that pipeline consequence. The table captures and
added/removed-node counts preserve this fact.

Selected lobe-mutual base has 312 measured members, 301 emitted extrema and 11
unresolved fiducials; subbase has 247 measured members, 240 emitted extrema and 7
unresolved fiducials. Respectively 159 and 111 measured members have nonzero
member-to-extremum offsets. These are explicit inferred state emissions, including
unknown states, rather than a cosmetic change to displayed accepted points.

Measured wall time is 80.25 s on the original graph and 88.41 s on lobe-mutual.
These are per-run observations with other work present, not a speed benchmark.
The frozen source already uses the verified compiled batch DTW dispatch; scalar
DTW, model configuration and every threshold were unchanged.

## Reproduce and inspect

```powershell
.venv/Scripts/python.exe scripts/experiment_extremum_states.py self-test
.venv/Scripts/python.exe scripts/experiment_extremum_states.py run --graph control --output exports/seeded-tracker/extremum-states/REPLAY-control
.venv/Scripts/python.exe scripts/experiment_extremum_states.py run --graph lobe-mutual --output exports/seeded-tracker/extremum-states/REPLAY-lobe-mutual
.venv/Scripts/python.exe scripts/experiment_extremum_states.py audit
.venv/Scripts/python.exe scripts/experiment_extremum_states.py summarize
```

Run refuses to overwrite a completed evaluation. The audit and summarize commands
read the preserved canonical runs; fresh replay directories can be compared with
those canonical evaluation observations. The executable files are
`scripts/experiment_extremum_states.py` and `scripts/extremum_state_model.py`.
Each run stores its exact script bytes, transformed executable functions,
pre-inference contract, radar-only graph inputs, member selections, input/seed/
configuration hashes, full source manifest and frozen scorer validation.
The source hash is
`e4b9c239079145ea74869629070252baa5cede65b3db31821600b83775d26361`.

`mandiali-short/comparison.json` lists all gains, losses and new accepted errors.
`mandiali-short/control-overlay.png` and `lobe-mutual-overlay.png` show all scored
reference observations, proposals and acceptances in the original native sample
coordinates. The lobe-mutual overlay was visually inspected after rendering.
The final artifact manifest records hashes of the evidence and current scripts.

References reach only the frozen scorer and subsequent diagnostics/plotting.
No production files or existing comparator scripts changed. These are previously
used development data; there is no untouched-road or thickness-accuracy claim.
