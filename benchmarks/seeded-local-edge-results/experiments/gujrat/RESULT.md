Frozen Gujrat local-patch edge transfer — reject product promotion
================================================================

The Mandiali-trained adjacent-patch CNN increases correct emitted proposals on
Gujrat, but it also emits many more wrong-lobe proposals and produces no correct
accepted observations. This transfer does not meet the tracker objective.

| Layer and arm | Correct proposals / fixed reviewed nonseed rows | Wrong signed-lobe proposals | Hidden proposals | Correct / accepted |
|---|---:|---:|---:|---:|
| Base, guided control | 302 / 1586 | 263 | 1001 | 0 / 0 |
| Base, local 2D NCC | 440 / 1586 | 369 | 750 | 0 / 0 |
| Base, local CNN | 515 / 1586 | 577 | 472 | 0 / 1 |
| Subbase, guided control | 88 / 1067 | 141 | 793 | 0 / 1 |
| Subbase, local 2D NCC | 88 / 1067 | 215 | 716 | 0 / 2 |
| Subbase, local CNN | 159 / 1067 | 293 | 550 | 0 / 2 |

Same-lobe timing-error proposals are reported separately in diagnosis/report.json:
base 20/27/22 and subbase 45/48/65 for control/NCC/CNN. The frozen tolerance remains
3.5 samples for base and 2.5 samples for subbase, plus the signed-lobe requirement.
At dt=0.029296875 ns these are 0.1025390625 ns and 0.0732421875 ns.
Unknown labels are excluded, not interpreted as absence. Operating seed rows are
excluded from every automatic accuracy count.

CNN correct/emitted proposal fractions decline from 302/585 (51.62%) to 515/1114
(46.23%) for base and from 88/274 (32.12%) to 159/517 (30.75%) for subbase. Correct
proposal counts alone therefore overstate the result. Paired CNN gains/losses
against the control are 222/9 base and 83/12 subbase. NCC gains/losses are 147/9
and 7/7. Every accepted error is in an extrapolated tail. The longest contiguous
wrong accepted reviewed footprint is 0.1 m in each arm with accepted errors.
Wrong-lobe observations are not independently identified semantic switch events.

Bracketed correct proposals are base 279/417/490 of 1352 and subbase 71/71/119 of
808; tail correct proposals are base 23/23/25 of 234 and subbase 17/17/40 of 259
(control/NCC/CNN). All arms have zero correct accepted coverage. The report also
contains 50 m block results; these are correlated development observations, not
independent road trials.

The post-inference diagnostic oracle finds agreeing candidates at 1582/1586 base
and 1052/1067 subbase observations. Keeping the same sample states, displacement
edges, surface and exact native anchors, maximum recoverable agreeing counts are
1456 and 912. Thus candidate/edge constraints lose 130 and 155 observations total;
objective selection remains the larger deficit. Oracle arrays are explicitly
diagnostic and were generated only after all deployment-style predictions and
frozen evaluation had completed. They are not inputs to any prediction.

Reproduction and provenance
---------------------------

From the repository root, the completed stages were:

```powershell
.venv/Scripts/python.exe scripts/experiment_local_patch_gujrat.py selftest
.venv/Scripts/python.exe scripts/experiment_local_patch_gujrat.py control
.venv/Scripts/python.exe scripts/experiment_local_patch_gujrat.py encode
.venv/Scripts/python.exe scripts/experiment_local_patch_gujrat.py edges
.venv/Scripts/python.exe scripts/experiment_local_patch_gujrat.py predict
.venv/Scripts/python.exe scripts/experiment_local_patch_gujrat.py evaluate
.venv/Scripts/python.exe exports/seeded-tracker/local-patch-edges-v1/gujrat/source/diagnose_local_patch_edges.py exports/seeded-tracker/local-patch-edges-v1/gujrat
```

Completed stages deliberately refuse overwrite. Preserve this directory and its
sibling gujrat-guide before any new experiment. Every stage checks the frozen
Jamshoro source/numerical dependencies, this wrapper's source hash, the original
model, training contract and fresh Gujrat guided control. Wrapper source and GUIDE
binding are recorded; the original Jamshoro runner and application src files were
not modified. Source snapshots are in both source directories.

Original three native seeds per interface come from
benchmarks/seeded-gujrat-second-seeds.json. Native trace/sample coordinates are
base (1280,226), (5584,226), (9888,249), subbase (1104,325), (5616,319), (9936,311).
Inference retains every fourth native trace, giving 2875 rows and 2874 links at
0.1 m; no seed is snapped. Physical road extent on that grid is 287.4 m.
Native dt=0.029296875 ns, dx=0.025 m, header time origin=-3 ns.

The fresh guide builds radar-only extrema/shoulder candidates plus exact seeds,
applies the original native patch usability mask and intersects packet support.
Pulse widths are estimated only from these operating seeds: base 14 samples and
subbase 10 samples. No DZX content is opened before evaluate. Previously frozen
aggregate coordinate audit metadata is copied, and the operating seed file is
read. Historical layer mapping was confirmed by the user.

The same 374,036 candidate encodings serve both layers. There are 2,182,268
supported pairs out of 2,182,385 eligible candidate pairs. The unchanged graph has
15 offsets (-7 through +7 samples) per link. NCC and CNN share the exact support
mask and unsupported cost 0.65. Native 21x65 two-channel patches, seed NCC unaries,
guide weight 1, geometry, movement bound and original acceptance gate are fixed.
Centered common-valid-pixel NCC uses cost 1-NCC; the learned timing head uses
2*(1-sigmoid(score)). Costs are multiplied by physical spacing in the same DP.
The classifier output and interval-normalized margin are uncalibrated.

The zero-edge control reproduces every original guided-control result array
exactly. Candidate arrays and exact seed values are identical in all three arms.
The adapter selftest rejects off-grid, out-of-range and nonintegral seeds.
Six direct numerical/graph tests passed in test_local_patch_edge_contract.py and
test_local_patch_graph_oracle.py; Ruff checks pass for the new wrapper.

Model SHA256:
`aac5a797f98eb4676927f5f088e2684d0a4689b0a9e210a5b79ac47a5f68575f`

Processed DZT SHA256:
`087182ae3b6fb0bd96a1a1b87a3483b490b02839a8a7d2bb9c3700b267bb2d26`

Evaluation DZX SHA256:
`ac9caed3599a66a1372a2c99194e4e534452fee2d78491d54b4e82aba1d54cb1`

The wrapper SHA256 is
`b8e789df1f78719643ec17a05063b5cf47258e0de66e1481452999dda8be47f8`.
The complete package/script/config/input/model hashes are in contract.json and
the guide contract. Prediction and stage hashes are in their respective manifests.

Measured stage runtimes: guide construction 13.64/13.84 s per layer, encoding
19.36 s, edges 89.30 s, each cached-evidence prediction 1.39–1.43 s. This excludes
fresh CNN evidence computation from the cached prediction latency. Observed peak
working set during edges was 754,864,128 bytes; the sampled process record is in
the sibling gujrat-edge-memory-observation.json. All stages were terminal at the
end of this transfer, with empty stderr logs.

Before/after radar overlays are diagnosis/layer2-comparison.png and
diagnosis/layer3-comparison.png. They distinguish accepted errors, unresolved
proposals, reviewed observations and operating seeds. No further correction
replay was added for this rejected mechanism. Available roads have development
history, and this report makes no untouched-road or physically measured thickness
claim. Retain the evidence and reusable wrapper; reject this CNN-edge
configuration for application promotion without a demonstrated selection and
acceptance improvement.
