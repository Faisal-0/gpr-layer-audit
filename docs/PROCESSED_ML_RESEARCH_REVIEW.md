# Independent review of the processed-label learning campaign

This review distinguishes verified implementation behavior, published evidence,
and risks that still require an experiment. It does not change the frozen scorer
or establish product reliability. External checks were performed on 2026-09-10;
download responses and SHA-256 fingerprints are in
`C:/Users/faisa/Desktop/gpr_campaign_artifacts/20260910/review/`.

## Verified starting limitations

| Mechanism | Verified behavior | Consequence for the new experiment |
|---|---|---|
| Remote seed conditioning | `ml/features.py::make_inputs` takes the pointwise maximum of nonnegative correlations across all seed waveforms. Positive/negative click maps contain individual pixels. | A window without clicks receives amplitude, envelope, layer number, and a collapsed correlation field. It cannot recover which seed supplied a score, seed count, seed distance, or the association between waveform and seed position. It still receives some remote waveform information, so calling it wholly unconditional would be inaccurate. |
| Support selection | `ml/training.py::select_support_labels` sorts by source hash and trace, then picks three evenly spaced indices per physical group. It is called again each epoch but chooses the same observations. | Shuffling chunks and changing augmentation RNG do not create new support/query episodes. Supports can span different acquisitions/variants in the same road group. |
| Sampling/context | `train_model` sets `dx = 0.4`; `build_dataset` also defaults to 0.4 m stacking. Training crops are 256 by 512. The nominal horizontal span is 102 m between first/last crop centers, but original fine lateral samples have already been stacked. | Large pixel crops are not proof that native waveform information survives. Compare retained native temporal samples and a finer lateral grid. Log both physical and pixel extents. |
| Automatically generated negatives | `competing_seed_clicks` chooses the strongest absolute-amplitude peak 6–24 samples from a positive observation. Both training and inference call it. | These are heuristic competing-lobe hints, not verified human negatives. Their semantic validity is unmeasured; do not relabel them authoritative negatives. |
| Processed/raw compatibility | `annotation_reason` deliberately rejects processed coordinates; `build_dataset` calibrates raw scans and shifts labels using the detected surface. | Reusing processed picks through this loader would violate its coordinate contract. A separate audited processed adapter is warranted. |
| Eligibility | `audit_dataset` requires at least three training groups, validation labels, and 96 training labels per layer. `train_model` repeats the restriction and enforces 1–50 epochs. | Three-group subbase leave-one-group-out cannot pass. A separate experimental entry point can allow two training groups without weakening production promotion or physical-group isolation. |
| Signal normalization | `make_inputs` scales each trace by its 99th percentile absolute amplitude over depth, clips, and derives the envelope from that view. | Strong shallow energy can dominate the scale for weak deep returns. This is a plausible information loss, not a measured failure on the new data. Preserve a signed signal view and compare normalization on identical folds. |
| Spatial/validity bookkeeping | Legacy `resample_grid` uses ceiling dimensions and zero padding, with no returned validity mask. Rounded labels can collide after coarse resampling; `make_targets` overwrites a target row when multiple labels map there. | The processed adapter must explicitly represent valid pixels and label collisions. Do not infer that these edge cases occurred in every legacy acquisition. |

The old boundary loss already ignores unreviewed traces and treats `not_visible`
separately from `absent`. Those sound masking distinctions should be preserved.
The new dataset must not invent either visibility class from missing DZX picks.

A small NumPy/SciPy source probe is preserved as
`review/legacy-source-behavior-probe.json`, with the four relevant source hashes.
Permuting or duplicating the template bank changes the old input by exactly zero;
the chosen remote crop has no nonzero click pixels. Resampling an all-ones
4-by-4 grid from spacing 1 to spacing 2 returns a 3-by-3 grid whose final row and
column are zero. These verify implementation properties, not real-road error
rates. The probe also reproduces the previously published interval arithmetic.

## Interval normalization is an existing measured limitation

`processing/seeded_challenger.py` divides the exact alternative-path cost increase
at a row by the full conditioned interval length:

`margin = alternative_cost_increase / ((right - left + 1) * dx_m)`.

`docs/DENSE_TRACKER_EXPERIMENTS.md` already records the controlled example:
the same local increase 0.23 yields 0.0227723 for 101 rows at 0.1 m spacing and
0.00229770 for 1,001 rows. The fixed 0.02 gate consequently changes its decision
because unrelated interval length increased. This is not a newly discovered
scorer defect. It is an uncalibrated acceptance statistic in a research decoder.

Removing that denominator or lowering the threshold cannot establish correct
identity: previous experiments contain many wrong proposals behind the gate.
The useful comparison is dense evidence, decoded proposal, then acceptance on
one unchanged reviewed non-seed denominator. Select any operating threshold on
training-road inner validation and report the whole precision/coverage curve.

## Difference from mechanisms already tried

The preceding 36,602-parameter local matcher used 21-by-65 native patches and
Mandiali examples separated by 0.1 m. Its learned outputs were inserted into
the existing candidate/displacement graph. It improved some proposal counts,
increased wrong proposals, and failed useful accepted coverage, including real
correction replay. Dense seeded DP, signed NCC edges, changed peak/gap states,
geometry costs, and constant-guide removal also have preserved negative results.

The proposed primary experiment is materially different only if it actually
trains a dense depth predictor on the eligible processed training roads,
retains individual remote seed identity and wider context, randomizes sparse
support episodes, and measures its evidence before the old graph. Merely
renaming the U-Net or enabling its loader would not supply that distinction.

## Selective original-source checks

### Peters, Granek, Haber

[Original paper](https://arxiv.org/html/1812.11092v1), accessible HTTP 200.
The abstract explicitly states: “Training uses all data without reserving some
for validation. Only the labels are split into training/testing.” Section 4
restricts loss/gradient contributions to known label pixels while using the full
image forward pass. Section 3 motivates multi-resolution context; the paper
compares horizon interpolation and extrapolation inside its seismic setting.

Adoptable ideas are large-context dense prediction and loss restricted to known
labels. Its fitting setup is transductive with respect to target imagery and is
not evidence of unseen-road generalization. Gaussian target widths encode
uncertainty in that paper; they do not justify widening this project's native
agreement tolerance. No paper accuracy is transferred to this benchmark.

### IRHMapNet

[Software record](https://doi.org/10.5281/zenodo.15111993) and
[dataset record](https://doi.org/10.5281/zenodo.13985741) both returned HTTP 200
through the original Zenodo API and declare CC BY 4.0. The software archive
`IRHMapNet_v2.0.zip` is 40,448 bytes, SHA-256
`009572d4e9ad2639e8a6ae67776009d4a73c3220a05aa9288cd8f18b3812c59e`.
It contains TensorFlow/Keras architecture/training utilities and no weights or
standalone LICENSE file. The original
[GitHub repository](https://github.com/Ham-Moqadam/IRHMapNet) is currently
accessible; GitHub reports no detected repository license. Attribute the
specific licensed Zenodo release rather than assuming all repository revisions
have an independently verified license.

The archive README describes fully mapped hand-labelled data combined with
image processing/thresholding and inferred layer slopes. Dataset metadata
describes 600 paired 512-by-512 CSV radargrams/binary masks: one means any IRH,
zero background or ice. The data archive is 563,856,363 bytes and was not
downloaded. This annotation scheme is not incomplete selected-interface
supervision and should not be copied onto missing pavement labels.

The inspected released `utils/data_loader.py` contains literal unresolved Git
merge conflict markers and a random patch `train_test_split`; no physical-road
group split is supplied. This archive is not a ready execution dependency.
The current repository already cites IRHMapNet for its U-Net. A new use would
need a controlled pretraining comparison or a concrete context/loss idea; another
generic U-Net citation adds no new experimental evidence.

The [journal DOI](https://doi.org/10.1029/2024JH000493) landing request returned
403. Original Crossref metadata verifies its title, June 2025 publication and
CC BY 4.0 version-of-record license. Full journal methods were therefore not
independently read in this check; detailed statements above are from the
accessible released code/README and dataset metadata.

### RITM

[Original paper](https://arxiv.org/html/2102.06583v1), accessible HTTP 200.
Sections 3.2–3.3 describe random initial clicks followed by iterative corrections
from current prediction errors, with the previous predicted mask as input.
Empty previous masks are used for first interactions. This motivates a
correction-aware challenger with self-generated prior predictions and preservation
of correct neighboring work.

The paper-linked [saic-vul repository](https://github.com/saic-vul/ritm_interactive_segmentation)
currently returns 404 through the GitHub API; the alternate SamsungLabs path
also returns 404. No usable upstream code, weights, or code license was verified
here. The idea can be independently implemented without that dependency.

The paper's evaluation click simulator chooses the largest ground-truth error
region. That is an analyst simulator, not an autonomous request policy with
hidden answers. It must not be copied into this campaign's test-time acquisition
policy. Ground-truth-guided corrective locations are appropriate only during
training or as an explicitly privileged diagnostic comparator.

## Current PyTorch API verification

Used `find-docs` at `C:/Users/faisa/.agents/skills/find-docs/SKILL.md` through
`npx --yes ctx7@latest`, with no global package installation. Commands:

```powershell
npx --yes ctx7@latest library pytorch "torch.amp autocast GradScaler CUDA mixed precision current usage float32 cross entropy backward unscale gradient clipping"
npx --yes ctx7@latest docs /websites/pytorch_2_13 "Current torch.amp.autocast and torch.amp.GradScaler CUDA float16 cross entropy, backward outside autocast and unscale before gradient clipping, save scaler checkpoint for resume"
```

The returned [AMP documentation](https://docs.pytorch.org/docs/2.13/amp.html)
recommends unified `torch.amp.autocast` and `torch.amp.GradScaler`, accepting the
device string. [Examples](https://docs.pytorch.org/docs/2.13/notes/amp_examples.html)
place backward outside the forward autocast context, call `scaler.unscale_`
before clipping, and then `scaler.step`/`scaler.update`. The separate
`torch.compile` backward context has additional constraints; this pilot does not
need compilation. Indexed docs are PyTorch 2.13, so the actual isolated runtime
version and working AMP/gradient sanity results remain necessary evidence.

## New-code review and independent numerical checks

`seed_context_model.py` provides a separately encoded coarse radar view and
individual seed embeddings, seed-relative sample/depth coordinates, learned
attention, and native-resolution skip features. Attention preserves the
association between each seed waveform and its coordinates before aggregation.
The layer-only control bypasses these inputs. There is no interpolated target
guide or synthesized negative-click map. After this review flagged ambiguous
zero padding, the model owner added explicit fine/coarse/seed-patch validity
channels and output masking. An entirely invalid trace still requires an explicit
unresolved inference result; softmax of equal masked logits is uniform.

The immutable processed dataset now contains the re-audited 35 processed and
26 non-Proc DZT files, 12 reviewed processed pairs, 555,439 stored observations,
and eight reviewed physical groups. A read-only check across all 12 cached
records verified label-count/mask consistency, native axes, and that every
valid label sits on a valid signal pixel. The manifest fingerprint checked is
`c81944434af55c8f2f231ccf6cacf42ccc2a33fb6e1ad29cb44370c6b772667a`.
Every reviewed record has native spacing 0.025 m and 0.029296875 ns, so differing
sample intervals do not currently invalidate sample-relative conditioning.
Those equal intervals must not be presumed for future inputs. The cache preserves
unknown label provenance and marks semantics outside Mandiali/Gujrat as the
historical numbering convention, not independently confirmed pavement identity.

The new dense decoder uses every valid native sample plus a gap state and exact
floating seed transitions, with no old candidate pruning or hard displacement
bound. An independent exhaustive enumeration of 20 random five-row/four-depth
graphs, including fractional seeds, missing samples and explicit breaks,
matched its optimum exactly in all cases and retained every seed unchanged.
The source fingerprint tested is
`95426e7b8ef820ae28c45e775c6bc55704e1f4878cc94bcc409cb40cf8aacb8b`.
Results are in `review/decoder-exhaustive-check.json`. This establishes the
stated optimization arithmetic, not that its objective selects correct interfaces.
Its per-trace log evidence and total-variation-in-time cost change relative
weighting when trace spacing changes; keep the research grid/config frozen or
explicitly test sampling-specific calibration. Local selected-depth probability
does not inherit the old interval-length denominator, but is still uncalibrated
for correctness.

The new evaluator retains the initial non-seed N, includes later supplied answers
in that N without automatic credit, reports undefined agreement at zero accepts,
and checks the frozen scorer's normalized source hash. Its accounting wrapper was
independently compared with unchanged `conventional.layer_metrics` on 12 random
integer-native cases; non-seed N, correct/total accepts, correct proposals and
accepted agreement matched exactly. Evidence and both source fingerprints are
in `review/scorer-parity-check.json`. This parity check uses integer references
matching the frozen DZX contract; it does not authorize altering that scorer.

`processed_ml_replay.py` selects requests using radar, predictions, existing seeds
and visited locations before looking up the exact answer. Unavailable answers
consume a request. It calls the real local merge/order guard and replay checkpoint
helpers. The source review prompted sorting wrong rows before observed-span
aggregation and preserving actual floating proposal samples in added timing
diagnostics. The original scorer remains unchanged. Replay and the dense
evaluator identify the consecutive-observed-bin footprint for wrong spans;
replay also reports endpoint distance separately. Unknown and analyst-supplied
rows break observed runs.

The first `processed_training.py` implementation rejects all outer-group source
and acquisition IDs before loading any fitting arrays. It verifies cache hashes,
uses known target-interface labels only, draws remote supports outside the entire
wide query context, and excludes support rows from the depth loss. Its corrected
65-trace input produces 65 unique contiguous native observations, a center-aligned
coarse view with four-native-trace spacing, and no temporal resampling. At the
reviewed native grid these span 1.6 m and 6.4 m; its 65-by-9 seed patch spans
1.875 ns by 0.2 m. The earlier two-step CPU smoke used a previous even-width
source and is only evidence that the plumbing ran, not a valid corrected-grid
learning result.

Two trainer issues were raised before larger jobs. Selecting a road before a
layer gave rare subbase much less optimization than other layers; the owner
changed sampling to layer first, then an eligible road, which this review
confirmed in the source. Buffered blocks in local acquisition coordinates do
not establish physical separation
across unregistered repeat passes or portions of the same road. Outer-road
isolation remains valid, but inner-validation calibration must either address
those overlaps conservatively or state its limitation. Checkpoints also need
atomic replacement to retain the last resumable state during an interrupted
save. The bare-probabilities evaluator CLI should bind and verify model, source,
seed and grid provenance before scoring rather than only hash the selected NPZ.
Those remaining findings were sent to the owners. The review does not claim
their fixes or any real-road learning result yet.
