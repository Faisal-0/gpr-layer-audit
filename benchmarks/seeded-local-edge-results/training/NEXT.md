# Next execution step

Training and both complete array audits are finished. Do not rerun preparation
or training into this directory. No predictor or common-graph experiment has
been implemented for this model yet. Model SHA256:
`aac5a797f98eb4676927f5f088e2684d0a4689b0a9e210a5b79ac47a5f68575f`.

The prospective contract is frozen in contract.json: existing architecture,
12 epochs, source at a reviewed Mandiali point exactly 0.1m before the target,
unchanged timing/signed-lobe labels, six nearby negatives prioritized for the
existing maximum step, no support rows as targets. All 55,695 pairs and 6,204
source patches were reconstructed exactly. No Jamshoro/Gujrat labels trained it.

Implement an isolated Jamshoro common-graph predictor next. Use the same three
native observations and masks from the prior frozen patch experiment, dense
sample graph, and the existing unit-weight endpoint guide. Compare classical
local patch NCC against the learned timing-head edge cost, with unit edge weight
per metre and the same unsupported cost (0.65) in both arms. Both arms must share
candidate and edge support; no pruning or acceptance threshold tuning.
The comparator should have the same local 2D radar context as the network.
The prospective NCC patch/mask numerical details need to be frozen before
running this new predictor, with a direct arithmetic test.

Patch encoding can be cached once: the learned source/target comparison is
layer-independent. Candidate patches are native +/-0.9375ns and +/-0.5m, 65x21,
amplitude plus validity. Run model.compare(target_encoding, source_encoding),
matching training orientation. Reference labels enter only after predictions.
The existing dense margin has a known interval-normalization defect; do not
lower its threshold to disguise poor path selection. Training alone is not a
keep/promotion result.

Current branch HEAD is 95a7bee. Application source is unchanged. The local-pair
trainer and two tests remain uncommitted pending the actual graph experiment;
their exact executed sources are saved here. Other completed experiments are
published and rejected, described in docs/BASIN_TRACKER_EXPERIMENT.md. All jobs
are terminal. All available roads remain development data.
