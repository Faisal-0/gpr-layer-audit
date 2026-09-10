# Native processed-coordinate learning dataset

The September 10 research cache trains directly on the paired processed DZT/DZX
coordinates. It does not use the raw-coordinate ML loader, calibrate the radar,
fit a reference offset, resample time, snap picks to peaks, or infer absence from
missing labels. The existing scorer and production tracker are unchanged.

The completed local dataset is
`C:\Users\faisa\Desktop\gpr_campaign_artifacts\20260910\processed_dataset\manifest.json`.
Numeric arrays occupy approximately 1.08 GB in this external artifact directory.
They are read-only NumPy memory maps during training, shared across jobs.
Its immutable manifest SHA256 is
`c81944434af55c8f2f231ccf6cacf42ccc2a33fb6e1ad29cb44370c6b772667a`.
The final builder snapshot and a note identifying formatting/docstring edits
after cache creation are stored as `builder-source-final.py` and
`builder-source-lineage.json`; the original manifest remains unchanged.

## Completed source audit

The supplied `gpr including proc files` tree contains 61 DZT files: 35 inside
Proc and 26 outside. Five processed and ten non-processed files are metal-plate
calibration acquisitions. Two Central/South Jhang raw acquisitions are exact
duplicate copies; 59 DZT hashes are unique. The processed Jhang versions differ
bytewise but retain one physical-road group. There are 24 filename acquisition
families after removing processing suffixes, 17 with processed variants.

There are 68 DZX files, 53 unique by hash; 12 contain reviewed observations and
56 have none. Identical empty metadata files occur on unrelated roads, so a DZX
hash alone never establishes acquisition identity. All 12 populated references
have paired processed DZT files and unique DZX hashes. They contain 555,439 stored
interface observations, all of which pass the adapter's stored amplitude, header
time, numerical validity, bounds and interface-order checks in this inventory.
Every scan has 512 native samples, 0.025 m native trace spacing, a sample interval
of 0.029296875 ns, and a time origin of -3 ns. Float32 conversion preserves every
stored amplitude exactly on all 12 scans (measured maximum absolute error zero).

| Physical road group | Asphalt convention | Base convention | Subbase convention |
|---|---:|---:|---:|
| Bahawalpur | 41,359 | 20,081 | 0 |
| Burewala | 59,826 | 27,015 | 0 |
| Daska | 45,302 | 22,649 | 0 |
| Gujrat | 51,330 | 20,083 | 21,191 |
| Jamshoro | 31,037 | 16,475 | 8,985 |
| Jhang | 32,251 | 15,211 | 0 |
| Mandiali | 43,107 | 16,506 | 13,659 |
| Pattoki | 44,167 | 25,205 | 0 |

Counts describe stored observations, not independent manual judgments or physical
thickness truth. Every stored point has vendor `interpretation_property=2` and
the populated layers have `pickType=0` and `link=1`. No verified enum definition
or per-point creation history establishes which points were individually drawn,
interpolated, or assisted by tracing software. That origin remains explicitly
unknown. The user's campaign authorizes learning from these processed analyst
interpretations; the uncertainty is retained instead of replacing this dataset
with fabricated manual labels. No explicit evaluation-only or prohibited label
file was found among the paired DZX files. Historical seed/checkpoint JSON and
workbook thickness values are not ingested as training labels.

Only Mandiali and Gujrat interface semantics have the existing user confirmation
in `benchmarks/seeded-label-confirmation.json`. Other groups retain the historical
RADAN number mapping (0/1/2 to asphalt/base/subbase); that distinction is recorded
per layer. No processed Talagang DZT or populated Talagang reference is present.

## API and array contract

```python
from gpr_layer_audit.ml.processed_data import (
    load_processed_manifest, open_processed_record, research_leave_one_group_out,
)

manifest_path = ".../processed_dataset/manifest.json"
manifest = load_processed_manifest(manifest_path)
data = open_processed_record(manifest_path, manifest["records"][0], verify_hashes=True)
radar = data["amplitudes"]       # [native_trace, native_sample], signed float32
labels = data["labels"]          # [3, native_trace], native sample coordinate; NaN unknown
known = data["label_valid"]      # [3, native_trace], bool
metadata = data["record"]
folds = research_leave_one_group_out(manifest, 3, min_training_groups=2)
```

Additional arrays are `sample_validity[trace,sample]`, `native_trace_indices`,
`native_sample_indices`, `distances_m`, `label_interpretation_property[3,trace]`,
and `label_signed_lobe[3,trace]`. Signed-lobe values are -1/0/+1 at the unchanged
stored observation. Source DZT and DZX fingerprints, array fingerprints,
physical-road group, acquisition family, processing variant, sample interval,
time origin, horizontal spacing, layer metadata and provenance accompany each
record. The cache has no learned normalization statistics. Model-specific
deterministic per-scan normalization must be recorded separately.

`label_valid` authorizes a selected-interface position target at that trace.
Other valid depths at that reviewed trace may be target-specific alternatives;
they do not label other reflectors absent. Unknown traces have no loss target.
No visibility, absence, break, or verified-negative-click labels are invented.
`sample_validity` uses the existing `processed_boundary_mask` on the radar alone:
exact boundary zero padding and repeated terminal runs are invalid, while
interior zero signal remains available. Label omissions never alter that mask.

Native time is `time_origin_ns + sample * sample_interval_ns`. Physical distance
is native trace times `horizontal_step_m`. `NativeWindow` provides invertible
crop/pad/integer horizontal decimation and axis reversal coordinates, including
fractional and remote seed coordinates. It preserves native temporal resolution
and polarity. Padding is masked; a returned numerical padding value cannot be a
measurement. Any model augmentation must transform signal, validity and labels
together. Vertical interpolation is deliberately unsupported by this adapter.

## Split safeguards and research eligibility

Every same-road processing variant, repeat pass, portion and potentially
overlapping acquisition stays in one physical-road group before crop assignment.
Acquisition IDs remove processing suffixes; geographic inter-acquisition offsets
and travel direction remain unknown. No registration is inferred from labels.
Training must balance road, spatial block and layer, including variants sharing
an acquisition. Inner validation must use training-road blocks with buffers that
cover every fitted context crop, including remote support contexts; the trainer
owns that episode policy.

The explicit research fold entry point permits two training groups, so all three
subbase folds are available: hold out Gujrat and train Mandiali/Jamshoro; hold out
Jamshoro and train Gujrat/Mandiali; hold out Mandiali and train Gujrat/Jamshoro.
This does not change the raw loader's production eligibility gate. Dataset
records and folds remain `production_eligible=false`. Records explicitly marked
prohibited/evaluation-only are excluded from this entry point. The build CLI
accepts source or label deny fingerprints through `--prohibited-hash`; it cannot
override them. Supplied seed/checkpoint files are outside this ingestion path.

All annotated roads have historical development use. Results must be called
grouped development generalization, with all outer-road pixels excluded from
weight fitting or pretraining in the strict inductive arm. This dataset cannot
establish independent final-road performance.

## Reproduction and verification

Build into a new directory; the builder refuses to replace existing evidence:

```powershell
$env:PYTHONPATH='C:\Users\faisa\Desktop\gpr_next_stage_20260910\src'
& 'C:\Users\faisa\Desktop\automate_gpr\.venv\Scripts\python.exe' scripts/build_processed_ml_dataset.py --data-root 'C:\Users\faisa\Desktop\automate_gpr\gpr including proc files' --output 'C:\Users\faisa\Desktop\gpr_campaign_artifacts\20260910\processed_dataset_rebuild' --overlays
& 'C:\Users\faisa\Desktop\automate_gpr\.venv\Scripts\python.exe' -m pytest tests/test_processed_ml_data.py -q
```

The 11 focused tests pass: signed native amplitudes/header times, valid/unknown
masks, conflicting duplicates, padding/time rejection, immutable outputs,
explicit prohibition, all two-training-group subbase folds, fractional click
round trips through crop/pad/decimation/reversal, cache hash tampering, and
incompatible preprocessing rejection. Ruff checks pass for these new files.
An independent read-only audit of all 12 cached records also passes declared
counts, authorized known labels, exact native axes, valid target pixels, and
shared dt/dx checks; see `../review/dataset-mask-consistency.json` relative to the
dataset directory.

The completed first/middle/last subbase context overlays for the longest
acquisition in each of Gujrat, Mandiali and Jamshoro are under the cache's
`overlays` directory. Their contexts were chosen by stored coordinate order,
independently of any model output. Native sample centers and header-time axes
coincide; missing observations remain gaps. A common whole-window amplitude
scale exposes the strong surface/deep-energy imbalance without modifying the
training signal. These overlays verify coordinates and do not demonstrate
prediction quality or confirm physical layer identity independently.
