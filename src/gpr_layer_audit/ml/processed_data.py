"""Native processed-coordinate research data, independent of the raw ML loader.

No calibration, reference-fitted offset, peak snapping, or negative-click synthesis
is performed here. RADAN observations are interpretation references; their density
does not establish the number of independent manual judgments.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gpr_layer_audit.conventional_reference import road_partition
from gpr_layer_audit.io.dzt import DZTFile, fingerprint_file
from gpr_layer_audit.io.dzx import read_dzx
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

SCHEMA = "processed-ml-dataset-v1"
PREPROCESSING_VERSION = "native-processed-signed-no-resampling-v1"
ARRAY_KEYS = (
    "amplitudes",
    "sample_validity",
    "labels",
    "label_valid",
    "native_trace_indices",
    "native_sample_indices",
    "distances_m",
    "label_interpretation_property",
    "label_signed_lobe",
)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _group(path):
    group, _ = road_partition(path)
    if group != "unresolved":
        return group
    for city in ("talagang", "rawalpindi"):
        if city in str(path).lower():
            return city
    return "unresolved-" + hashlib.sha256(str(path.parent).encode()).hexdigest()[:12]


def _acquisition(path):
    # Variants share this identifier; road grouping additionally joins repeat passes,
    # overlapping portions, and geographically unregistered acquisitions conservatively.
    stem = re.sub(r"\s+P_\d+$", "", path.stem, flags=re.IGNORECASE)
    return _group(path) + ":" + stem.lower()


def _inventory_file(path, data_root):
    source = DZTFile(path)
    h = source.header
    return {
        "path": str(path.resolve()),
        "relative_path": str(path.relative_to(data_root)),
        "source_sha256": fingerprint_file(path),
        "input_mode": "processed" if any(p.lower() == "proc" for p in path.parts) else "nonproc",
        "physical_road_group": _group(path),
        "acquisition_id": _acquisition(path),
        "processing_variant": (
            re.search(r"P_\d+$", path.stem, re.IGNORECASE).group(0)
            if re.search(r"P_\d+$", path.stem, re.IGNORECASE)
            else "none"
        ),
        "shape": [h.trace_count, h.samples_per_trace],
        "channels": h.channels,
        "sample_interval_ns": h.sample_interval_ns,
        "time_origin_ns": h.position_ns,
        "horizontal_step_m": h.distance_per_trace_m,
        "bits_per_sample": h.bits_per_sample,
        "historical_use": "development/calibration; no untouched annotated road",
    }


def _cache_record(source, metadata, output, *, training_use="research_allowed"):
    road = DZTFile(source["path"])
    h = road.header
    record_id = source["physical_road_group"] + "-" + source["source_sha256"][:12]
    destination = output / "records" / record_id
    if destination.exists():
        raise FileExistsError(f"Immutable record already exists: {destination}")
    destination.mkdir(parents=True)
    shape = tuple(source["shape"])
    amplitude = np.lib.format.open_memmap(
        destination / "amplitudes.npy",
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    validity = np.lib.format.open_memmap(
        destination / "sample_validity.npy",
        mode="w+",
        dtype=np.bool_,
        shape=shape,
    )
    quantization_max = 0.0
    for start, values in road.iter_channel_chunks(chunk_traces=2048):
        stop = start + len(values)
        amplitude[start:stop] = values
        validity[start:stop] = processed_boundary_mask(values)
        quantization_max = max(
            quantization_max,
            float(np.max(np.abs(values.astype(np.float64) - amplitude[start:stop]))),
        )
    amplitude.flush()
    validity.flush()
    labels = np.full((3, shape[0]), np.nan, np.float32)
    label_valid = np.zeros((3, shape[0]), bool)
    properties = np.full((3, shape[0]), -1, np.int32)
    signed_lobe = np.zeros((3, shape[0]), np.int8)
    exclusions = Counter()
    layer_reports = []
    conflicts = set()
    for layer in metadata.layers:
        if not layer.picks:
            continue
        if layer.number not in (0, 1, 2):
            exclusions["unmapped_layer_number"] += len(layer.picks)
            continue
        index = layer.number
        amplitudes, times, counts = [], [], Counter()
        for pick in layer.picks:
            counts["stored"] += 1
            if pick.channel != 0:
                exclusions["nonzero_channel"] += 1
                continue
            if not (0 <= pick.trace < shape[0] and 0 <= pick.sample < shape[1]):
                exclusions["coordinate_out_of_bounds"] += 1
                continue
            actual = float(road.channel()[pick.trace, pick.sample])
            residual = pick.time_ns - (h.position_ns + pick.sample * h.sample_interval_ns)
            amplitudes.append(abs(actual - pick.recorded_amplitude))
            times.append(residual)
            reason = None
            if abs(actual - pick.recorded_amplitude) > 1:
                reason = "recorded_amplitude_mismatch"
            elif abs(residual) >= h.sample_interval_ns / 4:
                reason = "recorded_time_mismatch"
            elif not validity[pick.trace, pick.sample]:
                reason = "numerical_padding_at_label"
            key = (index, pick.trace)
            if key in conflicts:
                reason = "conflicting_duplicate_label"
            if np.isfinite(labels[index, pick.trace]):
                if labels[index, pick.trace] != pick.sample:
                    conflicts.add(key)
                    labels[index, pick.trace] = np.nan
                    label_valid[index, pick.trace] = False
                    reason = "conflicting_duplicate_label"
                else:
                    counts["same_coordinate_duplicate"] += 1
                    continue
            if reason:
                exclusions[reason] += 1
                continue
            labels[index, pick.trace] = pick.sample
            label_valid[index, pick.trace] = training_use == "research_allowed"
            properties[index, pick.trace] = pick.interpretation_property
            signed_lobe[index, pick.trace] = int(np.sign(actual))
            counts["coordinate_valid"] += 1
        layer_reports.append(
            {
                "source_layer_number": layer.number,
                "source_layer_name": layer.name,
                "interface_order": index + 1,
                "semantic_name": ("asphalt", "base", "subbase")[index],
                "semantic_status": (
                    "user_confirmed_existing_convention"
                    if source["physical_road_group"] in ("mandiali", "gujrat")
                    else "historical_number_convention_not_independently_confirmed"
                ),
                "counts": dict(counts),
                "properties": dict(layer.properties),
                "amplitude_match_fraction": float(np.mean(np.asarray(amplitudes) <= 1))
                if amplitudes
                else None,
                "time_residual_ns_max_abs": float(np.max(np.abs(times))) if times else None,
                "interpretation_property_counts": dict(
                    Counter(str(p.interpretation_property) for p in layer.picks)
                ),
            }
        )
    # A contradiction between named interfaces is not silently repaired to fit order.
    ordering_conflicts = np.zeros_like(label_valid)
    for lower in (1, 2):
        both = np.isfinite(labels[lower - 1]) & np.isfinite(labels[lower])
        conflict = both & (labels[lower] <= labels[lower - 1])
        ordering_conflicts[lower - 1] |= conflict
        ordering_conflicts[lower] |= conflict
    exclusions["ordering_conflict_observations"] += int(ordering_conflicts.sum())
    label_valid[ordering_conflicts] = False
    labels[ordering_conflicts] = np.nan
    arrays = {
        "labels": labels,
        "label_valid": label_valid,
        "native_trace_indices": np.arange(shape[0], dtype=np.int64),
        "native_sample_indices": np.arange(shape[1], dtype=np.int64),
        "distances_m": np.arange(shape[0], dtype=np.float64) * h.distance_per_trace_m,
        "label_interpretation_property": properties,
        "label_signed_lobe": signed_lobe,
    }
    for key, value in arrays.items():
        np.save(destination / f"{key}.npy", value, allow_pickle=False)
    array_paths = {key: str((destination / f"{key}.npy").relative_to(output)) for key in ARRAY_KEYS}
    record = {
        **source,
        "record_id": record_id,
        "arrays": array_paths,
        "array_sha256": {key: fingerprint_file(output / path) for key, path in array_paths.items()},
        "label_source_path": metadata.source_path,
        "label_sha256": metadata.source_sha256,
        "training_use": training_use,
        "production_eligible": False,
        "label_counts": {str(i + 1): int(label_valid[i].sum()) for i in range(3)},
        "available_layers": [i + 1 for i in range(3) if label_valid[i].any()],
        "label_audit": layer_reports,
        "exclusions": dict(exclusions),
        "stored_observations": sum(len(layer.picks) for layer in metadata.layers),
        "coordinate_provenance": (
            "RADAN stored zero-based trace/sample; time=origin+sample*dt; no alignment fitted"
        ),
        "label_provenance": (
            "analyst interpretation export; individual manual/interpolated/automatic origin unknown"
        ),
        "missing_label_semantics": "unknown; never absence or background",
        "visibility_absence_labels": False,
        "float32_amplitude_max_absolute_quantization": quantization_max,
        "normalization": (
            "none; signed unenhanced stored signal; per-scan normalization belongs to model config"
        ),
        "geographic_registration": (
            "unknown; local stored trace origin; all same-road variants grouped"
        ),
        "processing_properties": list(metadata.processing_properties),
        "custom_fir": list(metadata.custom_fir),
        "system": metadata.system,
        "software_version": metadata.software_version,
        "scan_range": list(metadata.scan_range),
        "scan_range_matches": not metadata.scan_range or metadata.scan_range == (0, shape[0] - 1),
    }
    _write_json(destination / "record.json", record)
    return record


def build_processed_dataset(data_root, output_dir, *, prohibited_hashes=(), progress=None):
    """Audit source inventory and materialize unique reviewed processed DZT/DZX pairs.

    Explicitly prohibited/evaluation-only label fingerprints can be supplied as
    ``prohibited_hashes``. Their labels are excluded, never permission-overridden.
    Historical benchmark seed/checkpoint files are not training inputs.
    """
    data_root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new immutable dataset output directory: {output}")
    output.mkdir(parents=True)
    paths = sorted(p for p in data_root.rglob("*") if p.suffix.lower() == ".dzt")
    inventory, records, errors = [], [], []
    seen = {}
    denied = set(prohibited_hashes)
    for number, path in enumerate(paths, 1):
        if progress:
            progress(f"audit {number}/{len(paths)}: {path.name}")
        try:
            entry = _inventory_file(path, data_root)
        except (ValueError, OSError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        inventory.append(entry)
        if entry["input_mode"] != "processed":
            continue
        label_path = path.with_suffix(".DZX")
        if not label_path.exists():
            entry["label_status"] = "paired_dzx_missing"
            continue
        metadata = read_dzx(label_path)
        entry["label_sha256"] = metadata.source_sha256
        entry["stored_observations"] = sum(len(layer.picks) for layer in metadata.layers)
        if not entry["stored_observations"]:
            entry["label_status"] = "no_reviewed_layer_observations"
            continue
        if metadata.source_sha256 in denied or entry["source_sha256"] in denied:
            entry["label_status"] = "explicitly_prohibited_or_evaluation_only"
            continue
        if entry["horizontal_step_m"] is None or entry["horizontal_step_m"] <= 0:
            entry["label_status"] = "missing_physical_spacing"
            continue
        key = (entry["source_sha256"], metadata.source_sha256)
        if key in seen:
            entry["duplicate_of_record_id"] = seen[key]["record_id"]
            seen[key].setdefault("aliases", []).append(str(path))
            continue
        record = _cache_record(entry, metadata, output)
        seen[key] = record
        records.append(record)
        entry["label_status"] = "coordinate_audited_research_reference"
        if progress:
            progress(f"ready {record['record_id']} labels={record['label_counts']}")
    by_hash = defaultdict(list)
    for entry in inventory:
        by_hash[entry["source_sha256"]].append(entry)
    # Exact duplicates must not carry different fold identities.
    for duplicates in by_hash.values():
        groups = {item["physical_road_group"] for item in duplicates}
        if len(groups) > 1:
            raise ValueError(f"Exact duplicate source has conflicting physical groups: {groups}")
    support = {}
    for order in (1, 2, 3):
        counts = Counter()
        for record in records:
            counts[record["physical_road_group"]] += record["label_counts"][str(order)]
        support[str(order)] = {g: c for g, c in sorted(counts.items()) if c}
    manifest = {
        "schema": SCHEMA,
        "preprocessing_version": PREPROCESSING_VERSION,
        "data_root": str(data_root),
        "inventory": inventory,
        "records": records,
        "counts": {
            "dzt_files": len(inventory),
            "processed_dzt_files": sum(r["input_mode"] == "processed" for r in inventory),
            "nonproc_dzt_files": sum(r["input_mode"] == "nonproc" for r in inventory),
            "reviewed_processed_pairs_before_deduplication": sum(
                r.get("stored_observations", 0) > 0 for r in inventory
            ),
            "unique_reviewed_processed_pairs": len(records),
            "stored_interface_observations_before_deduplication": sum(
                r.get("stored_observations", 0) for r in inventory
            ),
            "unique_pair_stored_interface_observations": sum(
                r["stored_observations"] for r in records
            ),
            "reviewed_physical_road_groups": len({r["physical_road_group"] for r in records}),
        },
        "label_support_by_road_and_layer": support,
        "errors": errors,
        "duplicate_source_sets": [
            [x["relative_path"] for x in v] for v in by_hash.values() if len(v) > 1
        ],
        "split_policy": (
            "physical group held out before crops; all variants/repeats/portions share group; "
            "training-only inner validation"
        ),
        "evaluation_claim": "grouped development generalization, not untouched-test validation",
        "research_permission": (
            "current campaign authorizes validated processed DZX interpretation references; "
            "explicit deny hashes excluded"
        ),
        "prohibited_hashes": sorted(denied),
        "production_eligible": False,
        "source_code_sha256": fingerprint_file(__file__),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def load_processed_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported processed-coordinate dataset schema")
    if manifest.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ValueError("Unsupported processed-coordinate preprocessing")
    return manifest


def open_processed_record(manifest_path, record_or_id, *, verify_hashes=False):
    """Open immutable memory-mapped arrays. ``labels`` are native sample coordinates.

    ``label_valid`` authorizes only target-depth supervision at that reviewed trace.
    It does not label any unreviewed trace or provide visibility/absence supervision.
    """
    manifest_path = Path(manifest_path)
    manifest = load_processed_manifest(manifest_path)
    record_id = record_or_id if isinstance(record_or_id, str) else record_or_id["record_id"]
    records = [r for r in manifest["records"] if r["record_id"] == record_id]
    if len(records) != 1:
        raise ValueError(f"Unknown or ambiguous processed record: {record_id}")
    record = records[0]
    if record.get("input_mode") != "processed":
        raise ValueError("Processed model cannot consume raw-coordinate input")
    output = {"record": record}
    root = manifest_path.parent.resolve()
    for key in ARRAY_KEYS:
        path = (root / record["arrays"][key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Cached array path escapes dataset directory")
        if verify_hashes and fingerprint_file(path) != record["array_sha256"][key]:
            raise ValueError(f"Cached array fingerprint changed: {key}")
        output[key] = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(record["shape"])
    if output["amplitudes"].shape != shape or output["sample_validity"].shape != shape:
        raise ValueError("Processed radar and validity must match native shape")
    if output["labels"].shape != (3, shape[0]) or output["label_valid"].shape != (3, shape[0]):
        raise ValueError("Processed labels require three layer-by-native-trace rows")
    return output


def research_leave_one_group_out(manifest, layer_order, *, min_training_groups=2):
    """Explicit research entry point allowing the two other subbase groups.

    This grants no production promotion and excludes prohibited/evaluation-only
    records. Every processing variant of an outer road remains outside fitting.
    """
    if layer_order not in (1, 2, 3) or min_training_groups < 2:
        raise ValueError("Research folds require a known layer and at least two training groups")
    eligible = [
        r
        for r in manifest["records"]
        if r.get("training_use") == "research_allowed"
        and r["label_counts"].get(str(layer_order), 0) > 0
    ]
    groups = sorted({r["physical_road_group"] for r in eligible})
    folds = []
    for group in groups:
        training = [r for r in eligible if r["physical_road_group"] != group]
        training_groups = sorted({r["physical_road_group"] for r in training})
        if len(training_groups) < min_training_groups:
            continue
        folds.append(
            {
                "held_out_group": group,
                "training_groups": training_groups,
                "training_record_ids": [r["record_id"] for r in training],
                "evaluation_record_ids": [
                    r["record_id"] for r in eligible if r["physical_road_group"] == group
                ],
                "claim": "grouped development generalization",
                "production_eligible": False,
            }
        )
    return folds


@dataclass(frozen=True)
class NativeWindow:
    """Exact crop/pad/stride/translation/flip transform; no vertical interpolation.

    Horizontal stride is coordinate-preserving decimation, not signal averaging.
    A vertical flip or integer shift moves samples and labels together and never
    reverses polarity. Padding is explicitly masked and cannot become a measurement.
    """

    trace_start: int
    trace_count: int
    sample_start: int
    sample_count: int
    trace_stride: int = 1
    flip_trace: bool = False
    flip_sample: bool = False

    def axes(self):
        if min(self.trace_count, self.sample_count, self.trace_stride) <= 0:
            raise ValueError("Window sizes and stride must be positive")
        traces = self.trace_start + np.arange(self.trace_count) * self.trace_stride
        samples = self.sample_start + np.arange(self.sample_count)
        return (
            traces[::-1] if self.flip_trace else traces,
            samples[::-1] if self.flip_sample else samples,
        )

    def native_to_local(self, traces, samples):
        t = (np.asarray(traces) - self.trace_start) / self.trace_stride
        s = np.asarray(samples) - self.sample_start
        return (
            (self.trace_count - 1 - t) if self.flip_trace else t,
            (self.sample_count - 1 - s) if self.flip_sample else s,
        )

    def local_to_native(self, traces, samples):
        t = self.trace_count - 1 - np.asarray(traces) if self.flip_trace else np.asarray(traces)
        s = self.sample_count - 1 - np.asarray(samples) if self.flip_sample else np.asarray(samples)
        return self.trace_start + t * self.trace_stride, self.sample_start + s

    def extract(self, amplitudes, validity):
        traces, samples = self.axes()
        keep_t = (traces >= 0) & (traces < amplitudes.shape[0])
        keep_s = (samples >= 0) & (samples < amplitudes.shape[1])
        values = np.zeros((self.trace_count, self.sample_count), amplitudes.dtype)
        valid = np.zeros(values.shape, bool)
        values[np.ix_(keep_t, keep_s)] = amplitudes[np.ix_(traces[keep_t], samples[keep_s])]
        valid[np.ix_(keep_t, keep_s)] = validity[np.ix_(traces[keep_t], samples[keep_s])]
        return values, valid


def representative_overlays(manifest_path, output_dir):
    """Render one fixed first/middle/last label context per deep-label road group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_processed_manifest(manifest_path)
    selected = {}
    for record in manifest["records"]:
        if record["label_counts"]["3"]:
            old = selected.get(record["physical_road_group"])
            if old is None or record["shape"][0] > old["shape"][0]:
                selected[record["physical_road_group"]] = record
    outputs = []
    for group, record in sorted(selected.items()):
        data = open_processed_record(manifest_path, record)
        known = np.flatnonzero(data["label_valid"][2])
        centers = known[np.linspace(0, len(known) - 1, 3).astype(int)]
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
        dx, dt, origin = (
            record[k] for k in ("horizontal_step_m", "sample_interval_ns", "time_origin_ns")
        )
        for ax, center in zip(axes, centers, strict=True):
            start, stop = max(0, int(center) - 192), min(record["shape"][0], int(center) + 193)
            image = np.asarray(data["amplitudes"][start:stop])
            scale = max(float(np.percentile(np.abs(image), 99)), 1.0)
            ax.imshow(
                image.T,
                aspect="auto",
                cmap="gray",
                vmin=-scale,
                vmax=scale,
                extent=(
                    (start - 0.5) * dx,
                    (stop - 0.5) * dx,
                    origin + (image.shape[1] - 0.5) * dt,
                    origin - 0.5 * dt,
                ),
            )
            for index, color in enumerate(("#edc949", "#4e79a7", "#e15759")):
                rows = np.flatnonzero(data["label_valid"][index, start:stop]) + start
                ax.scatter(
                    rows * dx,
                    origin + data["labels"][index, rows] * dt,
                    s=3,
                    color=color,
                    label=("asphalt", "base", "subbase")[index],
                )
            ax.set(xlabel="Local stored distance (m)", ylabel="Header time (ns)")
        axes[0].legend(loc="lower right", ncols=3)
        fig.suptitle(
            f"{group}: unchanged processed amplitudes and native DZX observations\n"
            "First, middle, last subbase reference contexts; "
            "color names retain stated semantic limits"
        )
        path = output / f"{group}-native-coordinate-overlay.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        outputs.append(str(path.resolve()))
    return outputs
