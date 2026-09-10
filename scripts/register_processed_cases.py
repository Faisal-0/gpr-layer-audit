"""Freeze existing operating seeds against the audited processed dataset.

No seed coordinates are selected here. Four cases reuse existing seed artifacts;
Jamshoro copies only input.seeds from its original prediction manifest. Outputs
are immutable, and registration checks native coordinates without changing them.

Example::

    python scripts/register_processed_cases.py --manifest CACHE/manifest.json \
      --output CAMPAIGN/evaluation/cases.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def match_record(manifest, dzt_sha256, dzx_sha256):
    matches = [
        record for record in manifest["records"]
        if record["source_sha256"] == dzt_sha256
        and record["label_sha256"] == dzx_sha256
    ]
    if len(matches) != 1:
        raise ValueError("Case must match exactly one processed source/reference record")
    record = matches[0]
    if record["input_mode"] != "processed":
        raise ValueError("Evaluation registry requires processed coordinates")
    for path, expected in (
        (record["path"], dzt_sha256),
        (record["label_source_path"], dzx_sha256),
    ):
        if digest(path) != expected:
            raise ValueError(f"Processed source or reference fingerprint changed: {path}")
    return record


def validate_seeds(document, record, stride, dataset_root):
    """Verify only the already-authorized locations; never search for new seeds."""
    import numpy as np

    if document.get("schema") != "conventional-native-seeds-v1":
        raise ValueError("Expected native seed observations")
    if (document.get("mode") != "processed"
            or document.get("dzt_sha256") != record["source_sha256"]
            or document.get("dzx_sha256") != record["label_sha256"]):
        raise ValueError("Native seeds and processed dataset fingerprints differ")
    if type(stride) is not int or stride < 1:
        raise ValueError("The retained stride must be a positive integer")
    arrays = {}
    for name in ("labels", "label_valid", "sample_validity"):
        path = dataset_root / record["arrays"][name]
        if digest(path) != record["array_sha256"][name]:
            raise ValueError(f"Processed label/validity cache changed: {name}")
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
    layers = sorted(int(key) for key in document["observations"])
    counts = {}
    for layer in layers:
        if layer not in record["available_layers"]:
            raise ValueError("Seed interface is absent from the processed record")
        seen = set()
        points = document["observations"][str(layer)]
        if len(points) != 3:
            raise ValueError("This frozen campaign expects three initial seeds per interface")
        for point in points:
            trace, sample, channel = (point[key] for key in ("trace", "sample", "channel"))
            if any(type(value) is not int for value in (trace, sample, channel)):
                raise ValueError("Seeds must use exact native integer coordinates")
            if (channel != 0 or not 0 <= trace < record["shape"][0]
                    or not 0 <= sample < record["shape"][1] or trace % stride):
                raise ValueError("Native seed is outside the exact retained coordinate grid")
            if trace in seen:
                raise ValueError("Duplicate initial native seed")
            seen.add(trace)
            if (not arrays["label_valid"][layer - 1, trace]
                    or arrays["labels"][layer - 1, trace] != sample):
                raise ValueError("Frozen seed is not the identical stored reviewed observation")
            if not arrays["sample_validity"][trace, sample]:
                raise ValueError("Frozen seed has no valid measured signal")
        counts[str(layer)] = len(points)
    return layers, counts


def entry(case_id, record, stride, seed_path, layers, counts, lineage):
    return {
        "case_id": case_id,
        "record_id": record["record_id"],
        "road_group": record["physical_road_group"],
        "layers": layers,
        "stride": stride,
        "seed_path": str(seed_path.resolve()),
        "seed_sha256": digest(seed_path),
        "initial_seeds_by_layer": counts,
        "source_sha256": record["source_sha256"],
        "reference_sha256": record["label_sha256"],
        "native_shape": record["shape"],
        "sample_interval_ns": record["sample_interval_ns"],
        "time_origin_ns": record["time_origin_ns"],
        "native_dx_m": record["horizontal_step_m"],
        "working_dx_m": record["horizontal_step_m"] * stride,
        "seed_lineage": lineage,
        "exact_native_seeds_verified": True,
        "label_semantics": [
            {key: item[key] for key in ("interface_order", "semantic_name", "semantic_status")}
            for item in record["label_audit"] if item["interface_order"] in layers
        ],
    }


def scoring_pulses(document, record, stride, dataset_root):
    """Use only frozen pulse estimation and the authorized initial observations."""
    import numpy as np

    from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse

    measurement_path = dataset_root / record["arrays"]["amplitudes"]
    if digest(measurement_path) != record["array_sha256"]["amplitudes"]:
        raise ValueError("Processed measurement cache fingerprint changed")
    measurement = np.load(measurement_path, mmap_mode="r", allow_pickle=False)[::stride]
    valid = np.load(
        dataset_root / record["arrays"]["sample_validity"], mmap_mode="r", allow_pickle=False,
    )[::stride]
    return {
        layer: resolve_pulse(
            measurement, valid,
            {point["trace"] // stride: point["sample"] for point in points},
            {}, record["sample_interval_ns"], ConventionalConfig(),
        ).metadata()
        for layer, points in document["observations"].items()
    }


def jamshoro_document(prediction, source_path):
    original = prediction["input"]
    seeds = original["seeds"]
    # The input section contains native traces. Per-layer sections contain the
    # original retained rows, permitting an exact cross-check of stride four.
    stride = 4
    for layer, points in seeds.items():
        working = prediction["layers"][layer]["seeds"]
        expected = {str(int(trace) // stride): sample for trace, sample in points.items()}
        if any(int(trace) % stride for trace in points) or expected != working:
            raise ValueError("Jamshoro native and retained seed coordinates differ")
    return {
        "schema": "conventional-native-seeds-v1",
        "mode": "processed",
        "dzt_sha256": original["dzt_sha256"],
        "dzx_sha256": original["dzx_sha256"],
        "reference_label_mapping": {"0": 1, "1": 2, "2": 3},
        "observations": {
            layer: [
                {"trace": int(trace), "sample": sample, "channel": 0}
                for trace, sample in points.items()
            ] for layer, points in seeds.items()
        },
        "provenance": {
            "source_path": str(source_path.resolve()),
            "source_sha256": digest(source_path),
            "source_field": "input.seeds",
            "source_field_canonical_sha256": canonical_digest(seeds),
            "selection": "Exact copy of previously frozen native observations; no new selection",
            "original_working_stride": stride,
            "original_working_coordinates_cross_checked": True,
            "withheld_reference_used_to_select_seeds": False,
        },
    }, stride


def run(args):
    output = args.output.resolve()
    seed_output = output.parent / "seeds/jamshoro-native-seeds-v1.json"
    if output.exists() or seed_output.exists():
        raise FileExistsError(
            "Frozen registry or Jamshoro seeds exist; choose a new output location"
        )
    manifest_path = args.manifest.resolve()
    manifest = read(manifest_path)
    source_root = args.source_root.resolve()
    frozen_source = source_root / "exports/seeded-tracker/baseline-source/src"
    sys.path.insert(0, str(frozen_source))
    base_path = source_root / "benchmarks/seeded-evaluation-inputs.json"
    supplemental_path = (
        source_root / "exports/seeded-tracker/additional-inputs/"
        "supplemental-evaluation-inputs-v1.json"
    )
    jamshoro_path = args.jamshoro_manifest or (
        source_root / "exports/seeded-tracker/patchnet-v1/"
        "jamshoro-predictions/prediction-manifest.json"
    )
    sources = {path: read(path) for path in (base_path, supplemental_path, jamshoro_path)}
    cases = []
    for case_id, source_path in (
        ("gujrat-second", base_path), ("gujrat-first", supplemental_path),
        ("mandiali-short", base_path), ("mandiali-long", supplemental_path),
    ):
        frozen = sources[source_path]["cases"][case_id]
        record = match_record(manifest, frozen["dzt_sha256"], frozen["dzx_sha256"])
        if record["physical_road_group"] != frozen["physical_road_group"]:
            raise ValueError("Physical road identity differs from frozen case")
        seed_path = source_root / frozen["seed_source"]
        if digest(seed_path) != frozen["seed_sha256"]:
            raise ValueError("Existing frozen seed artifact changed")
        layers, counts = validate_seeds(
            read(seed_path), record, frozen["stride"], manifest_path.parent,
        )
        cases.append(entry(case_id, record, frozen["stride"], seed_path, layers, counts, {
            "manifest": str(source_path.resolve()), "manifest_sha256": digest(source_path),
            "case": case_id, "selection": "Existing frozen seed file reused without edits",
        }))
    jamshoro = sources[jamshoro_path]["input"]
    record = match_record(manifest, jamshoro["dzt_sha256"], jamshoro["dzx_sha256"])
    if record["physical_road_group"] != "jamshoro":
        raise ValueError("Jamshoro prediction source matched a different physical road")
    document, stride = jamshoro_document(sources[jamshoro_path], jamshoro_path)
    layers, counts = validate_seeds(document, record, stride, manifest_path.parent)
    write_new(seed_output, document)
    cases.append(entry("jamshoro", record, stride, seed_output, layers, counts, {
        "manifest": str(jamshoro_path.resolve()), "manifest_sha256": digest(jamshoro_path),
        "source_field": "input.seeds", "selection": "Exact prior native seeds exported",
    }))
    for case in cases:
        record = next(r for r in manifest["records"] if r["record_id"] == case["record_id"])
        pulses = scoring_pulses(
            read(case["seed_path"]), record, case["stride"], manifest_path.parent,
        )
        case["scoring_pulses"] = pulses
        case["tolerance_samples_by_layer"] = {
            layer: max(2.0, pulse["lobe_samples"] / 4) for layer, pulse in pulses.items()
        }
        if case["case_id"] == "jamshoro":
            for layer, pulse in pulses.items():
                if pulse != sources[jamshoro_path]["layers"][layer]["pulse"]:
                    raise ValueError("Fresh Jamshoro seed-only pulse differs from prior comparator")
    registry = {
        "schema": "processed-evaluation-case-registry-v1",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": digest(manifest_path),
        "script_sha256": digest(__file__),
        "source_manifests": {
            str(path.resolve()): digest(path) for path in sources
        },
        "coordinate_contract": (
            "Native integer trace/sample/channel; no snapping or new seed selection"
        ),
        "scoring_contract": "Unchanged signed lobe and max(2 samples, initial seed lobe width / 4)",
        "frozen_pulse_source": str(frozen_source.resolve()),
        "frozen_pulse_source_sha256": digest(
            frozen_source / "gpr_layer_audit/processing/conventional_config.py"
        ),
        "evaluation_claim": "Grouped development generalization; all roads historically used",
        "case_aliases": {"gujrat-short": "gujrat-second", "gujrat-long": "gujrat-first"},
        "cases": cases,
    }
    write_new(output, registry)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--jamshoro-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = run(args)
    print(json.dumps({
        "status": "complete", "output": str(args.output.resolve()),
        "cases": [
            {key: case[key] for key in ("case_id", "record_id", "layers", "stride")}
            for case in registry["cases"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
