"""Test the existing seed guide on Jamshoro with an exactly reproduced dense control.

Only guide_weight changes from 0 to the existing Mandiali value of 1. No learned
score, threshold tuning, new candidate, reference corridor or implicit snap is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from seeded_eval import sha, source_manifest, validate_frozen_helpers

from gpr_layer_audit.conventional import layer_metrics
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.io.dzx import read_dzx
from gpr_layer_audit.processing import seeded_challenger as dense
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "exports/seeded-tracker/patchnet-v2-mask/jamshoro-predictions"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def arrays(path):
    return dict(
        samples=path.samples,
        provisional=path.provisional_samples,
        visible=path.visible,
        alternate=path.alternate_samples,
        margin=path.confidence,
        candidate=path.candidate_components["audit_candidate_rank"],
        correlation=path.evidence["packet_seed_correlation"],
    )


def predict(output):
    if output.exists():
        raise ValueError("Preserve existing experiment")
    frozen = read(BASE / "prediction-manifest.json")
    entry = frozen["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    for path in (Path(__file__), Path(dense.__file__), Path(__file__).with_name("seeded_eval.py")):
        shutil.copy2(path, source / path.name)
    source_hash = source_manifest(ROOT / "src")
    manifest = {
        "input": entry,
        "baseline_manifest_sha256": sha(BASE / "prediction-manifest.json"),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source": source_hash,
        "script_sha256": sha(__file__),
        "frozen_helpers": validate_frozen_helpers(ROOT / "src"),
        "layers": frozen["layers"],
        "reference_opened_for_prediction": False,
        "candidate_contract": "Exact radar-only frozen observation mask; control equality required",
        "choice": "Existing Mandiali guide weight 1, fixed before this longer-road experiment",
        "claim": "Development interpretation; geometric prior is not independent identity evidence",
        "configurations": {str(w): asdict(dense.PacketConfig(guide_weight=w)) for w in (0, 1)},
        "runtime_s": {},
    }
    write(output / "contract.json", manifest)
    native = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)
    data = native[::4]
    valid = processed_boundary_mask(native)[::4]
    dt, dx = entry["dt_ns"], entry["dx_m"] * 4
    surface = int(np.floor(-entry["origin_ns"] / dt))
    original = dense._packet_correlations
    for order in (2, 3):
        filename = f"classical-layer{order}.npz"
        assert sha(BASE / filename) == frozen["prediction_hashes"][filename]
        reference = np.load(BASE / filename)
        observed = np.isfinite(reference["candidate"])
        anchors = {int(r): s for r, s in frozen["layers"][str(order)]["seeds"].items()}

        def correlations(measurement, mask, operating, radius, *, allowed=observed):
            bank, supported = original(measurement, mask, operating, radius)
            return bank, supported & allowed

        dense._packet_correlations = correlations
        try:
            for weight in (0, 1):
                start = time.perf_counter()
                result = dense.pick_seeded_packet(
                    data,
                    anchors,
                    valid=valid,
                    dt_ns=dt,
                    dx_m=dx,
                    pulse_width_samples=frozen["layers"][str(order)]["lobe_samples"],
                    reference_surface=surface,
                    config=dense.PacketConfig(guide_weight=weight),
                )
                values = arrays(result)
                if weight == 0:
                    for key, value in values.items():
                        np.testing.assert_array_equal(value, reference[key], err_msg=key)
                else:
                    np.testing.assert_array_equal(values["candidate"], reference["candidate"])
                for row, sample in anchors.items():
                    assert values["samples"][row] == sample
                name = f"guide{weight}-layer{order}"
                np.savez_compressed(output / f"{name}.npz", **values)
                manifest["runtime_s"][name] = time.perf_counter() - start
                print(name, manifest["runtime_s"][name], flush=True)
        finally:
            dense._packet_correlations = original
    manifest["control_arrays_exact"] = True
    manifest["prediction_hashes"] = {p.name: sha(p) for p in output.glob("*.npz")}
    assert source_manifest(ROOT / "src")["sha256"] == source_hash["sha256"]
    write(output / "prediction-manifest.json", manifest)


def evaluate(output):
    if (output / "evaluation.json").exists():
        raise ValueError("Preserve existing evaluation")
    manifest = read(output / "prediction-manifest.json")
    for name, digest in manifest["prediction_hashes"].items():
        assert sha(output / name) == digest
    entry = manifest["input"]
    assert sha(entry["dzx"]) == entry["dzx_sha256"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    validate_frozen_helpers(ROOT / "src")
    labels = read_dzx(entry["dzx"])
    data = np.asarray(DZTFile(entry["dzt"]).channel())[::4]
    results = {}
    for order in (2, 3):
        layer = next(layer for layer in labels.layers if layer.number + 1 == order)
        reviewed = [
            replace(p, trace=p.trace // 4)
            for p in layer.picks
            if p.channel == 0 and p.trace % 4 == 0
        ]
        anchors = {int(r): s for r, s in manifest["layers"][str(order)]["seeds"].items()}
        for weight in (0, 1):
            name = f"guide{weight}-layer{order}"
            values = np.load(output / f"{name}.npz")
            path = SimpleNamespace(
                samples=values["samples"],
                provisional_samples=values["provisional"],
                visible=values["visible"],
                candidate_components={"audit_candidate_rank": values["candidate"]},
                evidence={"hybrid_path_margin": values["margin"]},
            )
            result = layer_metrics(
                path,
                reviewed,
                anchors,
                data,
                manifest["layers"][str(order)]["lobe_samples"],
                entry["dx_m"] * 4,
            )
            results[name] = result
            print(name, {k: result[k] for k in ("accepted_agree", "accepted", "proposed_agree")})
    write(output / "evaluation.json", {"results": results, "manifest": manifest})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("predict", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args.output)
