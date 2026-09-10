"""Frozen adjacent local patch NCC/CNN comparison on a common Jamshoro graph.

Run encode, edges, predict, evaluate in order. No stage overwrites a completed
artifact. Prediction receives radar and operating seeds, never reviewed labels.
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
import torch
from experiment_jamshoro_guide import arrays
from local_patch_edge_model import common_patch_ncc, compare_forward, edge_costs, instrument
from patchnet_model import PATCH, PatchMatcher, patches
from seeded_eval import sha, source_manifest, validate_frozen_helpers

from gpr_layer_audit.conventional import layer_metrics
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.processing import seeded_challenger as dense
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "exports/seeded-tracker/local-patch-pairs-v1"
GUIDE = ROOT / "exports/seeded-tracker/jamshoro-guide-v1"
MODEL_SHA = "aac5a797f98eb4676927f5f088e2684d0a4689b0a9e210a5b79ac47a5f68575f"
SCRIPTS = (
    "experiment_local_patch_edges.py",
    "local_patch_edge_model.py",
    "patchnet_model.py",
    "experiment_jamshoro_guide.py",
    "seeded_eval.py",
)
STRIDE, BATCH = 4, 2048


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dependencies():
    return {
        "package": source_manifest(ROOT / "src"),
        "scripts": {name: sha(ROOT / "scripts" / name) for name in SCRIPTS},
        "model": sha(TRAIN / "model.pt"),
        "training_contract": sha(TRAIN / "contract.json"),
        "guide_manifest": sha(GUIDE / "prediction-manifest.json"),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def context(output):
    contract = read(output / "contract.json")
    if dependencies() != contract["dependencies"]:
        raise ValueError("Frozen source/model/dependencies changed")
    entry = contract["input"]
    if sha(entry["dzt"]) != entry["dzt_sha256"]:
        raise ValueError("Radar changed")
    native = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)
    valid = processed_boundary_mask(native)
    return contract, native, valid


def model():
    torch.set_num_threads(4)
    result = PatchMatcher().eval()
    if sha(TRAIN / "model.pt") != MODEL_SHA:
        raise ValueError("Wrong frozen trained model")
    result.load_state_dict(torch.load(TRAIN / "model.pt", weights_only=True))
    return result


def verify_stage(output, stage):
    manifest = read(output / f"{stage}.json")
    for name, digest in manifest["hashes"].items():
        if sha(output / name) != digest:
            raise ValueError(f"Stage artifact changed: {name}")
    return manifest


def encode(output):
    if output.exists():
        raise ValueError("Preserve existing experiment; output must be new")
    output.mkdir(parents=True)
    start = time.perf_counter()
    frozen = read(GUIDE / "prediction-manifest.json")
    contract = {
        "version": "local-patch-edges-v1",
        "input": frozen["input"],
        "layers": frozen["layers"],
        "patch": asdict(PATCH),
        "config": asdict(dense.PacketConfig(guide_weight=1)),
        "stride": STRIDE,
        "batch": BATCH,
        "edge_row_block": 32,
        "threads": 4,
        "dependencies": dependencies(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "claim": "Development interpretation only; no unused road or physical thickness claim",
        "reference_opened_for_encoding_edges_prediction": False,
        "supersedes": "Training legacy scores/graph text; planned_edge_cost retained",
        "candidate_bank": "Union of radar-only observable masks in exact frozen guided controls",
        "common_graph": "All sample states/offsets retained; shared evidence support; no pruning",
        "classical": {
            "normalization": "Existing per-lateral-A-scan temporal L2 native normalization",
            "ncc": "Float64 centered NCC over common valid amplitude pixels in 21x65 patches",
            "support": "Both patches usable; >=1024 common pixels; both centered energies >1e-12",
            "cost_per_metre": "1 - signed NCC; unsupported=0.65",
        },
        "learned": {
            "orientation": "model.compare(target at row+1, source at row)",
            "support": "Exactly the classical support; each encoder sees its own validity mask",
            "cost_per_metre": "2*(1-sigmoid(timing head)); unsupported=0.65",
            "model_score_is_calibrated": False,
        },
        "geometry_seed_unary_guide_gate": "Unchanged; exact same dense objective for margins",
        "known_gate_limit": "Legacy margin normalized by full interval; no threshold adjustment",
        "units": "edge[source_grid_row,offset_index,target_sample]; offset=target-source; cost*dx",
        "controls": frozen["prediction_hashes"],
    }
    assert contract["dependencies"]["model"] == MODEL_SHA
    write(output / "contract.json", contract)
    capture = output / "source"
    capture.mkdir()
    for name in SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, capture / name)
    shutil.copy2(dense.__file__, capture / "seeded_challenger.py")
    contract, native, valid = context(output)
    observed = np.zeros(native[::STRIDE].shape, bool)
    for order in (2, 3):
        name = f"guide1-layer{order}.npz"
        assert sha(GUIDE / name) == contract["controls"][name]
        with np.load(GUIDE / name) as values:
            observed |= np.isfinite(values["candidate"])
    rr, ss = np.where(observed)
    index = np.full(observed.shape, -1, np.int32)
    index[rr, ss] = np.arange(len(rr), dtype=np.int32)
    features = np.lib.format.open_memmap(
        output / "features.npy", mode="w+", dtype=np.float32, shape=(len(rr), 64)
    )
    usable = np.zeros(len(rr), bool)
    network = model()
    entry = contract["input"]
    with torch.inference_mode():
        for a in range(0, len(rr), BATCH):
            b = min(len(rr), a + BATCH)
            x, usable[a:b] = patches(
                native, valid, rr[a:b] * STRIDE, ss[a:b], entry["dt_ns"], entry["dx_m"]
            )
            features[a:b] = network.encoder(torch.from_numpy(x)).numpy()
            if a % (BATCH * 50) == 0:
                print("encode", b, "/", len(rr), round(time.perf_counter() - start, 1), flush=True)
    features.flush()
    np.save(output / "index.npy", index)
    np.save(output / "usable.npy", usable)
    np.savez_compressed(output / "coordinates.npz", row=rr, sample=ss)
    assert dependencies() == contract["dependencies"]
    write(
        output / "encode.json",
        {
            "candidates": len(rr),
            "usable": int(usable.sum()),
            "runtime_s": time.perf_counter() - start,
            "hashes": {
                name: sha(output / name)
                for name in ("features.npy", "index.npy", "usable.npy", "coordinates.npz")
            },
        },
    )


def edges(output):
    if (output / "edges.json").exists() or (output / "ncc-edges.npy").exists():
        raise ValueError("Preserve existing edge stage")
    start = time.perf_counter()
    verify_stage(output, "encode")
    contract, native, valid = context(output)
    entry = contract["input"]
    dx, dt = entry["dx_m"] * STRIDE, entry["dt_ns"]
    index = np.load(output / "index.npy")
    features = np.load(output / "features.npy", mmap_mode="r")
    usable = np.load(output / "usable.npy")
    rows, samples = index.shape
    shift = min(
        samples - 1, max(1, int(np.ceil(contract["config"]["max_slope_ns_per_m"] * dx / dt)))
    )
    shape = (rows - 1, 2 * shift + 1, samples)
    ncc = np.lib.format.open_memmap(
        output / "ncc-edges.npy", mode="w+", dtype=np.float32, shape=shape
    )
    cnn = np.lib.format.open_memmap(
        output / "cnn-edges.npy", mode="w+", dtype=np.float32, shape=shape
    )
    supported = np.lib.format.open_memmap(
        output / "edge-support.npy", mode="w+", dtype=bool, shape=shape
    )
    ncc[:] = cnn[:] = 0.65
    supported[:] = False
    network = model()
    count, eligible = 0, 0
    score_sum = np.zeros(2)
    with torch.inference_mode():
        for a in range(0, rows - 1, contract["edge_row_block"]):
            b = min(rows - 1, a + contract["edge_row_block"])
            rr, ss = np.where(index[a : b + 1] >= 0)
            ids = index[a : b + 1][rr, ss]
            raw, observed = patches(native, valid, (rr + a) * STRIDE, ss, dt, entry["dx_m"])
            np.testing.assert_array_equal(observed, usable[ids])
            local = np.full((b - a + 1, samples), -1, np.int32)
            local[rr, ss] = np.arange(len(rr), dtype=np.int32)
            for oi, offset in enumerate(range(-shift, shift + 1)):
                target = np.arange(max(0, offset), min(samples, samples + offset))
                source = target - offset
                pair = (local[:-1, source] >= 0) & (local[1:, target] >= 0)
                pr, pc = np.where(pair)
                tc, sc = target[pc], source[pc]
                eligible += len(pr)
                for p in range(0, len(pr), BATCH):
                    sl = slice(p, p + BATCH)
                    row, target_col, source_col = pr[sl], tc[sl], sc[sl]
                    si, ti = local[row, source_col], local[row + 1, target_col]
                    correlation, support = common_patch_ncc(
                        raw[si], raw[ti], observed[si], observed[ti]
                    )
                    sf = torch.from_numpy(np.array(features[ids[si]]))
                    tf = torch.from_numpy(np.array(features[ids[ti]]))
                    probability = torch.sigmoid(compare_forward(network, sf, tf)[:, 0]).numpy()
                    nc, cc = edge_costs(correlation, probability, support)
                    ncc[a + row, oi, target_col] = nc
                    cnn[a + row, oi, target_col] = cc
                    supported[a + row, oi, target_col] = support
                    count += int(support.sum())
                    score_sum += [correlation[support].sum(), probability[support].sum()]
            if a % (contract["edge_row_block"] * 10) == 0:
                print(
                    "edges",
                    b,
                    "/",
                    rows - 1,
                    "supported",
                    count,
                    "seconds",
                    round(time.perf_counter() - start, 1),
                    flush=True,
                )
    ncc.flush()
    cnn.flush()
    supported.flush()
    assert dependencies() == contract["dependencies"]
    write(
        output / "edges.json",
        {
            "shape": shape,
            "max_shift": shift,
            "dx_m": dx,
            "eligible_candidate_pairs": eligible,
            "supported_pairs": count,
            "mean_supported_scores": (score_sum / max(count, 1)).tolist(),
            "runtime_s": time.perf_counter() - start,
            "hashes": {
                name: sha(output / name)
                for name in ("ncc-edges.npy", "cnn-edges.npy", "edge-support.npy")
            },
        },
    )


def predict(output):
    if (output / "prediction.json").exists() or list(output.glob("*-layer?.npz")):
        raise ValueError("Preserve existing predictions")
    edge_manifest = verify_stage(output, "edges")
    contract, native, valid = context(output)
    entry = contract["input"]
    data, mask = native[::STRIDE], valid[::STRIDE]
    dt, dx = entry["dt_ns"], entry["dx_m"] * STRIDE
    surface = int(np.floor(-entry["origin_ns"] / dt))
    runtimes = {}
    original = dense._packet_correlations
    for order in (2, 3):
        name = f"guide1-layer{order}.npz"
        assert sha(GUIDE / name) == contract["controls"][name]
        reference = dict(np.load(GUIDE / name))
        observed = np.isfinite(reference["candidate"])
        anchors = {int(r): s for r, s in contract["layers"][str(order)]["seeds"].items()}

        def correlations(measurement, mask, operating, radius, *, allowed=observed):
            bank, supported = original(measurement, mask, operating, radius)
            return bank, supported & allowed

        dense._packet_correlations = correlations
        try:
            for arm in ("control", "ncc", "cnn"):
                start = time.perf_counter()
                costs = (
                    np.zeros(edge_manifest["shape"], np.float32)
                    if arm == "control"
                    else np.load(output / f"{arm}-edges.npy", mmap_mode="r")
                )
                picker = instrument(dense.pick_seeded_packet, costs)
                result = picker(
                    data,
                    anchors,
                    valid=mask,
                    dt_ns=dt,
                    dx_m=dx,
                    pulse_width_samples=contract["layers"][str(order)]["lobe_samples"],
                    reference_surface=surface,
                    config=dense.PacketConfig(**contract["config"]),
                )
                values = arrays(result)
                if arm == "control":
                    for key, value in values.items():
                        np.testing.assert_array_equal(value, reference[key], err_msg=key)
                np.testing.assert_array_equal(values["candidate"], reference["candidate"])
                for row, sample in anchors.items():
                    assert values["samples"][row] == sample
                filename = f"{arm}-layer{order}.npz"
                np.savez_compressed(output / filename, **values)
                runtimes[filename] = time.perf_counter() - start
                print(filename, runtimes[filename], flush=True)
        finally:
            dense._packet_correlations = original
    assert dependencies() == contract["dependencies"]
    write(
        output / "prediction.json",
        {
            "exact_control": True,
            "exact_candidates": True,
            "exact_native_seeds": True,
            "runtime_s": runtimes,
            "reference_opened": False,
            "hashes": {p.name: sha(p) for p in output.glob("*-layer?.npz")},
        },
    )


def evaluate(output):
    from gpr_layer_audit.io.dzx import read_dzx

    if (output / "evaluation.json").exists():
        raise ValueError("Preserve existing evaluation")
    verify_stage(output, "prediction")
    contract, native, _ = context(output)
    validate_frozen_helpers(ROOT / "src")
    entry = contract["input"]
    assert sha(entry["dzx"]) == entry["dzx_sha256"]
    labels = read_dzx(entry["dzx"])
    results = {}
    for order in (2, 3):
        layer = next(layer for layer in labels.layers if layer.number + 1 == order)
        reviewed = [
            replace(p, trace=p.trace // STRIDE)
            for p in layer.picks
            if p.channel == 0 and p.trace % STRIDE == 0
        ]
        anchors = {int(r): s for r, s in contract["layers"][str(order)]["seeds"].items()}
        for arm in ("control", "ncc", "cnn"):
            values = np.load(output / f"{arm}-layer{order}.npz")
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
                native[::STRIDE],
                contract["layers"][str(order)]["lobe_samples"],
                entry["dx_m"] * STRIDE,
            )
            results[f"{arm}-layer{order}"] = result
            print(
                arm,
                order,
                {k: result[k] for k in ("accepted_agree", "accepted", "proposed_agree")},
                flush=True,
            )
    write(
        output / "evaluation.json",
        {"results": results, "contract_sha256": sha(output / "contract.json")},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("encode", "edges", "predict", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args.output.resolve())
