"""Frozen Gujrat transfer of the unchanged adjacent-patch Jamshoro comparison.

Run selftest, control, encode, edges, predict, evaluate. The control uses radar
extrema/shoulders and native patch usability, exactly as the prior PatchNet
contract. Only operating seeds are read before evaluate. No training or tuning.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import experiment_local_patch_edges as runner
import numpy as np
from experiment_jamshoro_guide import arrays
from patchnet_model import native_candidates, patches
from seeded_eval import sha, source_manifest, validate_frozen_helpers

from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.processing import seeded_challenger as dense
from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "exports/seeded-tracker/local-patch-edges-v1"
GUIDE = BASE / "gujrat-guide"
OUTPUT = BASE / "gujrat"
MANIFEST = ROOT / "benchmarks/seeded-evaluation-inputs.json"
SEEDS = ROOT / "benchmarks/seeded-gujrat-second-seeds.json"
JAM = BASE / "jamshoro/contract.json"
SCRIPTS = runner.SCRIPTS + (Path(__file__).name,)
read, write = runner.read, runner.write


def seed_coordinates(observations, shape, stride):
    """Keep exact native coordinates; an off-grid observation cannot be snapped."""
    native = {}
    for point in observations:
        row, sample = point["trace"], point["sample"]
        if not isinstance(row, int) or not isinstance(sample, int):
            raise ValueError("Integral native seed coordinates required")
        if point["channel"] != 0 or not (0 <= row < shape[0] and 0 <= sample < shape[1]):
            raise ValueError("Seed channel/coordinates outside native radar")
        if row % stride:
            raise ValueError("Native seed is not on the retained grid; snapping forbidden")
        if row in native:
            raise ValueError("Duplicate operating seed trace")
        native[row] = sample
    if len(native) != 3:
        raise ValueError("Frozen transfer requires the original three observations per layer")
    return native, {row // stride: sample for row, sample in native.items()}


def dependencies():
    return {
        "package": source_manifest(ROOT / "src"),
        "scripts": {name: sha(ROOT / "scripts" / name) for name in SCRIPTS},
        "manifest": sha(MANIFEST),
        "seeds": sha(SEEDS),
        "jamshoro_contract": sha(JAM),
        "model": sha(runner.TRAIN / "model.pt"),
        "training_contract": sha(runner.TRAIN / "contract.json"),
        "guide_path": str(GUIDE.resolve()),
        "numpy": np.__version__,
        "torch": runner.torch.__version__,
    }


def verify_common_source():
    """Transfer is the identical experiment, including source and numerical settings."""
    frozen = read(JAM)
    current = dependencies()
    for key in ("package", "model", "training_contract", "numpy", "torch"):
        if current[key] != frozen["dependencies"][key]:
            raise ValueError(f"Jamshoro dependency changed: {key}")
    for name, digest in frozen["dependencies"]["scripts"].items():
        if current["scripts"][name] != digest:
            raise ValueError(f"Jamshoro source changed: {name}")
    assert frozen["stride"] == runner.STRIDE == 4
    assert frozen["config"] == asdict(dense.PacketConfig(guide_weight=1))
    assert frozen["patch"] == asdict(runner.PATCH)
    return current


def control():
    if GUIDE.exists():
        raise ValueError("Preserve existing Gujrat control; output must be new")
    dependency = verify_common_source()
    case = read(MANIFEST)["cases"]["gujrat-second"]
    assert case["seed_sha256"] == sha(SEEDS)
    assert (ROOT / case["seed_source"]).resolve() == SEEDS.resolve()
    seeds = read(SEEDS)
    assert seeds["dzt_sha256"] == case["dzt_sha256"]
    assert seeds["dzx_sha256"] == case["dzx_sha256"]
    dzt_path = ROOT / case["dzt"]
    assert sha(dzt_path) == case["dzt_sha256"]
    radar = DZTFile(dzt_path)
    native = np.asarray(radar.channel(), np.float32)
    valid_native = processed_boundary_mask(native)
    assert list(native.shape) == case["dimensions"]
    assert radar.header.sample_interval_ns == case["dt_ns"]
    assert radar.header.distance_per_trace_m == case["native_dx_m"]
    assert radar.header.position_ns == case["header_time_origin_ns"]
    dt, dx, stride = case["dt_ns"], case["native_dx_m"], runner.STRIDE
    assert dx * stride == 0.1
    entry = {
        "road": "gujrat",
        "dzt": str(dzt_path.resolve()),
        "dzx": str((ROOT / case["dzx"]).resolve()),
        "dzt_sha256": case["dzt_sha256"],
        "dzx_sha256": case["dzx_sha256"],
        "dt_ns": dt,
        "dx_m": dx,
        "origin_ns": case["header_time_origin_ns"],
        "dimensions": list(native.shape),
        "coordinate_audit": case["layer_audit"],
        "orientation": case["orientation"],
        "layer_mapping": "User confirmed Layer1=asphalt, Layer2=base, Layer3=subbase",
        "seeds": {},
    }
    layers = {}
    for order in (2, 3):
        operating, retained = seed_coordinates(
            seeds["observations"][str(order)], native.shape, stride
        )
        entry["seeds"][str(order)] = operating
        for point in seeds["observations"][str(order)]:
            row, sample = point["trace"], point["sample"]
            assert valid_native[row, sample]
            assert native[row, sample] == point["recorded_amplitude"]
            assert abs(entry["origin_ns"] + sample * dt - point["time_ns"]) < 1e-6
        pulse = resolve_pulse(native, valid_native, operating, {}, dt, ConventionalConfig())
        layers[str(order)] = {
            "seeds": retained,
            "pulse": pulse.metadata(),
            "lobe_samples": pulse.lobe_samples,
        }
    contract = {
        "version": "local-patch-gujrat-control-v1",
        "input": entry,
        "layers": layers,
        "dependencies": dependency,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "reference_opened_for_prediction": False,
        "metadata_only": (
            "Previously frozen coordinate audit copied; no DZX reads before evaluation"
        ),
        "candidate_contract": (
            "Radar valid-run extrema/shoulders plus exact seeds; native patches >=75% usable; "
            "intersect packet support"
        ),
        "pulse_contract": "Per-layer pulse from original native operating seeds only",
        "configurations": {"1": asdict(dense.PacketConfig(guide_weight=1))},
        "stride": stride,
        "claim": (
            "Previously used development road; interpretation only; no physical thickness claim"
        ),
        "runtime_s": {},
    }
    GUIDE.mkdir(parents=True)
    write(GUIDE / "contract.json", contract)
    source = GUIDE / "source"
    source.mkdir()
    for name in SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, source / name)
    shutil.copy2(dense.__file__, source / "seeded_challenger.py")
    data, valid = native[::stride], valid_native[::stride]
    surface = int(np.floor(-entry["origin_ns"] / dt))
    candidate = native_candidates(data, valid, surface)
    masks = {"radar_candidate": candidate, "native_valid": valid_native}
    original = dense._packet_correlations
    for order in (2, 3):
        start = time.perf_counter()
        anchors = layers[str(order)]["seeds"]
        allowed = candidate.copy()
        for row, sample in anchors.items():
            allowed[row, sample] = True
        rows, samples = np.nonzero(allowed)
        for a in range(0, len(rows), runner.BATCH):
            rr, ss = rows[a : a + runner.BATCH], samples[a : a + runner.BATCH]
            _, usable = patches(native, valid_native, rr * stride, ss, dt, dx)
            allowed[rr[~usable], ss[~usable]] = False
        assert all(allowed[row, sample] for row, sample in anchors.items())
        masks[f"allowed_layer{order}"] = allowed

        def correlations(measurement, mask, operating, radius, *, permitted=allowed):
            bank, supported = original(measurement, mask, operating, radius)
            return bank, supported & permitted

        dense._packet_correlations = correlations
        try:
            result = dense.pick_seeded_packet(
                data,
                anchors,
                valid=valid,
                dt_ns=dt,
                dx_m=dx * stride,
                pulse_width_samples=layers[str(order)]["lobe_samples"],
                reference_surface=surface,
                config=dense.PacketConfig(guide_weight=1),
            )
        finally:
            dense._packet_correlations = original
        values = arrays(result)
        assert all(values["samples"][row] == sample for row, sample in anchors.items())
        assert not np.any(np.isfinite(values["candidate"]) & ~allowed)
        filename = f"guide1-layer{order}.npz"
        np.savez_compressed(GUIDE / filename, **values)
        contract["runtime_s"][filename] = time.perf_counter() - start
        print(filename, contract["runtime_s"][filename], flush=True)
    np.savez_compressed(GUIDE / "masks.npz", **masks)
    assert verify_common_source() == dependency
    contract["prediction_hashes"] = {p.name: sha(p) for p in GUIDE.glob("*.npz")}
    contract["frozen_helpers"] = validate_frozen_helpers(ROOT / "src")
    write(GUIDE / "prediction-manifest.json", contract)


def transfer(stage):
    current = verify_common_source()
    frozen = read(GUIDE / "prediction-manifest.json")
    if current != frozen["dependencies"]:
        raise ValueError("Gujrat wrapper, guide location or control dependency changed")
    for name, digest in frozen["prediction_hashes"].items():
        if sha(GUIDE / name) != digest:
            raise ValueError(f"Gujrat control artifact changed: {name}")
    # The original runner's dependencies() resolves these globals every time.
    # Extending SCRIPTS both hashes and captures this wrapper at every stage.
    runner.GUIDE = GUIDE
    runner.SCRIPTS = SCRIPTS
    getattr(runner, stage)(OUTPUT)
    assert verify_common_source() == current


def selftest():
    points = [{"trace": r, "sample": s, "channel": 0} for r, s in ((4, 2), (12, 5), (20, 7))]
    native, retained = seed_coordinates(points, (24, 8), 4)
    assert native == {4: 2, 12: 5, 20: 7}
    assert retained == {1: 2, 3: 5, 5: 7}
    for change in ({"trace": 5}, {"trace": 24}, {"sample": 8}, {"channel": 1}, {"trace": 4.0}):
        bad = [dict(p) for p in points]
        bad[0].update(change)
        try:
            seed_coordinates(bad, (24, 8), 4)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid native coordinates accepted: {change}")
    verify_common_source()
    print("Exact-coordinate and unchanged Jamshoro numerical/source contracts pass", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("selftest", "control", "encode", "edges", "predict", "evaluate")
    )
    args = parser.parse_args()
    if args.mode in ("selftest", "control"):
        globals()[args.mode]()
    else:
        transfer(args.mode)
