"""One Mandiali-trained native CNN and common dense-graph development experiment.

prepare -> train -> predict -> evaluate. References are unavailable to predict.
All outputs are isolated; published controls and application dispatch are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import patchnet_model
import torch
from patchnet_model import PATCH, PatchMatcher, native_candidates, patches, targets
from seeded_eval import validate_frozen_helpers
from torch import nn

from gpr_layer_audit.conventional import layer_metrics
from gpr_layer_audit.conventional_reference import audit_reference, distributed_seeds
from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.io.dzx import read_dzx
from gpr_layer_audit.ml.processed_correspondence import interval_weights
from gpr_layer_audit.processing import seeded_challenger as dense
from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exports/seeded-tracker/patchnet-v1"
MANIFEST = ROOT / "benchmarks/seeded-evaluation-inputs.json"
CONTRACT = {
    "version": "native-patchnet-dense-v2-mask",
    "seed": 20260910,
    "training_roads": ["mandiali"],
    "development_calibration_roads": ["jamshoro"],
    "later_development_transfer_roads": ["gujrat"],
    "previous_use": "All roads have historical development use; no untouched-road claim",
    "patch": asdict(PATCH),
    "epochs": 12,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "maximum_rows_per_file_layer": 3000,
    "architecture": "shared three-convolution 64-dimensional patch encoder and two score heads",
    "targets": "Separate reviewed timing and signed-lobe targets; unknown rows ignored",
    "support": "Seed rows excluded; normalization uses measured samples and explicit masks",
    "candidate_mask": "Extrema computed inside contiguous finite valid runs only",
    "graph": "existing exact dense packet DP; same extrema/shoulder observable states in both arms",
    "numerical_gap": "Every sample remains a latent DP position; gaps emit no measurement",
    "scores": "0.5 NCC + 0.5 (2*CNN timing score-1), then existing interval seed blend",
    "confidence": "Classifier scores and path margins are uncalibrated; no probability claim",
    "selection": "Fixed final epoch; no architecture sweep or checkpoint selection on outcomes",
    "gate": "Require useful coverage at 95% accepted agreement without added wrong length",
    "dense_configuration": asdict(dense.PacketConfig()),
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_map(path, metadata, order, stride=1):
    original = read(MANIFEST)["cases"]["mandiali-short"]
    if path.resolve() == (ROOT / original["dzx"]).resolve():
        return {
            p["trace"]: p["sample"]
            for p in read(ROOT / original["seed_source"])["observations"][str(order)]
        }
    layer = next(layer for layer in metadata.layers if layer.number + 1 == order)
    eligible = sorted(
        (p for p in layer.picks if p.channel == 0 and p.trace % stride == 0), key=lambda p: p.trace
    )
    return {p.trace: p.sample for p in distributed_seeds(eligible)}


def prepare():
    if (OUT / "contract.json").exists():
        raise ValueError("Preserve prepared experiment")
    OUT.mkdir(parents=True, exist_ok=True)
    validate_frozen_helpers(ROOT / "src")
    write(OUT / "contract.json", CONTRACT)
    source = OUT / "source"
    source.mkdir()
    for script in (Path(__file__), Path(patchnet_model.__file__)):
        shutil.copy2(script, source / script.name)
    write(
        OUT / "source.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "scripts": {p.name: sha(p) for p in source.iterdir()},
            "frozen_helpers": validate_frozen_helpers(ROOT / "src"),
            "manifest_sha256": sha(MANIFEST),
        },
    )
    inventory = read(MANIFEST)["reviewed_inventory"]
    reports, inputs, x_parts, y_parts, partners, seed_parts = [], [], [], [], [], []
    seed_count = 0
    for record in inventory:
        if record["physical_road_group"] not in ("mandiali", "jamshoro"):
            continue
        path = ROOT / record["dzx"]
        assert sha(path) == record["sha256"]
        audit = audit_reference(path)
        assert not audit["issues"]
        assert all(
            not layer["issues"]
            and layer["amplitude_match_fraction"] == 1
            and layer["time_mapping_status"] == "verified_header"
            for layer in audit["layers"]
        )
        radar = DZTFile(path.with_suffix(".DZT"))
        data = np.asarray(radar.channel(), np.float32)
        valid = processed_boundary_mask(data)
        dt, dx = radar.header.sample_interval_ns, radar.header.distance_per_trace_m
        surface = int(np.floor(-radar.header.position_ns / dt))
        metadata = read_dzx(path)
        record_input = {
            "road": record["physical_road_group"],
            "dzt": str(path.with_suffix(".DZT")),
            "dzx": str(path),
            "dzt_sha256": sha(path.with_suffix(".DZT")),
            "dzx_sha256": sha(path),
            "dt_ns": dt,
            "dx_m": dx,
            "origin_ns": radar.header.position_ns,
            "dimensions": list(data.shape),
            "coordinate_audit": audit["layers"],
            "seeds": {},
        }
        observable = (
            native_candidates(data, valid, surface) if record_input["road"] == "mandiali" else None
        )
        for order in (2, 3):
            anchors = seed_map(
                path, metadata, order, 4 if record_input["road"] == "jamshoro" else 1
            )
            record_input["seeds"][str(order)] = anchors
            if record_input["road"] != "mandiali":
                continue
            pulse = resolve_pulse(data, valid, anchors, {}, dt, ConventionalConfig())
            tolerance = max(2, pulse.lobe_samples / 4)
            layer = next(layer for layer in metadata.layers if layer.number + 1 == order)
            reviewed = sorted(
                (p for p in layer.picks if p.channel == 0 and p.trace not in anchors),
                key=lambda p: p.trace,
            )
            skip = max(1, int(np.ceil(len(reviewed) / CONTRACT["maximum_rows_per_file_layer"])))
            reviewed = reviewed[::skip]
            rows, columns, labels, missing = [], [], [], 0
            for point in reviewed:
                candidates = np.flatnonzero(observable[point.trace])
                target = targets(data[point.trace], candidates, point.sample, tolerance)
                positive = np.flatnonzero(target[:, 0] == 1)
                negative = np.flatnonzero(
                    (target[:, 0] == 0) & (abs(candidates - point.sample) <= 4 * pulse.lobe_samples)
                )
                if not len(positive) or not len(negative):
                    missing += 1
                    continue
                # Balanced row contribution; nearest hard negatives include same-lobe timing misses.
                positive = positive[np.argsort(abs(candidates[positive] - point.sample))[:3]]
                negative = negative[np.argsort(abs(candidates[negative] - point.sample))[:6]]
                chosen = np.r_[positive, negative]
                rows.extend([point.trace] * len(chosen))
                columns.extend(candidates[chosen])
                labels.extend(target[chosen])
            rows, columns = np.asarray(rows), np.asarray(columns)
            x, usable = patches(data, valid, rows, columns, dt, dx)
            sr, ss = np.asarray(sorted(anchors.items())).T
            seed_x, seed_valid = patches(data, valid, sr, ss, dt, dx)
            assert np.all(seed_valid)
            mix = interval_weights(rows, sr)
            partner = np.argmax(mix, axis=1) + seed_count
            seed_count += len(seed_x)
            x_parts.append(x[usable])
            y_parts.append(np.asarray(labels)[usable])
            partners.append(partner[usable])
            seed_parts.append(seed_x)
            assert not set(rows) & set(anchors)
            reports.append(
                {
                    "file": str(path),
                    "layer": order,
                    "sampled_rows": len(reviewed),
                    "rows_without_training_pair": missing,
                    "pairs": int(usable.sum()),
                    "positive_timing": int(np.asarray(labels)[usable, 0].sum()),
                    "positive_signed_lobe": int(np.asarray(labels)[usable, 1].sum()),
                    "tolerance_samples": tolerance,
                    "pulse": pulse.metadata(),
                }
            )
            print(f"Prepared {path.stem} layer{order}: {usable.sum()} training pairs", flush=True)
        inputs.append(record_input)
    np.savez(
        OUT / "training.npz",
        x=np.concatenate(x_parts),
        y=np.concatenate(y_parts),
        partner=np.concatenate(partners),
        seeds=np.concatenate(seed_parts),
    )
    write(OUT / "inputs.json", inputs)
    write(
        OUT / "training-data.json",
        {
            "reports": reports,
            "training_sha256": sha(OUT / "training.npz"),
            "roads": ["mandiali"],
            "unknown_rows_used": False,
        },
    )


def train():
    if (OUT / "model.pt").exists():
        raise ValueError("Preserve trained model")
    assert read(OUT / "contract.json") == CONTRACT
    torch.manual_seed(CONTRACT["seed"])
    torch.set_num_threads(4)
    rng = np.random.default_rng(CONTRACT["seed"])
    data = np.load(OUT / "training.npz")
    assert sha(OUT / "training.npz") == read(OUT / "training-data.json")["training_sha256"]
    x, y, partner, seeds = [torch.from_numpy(data[key]) for key in ("x", "y", "partner", "seeds")]
    model = PatchMatcher()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONTRACT["learning_rate"], weight_decay=CONTRACT["weight_decay"]
    )
    positive_weight = (len(y) - y.sum(0)) / y.sum(0).clamp_min(1)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")
    history = []
    for epoch in range(CONTRACT["epochs"]):
        start = time.perf_counter()
        indices = rng.permutation(len(y))
        total = 0.0
        model.train()
        for offset in range(0, len(indices), CONTRACT["batch_size"]):
            use = torch.from_numpy(indices[offset : offset + CONTRACT["batch_size"]])
            logits = model(x[use], seeds[partner[use]])
            errors = loss_function(logits, y[use])
            loss = (errors[:, 0] + 0.25 * errors[:, 1]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(use)
        item = {
            "epoch": epoch + 1,
            "loss": total / len(y),
            "runtime_s": time.perf_counter() - start,
        }
        history.append(item)
        write(OUT / "training-progress.json", history)
        print(json.dumps(item), flush=True)
    torch.save(model.state_dict(), OUT / "model.pt")
    write(
        OUT / "training-result.json",
        {
            "model_sha256": sha(OUT / "model.pt"),
            "training_data_sha256": sha(OUT / "training.npz"),
            "epochs": history,
            "torch": torch.__version__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "no_evaluation_road_labels_used": True,
        },
    )


def predict():
    target = OUT / "jamshoro-predictions"
    if target.exists():
        raise ValueError("Preserve predictions")
    target.mkdir()
    entry = next(item for item in read(OUT / "inputs.json") if item["road"] == "jamshoro")
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    assert sha(OUT / "model.pt") == read(OUT / "training-result.json")["model_sha256"]
    torch.set_num_threads(4)
    model = PatchMatcher()
    model.load_state_dict(torch.load(OUT / "model.pt", weights_only=True))
    model.eval()
    native = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)
    native_valid = processed_boundary_mask(native)
    stride, dt, dx = 4, entry["dt_ns"], entry["dx_m"]
    data, valid = native[::stride], native_valid[::stride]
    surface = int(np.floor(-entry["origin_ns"] / dt))
    candidate = native_candidates(data, valid, surface)
    original_correlation = dense._packet_correlations
    provenance = {
        "input": entry,
        "model_sha256": sha(OUT / "model.pt"),
        "contract_sha256": sha(OUT / "contract.json"),
        "reference_opened_for_prediction": False,
        "layers": {},
    }
    for order in (2, 3):
        start = time.perf_counter()
        native_anchors = {int(row): sample for row, sample in entry["seeds"][str(order)].items()}
        anchors = {row // stride: sample for row, sample in native_anchors.items()}
        assert all(row % stride == 0 for row in native_anchors)
        pulse = resolve_pulse(native, native_valid, native_anchors, {}, dt, ConventionalConfig())
        allowed = candidate.copy()
        for row, sample in anchors.items():
            allowed[row, sample] = True
        rows, columns = np.nonzero(allowed)
        sr, ss = np.asarray(sorted(native_anchors.items())).T
        seed_x, seed_ok = patches(native, native_valid, sr, ss, dt, dx)
        assert seed_ok.all()
        learned = np.zeros((len(sr), *data.shape), np.float32)
        family = np.zeros_like(learned)
        with torch.inference_mode():
            seed_features = model.encoder(torch.from_numpy(seed_x))
            for offset in range(0, len(rows), 512):
                rr, cc = rows[offset : offset + 512], columns[offset : offset + 512]
                x, usable = patches(native, native_valid, rr * stride, cc, dt, dx)
                encoded = model.encoder(torch.from_numpy(x))
                for index, seed in enumerate(seed_features):
                    values = model.compare(encoded, seed.expand(len(x), -1)).sigmoid().numpy()
                    learned[index, rr, cc] = values[:, 0]
                    family[index, rr, cc] = values[:, 1]
                allowed[rr[~usable], cc[~usable]] = False
                if offset % 100000 < 512:
                    print(
                        f"Layer{order}: encoded {min(offset + 512, len(rows))}/{len(rows)}",
                        flush=True,
                    )
        for mode in ("classical", "learned"):

            def correlations(
                measurement,
                mask,
                operating,
                radius,
                *,
                arm=mode,
                predictions=learned,
                candidate_mask=allowed,
            ):
                bank, packet_valid = original_correlation(measurement, mask, operating, radius)
                if arm == "learned":
                    bank = {
                        row: (bank[row] + 2 * predictions[i] - 1) / 2
                        for i, row in enumerate(sorted(operating))
                    }
                return bank, packet_valid & candidate_mask

            dense._packet_correlations = correlations
            try:
                path = dense.pick_seeded_packet(
                    data,
                    anchors,
                    valid=valid,
                    dt_ns=dt,
                    dx_m=dx * stride,
                    pulse_width_samples=pulse.lobe_samples,
                    reference_surface=surface,
                )
            finally:
                dense._packet_correlations = original_correlation
            np.savez_compressed(
                target / f"{mode}-layer{order}.npz",
                samples=path.samples,
                provisional=path.provisional_samples,
                visible=path.visible,
                alternate=path.alternate_samples,
                margin=path.confidence,
                candidate=path.candidate_components["audit_candidate_rank"],
                correlation=path.evidence["packet_seed_correlation"],
            )
            assert all(path.samples[row] == sample for row, sample in anchors.items())
        np.savez_compressed(
            target / f"learned-evidence-layer{order}.npz", timing=learned, family=family
        )
        provenance["layers"][str(order)] = {
            "seeds": anchors,
            "pulse": pulse.metadata(),
            "lobe_samples": pulse.lobe_samples,
            "runtime_s": time.perf_counter() - start,
        }
        write(target / "progress.json", provenance)
    provenance["prediction_hashes"] = {p.name: sha(p) for p in target.glob("*.npz")}
    write(target / "prediction-manifest.json", provenance)


def evaluate():
    target = OUT / "jamshoro-predictions"
    if (target / "evaluation.json").exists():
        raise ValueError("Preserve scored result")
    provenance = read(target / "prediction-manifest.json")
    for name, expected in provenance["prediction_hashes"].items():
        assert sha(target / name) == expected
    entry = provenance["input"]
    assert sha(entry["dzx"]) == entry["dzx_sha256"]
    metadata = read_dzx(entry["dzx"])
    data = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)[::4]
    results = {}
    for order in (2, 3):
        layer = next(layer for layer in metadata.layers if layer.number + 1 == order)
        references = [
            replace(p, trace=p.trace // 4)
            for p in layer.picks
            if p.channel == 0 and p.trace % 4 == 0
        ]
        anchors = {
            int(row): sample for row, sample in provenance["layers"][str(order)]["seeds"].items()
        }
        for mode in ("classical", "learned"):
            arrays = np.load(target / f"{mode}-layer{order}.npz")
            path = SimpleNamespace(
                samples=arrays["samples"],
                provisional_samples=arrays["provisional"],
                visible=arrays["visible"],
                candidate_components={"audit_candidate_rank": arrays["candidate"]},
                evidence={"hybrid_path_margin": arrays["margin"]},
            )
            metrics = layer_metrics(
                path,
                references,
                anchors,
                data,
                provenance["layers"][str(order)]["lobe_samples"],
                entry["dx_m"] * 4,
            )
            results[f"{mode}-{order}"] = metrics
            print(
                mode,
                order,
                metrics["accepted_agree"],
                metrics["accepted"],
                metrics["correct_coverage"],
                flush=True,
            )
    write(
        target / "evaluation.json",
        {
            "results": results,
            "provenance": provenance,
            "claim": "Previously used Jamshoro interpretation; historical layer numbers only",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "predict", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    args = parser.parse_args()
    ROOT = args.workspace_root.resolve()
    MANIFEST = ROOT / "benchmarks/seeded-evaluation-inputs.json"
    OUT = args.output.resolve()
    globals()[args.mode]()
