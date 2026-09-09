"""Bounded offline learning; no application-time fitting or pseudo-labels."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.processing.hybrid_evidence import PREPROCESSING_VERSION

from .dataset import make_targets
from .features import (
    competing_seed_clicks,
    extract_template,
    make_inputs,
    resample_grid,
    tile_starts,
)


def select_support_labels(chunks, split, layer):
    """Three input observations per physical road, separate from target observations."""
    groups = defaultdict(list)
    for chunk in chunks:
        if chunk["split"] != split:
            continue
        for label in chunk["labels"]:
            if int(label["layer_order"]) == layer and label["visibility"] == "visible":
                groups[chunk["road_group"]].append((chunk, label))
    selected = {}
    for group, observations in groups.items():
        observations.sort(
            key=lambda pair: (pair[0]["source_sha256"], float(pair[1]["trace_index"]))
        )
        indices = np.unique(
            np.rint(np.linspace(0, len(observations) - 1, min(3, len(observations)))).astype(int)
        )
        selected[group] = [observations[i] for i in indices]
    return selected


def examples(dataset, split, layers, dt, dx, *, augment=False, seed=42):
    rng = np.random.default_rng(seed)
    chunks = [c for c in dataset["chunks"] if c["split"] == split]
    if augment:
        rng.shuffle(chunks)
    for layer in layers:
        support = select_support_labels(dataset["chunks"], split, layer)
        for chunk in chunks:
            raw = np.load(chunk["path"], mmap_mode="r", allow_pickle=False)
            data = resample_grid(
                raw, chunk["sample_interval_ns"], chunk["horizontal_step_m"], dt, dx
            )
            labels = []
            for label in chunk["labels"]:
                if int(label["layer_order"]) != layer:
                    continue
                labels.append(
                    {
                        **label,
                        "row": int(round(label["row"] * chunk["horizontal_step_m"] / dx)),
                        "sample_aligned": (
                            label["sample_aligned"] * chunk["sample_interval_ns"] / dt
                            if label["sample_aligned"] is not None
                            else None
                        ),
                        "pulse_width_samples": float(label.get("pulse_width_samples", 7))
                        * chunk["sample_interval_ns"]
                        / dt,
                    }
                )
            templates, clicks, excluded = [], [], set()
            for source, label in support.get(chunk["road_group"], []):
                support_data = resample_grid(
                    np.load(source["path"], mmap_mode="r", allow_pickle=False),
                    source["sample_interval_ns"],
                    source["horizontal_step_m"],
                    dt,
                    dx,
                )
                row = int(round(label["row"] * source["horizontal_step_m"] / dx))
                sample = label["sample_aligned"] * source["sample_interval_ns"] / dt
                templates.append(extract_template(support_data, row, sample))
                if source["path"] == chunk["path"]:
                    clicks.append((row, sample))
                    excluded.add(row)
            if not templates:
                continue
            if augment:
                shift = int(rng.integers(-3, 4))
                from gpr_layer_audit.processing.pipeline import _shift_sample_axis

                data = _shift_sample_axis(data, shift)
                data *= float(rng.uniform(0.7, 1.3))
                data += rng.normal(0, max(float(np.std(data)) * 0.015, 1e-9), data.shape).astype(
                    np.float32
                )
                for label in labels:
                    if label["sample_aligned"] is not None:
                        label["sample_aligned"] += shift
                clicks = [(row, sample + shift) for row, sample in clicks]
            inputs = make_inputs(
                data, layer, templates, clicks, competing_seed_clicks(data, clicks)
            )
            for x in tile_starts(len(data), 256):
                for y in tile_starts(data.shape[1], 512):
                    local_labels = [
                        {
                            **r,
                            "row": r["row"] - x,
                            "sample_aligned": r["sample_aligned"] - y
                            if r["sample_aligned"] is not None
                            else None,
                        }
                        for r in labels
                        if x <= r["row"] < x + 256
                        and (r["sample_aligned"] is None or y <= r["sample_aligned"] < y + 512)
                        and r["row"] not in excluded
                    ]
                    if not local_labels:
                        continue
                    values = inputs[:, x : x + 256, y : y + 512]
                    values = np.pad(
                        values, ((0, 0), (0, 256 - values.shape[1]), (0, 512 - values.shape[2]))
                    )
                    targets = make_targets((256, 512), local_labels, layer)
                    yield values, targets


def train_model(dataset_path, output_dir, *, layers=None, epochs=50, patience=8, device="auto"):
    dataset = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    eligible = [int(k) for k, v in dataset["eligibility"].items() if v["eligible_for_pilot"]]
    layers = sorted(eligible if layers is None else layers)
    if not layers or any(layer not in eligible for layer in layers):
        raise ValueError(
            "No eligible pilot: need 3 training roads, 96 verified aligned labels, "
            "and validation labels per layer"
        )
    if not dataset.get("chunks"):
        raise ValueError("Build the numeric dataset before training")
    if not 1 <= epochs <= 50 or not 1 <= patience <= 50:
        raise ValueError("Training is bounded to 1..50 epochs and patience")
    # Check physical-group isolation even if a manifest was edited after audit.
    group_splits = defaultdict(set)
    source_splits = defaultdict(set)
    for chunk in dataset["chunks"]:
        group_splits[chunk["road_group"]].add(chunk["split"])
        source_splits[chunk["source_sha256"]].add(chunk["split"])
        if any(
            r.get("training_use") != "allowed" or str(r.get("verified")).lower() != "true"
            for r in chunk["labels"]
        ):
            raise ValueError("Unverified or evaluation-only training annotation")
    if any(len(splits) != 1 for splits in (*group_splits.values(), *source_splits.values())):
        raise ValueError("Physical road leaked across dataset splits")
    import torch

    from .model import SeedUNet, masked_loss

    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeedUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    dt = float(
        np.median([c["sample_interval_ns"] for c in dataset["chunks"] if c["split"] == "train"])
    )
    dx = 0.4
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("inf"), 0
    for epoch in range(epochs):
        totals = {}
        for split in ("train", "validation"):
            model.train(split == "train")
            losses = []
            for values, targets in examples(
                dataset, split, layers, dt, dx, augment=split == "train", seed=42 + epoch
            ):
                tensors = [torch.from_numpy(np.asarray(v))[None].to(device) for v in targets]
                inputs = torch.from_numpy(values[None]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with (
                    torch.set_grad_enabled(split == "train"),
                    torch.amp.autocast("cuda", enabled=device.startswith("cuda")),
                ):
                    logits, visible = model(inputs)
                    loss = masked_loss(logits, visible, *tensors)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; no model promoted")
                if split == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                losses.append(float(loss.detach().cpu()))
            if not losses:
                raise ValueError(f"No non-seed supervised crops in {split}; more labels are needed")
            totals[split] = float(np.mean(losses))
        history.append({"epoch": epoch + 1, **totals})
        print(json.dumps(history[-1]), flush=True)
        if totals["validation"] < best:
            best, stale = totals["validation"], 0
            torch.save(model.state_dict(), output / "weights.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    manifest = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "architecture": "SeedUNet",
        "width": 16,
        "crop_shape": [256, 512],
        "sample_interval_ns": dt,
        "horizontal_step_m": dx,
        "supported_layers": layers,
        "weights": "weights.pt",
        "weights_sha256": fingerprint_file(output / "weights.pt"),
        "training_dataset_sha256": fingerprint_file(dataset_path),
        "training_source_sha256": sorted(
            {c["source_sha256"] for c in dataset["chunks"] if c["split"] == "train"}
        ),
        "training_road_groups": sorted(
            g for g, splits in group_splits.items() if "train" in splits
        ),
        "validation_road_groups": sorted(
            g for g, splits in group_splits.items() if "validation" in splits
        ),
        "history": history,
        "device": device,
        "calibration": {"temperature": 1.0, "fusion_weight": 0.20, "validated": False},
        "promotion": {"passed": False, "reason": "requires_held_out_hybrid_evaluation"},
    }
    (output / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
