"""Bounded overlapping inference with an explicit model-bundle contract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from gpr_layer_audit.io.dzt import fingerprint_file
from gpr_layer_audit.processing.hybrid_evidence import PREPROCESSING_VERSION, LearnedEvidence

from .features import (
    competing_seed_clicks,
    extract_template,
    make_inputs,
    resample_grid,
    tile_starts,
)


@lru_cache(maxsize=2)
def _load_model(weight_path, digest, width, device, file_stamp):
    import torch

    from .model import SeedUNet

    if fingerprint_file(weight_path) != digest:
        raise ValueError("Model weights fingerprint mismatch")
    model = SeedUNet(width)
    model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def infer_evidence(
    bundle, measurement, anchors, dt, dx, *, cancel=None, require_validated=True, device="auto"
):
    path = Path(bundle)
    manifest_path = path / "model.json" if path.is_dir() else path
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("preprocessing_version") != PREPROCESSING_VERSION
    ):
        raise ValueError("Incompatible model preprocessing or schema")
    if require_validated and metadata.get("promotion", {}).get("passed") is not True:
        raise ValueError("Model has not passed held-out hybrid promotion gates")
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    weights = manifest_path.parent / metadata["weights"]
    model = _load_model(
        str(weights.resolve()),
        metadata["weights_sha256"],
        metadata["width"],
        device,
        (weights.stat().st_mtime_ns, weights.stat().st_size),
    )
    grid_dt, grid_dx = metadata["sample_interval_ns"], metadata["horizontal_step_m"]
    grid = resample_grid(measurement, dt, dx, grid_dt, grid_dx)
    height, width = metadata.get("crop_shape", [256, 512])
    if (height, width) != (256, 512):
        raise ValueError("Unsupported inference crop shape")
    results = {}
    provenance = {
        "ml_status": "active" if require_validated else "experimental",
        "weights_sha256": metadata["weights_sha256"],
        "model_manifest_sha256": fingerprint_file(manifest_path),
        "device": device,
        "preprocessing_version": PREPROCESSING_VERSION,
        "calibration": metadata.get("calibration", {}),
        "acceptance_calibration": metadata.get("acceptance_calibration", {}),
    }
    for layer in metadata["supported_layers"]:
        layer = int(layer)
        if not anchors.get(layer):
            continue
        clicks = [
            (row * dx / grid_dx, sample * dt / grid_dt) for row, sample in anchors[layer].items()
        ]
        templates = [extract_template(grid, int(round(row)), sample) for row, sample in clicks]
        inputs = make_inputs(grid, layer, templates, clicks, competing_seed_clicks(grid, clicks))
        total, count = np.zeros_like(grid), np.zeros_like(grid)
        vis_total, vis_count = np.zeros(len(grid)), np.zeros(len(grid))
        for x in tile_starts(len(grid), height):
            for y in tile_starts(grid.shape[1], width):
                if cancel and cancel():
                    raise InterruptedError("Analysis cancelled")
                crop = inputs[:, x : x + height, y : y + width]
                nx, ny = crop.shape[1:]
                crop = np.pad(crop, ((0, 0), (0, height - nx), (0, width - ny)))
                with torch.inference_mode():
                    logits, visibility = model(torch.from_numpy(crop[None]).to(device))
                    temperature = max(
                        0.1, float(metadata.get("calibration", {}).get("temperature", 1))
                    )
                    predicted = (logits / temperature).sigmoid()[0, :nx, :ny].cpu().numpy()
                    visible = visibility.sigmoid()[0, :nx].cpu().numpy()
                # Strictly positive taper prevents uncovered seams at the edges.
                taper = np.maximum(0.05, np.outer(np.hanning(height), np.hanning(width)))[:nx, :ny]
                total[x : x + nx, y : y + ny] += predicted * taper
                count[x : x + nx, y : y + ny] += taper
                vis_total[x : x + nx] += visible
                vis_count[x : x + nx] += 1
        probability = total / np.maximum(count, 1e-8)
        coords = np.meshgrid(
            np.arange(len(measurement)) * dx / grid_dx,
            np.arange(measurement.shape[1]) * dt / grid_dt,
            indexing="ij",
        )
        restored = map_coordinates(probability, coords, order=1, mode="nearest")
        visibility = np.interp(
            np.arange(len(measurement)) * dx / grid_dx,
            np.arange(len(grid)),
            vis_total / np.maximum(vis_count, 1),
        )
        results[layer] = LearnedEvidence(
            restored.astype(np.float32),
            visibility.astype(np.float32),
            np.isfinite(measurement),
            layer,
            weight=float(metadata.get("calibration", {}).get("fusion_weight", 0.20)),
            provenance={**provenance, "layer_order": layer},
        )
        results[layer].validate(measurement.shape)
    return results, provenance
