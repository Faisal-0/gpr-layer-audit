"""Shared NumPy preprocessing for training and inference."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter1d
from scipy.signal import fftconvolve, hilbert


def resample_grid(data, source_dt, source_dx, target_dt, target_dx):
    if any(not np.isfinite(x) or x <= 0 for x in (source_dt, source_dx, target_dt, target_dx)):
        raise ValueError("Physical sampling intervals must be finite and positive")
    shape = (
        int(np.ceil((len(data) - 1) * source_dx / target_dx)) + 1,
        int(np.ceil((data.shape[1] - 1) * source_dt / target_dt)) + 1,
    )
    if shape[0] * shape[1] > max(1, data.size) * 16:
        raise ValueError("Model grid is incompatible with this acquisition")
    x = np.arange(shape[0]) * target_dx / source_dx
    y = np.arange(shape[1]) * target_dt / source_dt
    return map_coordinates(
        np.asarray(data, np.float32),
        np.meshgrid(x, y, indexing="ij"),
        order=1,
        mode="constant",
        cval=0,
    ).astype(np.float32)


def extract_template(data, row, sample, radius=10):
    sample = int(round(sample))
    if not 0 <= row < len(data) or not 0 <= sample < data.shape[1]:
        raise ValueError("Seed lies outside model input")
    snippet = np.pad(data[row], (radius, radius))[sample : sample + 2 * radius + 1].astype(
        np.float32
    )
    snippet -= snippet.mean()
    return snippet / max(float(np.linalg.norm(snippet)), 1e-9)


def make_inputs(data, layer, templates=(), positive=(), negative=()):
    """Six channels: signed amplitude, envelope, similarity, clicks +/- and layer."""
    data = np.asarray(data, np.float32)
    if data.ndim != 2 or not np.all(np.isfinite(data)) or layer not in (1, 2, 3):
        raise ValueError("Invalid model radargram or layer")
    scale = np.maximum(np.percentile(abs(data), 99, axis=1, keepdims=True), 1e-6)
    signed = np.clip(data / scale, -3, 3) / 3
    envelope = np.clip(abs(hilbert(signed, axis=1)), 0, 1)
    similarity = np.zeros_like(signed)
    for template in templates:
        template = np.asarray(template, np.float32)
        energy = np.sqrt(
            np.maximum(
                uniform_filter1d(signed * signed, len(template), axis=1) * len(template), 1e-12
            )
        )
        correlation = fftconvolve(signed, template[None, ::-1], mode="same", axes=1) / energy
        similarity = np.maximum(similarity, np.clip(correlation, 0, 1))
    click_maps = []
    for clicks in (positive, negative):
        values = np.zeros_like(signed)
        for row, sample in clicks:
            row, sample = int(round(row)), int(round(sample))
            if 0 <= row < len(data) and 0 <= sample < data.shape[1]:
                values[row, sample] = 1
        click_maps.append(values)
    return np.stack(
        [signed, envelope, similarity, *click_maps, np.full_like(signed, layer / 3)]
    ).astype(np.float32)


def tile_starts(length, size):
    if length <= size:
        return [0]
    return sorted({*range(0, length - size + 1, size // 2), length - size})


def competing_seed_clicks(data, positive):
    """Mark one nearby competing lobe per input click, never consulting target labels."""
    from scipy.signal import find_peaks

    negative = []
    for row, sample in positive:
        row = int(round(row))
        if not 0 <= row < len(data):
            continue
        peaks, _ = find_peaks(np.abs(data[row]))
        distance = abs(peaks - sample)
        candidates = peaks[(distance >= 6) & (distance <= 24)]
        if len(candidates):
            chosen = candidates[np.argmax(np.abs(data[row, candidates]))]
            negative.append((row, float(chosen)))
    return negative
