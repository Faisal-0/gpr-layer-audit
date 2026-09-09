"""Import explicit operating observations without opening an evaluation reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

from .io.dzt import DZTFile, fingerprint_file
from .models import SeedStation, VisibilityState


def load_native_observations(path, dzt_path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        document.get("schema") != "conventional-native-seeds-v1"
        or document.get("mode") != "processed"
    ):
        raise ValueError("Expected explicit native processed observations")
    if fingerprint_file(dzt_path) != document.get("dzt_sha256"):
        raise ValueError("Seed observations belong to a different processed DZT")
    dzt = DZTFile(dzt_path)
    dx = dzt.header.distance_per_trace_m
    if dx is None or dx <= 0:
        raise ValueError("Native observations require physical trace spacing")
    data = dzt.channel()
    stations = {}
    for key, observations in document["observations"].items():
        order = int(key)
        if order not in (1, 2, 3):
            raise ValueError("Unknown layer in native observations")
        for point in observations:
            row, sample, channel = (point[k] for k in ("trace", "sample", "channel"))
            if any(type(value) is not int for value in (row, sample, channel)):
                raise ValueError("Native trace/sample/channel must be integers")
            if not 0 <= row < len(data) or not 0 <= sample < data.shape[1] or channel != 0:
                raise ValueError("Native observation outside supported source/channel")
            station = stations.setdefault(row, SeedStation(f"native-{row}", row * dx))
            if order in station.samples:
                raise ValueError("Duplicate layer observation at native trace")
            trace = np.asarray(data[row], dtype=np.float64)
            polarity = int(np.sign(trace[sample]))
            phase = float(np.angle(hilbert(trace)[sample]))
            left = right = sample
            while left > 0 and np.sign(trace[left - 1]) == polarity and polarity:
                left -= 1
            while right + 1 < len(trace) and np.sign(trace[right + 1]) == polarity and polarity:
                right += 1
            station.samples[order] = float(sample)
            station.canonical_samples[order] = float(sample)
            station.user_confirmed[order] = True
            station.visibility[order] = VisibilityState.VISIBLE
            station.polarity[order] = polarity
            station.analytic_phase_rad[order] = phase
            station.phase_class[order] = int(((phase + np.pi) % (2 * np.pi)) * 8 / (2 * np.pi))
            station.selected_lobe[order] = "negative_trough" if polarity < 0 else "positive_peak"
            station.pulse_width_samples[order] = float(right - left + 1)
            station.event_ids[order] = f"native:{row}:{order}:{sample}"
            station.regime_ids[order] = "default"
    return [stations[row] for row in sorted(stations)]
