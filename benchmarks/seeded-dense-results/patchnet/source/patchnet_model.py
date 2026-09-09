"""Experimental native-patch correspondence; outputs are uncalibrated scores."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gpr_layer_audit.ml.processed_correspondence import PatchContract, extract_patches

PATCH = PatchContract(
    temporal_radius_ns=0.9375,
    temporal_points=65,
    lateral_radius_m=0.5,
    lateral_points=21,
    version="native-patchnet-v1",
)


def patches(data, valid, rows, samples, dt, dx):
    amplitude, support, usable = extract_patches(data, valid, rows, samples, dt, dx, PATCH)
    usable &= support.mean(axis=(1, 2)) >= 0.75
    return np.stack((amplitude, support), axis=1).astype(np.float32), usable


class PatchMatcher(nn.Module):
    """Shared encoder plus separate signed-lobe and timing-agreement heads."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 8, (3, 7), stride=(1, 2), padding=(1, 3)),
            nn.SiLU(),
            nn.Conv2d(8, 16, (3, 5), stride=2, padding=(1, 2)),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
            nn.Linear(240, 64),
            nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 2))

    def compare(self, candidate, seed):
        return self.head(torch.cat((candidate, seed, abs(candidate - seed), candidate * seed), 1))

    def forward(self, candidate, seed):
        return self.compare(self.encoder(candidate), self.encoder(seed))


def native_candidates(data, valid, surface):
    """Measured extrema and immediate shoulders, without consulting any labels."""
    from scipy.signal import find_peaks

    selected = np.zeros(data.shape, bool)
    for row, trace in enumerate(data):
        extrema = np.r_[find_peaks(trace)[0], find_peaks(-trace)[0]]
        positions = np.unique(np.r_[extrema - 1, extrema, extrema + 1])
        positions = positions[(positions > surface) & (positions < data.shape[1])]
        selected[row, positions] = True
    return selected & valid & np.isfinite(data) & (data != 0)


def targets(trace, candidates, reference, tolerance):
    """Column0: reviewed timing agreement; column1: signed-lobe membership.

    Called only on positive reviewed rows. Missing/absent rows are not targets.
    Same-lobe timing errors remain distinct from wrong-lobe negatives.
    """
    from gpr_layer_audit.conventional import _same_lobe

    if reference is None:
        return np.full((len(candidates), 2), -1, np.float32)
    family = np.array([_same_lobe(trace, int(c), int(reference)) for c in candidates])
    timing = family & (abs(candidates - reference) <= tolerance)
    return np.stack((timing, family), axis=1).astype(np.float32)
