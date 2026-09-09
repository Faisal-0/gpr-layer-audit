"""Signal validity and explicit, invertible coordinates for conventional tracing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SignalLayout:
    kind: str = "unresolved"
    invalid_prefix: int = 0
    invalid_suffix: int = 0
    provenance: str = "No destructive layout inference"

    def mask(self, data):
        if self.kind not in ("unresolved", "gssi_raw_sir30", "processed", "explicit"):
            raise ValueError("Unknown signal layout")
        if min(self.invalid_prefix, self.invalid_suffix) < 0:
            raise ValueError("Invalid boundary length")
        valid = np.isfinite(data)
        if self.kind == "unresolved" and (self.invalid_prefix or self.invalid_suffix):
            raise ValueError("Unresolved layout cannot remove samples")
        valid[:, : self.invalid_prefix] = False
        if self.invalid_suffix:
            valid[:, -self.invalid_suffix :] = False
        return valid


def source_layout(source):
    """Recognize acquisition provenance, not arbitrary amplitudes or filename suffixes."""
    from gpr_layer_audit.io.dzx import read_dzx

    path = source.path
    if any(p.casefold() == "proc" for p in path.parts):
        return SignalLayout(
            "processed", provenance="RADAN Proc directory; padding audited separately"
        )
    dzx = path.with_suffix(".DZX")
    metadata = read_dzx(dzx) if dzx.exists() else None
    import re

    if (
        metadata is not None
        and metadata.system == "SIR-30"
        and source.header.bits_per_sample == 32
        and source.header.position_ns == 0
        and re.search(r"_\d{3}$", path.stem)
        and not any(layer.picks for layer in metadata.layers)
    ):
        return SignalLayout(
            "gssi_raw_sir30", 2, provenance="SIR-30 acquisition DZX and 32-bit storage"
        )
    return SignalLayout()


def processed_boundary_mask(data, valid=None):
    """Exclude only exact zero padding and repeated terminal runs; retain interior zero signal."""
    valid = np.isfinite(data) if valid is None else valid.copy()
    for row, trace in enumerate(data):
        nonzero = np.flatnonzero(trace != 0)
        if not len(nonzero):
            valid[row] = False
            continue
        valid[row, : nonzero[0]] = False
        valid[row, nonzero[-1] + 1 :] = False
        # A repeated terminal run of >=4 values is storage padding, recorded separately.
        stop = int(nonzero[-1])
        start = stop
        while start > 0 and trace[start - 1] == trace[stop]:
            start -= 1
        if stop - start + 1 >= 4:
            valid[row, start:] = False
    return valid


def numerical_extension(data, valid):
    """Finite boundary-safe numerical values. The returned values never confer validity."""
    data = np.asarray(data, np.float32)
    if valid.shape != data.shape or valid.dtype != np.bool_:
        raise ValueError("Sample validity must be a boolean radar-grid mask")
    output = data.copy()
    axis = np.arange(data.shape[1])
    for row in range(len(data)):
        known = valid[row] & np.isfinite(data[row])
        output[row] = np.interp(axis, axis[known], data[row, known]) if np.any(known) else 0
    return output


def shift_validity(valid, shifts):
    output = np.zeros_like(valid)
    for row, shift in enumerate(shifts):
        indices = np.arange(valid.shape[1]) - int(shift)
        inside = (indices >= 0) & (indices < valid.shape[1])
        output[row, inside] = valid[row, indices[inside]]
    return output


def stack_valid(data, valid, size):
    if size < 1:
        raise ValueError("Stack size must be positive")
    values, masks, centres = [], [], []
    for start in range(0, len(data), size):
        stop = min(start + size, len(data))
        mask = valid[start:stop]
        # At least half the contributors must be measured; invalid values are excluded.
        supported = mask.sum(axis=0) >= (stop - start + 1) // 2
        block = np.ma.array(data[start:stop], mask=~mask)
        values.append(np.ma.median(block, axis=0).filled(0))
        masks.append(supported)
        centres.append((start + stop - 1) / 2)
    masks = np.asarray(masks, bool)
    return numerical_extension(np.asarray(values, np.float32), masks), masks, np.asarray(centres)


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    source_sha256: str
    target_sha256: str
    trace_scale: float = 1
    trace_offset: float = 0
    sample_scale: float = 1
    sample_offset: float = 0
    uncertainty_samples: float = float("inf")
    uncertainty_traces: float = float("inf")
    verified: bool = False
    basis: str = "unresolved"

    def __post_init__(self):
        values = [self.trace_scale, self.trace_offset, self.sample_scale, self.sample_offset]
        if not np.all(np.isfinite(values)) or self.trace_scale <= 0 or self.sample_scale <= 0:
            raise ValueError("Coordinate transform must be finite and invertible")
        if self.uncertainty_samples < 0 or self.uncertainty_traces < 0:
            raise ValueError("Negative transform uncertainty")

    def forward(self, trace, sample):
        return (
            np.asarray(trace) * self.trace_scale + self.trace_offset,
            np.asarray(sample) * self.sample_scale + self.sample_offset,
        )

    def inverse(self, trace, sample):
        return (
            (np.asarray(trace) - self.trace_offset) / self.trace_scale,
            (np.asarray(sample) - self.sample_offset) / self.sample_scale,
        )

    def require_scoring(self, source_sha256, target_sha256, tolerance):
        if (
            not self.verified
            or self.basis not in ("acquisition_metadata", "processing_stages", "radar_registration")
            or self.source_sha256 != source_sha256
            or self.target_sha256 != target_sha256
            or not self.uncertainty_samples <= tolerance / 2
            or not self.uncertainty_traces <= 0.5
        ):
            raise ValueError(
                "Raw accuracy scoring requires a fingerprint-bound verified radar-only mapping"
            )


def resample_valid(data, valid, trace_coordinates, sample_coordinates):
    """Linear resampling; every contributing endpoint must be valid."""
    from scipy.ndimage import map_coordinates

    coordinates = np.broadcast_arrays(
        np.asarray(trace_coordinates)[:, None], np.asarray(sample_coordinates)[None, :]
    )
    filled = numerical_extension(data, valid)
    values = map_coordinates(filled, coordinates, order=1, mode="constant", cval=0)
    mask = (
        map_coordinates(valid.astype(float), coordinates, order=1, mode="constant", cval=0)
        >= 1 - 1e-9
    )
    return numerical_extension(values, mask), mask
