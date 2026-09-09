"""Exact measured-extremum states; no shoulder-to-peak output conversion."""

from __future__ import annotations

import inspect

import numpy as np
from scipy.signal import find_peaks


def observed_extrema(data, valid):
    supported = np.asarray(valid, bool) & np.isfinite(data)
    output = np.zeros(data.shape, bool)
    for row, trace in enumerate(data):
        edges = np.flatnonzero(np.diff(np.r_[False, supported[row], False]))
        for lo, hi in edges.reshape(-1, 2):
            local = trace[lo:hi]
            output[row, lo + find_peaks(local)[0]] = True
            output[row, lo + find_peaks(-local)[0]] = True
    return output & supported & (data != 0)


def peak_mode_picker(original):
    source = inspect.getsource(original)
    before = "    correlations, packet_valid = _packet_correlations(data, valid, anchors, radius)\n"
    after = before + (
        "    extrema = _observed_extrema(data, valid)\n"
        "    if any(not extrema[r, s] for r, s in anchors.items()):\n"
        "        raise ValueError('This peak-mode experiment requires exact extremum observations')\n"
        "    packet_valid &= extrema\n"
    )
    if source.count(before) != 1:
        raise ValueError("Review changed packet candidate construction")
    namespace = dict(original.__globals__, _observed_extrema=observed_extrema)
    exec(compile(source.replace(before, after), "<dense-peak-mode>", "exec"), namespace)
    return namespace[original.__name__]
