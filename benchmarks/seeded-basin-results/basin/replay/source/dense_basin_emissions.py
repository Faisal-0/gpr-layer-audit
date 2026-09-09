"""Measured peak emissions on explicit dense waveform-member states.

Geometry and immutable-template NCC remain in member coordinates. Reported
measurements and exact competing events use each member's unique measured peak.
Unknown valleys and unsupported extrema stay gaps. No reference input is read.
"""

from __future__ import annotations

import inspect

import numpy as np
from dense_peak_modes import observed_extrema
from extremum_state_model import basin_map


def emission_map(data, valid, observable, peak_allowed=None):
    peaks = observed_extrema(data, valid) & observable
    if peak_allowed is not None:
        peaks &= peak_allowed
    mapping = np.array(
        [basin_map(np.where(ok, row, 0))[0] for row, ok in zip(data, valid, strict=True)]
    )
    peaks &= mapping == np.arange(data.shape[1])[None, :]
    rr = np.arange(len(data))[:, None]
    usable = (mapping >= 0) & peaks[rr, np.maximum(mapping, 0)] & valid
    return np.where(usable, mapping, -1).astype(np.int32), peaks


def other_events(costs, identities, selected):
    other = np.asarray(costs).copy()
    if identities[selected] >= 0:
        other[identities == identities[selected]] = np.inf
    else:
        other[selected] = np.inf
    return other


def basin_picker(original):
    source = inspect.getsource(original)
    changes = [
        ("    cancel=None,\n", "    cancel=None,\n    peak_allowed=None,\n"),
        (
            "    observable = packet_valid & (central_support >= 0.2) & (local_peak > 0)\n",
            "    observable = packet_valid & (central_support >= 0.2) & (local_peak > 0)\n"
            "    emissions, emission_peaks = _emission_map(data, valid, observable, peak_allowed)\n"
            "    if any(emissions[r, s] != s for r, s in anchors.items()):\n"
            "        raise ValueError('Seeds must be exact supported measured extrema')\n"
            "    observable &= emissions >= 0\n",
        ),
        (
            "                signs = np.sign(data[row])\n"
            "                changes = np.r_[True, (signs[1:] != signs[:-1]) | (signs[1:] == 0)]\n"
            "                lobes = np.cumsum(changes)\n"
            "                other = costs[i].copy()\n"
            "                other[lobes == lobes[sample]] = np.inf\n",
            "                other = _other_events(costs[i], emissions[row], sample)\n",
        ),
        (
            "    proposal[hidden] = -1\n",
            "    proposal[hidden] = -1\n"
            "    _rows = np.arange(rows)\n"
            "    proposal = np.where(proposal >= 0, "
            "emissions[_rows, np.maximum(proposal, 0)], -1)\n"
            "    alternate = np.where(alternate >= 0, "
            "emissions[_rows, np.maximum(alternate, 0)], -1)\n",
        ),
        (
            "    candidate = np.where(observable, 0.0, np.nan).astype(np.float32)\n",
            "    candidate = np.where(emission_peaks, 0.0, np.nan).astype(np.float32)\n",
        ),
        (
            '            "backend": "dense_seeded_packet_research",\n',
            '            "backend": "dense_basin_emission_research",\n'
            '            "state": "Dense waveform member with unique measured-peak emission",\n'
            '            "geometry_coordinates": "Original waveform member; no pooled cost",\n'
            '            "ambiguity": '
            '"Exact competing emitted peaks, including same-sign splits",\n',
        ),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError(f"Review changed dense source: {before!r}")
        source = source.replace(before, after)
    namespace = dict(original.__globals__, _emission_map=emission_map, _other_events=other_events)
    exec(compile(source, "<dense-basin-emissions>", "exec"), namespace)
    return namespace[original.__name__], source
