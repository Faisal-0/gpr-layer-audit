"""Research-only measured-basin emissions on exact waveform-member DAG states.

No reviewed observations enter this module. The member sample is the matching
coordinate; its local measured extremum is the emission coordinate. Scores and
edges belong to individual members, never to a pooled peak or signed lobe.
"""

# Exact frozen-source strings and recorded contract prose intentionally remain unwrapped.
# ruff: noqa: E501
from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
from scipy.signal import find_peaks

CONTRACT = {
    "version": "extremum-member-state-v1",
    "operating_fiducial": "All initial seeds must coincide with a unique measured amplitude extremum; exact seeds and immutable contexts are preserved. Incompatible seeds stop this experiment.",
    "state": "(row, original waveform-member sample, unique measured extremum basin). Distinct secondary extrema remain distinct even within the same signed lobe. Zero samples or members on an interpeak minimum have unresolved fiducial, emitted as a gap.",
    "basins": "Separate constant-sign runs; scipy local amplitude peaks with midpoint for a plateau. Consecutive peaks split at all minimum-amplitude samples between them; these valley samples have ambiguous assignment to both neighbors.",
    "candidate_cap": "Before the unchanged packet cap, close each packet over the measured peaks of its original members (both neighboring peaks for valley members), provided those peaks satisfy the original candidate union/masks. Original ranking and representative formulas remain, but closure changes their membership inputs and can change packet ranking/selection. After packet selection, keep each retained member plus its own basin peaks, deduplicated. Each peak gets independently computed waveform/features/edges/support.",
    "objective": "Unchanged frozen sum of member local scores times horizontal step, minus contracted internal (1-quality) costs and external correspondence/gap penalties. All motion, DTW and seed-distance geometry remains in the original waveform-member coordinates. One selected member per row; no marginal summation, peak bonus or pooled support.",
    "inference": "Solve original member DAG after peak closure. Emission is a fixed pre-inference attribute, not fitted from a selected route. Exact forward/backward max-marginals compare complete routes by emitted basin identity; every different peak competes regardless of frozen timing tolerance. Missing/unresolved emissions compete as gaps. Chain contraction and the objective remain unchanged.",
    "acceptance": "Original layer correspondence, path-margin, measurement, validity, anomaly and ordering gates and numeric thresholds. Correspondence is the selected member's own quality supporting only its unique basin. Emitted peak must additionally have valid full waveform context. Margins are recomputed for emitted states; no prior confidence reused. Frozen same-signed-lobe/time tolerance is scoring only.",
    "limits": "This tests a zero-offset seed fiducial and original member geometry, not general offset learning or fiducial-motion registration. Basin identity is a waveform mode, not physical-reflector semantic proof. Ambiguous valleys abstain rather than choosing an arbitrary adjacent peak. Packet modes with no retained member can still be cap losses.",
}


def basin_map(trace):
    """Return unique emission map and explicit ambiguous valley alternatives."""
    values = np.asarray(trace)
    sign = np.sign(values)
    emission = np.full(len(values), -1, np.int32)
    ambiguous = {}
    boundaries = np.r_[0, np.flatnonzero(sign[1:] != sign[:-1]) + 1, len(values)]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if sign[start] == 0:
            continue
        amplitude = abs(values[start:stop])
        peaks = find_peaks(np.r_[0, amplitude, 0], plateau_size=True)[0] - 1 + start
        if not len(peaks):
            continue
        emission[start:stop] = peaks[0]
        for left, right in zip(peaks[:-1], peaks[1:], strict=True):
            section = abs(values[left + 1 : right])
            valley = np.flatnonzero(section == section.min()) + left + 1
            emission[int(valley[-1]) + 1 : stop] = right
            emission[valley] = -1
            for sample in valley:
                ambiguous[int(sample)] = (int(left), int(right))
    return emission, ambiguous


def peak_closure(packets, emissions, ambiguous, allowed):
    """Extend membership before cap, without changing selected/canonical ranks."""
    result = []
    for selected, canonical, members in packets:
        peaks = set()
        for sample in members:
            options = ambiguous.get(sample, (int(emissions[sample]),))
            peaks.update(p for p in options if p >= 0 and allowed[p])
        result.append((selected, canonical, sorted(set(members) | peaks)))
    return result


def clone(function, substitutions, extras=None):
    source = inspect.getsource(function)
    for old, new in substitutions:
        if source.count(old) != 1:
            raise ValueError(f"Frozen source target is not unique: {old!r}")
        source = source.replace(old, new)
    namespace = dict(function.__globals__)
    namespace.update(extras or {})
    exec(compile(source, "<extremum-state-isolated>", "exec"), namespace)
    return namespace[function.__name__], source


def emit_path(table, member_path):
    rows = np.arange(len(member_path))
    mapping = table.component_maps["extremum_emission"]
    return np.where(member_path >= 0, mapping[rows, np.maximum(member_path, 0)], -1).astype(
        np.int32
    )


def emission_view(table):
    rows = np.arange(len(table.samples))[:, None]
    mapping = table.component_maps["extremum_emission"]
    samples = np.where(table.samples >= 0, mapping[rows, np.maximum(table.samples, 0)], -1)
    return replace(table, samples=samples.astype(np.int32))


def install(seed_graph, hybrid, ambiguity, output):
    """Install one isolated mechanism, returning exact transformed source text."""
    transformed = {}
    original_candidate = seed_graph._candidate_table
    candidate, transformed["candidate_table"] = clone(
        original_candidate,
        [
            (
                "    rows, sample_count = residual.shape\n",
                "    rows, sample_count = residual.shape\n"
                "    _basins = [basin_map(trace) for trace in original]\n",
            ),
            (
                "    capacity = limit * (1 + len(mode_maps) + len(hybrid_sources)) + int(\n"
                "        np.max(np.sum(raw_extrema, axis=1))\n    )\n",
                "    capacity = sample_count\n",
            ),
            (
                "        # Do not prune an event packet merely because a stronger ringing cycle\n",
                "        packets = peak_closure(packets, *_basins[row], candidate_union[row])\n"
                "        # Do not prune an event packet merely because a stronger ringing cycle\n",
            ),
            (
                "            retained.extend((sample, canonical, members) for sample in sorted(representatives))\n",
                "            for member in list(representatives):\n"
                "                peaks = _basins[row][1].get(member, (int(_basins[row][0][member]),))\n"
                "                representatives.update(p for p in peaks if p >= 0 and candidate_union[row, p])\n"
                "            retained.extend((sample, canonical, members) for sample in sorted(representatives))\n"
                "        retained = list({item[0]: item for item in retained}.values())\n",
            ),
        ],
        {"basin_map": basin_map, "peak_closure": peak_closure},
    )

    seed_reports = []

    def wrapped_candidate(*args, **kwargs):
        measurement, _, maps, _, _, _, anchors = args[:7]
        mapping = np.array([basin_map(trace)[0] for trace in measurement])
        report = [
            {
                "row": int(r),
                "sample": int(s),
                "measured_extremum": int(mapping[r, s]),
                "offset": int(s - mapping[r, s]),
            }
            for r, s in sorted(anchors.items())
        ]
        if any(item["offset"] != 0 or item["measured_extremum"] < 0 for item in report):
            raise ValueError(f"Unsupported operating fiducial; exact seeds disagree: {report}")
        table = candidate(*args, **kwargs)
        for row in range(len(mapping)):
            retained_peaks = np.zeros(measurement.shape[1], bool)
            retained_peaks[table.samples[row, table.valid[row] & (table.samples[row] >= 0)]] = True
            usable = (mapping[row] >= 0) & retained_peaks[np.maximum(mapping[row], 0)]
            mapping[row, ~usable] = -1
        table.component_maps["extremum_emission"] = mapping
        seed_reports.append(report)
        return table

    original_margin = ambiguity.component_path_margins
    strict_margin, transformed["component_path_margins"] = clone(
        original_margin, [("    tolerance = max(2, pulse_width / 4)\n", "    tolerance = 0\n")]
    )

    def member_margin(segments, edges, table, scores, anchors, selected, width, **kwargs):
        return strict_margin(
            segments,
            edges,
            emission_view(table),
            scores,
            anchors,
            emit_path(table, selected),
            width,
            **kwargs,
        )

    ambiguity.component_path_margins = member_margin

    component, transformed["solve_component"] = clone(
        hybrid._solve_component,
        [
            (
                "                abs(proposed[(proposed >= 0) & (old >= 0)] - old[(proposed >= 0) & (old >= 0)])\n"
                "                <= max(2, pulse_width / 4)\n",
                "                emit_path(table, proposed)[(proposed >= 0) & (old >= 0)]\n"
                "                == emit_path(table, old)[(proposed >= 0) & (old >= 0)]\n",
            )
        ],
        {"emit_path": emit_path},
    )
    hybrid._solve_component = component

    picker, transformed["pick_hybrid_interfaces"] = clone(
        hybrid.pick_hybrid_interfaces,
        [
            (
                "        _candidate_table,\n",
                "        _candidate_table as _unused_candidate_table,\n",
            ),
            (
                "        proposed = hypotheses[0]\n",
                "        member_hypotheses = hypotheses\n"
                "        hypotheses = [emit_path(table, path) for path in member_hypotheses]\n"
                "        proposed = hypotheses[0]\n",
            ),
            (
                "            indices = np.flatnonzero(table.valid[row] & (table.samples[row] == sample))\n",
                "            member_sample = int(member_hypotheses[0][row])\n"
                "            indices = np.flatnonzero(table.valid[row] & (table.samples[row] == member_sample))\n",
            ),
            (
                "            quality = float(correspondence[row, sample])\n",
                "            quality = float(correspondence[row, member_sample])\n",
            ),
            (
                "            canonical[row] = table.canonical_samples[row, col]\n",
                "            canonical[row] = sample\n",
            ),
            (
                "                valid[row, sample]\n",
                "                valid[row, sample]\n"
                "                and (not config.signal_validity or context_valid[row, sample])\n",
            ),
            (
                '            "backend": "seed_hybrid",\n',
                '            "backend": "seed_hybrid",\n'
                '            "extremum_state": {"contract": CONTRACT,\n'
                '                "member_hypothesis_samples": [p.tolist() for p in member_hypotheses]},\n',
            ),
        ],
        {"emit_path": emit_path, "CONTRACT": CONTRACT, "_candidate_table": wrapped_candidate},
    )
    hybrid.pick_hybrid_interfaces = picker
    for name, source in transformed.items():
        (output / f"executed-{name}.py").write_text(source, encoding="utf-8")
    return seed_reports, strict_margin
