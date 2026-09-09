"""Layer-specific observation requests from radar and current routes only.

The policy has deliberately no argument for reviewed/reference observations.
Hypothetical answers partition retained routes; their estimated benefit is a
ranking heuristic, not a calibrated prediction of recovered road length.
"""

from __future__ import annotations

import numpy as np


def request_observation(paths, measurement, valid, anchors, step, *, visited=(), policy="active"):
    if policy not in {"active", "midpoint"}:
        raise ValueError("Unknown observation policy")
    rows = len(measurement)
    visited = set(visited)
    requests = []
    for order, path in sorted(paths.items()):
        occupied = sorted(set(anchors.get(order, {})) | {r for o, r in visited if o == order})
        free = np.ones(rows, bool)
        free[occupied] = False
        proposed = path.provisional_samples
        if proposed is None:
            proposed = path.samples
        support = path.candidate_components.get("measurement_support")
        # This fallback checks measured signal, never synthetic extension samples.
        energy = np.max(np.where(valid, abs(measurement), 0), axis=1)
        free &= energy > 0
        if not np.any(free):
            continue
        distance = (
            np.min(abs(np.arange(rows)[:, None] - np.array(occupied)[None, :]), axis=1)
            if occupied
            else np.minimum(np.arange(rows) + 1, rows - np.arange(rows))
        ) * step
        if policy == "midpoint":
            candidates = np.flatnonzero(free)
            row = int(candidates[np.argmax(distance[candidates])])
            requests.append(
                dict(
                    layer_order=order,
                    row=row,
                    priority=float(distance[row]),
                    reason="Largest spacing between observations",
                    candidate_samples=[],
                )
            )
            continue
        hypotheses = [np.asarray(h) for h in path.provenance.get("hypothesis_samples", [])]
        hypotheses.extend([proposed, path.alternate_samples])
        alternate = path.evidence.get("hybrid_path_alternate")
        if alternate is not None:
            hypotheses.append(alternate)
        # Duplicate routes must not manufacture hypothetical information gain.
        unique = {
            np.asarray(h, dtype=np.int32).tobytes(): np.asarray(h, dtype=np.int32)
            for h in hypotheses
            if h is not None and len(h) == rows
        }
        hypotheses = list(unique.values())
        width = path.provenance.get("resolved_pulse", {}).get("lobe_samples", 8.0)
        tolerance = max(2.0, width / 4)
        scores = np.zeros(rows)
        choices = [set() for _ in range(rows)]
        for i, left in enumerate(hypotheses):
            for right in hypotheses[i + 1 :]:
                use = free & (left >= 0) & (right >= 0)
                use &= (left < measurement.shape[1]) & (right < measurement.shape[1])
                indices = np.flatnonzero(use)
                if not len(indices):
                    continue
                # Different timing inside one lobe still deserves a timing query.
                separated = abs(left[indices] - right[indices]) > tolerance
                observable = valid[indices, left[indices]] & valid[indices, right[indices]]
                observable &= measurement[indices, left[indices]] != 0
                observable &= measurement[indices, right[indices]] != 0
                indices = indices[separated & observable]
                disagreement_m = float(
                    np.count_nonzero((left >= 0) & (right >= 0) & (abs(left - right) > tolerance))
                    * step
                )
                if support is not None:
                    quality = np.minimum(
                        support[indices, left[indices]], support[indices, right[indices]]
                    )
                else:
                    quality = (
                        np.minimum(
                            abs(measurement[indices, left[indices]]),
                            abs(measurement[indices, right[indices]]),
                        )
                        / energy[indices]
                    )
                # Each observable answer separates this pair over its disputed span.
                scores[indices] += disagreement_m * np.sqrt(np.clip(quality, 0, 1))
                for row in indices:
                    choices[row].update((int(left[row]), int(right[row])))
        # Existing graph requests also cover unsupported endpoint frontiers.
        for item in path.provenance.get("suggested_observations", []):
            row = int(item["row"])
            if 0 <= row < rows and free[row]:
                scores[row] = max(scores[row], float(item.get("priority", 0)))
                choices[row].update(item.get("candidate_samples", []))
        contested = np.flatnonzero(free & (scores > 0))
        if len(contested):
            # Prefer separation from previous actions, without distance-decaying acceptance.
            values = scores[contested] * np.minimum(1, distance[contested] / max(step, 1.0))
            row = int(contested[np.argmax(values)])
            priority = float(np.max(values))
            reason = (
                "Observable competing routes; hypothetical answer separates retained hypotheses"
            )
        else:
            unresolved = free & ((path.samples < 0) | ~path.visible)
            candidates = np.flatnonzero(unresolved if np.any(unresolved) else free)
            # Inspect the most separated measurable location when no competitor survived.
            row = int(candidates[np.argmax(distance[candidates])])
            priority = float(distance[row])
            reason = (
                "Unsupported coverage frontier; absence of a retained competitor is not certainty"
            )
            if proposed[row] >= 0:
                choices[row].add(int(proposed[row]))
        requests.append(
            dict(
                layer_order=order,
                row=row,
                priority=priority,
                reason=reason,
                candidate_samples=sorted(choices[row]),
                benefit_is_calibrated=False,
            )
        )
    return (
        max(requests, key=lambda q: (q["priority"], -q["layer_order"], -q["row"]))
        if requests
        else None
    )
