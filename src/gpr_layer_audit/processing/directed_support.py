"""Seed-conditioned directed correspondence; components never confer identity."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def reciprocal_route_filter(links, samples, tolerance):
    """A direct match must agree with independently retained two-hop routes when available."""
    outgoing = defaultdict(list)
    by_pair = defaultdict(list)
    for source, target, quality, gap in links:
        outgoing[source].append((target, quality, gap))
        by_pair[source, target[0]].append(target)
    kept = []
    rejected = 0
    for source, target, quality, gap in links:
        routes = []
        for middle, q, first_gap in outgoing[source]:
            if first_gap >= gap or min(q, quality) < 0.75:
                continue
            routes.extend(by_pair.get((middle, target[0]), ()))
        if routes and not any(abs(samples[node] - samples[target]) <= tolerance for node in routes):
            rejected += 1
            continue
        kept.append((source, target, quality, gap))
    return kept, rejected


def directed_seed_support(nodes, links, anchors, samples, breaks=(), seed_observable=None):
    """Independent monotone sweeps from the bracketing seeds, bottleneck confidence.

    Between observations both endpoints must support the node. Outside their extent
    one monotone direction is available. No sweep reverses direction at a junction.
    """
    nodes = sorted(nodes)
    rows = samples.shape[0]
    forward_edges, backward_edges = defaultdict(list), defaultdict(list)
    for source, target, quality, _ in links:
        if source[0] >= target[0]:
            raise ValueError("Correspondence must form a directed observation DAG")
        forward_edges[source].append((target, quality))
        backward_edges[target].append((source, quality))
    boundaries = sorted({0, rows, *breaks})
    dense = np.zeros_like(samples, dtype=float)
    forward_dense, backward_dense = np.zeros_like(dense), np.zeros_like(dense)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        seeds = sorted(r for r in anchors if start <= r < stop)
        region_nodes = [n for n in nodes if start <= n[0] < stop]
        if not seeds:
            continue
        # Reset at every authoritative row: support from an earlier seed cannot
        # bypass an incompatible or unobservable nearer observation.
        for ordered, edges, output in (
            (region_nodes, forward_edges, forward_dense),
            (reversed(region_nodes), backward_edges, backward_dense),
        ):
            values = {}
            for node in ordered:
                row, col = node
                if row in anchors:
                    observable = seed_observable is None or seed_observable.get(row, False)
                    values[node] = float(samples[node] == anchors[row] and observable)
                value = values.get(node, 0.0)
                output[node] = value
                for target, quality in edges[node]:
                    if start <= target[0] < stop:
                        values[target] = max(values.get(target, 0.0), min(value, quality))
        for node in region_nodes:
            row = node[0]
            if row < seeds[0]:
                dense[node] = backward_dense[node]
            elif row > seeds[-1]:
                dense[node] = forward_dense[node]
            else:
                dense[node] = min(forward_dense[node], backward_dense[node])
    return dense, forward_dense, backward_dense
