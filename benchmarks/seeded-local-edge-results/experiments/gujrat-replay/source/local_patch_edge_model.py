"""Isolated local-patch edge evidence; no labels, pruning or gate changes."""

from __future__ import annotations

import inspect

import numpy as np


def common_patch_ncc(source_patches, target_patches, source_usable=None, target_usable=None):
    """Centered NCC over common measured pixels of normalized native patches.

    Inputs are [pair, amplitude/validity, road, time], as used by PatchMatcher.
    Each original A-scan window already has temporal L2 normalization. Centering
    here is over all common pixels, with float64 arithmetic. Both patches must
    be usable, >=75% of all pixels must be common, and each centered sum of
    squares must exceed 1e-12. Unsupported pairs return zero plus false support;
    callers MUST use the common fixed gap cost, not interpret zero as evidence.
    """
    source, target = np.asarray(source_patches), np.asarray(target_patches)
    if source.ndim != 4 or source.shape != target.shape or source.shape[1] != 2:
        raise ValueError("Equal [pair,2,road,time] patches required")
    common = (source[:, 1] > 0) & (target[:, 1] > 0)
    common &= np.isfinite(source[:, 0]) & np.isfinite(target[:, 0])
    count = common.sum(axis=(1, 2))
    a = np.where(common, source[:, 0], 0).astype(np.float64)
    b = np.where(common, target[:, 0], 0).astype(np.float64)
    divisor = np.maximum(count, 1)
    a -= (a.sum(axis=(1, 2)) / divisor)[:, None, None]
    b -= (b.sum(axis=(1, 2)) / divisor)[:, None, None]
    a = np.where(common, a, 0)
    b = np.where(common, b, 0)
    aa, bb = (a * a).sum(axis=(1, 2)), (b * b).sum(axis=(1, 2))
    support = (count >= 0.75 * common.shape[1] * common.shape[2]) & (aa > 1e-12) & (bb > 1e-12)
    for usable in (source_usable, target_usable):
        if usable is not None:
            usable = np.asarray(usable, bool)
            if usable.shape != (len(source),):
                raise ValueError("One usability flag per patch required")
            support &= usable
    dot = (a * b).sum(axis=(1, 2))
    correlation = np.clip(dot / np.sqrt(np.maximum(aa * bb, 1e-30)), -1, 1)
    return np.where(support, correlation, 0), support


def compare_forward(model, source_features, target_features):
    """Match training: candidate at row+1 first, preceding source second."""
    return model.compare(target_features, source_features)


def edge_costs(correlation, probability, support, gap_cost=0.65):
    """Unit-weight cost per metre; probabilities are classifier scores only."""
    correlation, probability, support = (
        np.asarray(correlation),
        np.asarray(probability),
        np.asarray(support, bool),
    )
    if correlation.shape != probability.shape or correlation.shape != support.shape:
        raise ValueError("Equal edge score/support shapes required")
    if not np.isfinite(correlation[support]).all() or not np.isfinite(probability[support]).all():
        raise ValueError("Supported scores must be finite")
    if np.any(abs(correlation[support]) > 1 + 1e-7) or np.any(
        (probability[support] < 0) | (probability[support] > 1)
    ):
        raise ValueError("NCC must lie in [-1,1] and classifier score in [0,1]")
    return (
        np.where(support, 1 - correlation, gap_cost),
        np.where(support, 2 * (1 - probability), gap_cost),
    )


def instrument(picker, costs_per_metre):
    """Source-checked isolated solver clone; selection and margins share costs.

    Edge axes: [source row, offset index, target sample], offset=target-source.
    The solver owns dx multiplication. A zero tensor exactly reproduces control.
    """
    source = inspect.getsource(picker)
    original = "    transition = config.geometry_weight * robust * dx_m\n"
    if source.count(original) != 1:
        raise ValueError("Dense comparator source changed; review instrumentation")
    costs = np.asarray(costs_per_metre)
    if costs.ndim != 3 or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Finite nonnegative edge costs required")
    replacement = original + (
        "    if transition.shape != _local_edges.shape:\n"
        "        raise ValueError('Local edge axes differ from dense transitions')\n"
        "    transition += _local_edges * dx_m\n"
    )
    namespace = dict(picker.__globals__, _local_edges=costs)
    exec(
        compile(source.replace(original, replacement), "<local-patch-edge-picker>", "exec"),
        namespace,
    )
    return namespace[picker.__name__]
