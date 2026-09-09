"""Experimental row-relative gap cost; observable masks and reports stay unchanged."""

from __future__ import annotations

import inspect

import numpy as np


def relative_gap_unary(correlation, observable, gap_cost, surface):
    """A gap pays the fixed penalty above the best eligible event in that row.

    On rows with measured alternatives, adding a common evidence cost cannot
    change the relative appeal of a gap. Rows without any event retain the
    original fixed gap cost. This is a prior, not an observation or probability.
    """
    observed = np.asarray(observable, bool).copy()
    observed[:, : max(0, surface + 1)] = False
    cost = 1 - correlation
    best = np.min(np.where(observed, cost, np.inf), axis=1)
    best = np.where(np.isfinite(best), best, 0)
    return np.where(observed, cost, best[:, None] + gap_cost)


def relative_gap_picker(original):
    source = inspect.getsource(original)
    before = "            unary = np.where(observable[rr], 1 - corr, config.gap_cost) * dx_m\n"
    after = (
        "            unary = _relative_gap_unary(\n"
        "                corr, observable[rr], config.gap_cost, reference_surface) * dx_m\n"
    )
    if source.count(before) != 1:
        raise ValueError("Review changed dense unary construction before instrumentation")
    namespace = dict(original.__globals__, _relative_gap_unary=relative_gap_unary)
    code = source.replace(before, after)
    exec(compile(code, "<dense-relative-gap>", "exec"), namespace)
    return namespace[original.__name__], code
