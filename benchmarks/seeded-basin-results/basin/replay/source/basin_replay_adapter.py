"""Research basin picker through the existing measured local correction adapter."""

from __future__ import annotations

import dense_replay_adapter as base
from dense_basin_emissions import basin_picker


def fit_dense(*args, **kwargs):
    if kwargs["config"].get("timing_ambiguity"):
        raise ValueError("Basin identity already defines exact distinct observation modes")
    original = base.pick_seeded_packet
    base.pick_seeded_packet = basin_picker(original)[0]
    try:
        paths = base.fit_dense(*args, **kwargs)
    finally:
        base.pick_seeded_packet = original
    for path in paths.values():
        path.provenance["backend"] = "dense_basin_emission_research"
        path.provenance["alternative_contract"] = (
            "Pointwise exact competing measured peaks; each has a constrained route, "
            "but alternatives at different rows are not one coherent route"
        )
    return paths
