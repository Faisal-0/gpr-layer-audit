"""Frozen local edge evidence through the real request/correct/local-merge loop."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from experiment_jamshoro_guide import arrays
from local_patch_edge_model import instrument
from seeded_eval import sha

from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.processing import seeded_challenger as dense
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask


@lru_cache(maxsize=1)
def context(folder, arm):
    folder = Path(folder)
    if arm not in ("cnn", "ncc"):
        raise ValueError("Frozen local edge arm required")
    contract = json.loads((folder / "contract.json").read_text())
    edge_manifest = json.loads((folder / "edges.json").read_text())
    predictions = json.loads((folder / "prediction.json").read_text())
    for name in (f"{arm}-edges.npy",):
        assert sha(folder / name) == edge_manifest["hashes"][name]
    for name, digest in contract["dependencies"]["scripts"].items():
        # The common mechanism/model files are captured in the replay source.
        local = Path(__file__).with_name(name)
        if local.exists():
            assert sha(local) == digest, name
    entry = contract["input"]
    assert sha(entry["dzt"]) == entry["dzt_sha256"]
    native = np.asarray(DZTFile(entry["dzt"]).channel(), np.float32)
    mask = processed_boundary_mask(native)[:: contract["stride"]]
    controls = {}
    for order in (2, 3):
        name = f"{arm}-layer{order}.npz"
        assert sha(folder / name) == predictions["hashes"][name]
        controls[order] = dict(np.load(folder / name))
    return (
        contract,
        native[:: contract["stride"]],
        mask,
        np.load(folder / f"{arm}-edges.npy", mmap_mode="r"),
        controls,
    )


def fit_dense(
    measurement,
    valid,
    surface,
    dt,
    step,
    layers,
    anchors,
    *,
    method,
    config,
    pulse_anchors=None,
    breaks=(),
    cancel=None,
    **unused,
):
    if method != "dense_packet_research":
        raise ValueError("Isolated research dispatch only")
    contract, expected_data, expected_mask, edges, controls = context(
        config["edge_experiment"], config["arm"]
    )
    np.testing.assert_array_equal(measurement, expected_data)
    np.testing.assert_array_equal(valid, expected_mask)
    entry = contract["input"]
    assert dt == entry["dt_ns"] and step == entry["dx_m"] * contract["stride"]
    assert surface == int(np.floor(-entry["origin_ns"] / dt))
    original = dense._packet_correlations
    paths = {}
    for layer in layers:
        order = layer.order
        frozen = contract["layers"][str(order)]
        initial = {int(r): s for r, s in frozen["seeds"].items()}
        assert (pulse_anchors or anchors)[order] == initial
        operating = anchors[order]
        allowed = np.isfinite(controls[order]["candidate"])

        def correlations(data, mask, seeds, radius, *, permitted=allowed):
            bank, supported = original(data, mask, seeds, radius)
            return bank, supported & permitted

        dense._packet_correlations = correlations
        try:
            path = instrument(dense.pick_seeded_packet, edges)(
                measurement,
                operating,
                valid=valid,
                dt_ns=dt,
                dx_m=step,
                pulse_width_samples=frozen["lobe_samples"],
                reference_surface=surface,
                config=dense.PacketConfig(**contract["config"]),
                breaks=breaks,
                cancel=cancel,
            )
        finally:
            dense._packet_correlations = original
        if operating == initial and not breaks:
            for key, value in arrays(path).items():
                np.testing.assert_array_equal(value, controls[order][key], err_msg=key)
        for row, sample in operating.items():
            assert path.samples[row] == sample
        np.testing.assert_array_equal(
            path.candidate_components["audit_candidate_rank"], controls[order]["candidate"]
        )
        path.provenance.update(
            {
                "backend": f"local_patch_{config['arm']}_edges_research",
                "resolved_pulse": frozen["pulse"],
                "acceptance_is_calibrated": False,
                "edge_experiment": config["edge_experiment"],
                "edge_sha256": json.loads(
                    (Path(config["edge_experiment"]) / "edges.json").read_text()
                )["hashes"][f"{config['arm']}-edges.npy"],
                "new_operating_observations_without_cached_edges": [
                    {"row": row, "sample": sample}
                    for row, sample in operating.items()
                    if not allowed[row, sample]
                ],
                "alternative_contract": "Pointwise exact competing events, not one coherent route",
            }
        )
        paths[order] = path
    return paths
