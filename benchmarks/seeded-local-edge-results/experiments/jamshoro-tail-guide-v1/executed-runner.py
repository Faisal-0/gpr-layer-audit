"""Isolate removal of the constant endpoint guide from learned-edge tails."""

from __future__ import annotations

import argparse
import inspect
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import experiment_local_patch_edges as base
import numpy as np
from experiment_jamshoro_guide import arrays
from seeded_eval import sha

from gpr_layer_audit.conventional import layer_metrics
from gpr_layer_audit.processing import seeded_challenger as dense


def instrument(picker, costs_per_metre):
    source = inspect.getsource(picker)
    guide = "            if config.guide_weight:\n"
    transition = "    transition = config.geometry_weight * robust * dx_m\n"
    if source.count(guide) != 1 or source.count(transition) != 1:
        raise ValueError("Frozen dense solver changed")
    costs = np.asarray(costs_per_metre)
    if costs.ndim != 3 or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Finite nonnegative edge tensor required")
    source = source.replace(
        guide, "            if config.guide_weight and left in anchors and right in anchors:\n"
    )
    source = source.replace(
        transition,
        transition
        + (
            "    if transition.shape != _local_edges.shape:\n"
            "        raise ValueError('Edge axes changed')\n"
            "    transition += _local_edges * dx_m\n"
        ),
    )
    namespace = dict(picker.__globals__, _local_edges=costs)
    exec(compile(source, "<learned-tail-guide-picker>", "exec"), namespace)
    return namespace[picker.__name__]


def predict(input_folder, output):
    if output.exists():
        raise ValueError("Preserve existing tail experiment")
    output.mkdir(parents=True)
    base.verify_stage(input_folder, "edges")
    base.verify_stage(input_folder, "prediction")
    contract, native, valid = base.context(input_folder)
    record = {
        "parent_folder": str(input_folder),
        "parent_contract_sha256": sha(input_folder / "contract.json"),
        "script_sha256": sha(__file__),
        "choice": "Omit guide only when span is not bracketed by seeds",
        "unchanged": "CNN edges, native seeds, candidates, seed unary, geometry and all gates",
        "reference_opened_for_prediction": False,
        "runtime_s": {},
    }
    base.write(output / "contract.json", record)
    (output / "executed-runner.py").write_bytes(Path(__file__).read_bytes())
    data, mask = native[:: base.STRIDE], valid[:: base.STRIDE]
    entry = contract["input"]
    costs = np.load(input_folder / "cnn-edges.npy", mmap_mode="r")
    original = dense._packet_correlations
    for order in (2, 3):
        control = np.load(input_folder / f"cnn-layer{order}.npz")
        observed = np.isfinite(control["candidate"])
        anchors = {int(r): s for r, s in contract["layers"][str(order)]["seeds"].items()}

        def correlations(measurement, mask, operating, radius, *, allowed=observed):
            bank, packet_valid = original(measurement, mask, operating, radius)
            return bank, packet_valid & allowed

        dense._packet_correlations = correlations
        try:
            start = time.perf_counter()
            picker = instrument(dense.pick_seeded_packet, costs)
            result = picker(
                data,
                anchors,
                valid=mask,
                dt_ns=entry["dt_ns"],
                dx_m=entry["dx_m"] * base.STRIDE,
                pulse_width_samples=contract["layers"][str(order)]["lobe_samples"],
                reference_surface=int(np.floor(-entry["origin_ns"] / entry["dt_ns"])),
                config=dense.PacketConfig(**contract["config"]),
            )
            values = arrays(result)
            rr = np.arange(len(data))
            bracketed = (rr > min(anchors)) & (rr < max(anchors))
            for key, value in values.items():
                np.testing.assert_array_equal(
                    value[bracketed], control[key][bracketed], err_msg=key
                )
            np.testing.assert_array_equal(values["candidate"], control["candidate"])
            for row, sample in anchors.items():
                assert values["samples"][row] == sample
            name = f"tail-guide-layer{order}.npz"
            np.savez_compressed(output / name, **values)
            record["runtime_s"][name] = time.perf_counter() - start
            print(name, record["runtime_s"][name], flush=True)
        finally:
            dense._packet_correlations = original
    assert sha(__file__) == record["script_sha256"]
    assert base.dependencies() == contract["dependencies"]
    record["bracketed_arrays_exact"] = True
    record["candidate_and_seeds_exact"] = True
    record["hashes"] = {p.name: sha(p) for p in output.glob("*.npz")}
    base.write(output / "prediction.json", record)


def evaluate(input_folder, output):
    from gpr_layer_audit.io.dzx import read_dzx

    if (output / "evaluation.json").exists():
        raise ValueError("Preserve evaluation")
    base.verify_stage(output, "prediction")
    contract, native, _ = base.context(input_folder)
    entry = contract["input"]
    assert sha(entry["dzx"]) == entry["dzx_sha256"]
    labels = read_dzx(entry["dzx"])
    results = {}
    for order in (2, 3):
        layer = next(layer for layer in labels.layers if layer.number + 1 == order)
        references = [
            replace(p, trace=p.trace // base.STRIDE)
            for p in layer.picks
            if p.channel == 0 and p.trace % base.STRIDE == 0
        ]
        anchors = {int(r): s for r, s in contract["layers"][str(order)]["seeds"].items()}
        values = np.load(output / f"tail-guide-layer{order}.npz")
        path = SimpleNamespace(
            samples=values["samples"],
            provisional_samples=values["provisional"],
            visible=values["visible"],
            candidate_components={"audit_candidate_rank": values["candidate"]},
            evidence={"hybrid_path_margin": values["margin"]},
        )
        result = layer_metrics(
            path,
            references,
            anchors,
            native[:: base.STRIDE],
            contract["layers"][str(order)]["lobe_samples"],
            entry["dx_m"] * base.STRIDE,
        )
        results[str(order)] = result
        print(
            order,
            {k: result[k] for k in ("proposed_agree", "accepted", "accepted_agree")},
            flush=True,
        )
    base.write(output / "evaluation.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("predict", "evaluate"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.input.resolve(), args.output.resolve())
