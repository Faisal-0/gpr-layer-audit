"""Train the existing patch architecture on adjacent reviewed reflector pairs.

This tests local correspondence rather than distant seed-template recognition.
Mandiali alone supplies training labels. Training is not tracking-performance
evidence; downstream comparison must use a frozen common graph and native seeds.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

import experiment_patchnet_dense as base
import numpy as np
from experiment_patchnet_far_negatives import dependency_contract
from patchnet_model import PATCH, native_candidates, patches, targets

from gpr_layer_audit.io.dzt import DZTFile
from gpr_layer_audit.io.dzx import read_dzx
from gpr_layer_audit.processing.conventional_config import ConventionalConfig, resolve_pulse
from gpr_layer_audit.processing.conventional_signal import processed_boundary_mask

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "exports/seeded-tracker/patchnet-v2-mask"


def eligible_pairs(points, operating_rows, lag):
    return np.array(
        sorted(r for r in points if r % lag == 0 and r - lag in points and r not in operating_rows),
        dtype=int,
    )


def negative_indices(candidates, target, reference, previous, width, maximum_shift):
    eligible = np.flatnonzero((target[:, 0] == 0) & (abs(candidates - reference) <= 4 * width))
    reachable = abs(candidates[eligible] - previous) <= maximum_shift
    order = np.lexsort((abs(candidates[eligible] - reference), ~reachable))
    return eligible[order[:6]]


def contract():
    return base.CONTRACT | {
        "version": "native-local-patch-pairs-v1",
        "pair_distance_m": 0.1,
        "training_support": (
            "Reviewed preceding point, exactly 0.1m earlier; Mandiali training only"
        ),
        "targets": "Existing independent timing and signed-lobe targets at the next reviewed row",
        "negative_sampling": (
            "Up to six nearest timing misses within four lobe widths; "
            "prioritize graph-reachable negatives"
        ),
        "initial_observations": "Initial operating seed rows excluded from training targets",
        "orientation": "Native forward trace order; no reversed or synthetic data",
        "selection": "Same architecture, patch geometry, optimizer and fixed 12 epochs",
        "planned_graph": (
            "Existing dense sample graph, guide weight 1; compare NCC versus learned adjacent edges"
        ),
        "planned_edge_cost": (
            "Unit weight per metre: 1-NCC versus 2*(1-CNN timing score); unsupported cost 0.65"
        ),
        "dense_configuration": asdict(base.dense.PacketConfig(guide_weight=1.0)),
        "status": "Model preparation/training only; no demonstrated tracker improvement",
        "patch": asdict(PATCH),
        "dependencies": dependency_contract(ROOT),
        "frozen_input_metadata_sha256": base.sha(FROZEN / "inputs.json"),
        "base_runner_sha256": base.sha(base.__file__),
        "local_runner_sha256": base.sha(__file__),
        "dependency_helper_sha256": base.sha(
            Path(__file__).with_name("experiment_patchnet_far_negatives.py")
        ),
    }


def prepare(output):
    if output.exists():
        raise ValueError("Preserve existing training experiment")
    output.mkdir(parents=True)
    base.write(output / "contract.json", contract())
    inputs = base.read(FROZEN / "inputs.json")
    assert inputs == base.read(ROOT / "benchmarks/seeded-dense-results/patchnet/inputs.json")
    reports, x_parts, y_parts, partner_parts, source_parts, coordinates = [], [], [], [], [], []
    source_count = 0
    for entry in inputs:
        if entry["road"] != "mandiali":
            continue
        assert base.sha(entry["dzt"]) == entry["dzt_sha256"]
        assert base.sha(entry["dzx"]) == entry["dzx_sha256"]
        radar = DZTFile(entry["dzt"])
        data = np.asarray(radar.channel(), np.float32)
        valid = processed_boundary_mask(data)
        dt, dx = radar.header.sample_interval_ns, radar.header.distance_per_trace_m
        lag = int(round(0.1 / dx))
        assert np.isclose(lag * dx, 0.1)
        maximum_shift = int(np.ceil(2 * 0.1 / dt))
        surface = int(np.floor(-radar.header.position_ns / dt))
        candidates = native_candidates(data, valid, surface)
        metadata = read_dzx(entry["dzx"])
        for order in (2, 3):
            operating = {int(r): s for r, s in entry["seeds"][str(order)].items()}
            pulse = resolve_pulse(data, valid, operating, {}, dt, ConventionalConfig())
            tolerance = max(2, pulse.lobe_samples / 4)
            layer = next(layer for layer in metadata.layers if layer.number + 1 == order)
            points = {p.trace: p.sample for p in layer.picks if p.channel == 0}
            rows = eligible_pairs(points, operating, lag)
            eligible_count = len(rows)
            if len(rows) > 3000:
                rows = rows[np.linspace(0, len(rows) - 1, 3000, dtype=int)]
            source_rows, source_samples = rows - lag, np.array([points[r - lag] for r in rows])
            source_x, source_usable = patches(data, valid, source_rows, source_samples, dt, dx)
            rr, cc, labels, partner = [], [], [], []
            for index, r in enumerate(rows):
                if not source_usable[index]:
                    continue
                available = np.flatnonzero(candidates[r])
                truth = targets(data[r], available, points[r], tolerance)
                positive = np.flatnonzero(truth[:, 0] == 1)
                positive = positive[
                    np.argsort(abs(available[positive] - points[r]), kind="stable")[:3]
                ]
                negative = negative_indices(
                    available, truth, points[r], points[r - lag], pulse.lobe_samples, maximum_shift
                )
                chosen = np.r_[positive, negative]
                rr.extend([r] * len(chosen))
                cc.extend(available[chosen])
                labels.extend(truth[chosen])
                partner.extend([index + source_count] * len(chosen))
            rr, cc = np.asarray(rr), np.asarray(cc)
            x, usable = patches(data, valid, rr, cc, dt, dx)
            y = np.asarray(labels)
            partner = np.asarray(partner, np.int64)
            assert not set(rr) & set(operating)
            assert np.all(y >= 0)
            source_parts.append(source_x)
            x_parts.append(x[usable])
            y_parts.append(y[usable])
            partner_parts.append(partner[usable])
            coordinates.append(
                np.column_stack(
                    (np.full(usable.sum(), len(reports)), rr[usable], cc[usable], rr[usable] - lag)
                )
            )
            source_count += len(source_x)
            reports.append(
                {
                    "input": entry,
                    "layer": order,
                    "eligible_paired_rows": eligible_count,
                    "sampled_rows": len(rows),
                    "pairs": int(usable.sum()),
                    "positive_timing": int(y[usable, 0].sum()),
                    "source_context_failures": int((~source_usable).sum()),
                    "lag_traces": lag,
                    "maximum_shift_samples": maximum_shift,
                    "pulse": pulse.metadata(),
                    "tolerance_samples": tolerance,
                }
            )
            print(entry["dzt"], order, int(usable.sum()), flush=True)
    np.savez(
        output / "training.npz",
        x=np.concatenate(x_parts),
        y=np.concatenate(y_parts),
        partner=np.concatenate(partner_parts),
        seeds=np.concatenate(source_parts),
    )
    np.savez_compressed(output / "training-coordinates.npz", pairs=np.concatenate(coordinates))
    base.write(
        output / "training-data.json",
        {
            "reports": reports,
            "roads": ["mandiali"],
            "training_sha256": base.sha(output / "training.npz"),
            "coordinates_sha256": base.sha(output / "training-coordinates.npz"),
            "unknown_rows_used": False,
        },
    )
    source = output / "source"
    source.mkdir()
    for name in (
        Path(__file__).name,
        "experiment_patchnet_dense.py",
        "patchnet_model.py",
        "experiment_patchnet_far_negatives.py",
        "seeded_eval.py",
    ):
        shutil.copy2(ROOT / "scripts" / name, source / name)
    base.write(output / "source.json", {p.name: base.sha(p) for p in source.iterdir()})


def train(output):
    frozen = base.read(output / "contract.json")
    if frozen != contract():
        raise ValueError("Training code/configuration differs from prepared experiment")
    base.OUT, base.CONTRACT = output.resolve(), frozen
    base.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.mode](args.output.resolve())
