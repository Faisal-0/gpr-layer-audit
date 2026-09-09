"""Replay two frozen dense ambiguity definitions through the real local retracker.

Uses an isolated copy of the existing replay worker. Only backend dispatch and
enabled seed layers differ; query selection, answer boundary, merge, local order
guard, frozen evaluation, action recording and checkpoints retain their code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from seeded_eval import sha, source_manifest, validate_frozen_helpers

from gpr_layer_audit.processing.seeded_challenger import PacketConfig

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, case, actions):
    if output.exists():
        raise ValueError("Preserve an existing experiment")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    worker = (ROOT / "scripts/replay_seeded_interaction.py").read_text()
    previous = '        anchors[int(order)] = {p["trace"] // stride: p["sample"] for p in picks}\n'
    replacement = (
        "        if int(order) in args.layers:\n"
        '            anchors[int(order)] = {p["trace"] // stride: p["sample"] for p in picks}\n'
    )
    if worker.count(previous) != 1:
        raise ValueError("Review changed replay seed construction before instrumentation")
    worker = worker.replace(previous, replacement)
    marker = "\n\ndef read(path):\n"
    worker = worker.replace(
        marker, "\n\nfrom dense_replay_adapter import fit_dense as fit_processed\n" + marker
    )
    # The legacy query phrase overstates what pointwise dense alternatives mean.
    marker = '        order, row = request["layer_order"], request["row"]\n'
    worker = worker.replace(
        marker,
        '        request["alternative_contract"] = '
        '"Pointwise min-marginal events; not a coherent route"\n'
        '        if args.policy == "active":\n'
        '            request["reason"] = '
        '"Observable alternative events or unsupported frontier; heuristic ranking"\n'
        + marker,
    )
    (source / "replay.py").write_text(worker, encoding="utf-8")
    for name in ("dense_replay_adapter.py", "experiment_dense_replay.py"):
        shutil.copy2(ROOT / "scripts" / name, source / name)
    manifest = source_manifest(ROOT / "src")
    contract = {
        "case": case,
        "actions": actions,
        "layers": [2, 3],
        "initial_seeds_per_layer": 3,
        "scope": "Same real local merge, inclusive +/-25m; outside predictions frozen",
        "pulse": "Fixed initial operating seeds; local answers cannot refit pulse scales",
        "methods": ["signed_lobe_ambiguity", "signed_lobe_and_timing_ambiguity"],
        "policies": ["active", "midpoint"],
        "source": manifest,
        "helpers": validate_frozen_helpers(ROOT / "src"),
        "scripts": {p.name: sha(p) for p in source.iterdir()},
        "worker_original_sha256": sha(ROOT / "scripts/replay_seeded_interaction.py"),
        "config": asdict(PacketConfig(guide_weight=1.0)),
        "claim": "Historically used development interpretation; no physical-thickness claim",
        "query_limitation": (
            "Current active heuristic uses pointwise alternatives, not coherent full routes"
        ),
        "selection": (
            "Fixed existing guide; no thresholds adjusted; failures do not erase scored rows"
        ),
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    for timing in (False, True):
        name = "timing" if timing else "legacy"
        config = asdict(PacketConfig(guide_weight=1.0)) | {"timing_ambiguity": timing}
        config_path = output / f"{name}-config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        for policy in ("active", "midpoint"):
            result = output / f"{name}-{policy}.json"
            command = [
                sys.executable,
                str(source / "replay.py"),
                "--workspace",
                str(ROOT),
                "--source",
                str(ROOT / "src"),
                "--case",
                case,
                "--config",
                str(config_path),
                "--output",
                str(result),
                "--method",
                "dense_packet_research",
                "--policy",
                policy,
                "--scope",
                "local",
                "--layers",
                "2",
                "3",
                "--actions",
                str(actions),
            ]
            with result.with_suffix(".log").open("w") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT
                )
            print(result, flush=True)
    assert source_manifest(ROOT / "src")["sha256"] == manifest["sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"), required=True)
    parser.add_argument("--actions", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output.resolve(), args.case, args.actions)
