"""Run four actual scoped requests using frozen learned edges and native seeds."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from seeded_eval import sha, source_manifest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "exports/seeded-tracker/dense-replay-v1/mandiali"


def run(edge_experiment, output, arm="cnn"):
    if output.exists():
        raise ValueError("Preserve previous replay")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    worker = (BASE / "source/replay.py").read_text()
    old = "from dense_replay_adapter import fit_dense as fit_processed\n"
    if worker.count(old) != 1:
        raise ValueError("Frozen worker changed")
    worker = worker.replace(
        old, "from local_edge_replay_adapter import fit_dense as fit_processed\n"
    )
    (source / "replay.py").write_text(worker, encoding="utf-8")
    for name in (
        "local_edge_replay_adapter.py",
        "local_patch_edge_model.py",
        "seeded_eval.py",
        "experiment_jamshoro_guide.py",
        "experiment_local_edge_replay.py",
    ):
        shutil.copy2(ROOT / "scripts" / name, source / name)
    config = {"edge_experiment": str(edge_experiment), "arm": arm}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    package = source_manifest(ROOT / "src")
    prediction = edge_experiment / "prediction.json"
    contract = {
        "case": "gujrat-second",
        "actions": 4,
        "layers": [2, 3],
        "source": package,
        "prediction_manifest_sha256": sha(prediction),
        "edge_contract_sha256": sha(edge_experiment / "contract.json"),
        "configuration_sha256": sha(output / "config.json"),
        "source_hashes": {p.name: sha(p) for p in source.iterdir()},
        "frozen_worker_sha256": sha(BASE / "source/replay.py"),
        "query_answer_merge_scope": "Unchanged actual replay worker; local +/-25m merge",
        "initial_arrays": "Every fit with initial seeds must exactly reproduce the frozen arm",
        "candidate_edges": "Fixed radar evidence; off-bank answers retain exact seed",
        "claim": "Development interpretation; unresolved proposals cannot generate measurements",
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    for policy in ("active", "midpoint"):
        destination = output / f"{policy}.json"
        command = [
            sys.executable,
            str(source / "replay.py"),
            "--workspace",
            str(ROOT),
            "--source",
            str(ROOT / "src"),
            "--case",
            "gujrat-second",
            "--config",
            str(output / "config.json"),
            "--output",
            str(destination),
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
            "4",
        ]
        with destination.with_suffix(".log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        print(destination, flush=True)
    assert source_manifest(ROOT / "src")["sha256"] == package["sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_experiment", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", choices=("cnn", "ncc"), default="cnn")
    args = parser.parse_args()
    run(args.edge_experiment.resolve(), args.output.resolve(), args.arm)
