"""Four measured corrections under the explicit dense basin-emission contract."""

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


def run(output, case="mandiali-short"):
    if output.exists():
        raise ValueError("Preserve existing replay")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    code = (BASE / "source/replay.py").read_text()
    old = "from dense_replay_adapter import fit_dense as fit_processed\n"
    assert code.count(old) == 1
    code = code.replace(old, "from basin_replay_adapter import fit_dense as fit_processed\n")
    (source / "replay.py").write_text(code, encoding="utf-8")
    for name in (
        "basin_replay_adapter.py",
        "dense_replay_adapter.py",
        "dense_basin_emissions.py",
        "dense_peak_modes.py",
        "extremum_state_model.py",
        "experiment_basin_replay.py",
    ):
        shutil.copy2(ROOT / "scripts" / name, source / name)
    config = output / "config.json"
    shutil.copy2(BASE / "legacy-config.json", config)
    package = source_manifest(ROOT / "src")
    contract = {
        "case": case,
        "layers": [2, 3],
        "actions": 4,
        "source": package,
        "scripts": {p.name: sha(p) for p in source.iterdir()},
        "baseline_worker_sha256": sha(BASE / "source/replay.py"),
        "configuration_sha256": sha(config),
        "change": (
            "Explicit basin emissions and recomputed identity margins; same queries and merge"
        ),
        "claim": "Development only; action density is recorded using the actual case length",
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
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
            case,
            "--config",
            str(config),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case", choices=("mandiali-short", "gujrat-second"), default="mandiali-short"
    )
    args = parser.parse_args()
    run(args.output.resolve(), args.case)
