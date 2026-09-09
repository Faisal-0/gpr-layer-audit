"""One explicit waveform-state / measured-event contract on the frozen short road."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from dense_basin_emissions import basin_picker
from experiment_seeded_challenger import run_case
from seeded_eval import sha, source_manifest, validate_frozen_helpers

from gpr_layer_audit.processing import seeded_challenger as dense

ROOT = Path(__file__).resolve().parents[1]


def run(case, output):
    if output.exists():
        raise ValueError("Preserve existing experiment")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    original = dense.pick_seeded_packet
    changed, code = basin_picker(original)
    config = dense.PacketConfig(guide_weight=1.0)
    for path in (
        Path(__file__),
        Path(__file__).with_name("dense_basin_emissions.py"),
        Path(__file__).with_name("dense_peak_modes.py"),
        Path(__file__).with_name("extremum_state_model.py"),
        ROOT / "scripts/experiment_seeded_challenger.py",
        Path(inspect.getfile(original)),
    ):
        shutil.copy2(path, source / path.name)
    (source / "executed-picker.py").write_text(code, encoding="utf-8")
    package = source_manifest(ROOT / "src")
    contract = {
        "case": case,
        "config": asdict(config),
        "source_hashes": {p.name: sha(p) for p in source.iterdir()},
        "package": package,
        "frozen_helpers": validate_frozen_helpers(ROOT / "src"),
        "state": "Original dense waveform-member position with unique measured peak emission",
        "geometry": "Unchanged member-coordinate NCC, slope, guide, gaps and displacement edges",
        "observations": "Supported basin members emit a measured amplitude extremum; valleys abstain",
        "ambiguity": "Recomputed exact alternative peak classes, including same-sign split modes",
        "gate": "Original thresholds and interval normalization, explicitly uncalibrated",
        "claim": "Development interpretation; no physical-reflector identity guarantee",
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    for arm, picker in (("member", original), ("emission", changed)):
        dense.pick_seeded_packet = picker
        try:
            run_case(case, output / f"{arm}.json", config=config, layers=(2, 3))
        finally:
            dense.pick_seeded_packet = original
    assert source_manifest(ROOT / "src")["sha256"] == package["sha256"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.output)
