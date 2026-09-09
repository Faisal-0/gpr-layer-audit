"""A fixed comparison of sample versus actual extremum states under the seed guide."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from dense_peak_modes import peak_mode_picker
from experiment_seeded_challenger import run_case
from seeded_eval import sha, validate_frozen_helpers

from gpr_layer_audit.processing import seeded_challenger as dense

ROOT = Path(__file__).resolve().parents[1]


def run(case, output):
    if output.exists():
        raise ValueError("Preserve existing experiment")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    original = dense.pick_seeded_packet
    config = dense.PacketConfig(guide_weight=1.0)
    for path in (
        Path(__file__),
        Path(__file__).with_name("dense_peak_modes.py"),
        ROOT / "scripts/experiment_seeded_challenger.py",
        Path(inspect.getfile(original)),
    ):
        shutil.copy2(path, source / path.name)
    contract = {
        "case": case,
        "config": asdict(config),
        "source_hashes": {p.name: sha(p) for p in source.iterdir()},
        "frozen_helpers": validate_frozen_helpers(ROOT / "src"),
        "mechanism": (
            "Select measured extrema directly; retain other samples as unreported latent positions"
        ),
        "precondition": (
            "All supplied initial observations are exact radar extrema, checked without references"
        ),
        "fixed": (
            "Same signed NCC, immutable seed guide, geometry, motion bounds and acceptance gate"
        ),
        "claim": "Development interpretation only; no physical accuracy claim",
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    for arm in ("sample", "peak"):
        dense.pick_seeded_packet = original if arm == "sample" else peak_mode_picker(original)
        try:
            run_case(case, output / f"{arm}.json", config=config, layers=(2, 3))
        finally:
            dense.pick_seeded_packet = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.output)
