"""One fixed adjacent-packet objective versus the unchanged dense comparator."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

from dense_pairwise_model import instrument
from experiment_seeded_challenger import run_case
from seeded_eval import sha, validate_frozen_helpers

from gpr_layer_audit.processing import seeded_challenger as dense

ROOT = Path(__file__).resolve().parents[1]


def run(case, output):
    if output.exists():
        raise ValueError("Preserve existing experiment")
    output.mkdir(parents=True)
    original = dense.pick_seeded_packet
    helper = validate_frozen_helpers(ROOT / "src")
    source = output / "source"
    source.mkdir()
    for path in (
        Path(__file__),
        Path(__file__).with_name("dense_pairwise_model.py"),
        ROOT / "scripts/experiment_seeded_challenger.py",
        Path(inspect.getfile(original)),
    ):
        shutil.copy2(path, source / path.name)
    contract = {
        "case": case,
        "claim": "Previously used development interpretation; no physical thickness claim",
        "source_hashes": {p.name: sha(p) for p in source.iterdir()},
        "frozen_helpers": helper,
        "mechanism": "Add unit-weight adjacent-packet signed NCC cost per metre to every edge",
        "fixed": (
            "Same dense states, native seeds, seed unary, tensor, slope bounds and margin gate"
        ),
        "missing": "Unsupported pairs cost 0.65/m and provide no correspondence evidence",
        "selection": (
            "Two arms only, no learned model or threshold changes; reject added wrong length"
        ),
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    for arm in ("classical", "pairwise"):
        dense.pick_seeded_packet = original if arm == "classical" else instrument(original)
        try:
            run_case(case, output / f"{arm}.json", layers=(2, 3))
        finally:
            dense.pick_seeded_packet = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("mandiali-short", "gujrat-second"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.output)
