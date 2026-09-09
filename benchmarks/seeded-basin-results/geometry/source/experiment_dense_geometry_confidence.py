"""A fixed correction to the existing slope-confidence contract."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from dense_geometry_confidence import confidence_picker
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
    changed, code = confidence_picker(original)
    for path in (
        Path(__file__),
        Path(dense.__file__),
        ROOT / "scripts/dense_geometry_confidence.py",
        ROOT / "scripts/experiment_seeded_challenger.py",
    ):
        shutil.copy2(path, source / path.name)
    (source / "executed-picker.py").write_text(code, encoding="utf-8")
    config = dense.PacketConfig(guide_weight=1.0)
    package = source_manifest(ROOT / "src")
    contract = {
        "case": case,
        "configuration": asdict(config),
        "source": package,
        "scripts": {p.name: sha(p) for p in source.iterdir()},
        "frozen_helpers": validate_frozen_helpers(ROOT / "src"),
        "change": "Confidence weights the motion penalty instead of shrinking expected motion",
        "fixed": "Same samples, NCC, gaps, seeds, guide, graph, margin and thresholds",
        "claim": (
            "Development interpretation; geometry confidence is not calibrated identity probability"
        ),
    }
    (output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    for arm, picker in (("original", original), ("weighted", changed)):
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
