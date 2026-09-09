"""Run independent processes for frozen/current conventional road comparisons.

Example: uv run python scripts/evaluate_conventional_development.py --stage repaired
The frozen package is the captured pre-edit working tree, including uncommitted work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("baseline", "signal", "pulse", "directed", "repaired"), required=True
    )
    parser.add_argument("--road", default="all")
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed-source", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "exports/conventional")
    args = parser.parse_args()
    if args.seed_source and args.stage == "baseline":
        parser.error("Seed reuse requires a current helper; the frozen package remains immutable")
    sys.path.insert(0, str(ROOT / "src"))
    from gpr_layer_audit.conventional_reference import road_partition
    from gpr_layer_audit.io.dzx import read_dzx

    config = {
        "signal_validity": True,
        "physical_pulse": args.stage in ("pulse", "directed", "repaired"),
        "directed_correspondence": args.stage in ("directed", "repaired"),
        "complete_paths": args.stage == "repaired",
        "path_acceptance": args.stage == "repaired",
    }
    if args.config:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    package = ROOT / "exports/conventional/frozen"
    if args.stage != "baseline":
        source = ROOT / "src/gpr_layer_audit"
        digest = hashlib.sha256()
        for file in sorted(source.rglob("*.py")):
            digest.update(str(file.relative_to(source)).encode())
            digest.update(file.read_bytes())
        package = ROOT / "exports/conventional" / ("source-" + digest.hexdigest()[:16])
        if not package.exists():
            shutil.copytree(
                source, package / "gpr_layer_audit", ignore=shutil.ignore_patterns("__pycache__")
            )
    for path in sorted((ROOT / "gpr including proc files").rglob("*.DZX")):
        group, split = road_partition(path)
        if args.source_filter and args.source_filter not in path.stem:
            continue
        if split not in ("development", "calibration") or (
            args.road != "all" and group != args.road
        ):
            continue
        if not any(layer.picks for layer in read_dzx(path).layers):
            continue
        stride = 1 if "P_21" in path.stem else args.stride
        output = args.output_root / args.stage / (path.stem + ".json")
        if output.exists():
            cached = json.loads(output.read_text(encoding="utf-8"))
            expected_seed_hash = (
                hashlib.sha256(args.seed_source.read_bytes()).hexdigest()
                if args.seed_source
                else None
            )
            if (
                cached.get("stride") != stride
                or cached.get("config") != ({} if args.stage == "baseline" else config)
                or cached.get("seed_selection", {}).get("sha256") != expected_seed_hash
            ):
                raise ValueError("Cached settings or seed source differ; use a new output root")
            print(f"Cached {output.name}", flush=True)
            continue
        code = (
            "import sys,json; sys.path.insert(0,sys.argv[1]); "
            "from gpr_layer_audit.conventional import evaluate_reference; "
            "evaluate_reference(sys.argv[2], output=sys.argv[3], stride=int(sys.argv[4]), "
            "config=json.loads(sys.argv[5]), methods=json.loads(sys.argv[6])"
            + (", seed_source=sys.argv[7])" if args.seed_source else ")")
        )
        print(f"Running {args.stage}: {path.stem} (stride {stride})", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                str(package),
                str(path),
                str(output),
                str(stride),
                json.dumps(None if args.stage == "baseline" else config),
                json.dumps(
                    ["joint_seed_adaptive", "seed_hybrid"]
                    if args.stage == "baseline"
                    else ["seed_hybrid"]
                ),
                *([str(args.seed_source.resolve())] if args.seed_source else []),
            ],
            cwd=ROOT,
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
        values = json.loads(output.read_text())
        print(
            {
                method: {
                    order: (v["accepted_pick_agreement"], v["correct_coverage"])
                    for order, v in data.get("layers", {}).items()
                }
                for method, data in values["methods"].items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
