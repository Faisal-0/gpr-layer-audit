"""Build the immutable processed-coordinate ML cache without loading PyTorch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpr_layer_audit.ml.processed_data import (
    build_processed_dataset,
    representative_overlays,
    research_leave_one_group_out,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prohibited-hash", action="append", default=[])
    parser.add_argument("--overlays", action="store_true")
    args = parser.parse_args()
    manifest = build_processed_dataset(
        args.data_root,
        args.output,
        prohibited_hashes=args.prohibited_hash,
        progress=lambda text: print(text, flush=True),
    )
    print(json.dumps(manifest["counts"], indent=2), flush=True)
    folds = research_leave_one_group_out(manifest, 3)
    (args.output / "subbase-folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    if args.overlays:
        print(
            json.dumps(
                representative_overlays(args.output / "manifest.json", args.output / "overlays")
            )
        )


if __name__ == "__main__":
    main()
