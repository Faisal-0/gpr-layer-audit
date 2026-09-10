"""Run a native processed-coordinate research fold with explicit bounded budgets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpr_layer_audit.ml.processed_training import TrainingConfig, train_processed_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--fine-width", type=int, default=65)
    parser.add_argument("--coarse-span-ratio", type=int, default=4)
    parser.add_argument("--block-m", type=float, default=64)
    parser.add_argument(
        "--normalization", choices=["scan_rms", "depth_rms", "none"], default="scan_rms"
    )
    parser.add_argument("--layer-only", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--correction-aware", action="store_true")
    parser.add_argument("--small-overfit-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = TrainingConfig(
        held_out_group=args.held_out,
        layers=tuple(args.layers),
        steps=args.steps,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        patience=args.patience,
        learning_rate=args.learning_rate,
        width=args.width,
        fine_width=args.fine_width,
        coarse_span_ratio=args.coarse_span_ratio,
        block_m=args.block_m,
        normalization=args.normalization,
        conditioned=not args.layer_only,
        use_context=not args.no_context,
        correction_aware=args.correction_aware,
        small_overfit_episodes=args.small_overfit_episodes,
        seed=args.seed,
    )
    result = train_processed_model(
        args.dataset, args.output, config=config, device=args.device, resume=args.resume
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_steps": result["completed_steps"],
                "output": str(Path(args.output).resolve()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
