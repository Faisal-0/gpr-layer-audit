"""CLI registration remains usable without importing torch."""

from __future__ import annotations

import json
from pathlib import Path


def add_commands(subparsers):
    dataset = subparsers.add_parser("dataset", help="Audit and build verified radar training data")
    commands = dataset.add_subparsers(dest="operation", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("data", type=Path)
    audit.add_argument("--seeds", type=Path)
    audit.add_argument("--annotations", type=Path)
    audit.add_argument("--groups", type=Path, help="Optional survey-id to physical-road-group JSON")
    audit.add_argument("--output", required=True, type=Path)
    build = commands.add_parser("build")
    build.add_argument("manifest", type=Path)
    build.add_argument("--output", required=True, type=Path)
    ml = subparsers.add_parser("ml", help="Offline optional model training and calibration")
    commands = ml.add_subparsers(dest="operation", required=True)
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--layers", type=int, nargs="+", choices=(1, 2, 3))
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("report", type=Path)
    calibration.add_argument("--output", required=True, type=Path)
    promote = commands.add_parser("promote")
    promote.add_argument("model", type=Path)
    promote.add_argument("report", type=Path)
    hybrid = subparsers.add_parser("hybrid", help="Evaluate the fused tracing system")
    commands = hybrid.add_subparsers(dest="operation", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("--model", type=Path)
    evaluate.add_argument("--calibration", type=Path)
    evaluate.add_argument("--split", default="test", choices=("test", "validation"))
    evaluate.add_argument("--output", type=Path, required=True)


def _write(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")


def run_command(args):
    try:
        if args.command == "dataset":
            from .dataset import audit_dataset, build_dataset

            if args.operation == "audit":
                groups = json.loads(args.groups.read_text()) if args.groups else None
                result = audit_dataset(
                    args.data,
                    seed_root=args.seeds,
                    annotations=args.annotations,
                    group_overrides=groups,
                )
                _write(args.output, result)
                print(
                    json.dumps(
                        {"sources": len(result["sources"]), "eligibility": result["eligibility"]}
                    )
                )
            else:
                result = build_dataset(args.manifest, args.output)
                print(
                    json.dumps(
                        {"chunks": len(result["chunks"]), "path": str(args.output / "dataset.json")}
                    )
                )
            return 0
        if args.command == "hybrid":
            from .evaluation import evaluate_hybrid

            result = evaluate_hybrid(
                args.manifest, model=args.model, split=args.split, calibration=args.calibration
            )
            _write(args.output, result)
            print(json.dumps({"observations": len(result["records"]), "layers": result["layers"]}))
            return 0 if result["records"] else 2
        if args.operation == "train":
            from .training import train_model

            result = train_model(
                args.manifest,
                args.output,
                layers=args.layers,
                epochs=args.epochs,
                patience=args.patience,
                device=args.device,
            )
            print(json.dumps({"supported_layers": result["supported_layers"], "promoted": False}))
            return 0
        if args.operation == "calibrate":
            from .evaluation import calibrate_acceptance

            result = calibrate_acceptance(json.loads(args.report.read_text(encoding="utf-8")))
            _write(args.output, result)
            return 0 if result["validated"] else 2
        if args.operation == "promote":
            from gpr_layer_audit.io.dzt import fingerprint_file

            from .evaluation import promotion_report

            report = json.loads(args.report.read_text(encoding="utf-8"))
            manifest_path = args.model / "model.json" if args.model.is_dir() else args.model
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                report.get("split") != "test"
                or report.get("model_weights_sha256") != manifest["weights_sha256"]
            ):
                raise ValueError("Promotion requires held-out evaluation of these exact weights")
            calibration = report.get("acceptance_calibration", {})
            if (
                not calibration.get("validated")
                or calibration.get("model_weights_sha256") != manifest["weights_sha256"]
            ):
                raise ValueError(
                    "Promotion requires frozen validation calibration for these weights"
                )
            layers = promotion_report(report["records"], report["baseline_records"])
            passed = [
                order
                for order in manifest["supported_layers"]
                if order in (2, 3)
                and layers[str(order)]["passed"]
                and str(order) in calibration.get("layers", {})
            ]
            if not passed:
                raise ValueError("No interface passes the precision, coverage and road-count gates")
            manifest["supported_layers"] = passed
            manifest["promotion"] = {
                "passed": True,
                "report_sha256": fingerprint_file(args.report),
                "layers": {str(order): layers[str(order)] for order in passed},
            }
            manifest["acceptance_calibration"] = report.get("acceptance_calibration", {})
            _write(manifest_path, manifest)
            return 0
    except (ValueError, OSError, ImportError) as exc:
        outcome = {"status": "not_run_or_not_promoted", "reason": str(exc)}
        print(json.dumps(outcome))
        if getattr(args, "output", None):
            output = args.output if args.output.suffix == ".json" else args.output / "outcome.json"
            _write(output, outcome)
        return 2
    return 2
