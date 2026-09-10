"""Select native ML operating thresholds on buffered TRAIN-road episodes only.

The root orchestrator owns the single GPU queue; importing this module never
starts inference. No outer-road array is opened by this command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_layer_audit.ml.processed_dense_decoder import (  # noqa: E402
    DenseDecoderConfig,
    decode_dense_path,
    direct_predictions,
)
from gpr_layer_audit.ml.processed_dense_eval import (  # noqa: E402
    aggregate_reports,
    canonical_sha256,
    evaluate_native_predictions,
    frozen_initial_tolerance,
    scorer_provenance,
    select_training_threshold,
)


def _plot_curves(document, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(document["calibrations"], key=int)
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), squeeze=False)
    for axis, layer in zip(axes[0], layers, strict=True):
        for stage, result in document["calibrations"][layer].items():
            points = [p for p in result["curve"] if p["accepted_agreement"] is not None]
            axis.plot(
                [p["macro_road_correct_coverage"] for p in points],
                [p["accepted_agreement"] for p in points],
                label=stage,
            )
        axis.axhline(document["target_agreement"], color="gray", linestyle="--", linewidth=0.8)
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Macro road correct coverage",
            ylabel="Pooled accepted agreement",
            title=f"TRAIN inner validation: layer {layer}",
        )
        axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def calibrate(
    model_path,
    manifest_path,
    output,
    *,
    device="cpu",
    episodes=96,
    trace_stride=4,
    target_agreement=0.95,
    min_accepted=20,
    cpu_threads=4,
    decoder_config=None,
):
    import torch

    from gpr_layer_audit.ml.processed_training import (
        _MODEL_KEYS,
        EpisodeSampler,
        TrainingConfig,
        episode_tensors,
        file_sha256,
    )
    from gpr_layer_audit.ml.seed_context_model import SeedContextUNet

    output = Path(output)
    if output.exists():
        raise FileExistsError("Calibration output exists; choose a new immutable artifact")
    if episodes < 1 or trace_stride < 1 or cpu_threads < 1:
        raise ValueError("Positive episode, native stride, and CPU thread budgets required")
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path /= "model.json"
    metadata = json.loads(model_path.read_text(encoding="utf-8"))
    config = TrainingConfig(**metadata["config"])
    config.validate()
    if metadata.get("status") != "completed":
        raise ValueError("Calibrate a completed immutable training run")
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != metadata["dataset_sha256"]:
        raise ValueError("Training and calibration dataset hashes differ")
    current_code = {
        key: file_sha256(ROOT / "src/gpr_layer_audit/ml" / key) for key in metadata["code_hashes"]
    }
    if current_code != metadata["code_hashes"]:
        raise ValueError("Calibration must use the exact training data/model/sampler source")
    weights = model_path.parent / metadata["weights"]
    weights_sha = file_sha256(weights)
    if weights_sha != metadata["weights_sha256"]:
        raise ValueError("Model weights do not match saved training provenance")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cpu_threads)
    model = SeedContextUNet(
        width=config.width,
        conditioned=config.conditioned,
        use_context=config.use_context,
    ).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    sampler = EpisodeSampler(manifest_path, config)
    if any(r["physical_road_group"] == config.held_out_group for r in sampler.records.values()):
        raise ValueError("Outer-road record entered calibration")
    rng = np.random.default_rng(config.seed + 100000)
    decoder_config = decoder_config or DenseDecoderConfig()
    started = time.perf_counter()
    reports = {str(layer): {"direct": [], "dense": []} for layer in config.layers}
    descriptors = []
    span_diagnostics = {}
    for index in range(episodes):
        episode = sampler.sample(rng, validation=True)
        layer = int(episode["layer"])
        key = episode["record_id"]
        data, record = sampler.arrays[key], sampler.records[key]
        anchors = {
            int(row): int(sample)
            for row, sample in zip(episode["support_rows"], episode["support_samples"], strict=True)
        }
        if any(
            sample != anchors[int(row)]
            for row, sample in zip(episode["support_rows"], episode["support_samples"], strict=True)
        ):
            raise ValueError("Pulse scoring requires exact stored native support samples")
        tolerance, pulse = frozen_initial_tolerance(
            data["amplitudes"], data["sample_validity"], anchors, record["sample_interval_ns"]
        )
        batch = episode_tensors(episode, device)
        # Match ProcessedPredictor's FP32 forward pass; calibration precision is
        # part of the operating contract, independent of BF16 training updates.
        with torch.inference_mode():
            logits = model(
                **{k: batch[k] for k in _MODEL_KEYS}, coarse_span_ratio=config.coarse_span_ratio
            )
            probabilities = logits.float().softmax(dim=1)[0].transpose(0, 1).cpu().numpy()
        # Keep exactly the same retained native grid as the prospective road run.
        rows = episode["rows"]
        take = (rows >= 0) & (rows < len(data["amplitudes"])) & (rows % trace_stride == 0)
        native = rows[take]
        p = probabilities[take]
        radar = data["amplitudes"][native]
        validity = data["sample_validity"][native]
        refs = np.where(
            episode["valid"] & ~episode["support_mask"], episode["target_depth"], np.nan
        )[take]
        if not np.isfinite(refs).any():
            descriptors.append(
                {
                    "index": index,
                    "record_id": key,
                    "layer": layer,
                    "status": "no reviewed nonseed row on retained grid",
                }
            )
            continue
        local = {i: anchors[int(row)] for i, row in enumerate(native) if int(row) in anchors}
        direct = direct_predictions(p, sample_valid=validity)
        dense = decode_dense_path(
            p,
            local,
            sample_valid=validity,
            dx_m=record["horizontal_step_m"] * trace_stride,
            dt_ns=record["sample_interval_ns"],
            config=decoder_config,
        )
        if str(layer) not in span_diagnostics and not local:
            # Duplicate already computed TRAIN evidence, without refitting or
            # reading targets. This isolates length effects in decoder/gating;
            # it does not pretend the duplicated scan is another physical road.
            repeated = []
            for factor in (2, 4):
                expanded = decode_dense_path(
                    np.tile(p, (factor, 1)),
                    {},
                    sample_valid=np.tile(validity, (factor, 1)),
                    dx_m=record["horizontal_step_m"] * trace_stride,
                    dt_ns=record["sample_interval_ns"],
                    config=decoder_config,
                )
                unchanged = expanded.proposed_samples[: len(p)] == dense.proposed_samples
                repeated.append(
                    {
                        "length_factor": factor,
                        "changed_path_rows_in_original_copy": int(np.count_nonzero(~unchanged)),
                        "maximum_confidence_difference_where_path_unchanged": (
                            float(
                                np.max(
                                    abs(
                                        expanded.confidence[: len(p)][unchanged]
                                        - dense.confidence[unchanged]
                                    )
                                )
                            )
                            if unchanged.any()
                            else None
                        ),
                    }
                )
            span_diagnostics[str(layer)] = {
                "episode_index": index,
                "original_rows": len(p),
                "experiment": "duplicate frozen TRAIN probability rows; no model refitting",
                "original_internal_seed_count": len(local),
                "accumulated_objective_margin_used_as_confidence": False,
                "comparisons": repeated,
            }
        descriptor = {
            "index": index,
            "record_id": key,
            "road_group": episode["group"],
            "layer": layer,
            "block": int(episode["block"]),
            "center": int(episode["center"]),
            "native_rows": native.tolist(),
            "support_native_rows": list(anchors),
            "support_native_samples": list(anchors.values()),
            "tolerance_samples": tolerance,
            "initial_pulse": pulse,
            "support_query_overlap": len(local),
        }
        descriptors.append(descriptor)
        for stage, result in (("direct", direct), ("dense", dense)):
            report = evaluate_native_predictions(
                radar,
                refs,
                result.proposed_samples,
                result.accepted,
                accepted_samples=result.samples,
                initial_seeds=local,
                native_dx_m=record["horizontal_step_m"],
                dt_ns=record["sample_interval_ns"],
                tolerance_samples=tolerance,
                road_group=episode["group"],
                layer=layer,
                native_trace_indices=native,
                confidence=result.confidence,
            )
            report.update(
                split_role="inner_validation",
                spatial_buffer_verified=True,
                record_id=key,
                episode_index=index,
                remote_support_native_rows=list(anchors),
            )
            reports[str(layer)][stage].append(report)
        if (index + 1) % 24 == 0:
            print(
                json.dumps(
                    {
                        "calibration_episodes": index + 1,
                        "requested": episodes,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    thresholds = np.unique(np.r_[0, np.geomspace(1e-4, 1, 201), 1.000001])
    calibrations = {}
    for layer in config.layers:
        layer_key = str(layer)
        expected_groups = sorted(
            group for group in sampler.index[1] if layer in sampler.index[1][group]
        )
        calibrations[layer_key] = {}
        for stage, members in reports[layer_key].items():
            observed_groups = sorted({r["road_group"] for r in members})
            if observed_groups != expected_groups:
                calibration = {
                    "schema": "processed-dense-train-only-threshold-v1",
                    "selected_threshold": None,
                    "objective_met": False,
                    "training_groups": expected_groups,
                    "excluded_outer_groups": [config.held_out_group],
                    "curve": [],
                    "reason": "Fixed episode budget did not cover every eligible training road",
                    "observed_groups": observed_groups,
                }
            else:
                calibration = select_training_threshold(
                    members,
                    training_groups=expected_groups,
                    excluded_groups=[config.held_out_group],
                    thresholds=thresholds,
                    target_agreement=target_agreement,
                    min_accepted=min_accepted,
                )
            calibration["ungated_per_road_and_pooled"] = aggregate_reports(members)
            calibration["observed_episodes"] = len(members)
            calibration["unique_native_reference_observations"] = len(
                {
                    (r["record_id"], o["native_trace"])
                    for r in members
                    for o in r["evaluation_observations"]
                }
            )
            calibrations[layer_key][stage] = calibration
    if file_sha256(weights) != weights_sha:
        raise ValueError("Weights changed during calibration")
    if any(file_sha256(ROOT / "src/gpr_layer_audit/ml" / k) != v for k, v in current_code.items()):
        raise ValueError("Training/model source changed during calibration")
    document = {
        "schema": "processed-ml-operating-calibration-v1",
        "calibrations": calibrations,
        "model_path": str(model_path.resolve()),
        "model_sha256": file_sha256(model_path),
        "weights_sha256": weights_sha,
        "dataset_sha256": manifest_sha,
        "training_source_sha256": current_code,
        "calibration_source_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                Path(__file__),
                ROOT / "src/gpr_layer_audit/ml/processed_dense_eval.py",
                ROOT / "src/gpr_layer_audit/ml/processed_dense_decoder.py",
            )
        },
        "scorer": scorer_provenance(),
        "validation_cohort_sha256": canonical_sha256(descriptors),
        "validation_episodes": descriptors,
        "requested_episodes": episodes,
        "rng_seed": config.seed + 100000,
        "trace_stride": trace_stride,
        "decoder_config": asdict(decoder_config),
        "decoder_interval_length_diagnostics": span_diagnostics,
        "target_agreement": target_agreement,
        "min_accepted_per_training_road": min_accepted,
        "held_out_group": config.held_out_group,
        "runtime_seconds": time.perf_counter() - started,
        "device": device,
        "inference_precision": "float32; autocast disabled; matches ProcessedPredictor",
        "selection_reads_outer_arrays": False,
        "counts_unit": (
            "episode-query observations; repeated reference coordinates reported separately"
        ),
        "limitations": [
            "TRAIN inner-validation thresholds are a research operating point, not a guarantee.",
            "Dense calibration uses short query windows conditioned on distant seeds; full-road "
            "decoding can change selected paths and must be evaluated separately.",
            "No support row contributes query accuracy or automatic coverage.",
        ],
        "production_promotion": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    _plot_curves(document, output.with_suffix(".png"))
    print(
        json.dumps(
            {
                layer: {stage: value["selected_threshold"] for stage, value in stages.items()}
                for layer, stages in calibrations.items()
            },
            indent=2,
        )
    )
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Completed model.json or run directory"
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--trace-stride", type=int, default=4)
    parser.add_argument("--target-agreement", type=float, default=0.95)
    parser.add_argument("--min-accepted", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--decoder-config", type=Path)
    args = parser.parse_args()
    decoder = (
        DenseDecoderConfig(**json.loads(args.decoder_config.read_text()))
        if args.decoder_config
        else None
    )
    calibrate(
        args.model,
        args.dataset_manifest,
        args.output,
        device=args.device,
        episodes=args.episodes,
        trace_stride=args.trace_stride,
        target_agreement=args.target_agreement,
        min_accepted=args.min_accepted,
        cpu_threads=args.cpu_threads,
        decoder_config=decoder,
    )


if __name__ == "__main__":
    main()
