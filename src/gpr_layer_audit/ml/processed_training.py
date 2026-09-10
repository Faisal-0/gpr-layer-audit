"""Experimental grouped training on native, signed processed radar observations.

No outer-road image or label enters fitting. Seeds are sampled once per acquisition
episode, outside the query's entire context, and never contribute query loss.
The inner split uses buffered spatial blocks; support patches come from fitting
blocks even when their query is inner validation. Unknown observations stay unknown.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    held_out_group: str = "gujrat"
    layers: tuple[int, ...] = (1, 2, 3)
    steps: int = 1200
    validation_interval: int = 100
    validation_episodes: int = 24
    patience: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    width: int = 12
    fine_width: int = 65
    coarse_span_ratio: int = 4
    patch_depth: int = 65
    patch_width: int = 9
    block_m: float = 64.0
    support_min: int = 1
    support_max: int = 5
    normalization: str = "scan_rms"
    conditioned: bool = True
    use_context: bool = True
    correction_aware: bool = False
    correction_probability: float = 0.5
    seed: int = 42
    small_overfit_episodes: int = 0
    cpu_threads: int = 4

    def validate(self):
        if not self.held_out_group or not self.layers or set(self.layers) - {1, 2, 3}:
            raise ValueError("An explicit held-out physical group and layers 1..3 are required")
        if min(self.steps, self.validation_interval, self.validation_episodes, self.patience) < 1:
            raise ValueError("Positive optimization and validation budgets are required")
        if self.fine_width not in (33, 65, 129) or self.coarse_span_ratio < 1:
            raise ValueError("Use odd native fine width 33/65/129 for exact multiscale centering")
        if min(self.patch_depth, self.patch_width) < 3 or not (
            self.patch_depth % 2 and self.patch_width % 2
        ):
            raise ValueError("Support patch dimensions must be odd and >=3")
        if not 1 <= self.support_min <= self.support_max <= 5:
            raise ValueError("Acquisition-level operating support budget is 1..5")
        if self.normalization not in ("scan_rms", "depth_rms", "none"):
            raise ValueError("Unknown signed normalization ablation")
        if not 0 <= self.correction_probability <= 1 or self.small_overfit_episodes < 0:
            raise ValueError("Invalid correction or overfit budget")
        if self.learning_rate <= 0 or self.block_m <= 0:
            raise ValueError("Positive learning rate and physical block extent required")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_torch(torch, path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def scan_normalization(amplitudes, sample_validity, mode="scan_rms", chunk_traces=4096):
    """Deterministic label-free scale, computed separately for each complete scan.

    A positive divisor preserves signal polarity. Depth-RMS is an explicit ablation
    that compensates depth attenuation; there is no enhanced/rectified replacement.
    No dataset-wide statistic or optimization on an evaluation image is performed.
    """
    if mode not in ("scan_rms", "depth_rms", "none"):
        raise ValueError("Unknown normalization mode")
    if amplitudes.ndim != 2 or sample_validity.shape != amplitudes.shape:
        raise ValueError("A native radar grid and matching validity mask are required")
    if mode == "none":
        return np.ones(amplitudes.shape[1], dtype=np.float32)
    total = np.zeros(amplitudes.shape[1], dtype=np.float64)
    count = np.zeros(amplitudes.shape[1], dtype=np.int64)
    for start in range(0, len(amplitudes), chunk_traces):
        values = np.asarray(amplitudes[start : start + chunk_traces], dtype=np.float64)
        valid = sample_validity[start : start + chunk_traces] & np.isfinite(values)
        values = np.where(valid, values, 0)
        total += np.sum(values * values, axis=0)
        count += valid.sum(axis=0)
    global_rms = float(np.sqrt(total.sum() / max(int(count.sum()), 1)))
    if not np.isfinite(global_rms) or global_rms <= 0:
        raise ValueError("Processed scan has no nonzero finite observations")
    if mode == "scan_rms":
        return np.full(amplitudes.shape[1], global_rms, dtype=np.float32)
    rms = np.sqrt(total / np.maximum(count, 1))
    return np.maximum(rms, global_rms * 0.01).astype(np.float32)


def context_rows(center, width, span_ratio=1):
    # Select actual native observations laterally. Temporal samples never resample.
    fine_start = int(center) - width // 2
    if span_ratio == 1:
        return fine_start + np.arange(width)
    physical_center = fine_start + (width - 1) / 2
    return np.rint(physical_center + (np.arange(width) - (width - 1) / 2) * span_ratio).astype(int)


def spatial_blocks(distances_m, config):
    """Return split and block per trace, excluding the entire context at boundaries.

    Every fifth block is validation; a short scan reserves its last full/partial
    block. All variants use the same physical distance rule. Unregistered repeat
    passes remain grouped for the outer split; this is spatial inner validation,
    not a claim of separately registered nonoverlapping physical road sections.
    """
    distances = np.asarray(distances_m, float)
    if len(distances) < 2 or not np.isfinite(distances).all() or np.any(np.diff(distances) <= 0):
        raise ValueError("Monotonic native physical distances required")
    dx = float(np.median(np.diff(distances)))
    block = np.floor((distances - distances[0]) / config.block_m).astype(int)
    val_blocks = (
        (set(range(4, int(block.max()) + 1, 5)) or {int(block.max())}) if block.max() else set()
    )
    extent = (
        max(
            (config.fine_width - 1) * config.coarse_span_ratio / 2 + 1,
            config.patch_width // 2,
        )
        * dx
    )
    if config.block_m <= 2 * extent:
        raise ValueError("Spatial block must exceed twice the full context radius")
    left = distances[0] + block * config.block_m
    right = np.minimum(left + config.block_m, distances[-1])
    inside = (distances >= left + extent) & (distances <= right - extent)
    split = np.full(len(distances), -1, dtype=np.int8)
    split[inside] = np.where(np.isin(block[inside], list(val_blocks)), 1, 0)
    return (
        split,
        block,
        {"context_buffer_m": extent, "native_dx_m": dx, "validation_blocks": sorted(val_blocks)},
    )


def _gather(data, rows, samples, scale):
    values = data["amplitudes"]
    valid_grid = data["sample_validity"]
    rows, samples = np.asarray(rows, int), np.asarray(samples, int)
    inside = (
        (rows[:, None] >= 0)
        & (rows[:, None] < len(values))
        & (samples[None] >= 0)
        & (samples[None] < values.shape[1])
    )
    rr = np.clip(rows, 0, len(values) - 1)
    ss = np.clip(samples, 0, values.shape[1] - 1)
    raw = np.asarray(values[np.ix_(rr, ss)], dtype=np.float32)
    valid = inside & valid_grid[np.ix_(rr, ss)] & np.isfinite(raw)
    normalized = np.where(valid, raw / scale[ss][None], 0).astype(np.float32)
    return normalized.T[None], valid.T[None]


def extract_episode(data, *, center, support_rows, support_samples, layer, config, scale=None):
    """Label-free input builder shared by training and ordinary frozen inference.

    Caller supplies authorized native support coordinates. Returned arrays have no
    batch dimension. Query targets are intentionally absent. Fine rows are contiguous
    native traces; coarse rows span a wider centered neighborhood without vertical
    resampling. The native selected seed sample remains fractional in coordinates;
    patch centering selects native observations and never alters that observation.
    """
    if scale is None:
        scale = scan_normalization(
            data["amplitudes"], data["sample_validity"], config.normalization
        )
    rows = context_rows(center, config.fine_width)
    wide_rows = context_rows(center, config.fine_width, config.coarse_span_ratio)
    samples = np.arange(data["amplitudes"].shape[1])
    fine, fine_valid = _gather(data, rows, samples, scale)
    coarse, coarse_valid = _gather(data, wide_rows, samples, scale)
    support_rows = np.asarray(support_rows)
    if not np.isfinite(support_rows).all() or not np.equal(
        support_rows, np.rint(support_rows)
    ).all():
        raise ValueError("Support trace coordinates must be integral native observations")
    support_rows = support_rows.astype(int)
    support_samples = np.asarray(support_samples, dtype=float)
    if support_rows.shape != support_samples.shape or support_rows.ndim != 1:
        raise ValueError("One native target sample per support trace required")
    if np.any((support_rows < 0) | (support_rows >= len(data["amplitudes"]))) or not (
        np.isfinite(support_samples).all()
        and np.all((support_samples >= 0) & (support_samples <= samples[-1]))
    ):
        raise ValueError("Authorized support coordinate outside native radar")
    patches, patch_valid = [], []
    for row, sample in zip(support_rows, support_samples, strict=True):
        patch, valid = _gather(
            data,
            row + np.arange(config.patch_width) - config.patch_width // 2,
            int(np.rint(sample)) + np.arange(config.patch_depth) - config.patch_depth // 2,
            scale,
        )
        patches.append(patch)
        patch_valid.append(valid)
    distances = np.asarray(data["distances_m"])
    dx = float(np.median(np.diff(distances)))
    query_distances = distances[0] + rows * dx
    return {
        "fine": fine,
        "coarse": coarse,
        "fine_valid": fine_valid,
        "coarse_valid": coarse_valid,
        "seed_patches": np.asarray(patches, dtype=np.float32).reshape(
            -1, 1, config.patch_depth, config.patch_width
        ),
        "seed_patch_valid": np.asarray(patch_valid, dtype=bool).reshape(
            -1, 1, config.patch_depth, config.patch_width
        ),
        "relative_depth": (samples[None, :, None] - support_samples[:, None, None]).astype(
            np.float32
        ),
        "relative_dx_m": (query_distances[None] - distances[support_rows, None]).astype(np.float32),
        "seed_mask": np.ones(len(support_rows), dtype=bool),
        "layer": np.asarray(layer, dtype=np.int64),
        "rows": rows,
        "wide_rows": wide_rows,
        "support_rows": support_rows,
        "support_samples": support_samples,
    }


class EpisodeSampler:
    """Balanced layer -> eligible road -> acquisition -> spatial block -> variant episodes."""

    def __init__(self, manifest_path, config):
        from .processed_data import load_processed_manifest, open_processed_record

        config.validate()
        self.config = config
        self.manifest_path = str(Path(manifest_path).resolve())
        self.manifest = load_processed_manifest(manifest_path)
        all_records = self.manifest["records"]
        held = [r for r in all_records if r["physical_road_group"] == config.held_out_group]
        if not held:
            raise ValueError("Held-out group is not represented in this dataset")
        held_sources = {r["source_sha256"] for r in held}
        held_acquisitions = {r["acquisition_id"] for r in held}
        records = [
            r
            for r in all_records
            if r["physical_road_group"] != config.held_out_group
            and r.get("training_use") == "research_allowed"
        ]
        if any(
            r["source_sha256"] in held_sources or r["acquisition_id"] in held_acquisitions
            for r in records
        ):
            raise ValueError("Outer-road source/acquisition leaked into fitting")
        # Different acquisitions on a physical road have no registered overlap map.
        # Restrict this arm to one representative acquisition/variant per group so
        # buffered validation sections cannot reappear via an unregistered repeat.
        # The fixed selection is longest eligible native extent, then source hash;
        # neither target geometry nor outer labels participate in this choice.
        by_group = defaultdict(list)
        for record in records:
            if any(record["label_counts"].get(str(layer), 0) for layer in config.layers):
                by_group[record["physical_road_group"]].append(record)
        canonical = {
            group: max(
                acquisitions,
                key=lambda record: (
                    (record["shape"][0] - 1) * record["horizontal_step_m"],
                    record["source_sha256"],
                ),
            )["record_id"]
            for group, acquisitions in by_group.items()
        }
        self.excluded_records = [
            {
                "record_id": r["record_id"],
                "label_counts": r["label_counts"],
                "reason": (
                    "noncanonical acquisition/variant; preserve verified inner spatial separation"
                ),
            }
            for r in records
            if r["record_id"] != canonical.get(r["physical_road_group"])
        ]
        records = [r for r in records if r["record_id"] == canonical.get(r["physical_road_group"])]
        for layer in config.layers:
            groups = {
                r["physical_road_group"] for r in records if r["label_counts"].get(str(layer), 0)
            }
            if len(groups) < 2:
                raise ValueError(f"Research layer {layer} requires at least two training groups")
        self.records, self.arrays, self.scales, self.splits = {}, {}, {}, {}
        self.index = {
            0: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
            1: defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        }
        self.audit = []
        for record in records:
            if not any(record["label_counts"].get(str(layer), 0) for layer in config.layers):
                continue
            key = record["record_id"]
            data = open_processed_record(manifest_path, record, verify_hashes=True)
            # The processed adapter guarantees contiguous native source coordinates.
            if not np.array_equal(
                data["native_sample_indices"], np.arange(data["amplitudes"].shape[1])
            ):
                raise ValueError("Training requires contiguous native sample coordinates")
            split, blocks, audit = spatial_blocks(data["distances_m"], config)
            self.records[key], self.arrays[key], self.splits[key] = record, data, (split, blocks)
            self.scales[key] = scan_normalization(
                data["amplitudes"], data["sample_validity"], config.normalization
            )
            audit.update(
                {
                    "record_id": key,
                    "physical_road_group": record["physical_road_group"],
                    "acquisition_id": record["acquisition_id"],
                    "layers": {},
                }
            )
            for layer in config.layers:
                valid = data["label_valid"][layer - 1] & np.isfinite(data["labels"][layer - 1])
                support = np.flatnonzero(valid & (split == 0))
                audit["layers"][str(layer)] = {
                    "train_observations": int(len(support)),
                    "inner_validation_observations": int((valid & (split == 1)).sum()),
                }
                if not len(support):
                    continue
                for part in (0, 1):
                    centers = np.flatnonzero(valid & (split == part))
                    for block in np.unique(blocks[centers]):
                        selected = centers[blocks[centers] == block]
                        self.index[part][record["physical_road_group"]][layer][
                            record["acquisition_id"]
                        ].append((int(block), key, selected, support))
            self.audit.append(audit)
        if not self.index[0] or not self.index[1]:
            raise ValueError("No usable buffered fitting/inner-validation episodes")

    def sample(self, rng, *, validation=False):
        part = int(validation)
        index = self.index[part]
        for _ in range(100):
            layer = int(rng.choice(sorted({layer for group in index for layer in index[group]})))
            group = str(rng.choice(sorted(group for group in index if layer in index[group])))
            acquisition = str(rng.choice(sorted(index[group][layer])))
            entries = index[group][layer][acquisition]
            block = int(rng.choice(sorted({entry[0] for entry in entries})))
            variants = [entry for entry in entries if entry[0] == block]
            _, key, centers, support = variants[int(rng.integers(len(variants)))]
            center = int(rng.choice(centers))
            # No seed lies in the fine OR wide query context. It identifies a distant
            # target using its own radar patch, never three privileged labels per crop.
            wide = context_rows(center, self.config.fine_width, self.config.coarse_span_ratio)
            radius = self.config.patch_width // 2
            eligible = support[(support < wide.min() - radius) | (support > wide.max() + radius)]
            if not len(eligible):
                continue
            count = int(rng.integers(self.config.support_min, self.config.support_max + 1))
            count = min(count, len(eligible))
            support_rows = np.sort(rng.choice(eligible, count, replace=False))
            data = self.arrays[key]
            episode = extract_episode(
                data,
                center=center,
                support_rows=support_rows,
                support_samples=data["labels"][layer - 1, support_rows],
                layer=layer,
                config=self.config,
                scale=self.scales[key],
            )
            rows = episode["rows"]
            in_bounds = (rows >= 0) & (rows < len(data["amplitudes"]))
            clipped = np.clip(rows, 0, len(data["amplitudes"]) - 1)
            target = data["labels"][layer - 1, clipped].astype(np.float32)
            valid = in_bounds & data["label_valid"][layer - 1, clipped]
            valid &= self.splits[key][0][clipped] == part
            valid &= episode["fine_valid"][0].any(axis=0)
            target_index = np.clip(
                np.rint(np.nan_to_num(target)).astype(int), 0, data["amplitudes"].shape[1] - 1
            )
            valid &= episode["fine_valid"][0, target_index, np.arange(len(rows))]
            episode.update(
                {
                    "target_depth": target,
                    "valid": valid,
                    "support_mask": np.isin(rows, support_rows),
                    "record_id": key,
                    "group": group,
                    "block": block,
                    "center": center,
                }
            )
            if valid.any():
                return episode
        raise ValueError("Cannot draw nonempty query with distant acquisition supports")


_MODEL_KEYS = (
    "fine",
    "coarse",
    "fine_valid",
    "coarse_valid",
    "seed_patches",
    "seed_patch_valid",
    "relative_depth",
    "relative_dx_m",
    "seed_mask",
    "layer",
)


def episode_tensors(episode, device):
    import torch

    return {
        key: torch.from_numpy(np.asarray(episode[key]).copy()).unsqueeze(0).to(device)
        for key in (*_MODEL_KEYS, "target_depth", "valid", "support_mask")
        if key in episode
    }


def _rng_state(torch, rng):
    return {
        "numpy": rng.bit_generator.state,
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def train_processed_model(manifest_path, output_dir, *, config=None, device="auto", resume=None):
    """Train a bounded research run; checkpoints never authorize production use."""
    import torch

    from .seed_context_model import SeedContextUNet, masked_depth_cross_entropy

    config = config or TrainingConfig()
    config.validate()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoint.pt").exists() and resume is None:
        raise FileExistsError(
            "Run already has a checkpoint; choose a new output or explicit resume"
        )
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeedContextUNet(
        width=config.width, conditioned=config.conditioned, use_context=config.use_context
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # The first real CUDA episode overflowed FP16's scaled backward pass.
    # Prefer BF16's FP32 exponent range on supported devices; otherwise use FP32.
    # This choice is fixed by hardware, without consulting evaluation labels.
    use_bf16 = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    sampler = EpisodeSampler(manifest_path, config)
    validation_rng = np.random.default_rng(config.seed + 100000)
    validation = [
        sampler.sample(validation_rng, validation=True) for _ in range(config.validation_episodes)
    ]
    overfit_rng = np.random.default_rng(config.seed + 200000)
    overfit = [sampler.sample(overfit_rng) for _ in range(config.small_overfit_episodes)]
    code_hashes = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ("processed_training.py", "seed_context_model.py", "processed_data.py")
    }
    dataset_hash = file_sha256(manifest_path)
    history, best, stale, start, prior_diagnostics, best_step = [], float("inf"), 0, 0, [], 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        old = dict(state["config"])
        current = asdict(config)
        # Extending the logged total budget is permitted, changing the experiment is not.
        old.pop("steps")
        current.pop("steps")
        if (
            old != current
            or state["dataset_sha256"] != dataset_hash
            or state["code_hashes"] != code_hashes
        ):
            raise ValueError("Resume config, source code, or dataset identity changed")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        start, best, stale, history = state["step"], state["best"], state["stale"], state["history"]
        prior_diagnostics, best_step = state.get("diagnostics", []), state.get("best_step", 0)
        if config.steps <= start:
            raise ValueError("Resumed optimization budget must exceed the completed step")
        rng.bit_generator.state = state["rng_state"]["numpy"]
        random.setstate(state["rng_state"]["python"])
        torch.set_rng_state(state["rng_state"]["torch"].cpu())
        if state["rng_state"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state["rng_state"]["cuda"]])
    metadata = {
        "schema": "processed-seed-context-training-v1",
        "config": asdict(config),
        "architecture": "SeedContextUNet",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "dataset_path": str(Path(manifest_path).resolve()),
        "dataset_sha256": dataset_hash,
        "code_hashes": code_hashes,
        "held_out_group": config.held_out_group,
        "training_road_groups": sorted(
            {r["physical_road_group"] for r in sampler.records.values()}
        ),
        "training_record_ids": sorted(sampler.records),
        "excluded_training_records": sampler.excluded_records,
        "training_source_sha256": sorted({r["source_sha256"] for r in sampler.records.values()}),
        "support_and_inner_split_audit": sampler.audit,
        "normalization": {
            "mode": config.normalization,
            "scope": "deterministic per scan; no labels",
            "scales": {k: v.tolist() for k, v in sampler.scales.items()},
        },
        "physical_contexts": [
            {
                "record_id": k,
                "fine_width_m": (config.fine_width - 1) * r["horizontal_step_m"],
                "coarse_width_m": (config.fine_width - 1)
                * config.coarse_span_ratio
                * r["horizontal_step_m"],
                "support_width_m": (config.patch_width - 1) * r["horizontal_step_m"],
                "support_depth_ns": (config.patch_depth - 1) * r["sample_interval_ns"],
                "native_dt_ns": r["sample_interval_ns"],
                "depth_samples": r["shape"][1],
            }
            for k, r in sampler.records.items()
        ],
        "claim": "grouped development generalization; no untouched road",
        "promotion": {
            "passed": False,
            "reason": "research only; held-out scoring and product gates required",
        },
        "device": device,
        "torch_version": torch.__version__,
        "arithmetic": "cuda_bfloat16_unscaled" if use_bf16 else "float32_unscaled",
        "history": history,
        "weights": "weights.pt",
        "status": "running",
        "diagnostics": prior_diagnostics,
        "best_step": best_step,
    }
    _json(output / "model.json", metadata)
    started, train_losses = time.perf_counter(), []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for step in range(start + 1, config.steps + 1):
        model.train()
        episode = overfit[(step - 1) % len(overfit)] if overfit else sampler.sample(rng)
        tensors = episode_tensors(episode, device)
        inputs = {key: tensors[key] for key in _MODEL_KEYS}
        optimizer.zero_grad(set_to_none=True)
        correction_row = None
        if (
            config.correction_aware
            and int(episode["valid"].sum()) > 1
            and rng.random() < config.correction_probability
        ):
            # TRAINING ONLY: use actual current errors to select one positive answer.
            # This challenger has an explicitly larger support budget for that action.
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                previous = model(**inputs, coarse_span_ratio=config.coarse_span_ratio).softmax(
                    dim=1
                )
            predicted = previous.argmax(dim=1)[0].cpu().numpy()
            errors = np.where(episode["valid"], np.abs(predicted - episode["target_depth"]), -1)
            local = int(np.argmax(errors))
            correction_row = int(episode["rows"][local])
            revised = extract_episode(
                sampler.arrays[episode["record_id"]],
                center=episode["center"],
                support_rows=np.r_[episode["support_rows"], correction_row],
                support_samples=np.r_[episode["support_samples"], episode["target_depth"][local]],
                layer=int(episode["layer"]),
                config=config,
                scale=sampler.scales[episode["record_id"]],
            )
            revised.update({key: episode[key] for key in ("target_depth", "valid", "support_mask")})
            revised["support_mask"] = revised["support_mask"].copy()
            revised["support_mask"][local] = True
            tensors = episode_tensors(revised, device)
            inputs = {key: tensors[key] for key in _MODEL_KEYS}
            inputs["previous_probability"] = previous.detach()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(**inputs, coarse_span_ratio=config.coarse_span_ratio)
            loss = masked_depth_cross_entropy(
                logits,
                tensors["target_depth"],
                tensors["valid"],
                support_mask=tensors["support_mask"],
            )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss; checkpoint is not promoted")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0:
            raise ValueError("Nonfinite or zero gradient; diagnose before further training")
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(float(loss.detach().cpu()))
        if step <= 5 or step % config.validation_interval == 0:
            known = tensors["valid"] & ~tensors["support_mask"]
            pred = logits.detach().argmax(dim=1)
            mae = (pred[known] - tensors["target_depth"][known]).abs().float().mean()
            metadata["diagnostics"].append(
                {
                    "step": step,
                    "loss": train_losses[-1],
                    "gradient_norm": float(grad_norm),
                    "supervised_nonseed_traces": int(known.sum()),
                    "native_sample_mae": float(mae),
                    "support_count": len(episode["support_rows"]),
                    "record_id": episode["record_id"],
                    "correction_row": correction_row,
                }
            )
        if step % config.validation_interval == 0 or step == config.steps:
            model.eval()
            losses, errors = [], []
            with torch.no_grad():
                for val in validation:
                    batch = episode_tensors(val, device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                        pred = model(
                            **{key: batch[key] for key in _MODEL_KEYS},
                            coarse_span_ratio=config.coarse_span_ratio,
                        )
                        val_loss = masked_depth_cross_entropy(
                            pred,
                            batch["target_depth"],
                            batch["valid"],
                            support_mask=batch["support_mask"],
                        )
                    mask = batch["valid"] & ~batch["support_mask"]
                    losses.append(float(val_loss))
                    errors.append(
                        float((pred.argmax(dim=1)[mask] - batch["target_depth"][mask]).abs().mean())
                    )
            validation_loss = float(np.mean(losses))
            if not np.isfinite(validation_loss):
                raise ValueError("Nonfinite inner-validation loss")
            row = {
                "step": step,
                "train_loss": float(np.mean(train_losses)),
                "inner_validation_loss": validation_loss,
                "inner_validation_mae_samples": float(np.mean(errors)),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            train_losses = []
            if validation_loss < best:
                best, stale = validation_loss, 0
                _save_torch(torch, output / "weights.pt", model.state_dict())
                metadata["best_step"] = step
            else:
                stale += 1
            state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "step": step,
                "best": best,
                "stale": stale,
                "history": history,
                "rng_state": _rng_state(torch, rng),
                "config": asdict(config),
                "dataset_sha256": dataset_hash,
                "code_hashes": code_hashes,
                "diagnostics": metadata["diagnostics"],
                "best_step": metadata["best_step"],
            }
            _save_torch(torch, output / "checkpoint.pt", state)
            metadata.update(
                {
                    "history": history,
                    "completed_steps": step,
                    "weights_sha256": file_sha256(output / "weights.pt"),
                    "checkpoint_sha256": file_sha256(output / "checkpoint.pt"),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            _json(output / "model.json", metadata)
            print(json.dumps(row), flush=True)
            if stale >= config.patience:
                metadata["stop_reason"] = "training-road inner-validation patience"
                break
    metadata["status"] = "completed"
    metadata.setdefault("stop_reason", "configured optimization budget")
    metadata["peak_cuda_allocated_bytes"] = (
        torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    )
    _json(output / "model.json", metadata)
    return metadata
