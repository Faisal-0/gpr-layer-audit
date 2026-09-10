"""Frozen, radar-only native processed inference for the explicit research backend.

Importing this module does not import PyTorch. Signals and selected seed samples
are never vertically resized, rectified, snapped or fitted to reference labels.
Confidence is local model evidence; acceptance is a separate research decision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .processed_dense_decoder import DenseDecoderConfig, decode_dense_path
from .processed_training import (
    _MODEL_KEYS,
    TrainingConfig,
    extract_episode,
    file_sha256,
    scan_normalization,
)

RADAR_KEYS = (
    "record",
    "amplitudes",
    "sample_validity",
    "native_trace_indices",
    "native_sample_indices",
    "distances_m",
)


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _threshold(value):
    value = float(value)
    if np.isnan(value) or value < 0:
        raise ValueError("Acceptance threshold must be nonnegative or positive infinity")
    return value


class ProcessedPredictor:
    """One frozen checkpoint, native acquisition and explicit retained trace grid.

    Anchors map layer order to {retained row: native floating sample}. Changing
    support observations reruns the frozen model. Correction-aware checkpoints
    additionally receive the last successful ``predict`` evidence, detached, at
    all native tile positions. ``predict_evidence`` is stateless for ablations.
    Inference uses bounded native tile batches, and retains full native history only
    for correction-aware checkpoints, under an explicit bounded memory budget.
    """

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint,
        loaded_data_dict,
        device="cpu",
        trace_stride=4,
        *,
        acceptance_threshold=float("inf"),
        cancel=None,
        allow_training_records=False,
        decoder_config=None,
        max_history_bytes=512 * 1024 * 1024,
        batch_size=4,
    ):
        location = Path(checkpoint).resolve()
        if not location.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {location}")
        directory = location if location.is_dir() else location.parent
        metadata_path = directory / "model.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema") != "processed-seed-context-training-v1"
            or metadata.get("architecture") != "SeedContextUNet"
        ):
            raise ValueError("Checkpoint is not a processed SeedContextUNet research model")
        config = TrainingConfig(**metadata["config"])
        config.validate()
        if int(trace_stride) != trace_stride or trace_stride < 1:
            raise ValueError("Trace stride must be an exact positive integer")
        if max_history_bytes < 1:
            raise ValueError("History memory budget must be positive")
        if int(batch_size) != batch_size or not 1 <= batch_size <= 32:
            raise ValueError("Inference tile batch size must be an integer from 1 to 32")
        # Drop label and reference arrays before retaining any input in this object.
        data = {key: loaded_data_dict[key] for key in RADAR_KEYS}
        record = dict(data["record"])
        if record.get("input_mode") != "processed":
            raise ValueError("Processed checkpoint is incompatible with raw-coordinate input")
        group = record.get("physical_road_group")
        held = metadata.get("held_out_group", config.held_out_group)
        if held != config.held_out_group:
            raise ValueError("Checkpoint held-out fold metadata is contradictory")
        trained = (
            group in metadata.get("training_road_groups", [])
            or record.get("record_id") in metadata.get("training_record_ids", [])
            or record.get("source_sha256") in metadata.get("training_source_sha256", [])
        )
        if not allow_training_records and (trained or group != held):
            raise ValueError("Input is outside the checkpoint held-out fold or overlaps fitting")
        values, valid = data["amplitudes"], data["sample_validity"]
        if (
            values.ndim != 2
            or min(values.shape) < 2
            or values.shape[1] < 4
            or valid.shape != values.shape
            or tuple(record["shape"]) != values.shape
        ):
            raise ValueError("Radar/validity arrays must match the full native processed shape")
        if valid.dtype != np.bool_:
            raise ValueError("Sample validity must be an explicit boolean native mask")
        if not np.array_equal(data["native_trace_indices"], np.arange(len(values))):
            raise ValueError("Input must retain all contiguous native traces before inference")
        if not np.array_equal(data["native_sample_indices"], np.arange(values.shape[1])):
            raise ValueError("Native sample coordinates must be contiguous and unresampled")
        distances = np.asarray(data["distances_m"], dtype=float)
        dt, native_dx = float(record["sample_interval_ns"]), float(record["horizontal_step_m"])
        if (
            not np.isfinite(dt)
            or dt <= 0
            or not np.isfinite(native_dx)
            or native_dx <= 0
            or distances.shape != (len(values),)
            or not np.isfinite(distances).all()
            or not np.allclose(np.diff(distances), native_dx, rtol=1e-6, atol=1e-8)
        ):
            raise ValueError(
                "Positive native sample timing and regular physical trace spacing required"
            )
        contexts = metadata.get("physical_contexts", [])
        if contexts and not any(
            int(item["depth_samples"]) == values.shape[1]
            and np.isclose(float(item["native_dt_ns"]), dt, rtol=1e-6)
            for item in contexts
        ):
            raise ValueError("Input depth/time grid is outside checkpoint training compatibility")
        weight_name = metadata.get("weights", "weights.pt")
        weights_path = (directory / weight_name).resolve()
        if not weights_path.is_relative_to(directory):
            raise ValueError("Checkpoint weight path escapes checkpoint directory")
        if not weights_path.is_file():
            raise FileNotFoundError(
                "Best validation weights are missing; resumable state is not inference"
            )
        weights_hash = file_sha256(weights_path)
        if metadata.get("weights_sha256", weights_hash) != weights_hash:
            raise ValueError("Best checkpoint weight fingerprint changed")
        threshold = _threshold(acceptance_threshold)
        if cancel is not None and cancel():
            raise InterruptedError("Processed inference cancelled")
        import torch

        from .seed_context_model import SeedContextUNet

        device = str(device)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            torch.set_num_threads(config.cpu_threads)
        model = SeedContextUNet(
            width=config.width, conditioned=config.conditioned, use_context=config.use_context
        )
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device).eval().requires_grad_(False)
        if file_sha256(weights_path) != weights_hash:
            raise ValueError("Checkpoint changed while loading frozen weights")
        self = cls()
        self.model, self.config, self.metadata, self.device = model, config, metadata, device
        self.data, self.record = data, record
        self.trace_stride = int(trace_stride)
        self.measurement = values[:: self.trace_stride]
        self.valid = valid[:: self.trace_stride]
        self.native_trace_indices = np.asarray(data["native_trace_indices"])[:: self.trace_stride]
        self.distances_m = distances[:: self.trace_stride]
        self.dt_ns, self.dx_m = dt, native_dx * self.trace_stride
        self.acceptance_threshold, self.cancel = threshold, cancel
        self.decoder_config = decoder_config or DenseDecoderConfig()
        self.weights_path, self.weights_sha256 = weights_path, weights_hash
        self.max_history_bytes = int(max_history_bytes)
        self.batch_size = int(batch_size)
        self._previous_probability = {}
        self.scale = scan_normalization(values, valid, config.normalization)
        source_hashes = {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "processed_inference.py",
                "processed_training.py",
                "seed_context_model.py",
                "processed_data.py",
                "processed_dense_decoder.py",
            )
        }
        preprocessing = {
            "coordinate_mode": "processed",
            "normalization": config.normalization,
            "normalization_scale_sha256": hashlib.sha256(self.scale.tobytes()).hexdigest(),
            "fine_width": config.fine_width,
            "fine_trace_stride": 1,
            "coarse_span_ratio": config.coarse_span_ratio,
            "output_trace_stride": self.trace_stride,
            "native_dt_ns": dt,
            "native_dx_m": native_dx,
            "native_shape": list(values.shape),
            "sample_resampling": False,
        }
        self.provenance = {
            "backend": "processed_seed_context_research",
            "coordinate_mode": "processed",
            "production_eligible": False,
            "record_id": record.get("record_id"),
            "physical_road_group": group,
            "input_sha256": record.get("source_sha256"),
            "model_sha256": weights_hash,
            "model_metadata_sha256": file_sha256(metadata_path),
            "model_path": str(weights_path),
            "selected_weights": "best inner validation",
            "held_out_group": held,
            "allow_training_records": bool(allow_training_records),
            "training_overlap": bool(trained),
            "weights_refitted": False,
            "source_hashes": source_hashes,
            "source_sha256": _hash_json(source_hashes),
            "preprocessing": preprocessing,
            "preprocessing_sha256": _hash_json(preprocessing),
            "confidence_is_correctness_probability": False,
            "tile_batch_size": self.batch_size,
        }
        return self

    def _anchors(self, anchors):
        if not isinstance(anchors, Mapping) or not anchors:
            raise ValueError("Explicit layer-to-seed mapping is required")
        output = {}
        for layer, observations in anchors.items():
            if int(layer) != layer or int(layer) not in self.config.layers:
                raise ValueError("Requested layer is outside the checkpoint trained layers")
            if not isinstance(observations, Mapping):
                raise ValueError("Layer observations must map retained rows to native samples")
            selected = {}
            for row, sample in observations.items():
                if int(row) != row or not 0 <= row < len(self.measurement):
                    raise ValueError("Seed trace must be an exact retained native-grid row")
                if not np.isfinite(sample) or not 0 <= sample <= self.measurement.shape[1] - 1:
                    raise ValueError("Seed sample lies outside the native depth grid")
                selected[int(row)] = float(sample)
            output[int(layer)] = selected
        return output

    def reset_history(self):
        """Start an independent correction session without changing model weights."""
        self._previous_probability.clear()

    def export_replay_state(self):
        """Detached evidence snapshot for the trusted local replay serializer.

        Arrays are read-only and replaced after inference, so exporting does not
        double full-road memory. No model parameters or reference labels occur.
        """
        for value in self._previous_probability.values():
            value.flags.writeable = False
        return {
            "schema": "processed-predictor-replay-state-v1",
            "model_sha256": self.weights_sha256,
            "record_id": self.record.get("record_id"),
            "input_sha256": self.record.get("source_sha256"),
            "trace_stride": self.trace_stride,
            "previous_probability": dict(self._previous_probability),
        }

    def restore_replay_state(self, state):
        """Restore validated guidance after resume, cancellation or rejected edits."""
        expected = self.export_replay_state()
        for key in ("schema", "model_sha256", "record_id", "input_sha256", "trace_stride"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Predictor replay state mismatch: {key}")
        history = state.get("previous_probability", {})
        if not self.config.correction_aware and history:
            raise ValueError("Stateless checkpoint cannot restore correction guidance")
        if (
            sum(np.asarray(value).nbytes for value in history.values())
            > self.max_history_bytes // 2
        ):
            raise MemoryError("Restored correction guidance exceeds history memory budget")
        restored = {}
        for layer, value in history.items():
            array = np.asarray(value)
            if (
                layer not in self.config.layers
                or array.dtype != np.float32
                or array.shape != self.data["amplitudes"].shape
                or not np.isfinite(array).all()
                or np.any(array < 0)
            ):
                raise ValueError("Invalid native probability in predictor replay state")
            array.flags.writeable = False
            restored[layer] = array
        self._previous_probability = restored

    def commit_replay_scope(self, previous_state, order, start_row, stop_row):
        """Commit guidance only inside the inclusive retained-row correction scope."""
        if (
            int(start_row) != start_row
            or int(stop_row) != stop_row
            or not 0 <= start_row <= stop_row < len(self.measurement)
        ):
            raise ValueError("Replay guidance scope must use exact retained rows")
        current = self._previous_probability
        self.restore_replay_state(previous_state)
        if not self.config.correction_aware:
            return
        if order not in current:
            raise ValueError("No newly inferred probability for the corrected layer")
        # Native hidden guidance follows the same physical interval as retained output.
        start = int(start_row) * self.trace_stride
        stop = int(stop_row) * self.trace_stride + 1
        if start_row == 0 and stop_row == len(self.measurement) - 1:
            stop = len(self.data["amplitudes"])
        previous = self._previous_probability.get(order)
        result = np.zeros_like(current[order]) if previous is None else previous.copy()
        result[start:stop] = current[order][start:stop]
        result.flags.writeable = False
        self._previous_probability = {**self._previous_probability, order: result}

    def _infer(
        self,
        anchors,
        *,
        conditioning=None,
        use_context=None,
        previous_probability=None,
        retain_history=False,
        cancel=None,
    ):
        import torch

        anchors = self._anchors(anchors)
        cancel = cancel if cancel is not None else self.cancel
        history = {}
        n, depth = self.data["amplitudes"].shape
        if retain_history:
            # Includes old and new full-grid histories during a successful replacement.
            required = 2 * len(set(anchors) | set(self._previous_probability)) * n * depth * 4
            if required > self.max_history_bytes:
                raise MemoryError(
                    "Correction guidance exceeds the configured history memory budget"
                )
        old_conditioned, old_context = self.model.conditioned, self.model.use_context
        self.model.conditioned = old_conditioned if conditioning is None else bool(conditioning)
        self.model.use_context = old_context if use_context is None else bool(use_context)
        output = {}
        # Halo protects stitched tile edges; no lateral or temporal output interpolation.
        core = self.config.fine_width // 2 + 1
        half = self.config.fine_width // 2
        try:
            with torch.inference_mode():
                for layer, selected in sorted(anchors.items()):
                    probabilities = np.zeros((len(self.measurement), depth), np.float32)
                    native_history = np.zeros((n, depth), np.float32) if retain_history else None
                    supports = sorted(selected)
                    previous = (previous_probability or {}).get(layer)
                    if previous is not None:
                        previous = np.asarray(previous)
                        if previous.shape not in ((n, depth), probabilities.shape):
                            raise ValueError(
                                "Previous probability must use native or retained radar grid"
                            )
                        if not np.isfinite(previous).all() or np.any(previous < 0):
                            raise ValueError("Previous probability must be finite and nonnegative")
                    tile_starts = list(range(0, n, core))
                    for batch_start in range(0, len(tile_starts), self.batch_size):
                        if cancel is not None and cancel():
                            raise InterruptedError("Processed inference cancelled")
                        episodes, layouts, guidance_batch = [], [], []
                        for start in tile_starts[batch_start : batch_start + self.batch_size]:
                            stop = min(start + core, n)
                            retained = np.arange(
                                ((start + self.trace_stride - 1) // self.trace_stride)
                                * self.trace_stride,
                                stop,
                                self.trace_stride,
                            )
                            if not len(retained) and not retain_history:
                                continue
                            center = start + core // 2
                            episode = extract_episode(
                                self.data,
                                center=center,
                                support_rows=[row * self.trace_stride for row in supports],
                                support_samples=[selected[row] for row in supports],
                                layer=layer,
                                config=self.config,
                                scale=self.scale,
                            )
                            episodes.append(episode)
                            layouts.append((start, stop, center, retained))
                            if previous is not None:
                                guidance = np.zeros((depth, self.config.fine_width), np.float32)
                                rows = episode["rows"]
                                inside = (rows >= 0) & (rows < n)
                                if previous.shape == (n, depth):
                                    guidance[:, inside] = previous[rows[inside]].T
                                else:
                                    inside &= rows % self.trace_stride == 0
                                    guidance[:, inside] = previous[
                                        rows[inside] // self.trace_stride
                                    ].T
                                guidance_batch.append(guidance)
                        if not episodes:
                            continue
                        inputs = {
                            key: torch.from_numpy(np.stack([e[key] for e in episodes])).to(
                                self.device
                            )
                            for key in _MODEL_KEYS
                        }
                        if previous is not None:
                            inputs["previous_probability"] = (
                                torch.from_numpy(np.stack(guidance_batch)).to(self.device).detach()
                            )
                        logits = self.model(
                            **inputs, coarse_span_ratio=self.config.coarse_span_ratio
                        )
                        batch_p = logits.float().softmax(dim=1).transpose(1, 2).cpu().numpy()
                        for p, (start, stop, center, retained) in zip(
                            batch_p, layouts, strict=True
                        ):
                            first = start - (center - half)
                            slab = p[first : first + stop - start].copy()
                            mask = self.data["sample_validity"][start:stop]
                            slab[~mask] = 0
                            totals = slab.sum(axis=1, keepdims=True)
                            np.divide(slab, totals, out=slab, where=totals > 0)
                            if not np.isfinite(slab).all():
                                raise ValueError("Model produced nonfinite native depth evidence")
                            probabilities[retained // self.trace_stride] = slab[retained - start]
                            if native_history is not None:
                                native_history[start:stop] = slab
                    output[layer] = probabilities
                    if native_history is not None:
                        history[layer] = native_history
        finally:
            self.model.conditioned, self.model.use_context = old_conditioned, old_context
        return output, history

    def predict_evidence(
        self,
        anchors,
        conditioning=None,
        use_context=None,
        *,
        previous_probability=None,
        cancel=None,
    ):
        """Direct native-depth evidence; no decoding, seed overwrite or state update."""
        return self._infer(
            anchors,
            conditioning=conditioning,
            use_context=use_context,
            previous_probability=previous_probability,
            cancel=cancel,
        )[0]

    def predict(
        self, anchors, *, acceptance_threshold=None, previous_probability=None, cancel=None
    ):
        from gpr_layer_audit.processing.seed_graph import SeedConditionedPath

        anchors = self._anchors(anchors)
        threshold = _threshold(
            self.acceptance_threshold if acceptance_threshold is None else acceptance_threshold
        )
        previous = previous_probability
        if previous is None and self.config.correction_aware:
            previous = self._previous_probability
        probabilities, history = self._infer(
            anchors,
            previous_probability=previous,
            retain_history=self.config.correction_aware,
            cancel=cancel,
        )
        output = {}
        for layer, p in probabilities.items():
            # Decoder requires finite thresholds. Two exceeds every normalized model score.
            decoder_config = replace(
                self.decoder_config,
                acceptance_threshold=(2.0 if np.isposinf(threshold) else threshold),
            )
            decoded = decode_dense_path(
                p,
                anchors[layer],
                sample_valid=self.valid,
                dx_m=self.dx_m,
                dt_ns=self.dt_ns,
                config=decoder_config,
                cancel=cancel if cancel is not None else self.cancel,
            )
            for row, sample in anchors[layer].items():
                if decoded.samples[row] != sample or not decoded.accepted[row]:
                    raise RuntimeError("Dense decoder changed an exact authorized seed")
            direct = np.argmax(p, axis=1).astype(float)
            direct[p.sum(axis=1) == 0] = -1
            zeros = np.zeros(len(p), bool)
            provenance = {
                **self.provenance,
                "layer": layer,
                "acceptance_threshold": "withheld_uncalibrated"
                if np.isposinf(threshold)
                else threshold,
                "decoder": asdict(decoder_config),
                "decoder_diagnostics": decoded.diagnostics,
                "seed_observations": {str(k): v for k, v in sorted(anchors[layer].items())},
                "previous_probability_guidance": previous is not None and layer in previous,
                "correction_aware_checkpoint": self.config.correction_aware,
            }
            output[layer] = SeedConditionedPath(
                samples=decoded.samples,
                confidence=decoded.confidence,
                feature=p,
                alternate_samples=np.full(len(p), -1.0),
                visible=decoded.accepted,
                interpolated=zeros.copy(),
                evidence={
                    "model_depth_probability": decoded.confidence,
                    "normalized_entropy": decoded.entropy,
                    "manual_seed": decoded.manual,
                    "gap_mask": decoded.gap_mask,
                },
                signal_only_samples=direct,
                design_guided_samples=decoded.proposed_samples.copy(),
                design_conflict=zeros.copy(),
                design_constrained=False,
                candidate_components={"processed_ml_probability": p},
                provisional_samples=decoded.proposed_samples,
                provenance=provenance,
            )
        if self.config.correction_aware:
            self._previous_probability = {**self._previous_probability, **history}
        return output
