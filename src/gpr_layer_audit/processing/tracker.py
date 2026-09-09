"""Public entry point for the current event-family tracker."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d

from gpr_layer_audit.models import LayerSpec, SearchCorridor

from .seed_graph import SeedConditionedPath, pick_seed_conditioned_interfaces

TRACKER_METHODS = ("joint_seed_adaptive", "seed_hybrid")
PickPath = SeedConditionedPath


def propose_seed_rows(
    candidate_feature: NDArray[np.floating],
    count: int = 3,
    occupied_rows: NDArray[np.integer] | None = None,
) -> NDArray[np.int32]:
    """Choose distributed, high-information stations without consulting labels.

    When observations already exist, prefer high-quality rows in the largest
    uncovered spans. This prevents a follow-up request from simply returning
    the strongest trace next to a seed that is already known.
    """
    rows = candidate_feature.shape[0]
    if rows == 0 or count <= 0:
        return np.empty(0, dtype=np.int32)
    quality = np.percentile(candidate_feature, 98, axis=1) - np.median(candidate_feature, axis=1)
    quality = gaussian_filter1d(quality.astype(float), sigma=max(1.0, rows / 400.0))

    occupied = np.asarray([] if occupied_rows is None else occupied_rows, dtype=np.int32)
    occupied = occupied[(occupied >= 0) & (occupied < rows)]
    if len(occupied):
        # First retain a radar-supported representative from each small road
        # segment, then greedily cover the largest distance from all existing
        # and newly selected observations. Distance is primary; radar quality
        # breaks ties without using design or workbook information.
        candidate_count = min(rows, max(12, 6 * count))
        edges = np.linspace(0, rows, candidate_count + 1, dtype=int)
        candidates: list[int] = []
        for start, stop in zip(edges[:-1], edges[1:], strict=False):
            if stop <= start:
                continue
            candidates.append(start + int(np.argmax(quality[start:stop])))
        available = np.asarray(sorted(set(candidates) - set(occupied.tolist())), dtype=np.int32)
        selected: list[int] = []
        quality_span = float(np.ptp(quality[available])) if len(available) else 0.0
        quality_score = (
            (quality[available] - float(np.min(quality[available]))) / quality_span
            if len(available) and quality_span > 1e-12
            else np.zeros(len(available), dtype=float)
        )
        while len(selected) < count and len(available):
            references = np.asarray([*occupied.tolist(), *selected], dtype=float)
            distance = np.min(
                np.abs(available[:, None].astype(float) - references[None, :]), axis=1
            ) / max(float(rows - 1), 1.0)
            score = 0.75 * distance + 0.25 * quality_score
            chosen_position = int(np.argmax(score))
            selected.append(int(available[chosen_position]))
            available = np.delete(available, chosen_position)
            quality_score = np.delete(quality_score, chosen_position)
        return np.asarray(sorted(selected), dtype=np.int32)

    selected: list[int] = []
    edges = np.linspace(0, rows, min(count, rows) + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:], strict=False):
        if stop <= start:
            continue
        margin = int(0.15 * (stop - start))
        search_start = min(stop - 1, start + margin)
        search_stop = max(search_start + 1, stop - margin)
        selected.append(search_start + int(np.argmax(quality[search_start:search_stop])))
    return np.asarray(sorted(set(selected)), dtype=np.int32)


def _validate_anchors(
    anchor_samples: dict[int, dict[int, int]],
    layers: list[LayerSpec],
    row_count: int,
    sample_count: int,
) -> None:
    enabled = {item.order: item for item in layers if item.analysis_enabled}
    by_row: dict[int, dict[int, int]] = {}
    for order, anchors in anchor_samples.items():
        if order not in enabled:
            continue
        for row, sample in anchors.items():
            if not 0 <= row < row_count or not 0 <= sample < sample_count:
                raise ValueError(
                    f"Layer {order} seed ({row}, {sample}) lies outside the radargram."
                )
            by_row.setdefault(row, {})[order] = sample
    for row, values in by_row.items():
        previous_sample = None
        previous_order = None
        for order in sorted(values):
            sample = values[order]
            if (
                previous_sample is not None
                and sample < previous_sample + enabled[order].min_gap_samples
            ):
                raise ValueError(
                    "Seed interfaces are out of order at stacked row "
                    f"{row}: layer {order} must be at least {enabled[order].min_gap_samples} "
                    f"samples below layer {previous_order}."
                )
            previous_order, previous_sample = order, sample


def pick_interfaces(
    radargram: NDArray[np.floating],
    reference_surface_sample: int,
    layers: list[LayerSpec],
    *,
    anchor_samples: dict[int, dict[int, int]] | None = None,
    seed_metadata: dict[int, dict[int, dict[str, object]]] | None = None,
    matched_template: NDArray[np.floating] | None = None,
    feature_branches: dict[str, NDArray[np.floating]] | None = None,
    design_prior_samples: dict[int, NDArray[np.floating]] | None = None,
    design_prior_widths: dict[int, NDArray[np.floating]] | None = None,
    design_weight: float = 0.10,
    search_corridors: dict[int, SearchCorridor] | None = None,
    pulse_width_samples: float = 7.0,
    break_rows: set[int] | None = None,
    anomaly_mask: NDArray[np.bool_] | None = None,
    max_interpolation_rows: int = 2,
    horizontal_step_m: float = 0.4,
    method: str = "joint_seed_adaptive",
    hybrid_evidence=None,
    model_path: str | None = None,
    ml_policy: str = "auto",
    sample_interval_ns: float = 1.0,
    calibration_path: str | None = None,
    conventional_config=None,
    coordinate_transforms=None,
    cancel: Callable[[], bool] | None = None,
) -> dict[int, PickPath]:
    """Run the established tracker or the experimental fused evidence tracker."""
    del matched_template, design_prior_samples, design_prior_widths
    if method not in TRACKER_METHODS:
        raise ValueError(f"Unknown tracker method {method!r}; choose {TRACKER_METHODS[0]!r}.")
    data = np.asarray(radargram, dtype=np.float32)
    anchors = anchor_samples or {}
    _validate_anchors(anchors, layers, *data.shape)
    anomaly = (
        np.asarray(anomaly_mask, dtype=bool)
        if anomaly_mask is not None
        else np.zeros(len(data), dtype=bool)
    )
    if anomaly.shape != (len(data),):
        raise ValueError("anomaly_mask must contain one value per horizontal bin")
    if method == "seed_hybrid":
        from .conventional_config import resolve_config
        from .hybrid import pick_hybrid_interfaces
        from .hybrid_evidence import HybridEvidence

        conventional = resolve_config(conventional_config)

        if ml_policy not in ("auto", "off", "require"):
            raise ValueError("ML policy must be auto, off, or require")
        if hybrid_evidence is None:
            raw = (feature_branches or {}).get("hybrid_measurement", data)
            hybrid_evidence = HybridEvidence(
                np.asarray(raw, np.float32),
                sample_interval_ns,
                horizontal_step_m,
                provenance={"ml_status": "no_model" if ml_policy != "off" else "disabled"},
                valid=(feature_branches or {}).get("sample_validity"),
                coordinate_transforms=coordinate_transforms or [],
            )
            if ml_policy != "off" and model_path:
                from gpr_layer_audit.ml.inference import infer_evidence

                try:
                    hybrid_evidence.learned, metadata = infer_evidence(
                        model_path,
                        hybrid_evidence.measurement,
                        anchors,
                        sample_interval_ns,
                        horizontal_step_m,
                        cancel=cancel,
                        require_validated=ml_policy != "require",
                    )
                    hybrid_evidence.provenance.update(metadata)
                except InterruptedError:
                    raise
                except (ValueError, OSError, ImportError, RuntimeError) as exc:
                    if ml_policy == "require":
                        raise
                    hybrid_evidence.provenance.update(ml_status="unavailable", reason=str(exc))
            elif ml_policy == "require":
                raise ValueError("Explicit ML evaluation requires a model bundle")
        if calibration_path:
            import json
            from pathlib import Path

            from .hybrid_evidence import PREPROCESSING_VERSION

            calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            if calibration.get("preprocessing_version") != PREPROCESSING_VERSION:
                raise ValueError("Acceptance calibration preprocessing mismatch")
            if calibration.get("model_weights_sha256") != hybrid_evidence.provenance.get(
                "weights_sha256"
            ):
                raise ValueError("Acceptance calibration belongs to different model weights")
            hybrid_evidence.provenance["acceptance_calibration"] = calibration
        if ml_policy == "off" and hybrid_evidence.learned:
            from dataclasses import replace

            hybrid_evidence = replace(
                hybrid_evidence,
                learned={},
                provenance={**hybrid_evidence.provenance, "ml_status": "disabled"},
            )

        paths = pick_hybrid_interfaces(
            data,
            reference_surface_sample,
            layers,
            anchor_samples=anchors,
            seed_metadata=seed_metadata,
            feature_branches=feature_branches,
            search_corridors=search_corridors or {},
            design_weight=design_weight,
            pulse_width_samples=pulse_width_samples,
            break_rows=break_rows or set(),
            anomaly_mask=anomaly,
            horizontal_step_m=horizontal_step_m,
            cancel=cancel,
            evidence=hybrid_evidence,
            config=conventional,
        )
        for path in paths.values():
            path.provenance.update(hybrid_evidence.provenance)
        return paths
    return pick_seed_conditioned_interfaces(
        data,
        reference_surface_sample,
        layers,
        anchor_samples=anchors,
        seed_metadata=seed_metadata,
        feature_branches=feature_branches,
        search_corridors=search_corridors or {},
        design_weight=design_weight,
        pulse_width_samples=pulse_width_samples,
        break_rows=break_rows or set(),
        anomaly_mask=anomaly,
        max_interpolation_rows=max_interpolation_rows,
        horizontal_step_m=horizontal_step_m,
        cancel=cancel,
    )
