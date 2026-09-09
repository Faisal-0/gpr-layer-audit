from __future__ import annotations

import json

import numpy as np
import pytest

from gpr_layer_audit.ml.features import extract_template, make_inputs, resample_grid


def test_physical_grid_preserves_event_time():
    data = np.zeros((5, 100), np.float32)
    data[:, 40] = 1
    resampled = resample_grid(data, 0.1, 0.4, 0.05, 0.2)
    assert np.argmax(resampled[4]) == 80
    assert resampled.shape[0] == 9


def test_seed_template_conditions_crops_without_clicks():
    data = np.zeros((40, 128), np.float32)
    data[:, 60] = -1
    template = extract_template(data, 0, 60)
    inputs = make_inputs(data[20:], 2, [template])
    assert inputs.shape == (6, 20, 128)
    assert not inputs[3].any()
    assert np.argmax(inputs[2, 10]) == 60


def test_unet_forward_backward_ignores_unknown_labels():
    torch = pytest.importorskip("torch")
    from gpr_layer_audit.ml.model import SeedUNet, masked_loss

    torch.set_num_threads(2)
    model = SeedUNet()
    values = torch.randn(1, 6, 33, 65)
    logits, visible = model(values)
    assert logits.shape == (1, 33, 65) and visible.shape == (1, 33)
    target = torch.zeros_like(logits)
    valid = torch.zeros_like(logits, dtype=torch.bool)
    valid[:, 10] = True
    target[:, 10, 30] = 1
    observed = torch.zeros_like(visible, dtype=torch.bool)
    observed[:, 10] = True
    loss = masked_loss(logits, visible, target, valid, observed.float(), observed)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    changed = target.clone()
    changed[:, 20] = 1
    assert torch.allclose(
        loss, masked_loss(logits, visible, changed, valid, observed.float(), observed)
    )


def bundle(tmp_path):
    torch = pytest.importorskip("torch")
    from gpr_layer_audit.io.dzt import fingerprint_file
    from gpr_layer_audit.ml.model import SeedUNet
    from gpr_layer_audit.processing.hybrid_evidence import PREPROCESSING_VERSION

    torch.save(SeedUNet().state_dict(), tmp_path / "weights.pt")
    metadata = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "weights": "weights.pt",
        "weights_sha256": fingerprint_file(tmp_path / "weights.pt"),
        "width": 16,
        "crop_shape": [256, 512],
        "sample_interval_ns": 0.05,
        "horizontal_step_m": 0.4,
        "supported_layers": [2],
        "promotion": {"passed": False},
    }
    (tmp_path / "model.json").write_text(json.dumps(metadata))
    return metadata


def test_unvalidated_bundle_only_runs_in_explicit_research_mode(tmp_path):
    from gpr_layer_audit.ml.inference import infer_evidence

    bundle(tmp_path)
    data = np.zeros((33, 140), np.float32)
    data[:, 60] = -1
    with pytest.raises(ValueError, match="promotion gates"):
        infer_evidence(tmp_path, data, {2: {2: 60}}, 0.05, 0.4)
    evidence, provenance = infer_evidence(
        tmp_path, data, {2: {2: 60}}, 0.05, 0.4, require_validated=False, device="cpu"
    )
    evidence[2].validate(data.shape)
    assert provenance["ml_status"] == "experimental"


def test_ml_cancellation_not_swallowed_by_auto_fallback(tmp_path):
    from gpr_layer_audit.ml.inference import infer_evidence

    bundle(tmp_path)
    with pytest.raises(InterruptedError):
        infer_evidence(
            tmp_path,
            np.ones((33, 140), np.float32),
            {2: {2: 60}},
            0.05,
            0.4,
            require_validated=False,
            cancel=lambda: True,
        )


def test_training_support_never_reads_test_roads():
    from gpr_layer_audit.ml.training import select_support_labels

    chunks = [
        {
            "split": split,
            "road_group": split,
            "source_sha256": split,
            "labels": [
                {"layer_order": 2, "visibility": "visible", "trace_index": r}
                for r in (1, 10, 20, 30)
            ],
        }
        for split in ("train", "validation", "test")
    ]
    support = select_support_labels(chunks, "train", 2)
    assert set(support) == {"train"}
    assert len(support["train"]) == 3


def test_negative_clicks_use_only_input_seed_rows():
    from gpr_layer_audit.ml.features import competing_seed_clicks

    data = np.zeros((20, 128), np.float32)
    data[:, 60] = -1
    data[:, 75] = -3
    assert competing_seed_clicks(data, [(3, 60)]) == [(3, 75.0)]


def test_one_epoch_training_bundle_smoke(tmp_path):
    from gpr_layer_audit.ml.inference import infer_evidence
    from gpr_layer_audit.ml.training import train_model

    chunks = []
    x = np.arange(128)
    data = np.tile(-np.exp(-(((x - 60) / 3) ** 2)), (64, 1)).astype(np.float32)
    for index, split in enumerate(("train", "train", "train", "validation")):
        path = tmp_path / f"chunk-{index}.npy"
        np.save(path, data)
        chunks.append(
            {
                "path": str(path),
                "split": split,
                "road_group": str(index),
                "source_sha256": str(index),
                "sample_interval_ns": 0.05,
                "horizontal_step_m": 0.4,
                "labels": [
                    dict(
                        row=row,
                        trace_index=row,
                        layer_order=2,
                        sample_aligned=60.0,
                        visibility="visible",
                        training_use="allowed",
                        verified=True,
                        origin="manual",
                    )
                    for row in range(32)
                ],
            }
        )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        json.dumps({"chunks": chunks, "eligibility": {"2": {"eligible_for_pilot": True}}})
    )
    result = train_model(dataset, tmp_path / "trained", epochs=1, device="cpu")
    assert len(result["history"]) == 1
    assert result["promotion"]["passed"] is False
    evidence, _ = infer_evidence(
        tmp_path / "trained", data, {2: {0: 60}}, 0.05, 0.4, require_validated=False, device="cpu"
    )
    assert evidence[2].likelihood.shape == data.shape
