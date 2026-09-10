from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)


def _new_model(**kwargs):
    from gpr_layer_audit.ml.seed_context_model import SeedContextUNet

    return SeedContextUNet(**kwargs)


def _loss(*args, **kwargs):
    from gpr_layer_audit.ml.seed_context_model import masked_depth_cross_entropy

    return masked_depth_cross_entropy(*args, **kwargs)


def _case(*, batch: int = 2, depth: int = 17, traces: int = 19, seeds: int = 3):
    generator = torch.Generator().manual_seed(1234)
    fine = torch.randn((batch, 1, depth, traces), generator=generator) * 0.15
    coarse = torch.randn((batch, 1, depth, 61), generator=generator) * 0.15
    patches = torch.randn((batch, seeds, 1, 9, 5), generator=generator) * 0.2
    depth_axis = torch.arange(depth, dtype=torch.float32)[None, None, :, None]
    depth_delta = depth_axis - torch.tensor([4.0, 9.0, 13.0])[None, :, None, None]
    depth_delta = depth_delta.expand(batch, seeds, depth, traces).clone()
    dx = torch.linspace(-31.0, 31.0, traces)[None, None, :].expand(batch, seeds, traces).clone()
    mask = torch.tensor([[True, False, True], [True, True, False]])
    layer = torch.tensor([1, 3], dtype=torch.long)
    return {
        "fine": fine,
        "coarse": coarse,
        "seed_patches": patches,
        "relative_depth": depth_delta,
        "relative_dx_m": dx,
        "seed_mask": mask,
        "layer": layer,
    }


def _clone_case(case):
    return {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in case.items()
    }


def test_native_odd_shape_and_backward_are_finite():
    torch.manual_seed(11)
    model = _new_model(width=12, embedding_dim=12, num_layers=3)
    model.eval()
    case = _case()
    logits = model(**case)
    assert logits.shape == (2, 17, 19)

    target = torch.full((2, 19), 7.25)
    target[0, 0] = 4.4
    target[1, 1] = 11.6
    reviewed = torch.ones((2, 19), dtype=torch.bool)
    reviewed[:, -1] = False
    loss = _loss(logits, target, reviewed)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_seed_order_is_invariant_and_remote_seed_changes_evidence():
    torch.manual_seed(12)
    model = _new_model(width=12, embedding_dim=12, num_layers=3).eval()
    case = _case()
    reference = model(**case)

    permutation = torch.tensor([2, 0, 1])
    permuted = _clone_case(case)
    for name in ("seed_patches", "relative_depth", "relative_dx_m", "seed_mask"):
        permuted[name] = permuted[name][:, permutation]
    assert torch.allclose(reference, model(**permuted), rtol=1e-5, atol=1e-5)

    changed_patch = _clone_case(case)
    changed_patch["seed_patches"][:, 0] += 10.0
    patch_delta = (reference - model(**changed_patch)).abs().max()
    assert patch_delta > 1e-6

    changed_geometry = _clone_case(case)
    changed_geometry["relative_dx_m"][:, 0] += 200.0
    geometry_delta = (reference - model(**changed_geometry)).abs().max()
    assert geometry_delta > 1e-6


def test_masked_seed_values_are_ignored_and_all_masked_equals_omitted():
    torch.manual_seed(13)
    model = _new_model(width=12, embedding_dim=12, num_layers=3).eval()
    case = _case()
    reference = model(**case)

    changed_masked = _clone_case(case)
    changed_masked["seed_patches"][0, 1] = 1e7
    changed_masked["seed_patches"][1, 2] = 1e7
    changed_masked["relative_depth"][0, 1] = -1e7
    changed_masked["relative_depth"][1, 2] = -1e7
    changed_masked["relative_dx_m"][0, 1] = 1e7
    changed_masked["relative_dx_m"][1, 2] = 1e7
    assert torch.allclose(reference, model(**changed_masked), rtol=1e-5, atol=1e-5)

    all_masked = _clone_case(case)
    all_masked["seed_mask"] = torch.zeros_like(all_masked["seed_mask"])
    all_masked["seed_patches"][:] = 1e7
    all_masked["relative_depth"][:] = -1e7
    all_masked["relative_dx_m"][:] = 1e7
    omitted = {name: case[name] for name in ("fine", "coarse", "layer")}
    assert torch.allclose(model(**omitted), model(**all_masked), rtol=1e-5, atol=1e-5)


def test_conditioned_false_ignores_seed_inputs_and_narrow_model_ignores_coarse():
    torch.manual_seed(14)
    case = _case()
    unconditional = _new_model(width=12, embedding_dim=12, conditioned=False).eval()
    changed = _clone_case(case)
    changed["seed_patches"][:] = 1e7
    changed["relative_depth"][:] = -1e7
    changed["relative_dx_m"][:] = 1e7
    changed["seed_mask"] = ~changed["seed_mask"]
    assert torch.allclose(unconditional(**case), unconditional(**changed), rtol=1e-5, atol=1e-5)

    narrow = _new_model(width=12, embedding_dim=12, use_context=False).eval()
    coarse_changed = _clone_case(case)
    coarse_changed["coarse"][:] = 1e7
    assert torch.allclose(narrow(**case), narrow(**coarse_changed), rtol=1e-5, atol=1e-5)
    no_coarse = _clone_case(case)
    no_coarse["coarse"] = None
    assert torch.allclose(narrow(**case), narrow(**no_coarse), rtol=1e-5, atol=1e-5)


def test_validity_masks_sanitize_nan_padding():
    torch.manual_seed(15)
    model = _new_model(width=12, embedding_dim=12, num_layers=3).eval()
    case = _case()
    fine_valid = torch.ones_like(case["fine"], dtype=torch.bool)
    coarse_valid = torch.ones_like(case["coarse"], dtype=torch.bool)
    seed_patch_valid = torch.ones_like(case["seed_patches"], dtype=torch.bool)
    fine_valid[:, :, 0, 0] = False
    coarse_valid[:, :, 0, 0] = False
    seed_patch_valid[:, :, :, 0, 0] = False
    masked = _clone_case(case)
    masked["fine"][:, :, 0, 0] = torch.nan
    masked["coarse"][:, :, 0, 0] = torch.nan
    masked["seed_patches"][:, :, :, 0, 0] = torch.nan
    masked_logits = model(
        **masked,
        fine_valid=fine_valid,
        coarse_valid=coarse_valid,
        seed_patch_valid=seed_patch_valid,
    )
    sanitized = _clone_case(case)
    sanitized["fine"][:, :, 0, 0] = 0
    sanitized["coarse"][:, :, 0, 0] = 0
    sanitized["seed_patches"][:, :, :, 0, 0] = 0
    sanitized_logits = model(
        **sanitized,
        fine_valid=fine_valid,
        coarse_valid=coarse_valid,
        seed_patch_valid=seed_patch_valid,
    )
    assert torch.isfinite(masked_logits).all()
    assert torch.allclose(masked_logits, sanitized_logits, rtol=1e-5, atol=1e-5)


def test_masked_depth_cross_entropy_ignores_unknown_and_support_rows():
    from torch.nn import functional as F

    logits = torch.randn((1, 5, 5), generator=torch.Generator().manual_seed(16), requires_grad=True)
    target = torch.tensor([[1.4, float("nan"), 2.6, 3.49, 4.0]])
    reviewed = torch.ones((1, 5), dtype=torch.bool)
    reviewed[:, 4] = False
    support = torch.tensor([[False, False, True, False, False]])
    loss = _loss(logits, target, reviewed, support_mask=support)
    expected = F.cross_entropy(logits[:, :, [0, 3]], torch.tensor([[1, 3]]))
    assert torch.allclose(loss, expected)

    altered = logits.detach().clone()
    altered[:, :, 1] = float("nan")
    altered[:, :, 2] = float("nan")
    altered[:, :, 4] = float("nan")
    altered.requires_grad_()
    ignored_loss = _loss(altered, target, reviewed, support_mask=support)
    assert torch.allclose(loss, ignored_loss)

    unknown_logits = torch.full((1, 5, 5), float("nan"), requires_grad=True)
    unknown_target = torch.full((1, 5), float("nan"))
    unknown = _loss(
        unknown_logits, unknown_target, torch.zeros((1, 5), dtype=torch.bool)
    )
    assert torch.isfinite(unknown) and unknown.item() == 0.0
    unknown.backward()
    assert torch.equal(unknown_logits.grad, torch.zeros_like(unknown_logits))


def test_small_synthetic_episode_learns_seed_identity():
    torch.manual_seed(391)
    depth, traces = 48, 24
    wavelet_axis = torch.arange(9, dtype=torch.float32) - 4
    wavelet = (1 - (wavelet_axis / 2.2) ** 2) * torch.exp(-0.5 * (wavelet_axis / 2.2) ** 2)
    upper = torch.zeros(depth)
    lower = torch.zeros(depth)
    upper[9:18] = wavelet
    lower[30:39] = -wavelet
    radar = (upper + lower)[None, :, None].expand(2, depth, traces).contiguous()
    fine = radar[:, None]
    seed_depth = (13, 34)
    patches = torch.stack(
        [fine[index, :, center - 4 : center + 5, 10:15] for index, center in enumerate(seed_depth)]
    )[:, None]
    target = torch.tensor(seed_depth, dtype=torch.long)[:, None].expand(2, traces)
    relative_depth = torch.zeros((2, 1, depth, 1))
    relative_dx = torch.full((2, 1, traces), 50.0)
    layer = torch.full((2,), 2, dtype=torch.long)

    model = _new_model(width=8, embedding_dim=8).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    valid = torch.ones_like(target, dtype=torch.bool)
    losses = []
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            fine,
            seed_patches=patches,
            relative_depth=relative_depth,
            relative_dx_m=relative_dx,
            layer=layer,
        )
        loss = _loss(logits, target.float(), valid)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    assert losses[-1] < losses[0] * 0.25
    learned = model(
        fine,
        seed_patches=patches,
        relative_depth=relative_depth,
        relative_dx_m=relative_dx,
        layer=layer,
    )
    assert torch.equal(learned.argmax(1), target)

    swapped = model(
        fine,
        seed_patches=patches.flip(0),
        relative_depth=relative_depth,
        relative_dx_m=relative_dx,
        layer=layer,
    )
    assert torch.equal(swapped.argmax(1), target.flip(0))
    removed = model(fine, relative_depth=None, relative_dx_m=None, layer=layer)
    assert torch.allclose(removed[0], removed[1], rtol=1e-5, atol=1e-5)
