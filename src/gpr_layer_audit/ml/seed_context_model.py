"""Experimental native-depth predictor with remote, individual seed conditioning.

Unlike ``SeedUNet``, this model receives a separately encoded wider radar view
and a set of authorized seed patches for *every* tile.  Learned attention keeps
each seed's waveform embedding and relative geometry together.  No interpolated
seed path, automatically generated negative click, or visibility target is used.

Inputs use ``[batch, channel, sample, trace]`` ordering.  The model returns depth
logits ``[batch, sample, trace]``; callers use depth-wise softmax or discrete argmax,
never soft-argmax.  Pooling in the encoder is followed by native-depth skip
features and an output at exactly the original sample/trace coordinates.

This optional PyTorch module is research code, not a production model promotion.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_VERSION = "seed-context-native-depth-v1"


class _ResidualBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        groups = math.gcd(4, outgoing)
        self.body = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outgoing),
        )
        self.skip = nn.Identity() if incoming == outgoing else nn.Conv2d(incoming, outgoing, 1)

    def forward(self, values: Tensor) -> Tensor:
        return F.silu(self.body(values) + self.skip(values))


class SeedContextUNet(nn.Module):
    """A small two-level ResUNet and permutation-invariant seed attention.

    ``coarse`` covers a center-aligned, wider lateral span with the same temporal
    extent as ``fine``. ``coarse_span_ratio`` is the ratio of physical lateral
    extents (last minus first trace), not the ratio of array widths. The centered
    portion is sampled from the encoded context using this ratio. Padding and
    acquisition-boundary handling belong to the coordinate-aware data adapter.

    Seed patches have odd dimensions and are centered on authorized observations,
    including observations outside the tile. ``relative_depth[b,s,d,w]`` is output
    sample minus seed sample; the final dimension may be 1 for broadcasting.
    ``relative_dx_m[b,s,w]`` is output chainage minus seed chainage in metres.
    These are measured offsets, not an interpolated target surface. Seed order
    does not affect the result. Missing/all-masked seeds give finite, unconditioned
    features. A masked padded seed may contain NaNs; it is removed before encoding.

    ``layer`` contains optional semantic indices 1..3; 0 means unspecified.
    ``conditioned=False`` is the layer-only control and ignores all seed inputs.
    ``use_context=False`` is the narrow-context ablation and ignores ``coarse``.
    ``previous_probability`` optionally supplies a previous depth distribution
    for a separately trained correction-aware experiment; its default is zero.
    Optional ``fine_valid``, ``coarse_valid``, and ``seed_patch_valid`` have the
    same shapes as their signals. They remove padded values before encoding and
    supply an explicit validity channel. Invalid fine pixels receive -1e4 logits;
    callers must still mark all-invalid traces unresolved. Omitting masks means
    every supplied pixel is valid, so callers that pad must provide these masks.
    """

    def __init__(
        self,
        width: int = 12,
        embedding_dim: int = 12,
        conditioned: bool = True,
        use_context: bool = True,
        num_layers: int = 3,
    ):
        super().__init__()
        if width < 4 or embedding_dim < 2 or num_layers < 1:
            raise ValueError("width >= 4, embedding_dim >= 2 and num_layers >= 1 are required")
        self.width = width
        self.embedding_dim = embedding_dim
        self.conditioned = conditioned
        self.use_context = use_context
        self.num_layers = num_layers
        self.stem = _ResidualBlock(2, width)
        self.encoder1 = _ResidualBlock(width, width * 2)
        self.encoder2 = _ResidualBlock(width * 2, width * 4)
        self.context_encoder = nn.Sequential(
            _ResidualBlock(2, width),
            nn.AvgPool2d(2, ceil_mode=True),
            _ResidualBlock(width, width * 2),
            nn.AvgPool2d(2, ceil_mode=True),
            _ResidualBlock(width * 2, width * 2),
        )
        self.bridge = _ResidualBlock(width * 6, width * 4)
        self.decoder1 = _ResidualBlock(width * 6, width * 2)
        self.decoder0 = _ResidualBlock(width * 3, width)
        # The local stem is shared by radar and seed patches; the seed embedding
        # retains the exact center separately from the wider patch context.
        self.seed_embedding = nn.Linear(width * 2, embedding_dim)
        self.query_embedding = nn.Conv2d(width * 2, embedding_dim, 1)
        self.seed_attention = nn.Sequential(
            nn.Conv2d(5, max(4, width // 2), 1),
            nn.SiLU(),
            nn.Conv2d(max(4, width // 2), 1, 1),
        )
        self.layer_embedding = nn.Embedding(num_layers + 1, width, padding_idx=0)
        # Features: native radar + attended seed embedding + five attended pair
        # features + any-seed flag + layer embedding + previous probability.
        self.readout = nn.Sequential(
            _ResidualBlock(width * 2 + embedding_dim + 7, width),
            nn.Conv2d(width, 1, 1),
        )

    @staticmethod
    def _with_validity(values: Tensor, valid: Tensor | None, name: str) -> Tensor:
        if valid is None:
            valid = torch.ones_like(values, dtype=torch.bool)
        elif valid.shape != values.shape:
            raise ValueError(f"{name} must have the same shape as its signal")
        valid = valid.to(device=values.device, dtype=torch.bool)
        return torch.cat((torch.where(valid, values, 0), valid.to(values.dtype)), dim=1)

    def _aligned_context(
        self,
        fine: Tensor,
        coarse: Tensor | None,
        size: tuple[int, int],
        ratio: float,
        coarse_valid: Tensor | None,
    ) -> Tensor:
        batch = fine.shape[0]
        if not self.use_context or coarse is None:
            return fine.new_zeros((batch, self.width * 2, *size))
        if coarse.ndim != 4 or coarse.shape[:2] != fine.shape[:2]:
            raise ValueError("coarse must have shape [batch, 1, depth, trace]")
        if coarse.shape[2] != fine.shape[2]:
            raise ValueError("fine and coarse must cover the same native depth coordinates")
        if not math.isfinite(ratio) or ratio < 1:
            raise ValueError("coarse_span_ratio must be finite and >= 1")
        encoded = self.context_encoder(self._with_validity(coarse, coarse_valid, "coarse_valid"))
        # Compute sampling coordinates in float32, including under autocast;
        # grid_sample accepts both input and grid in float32 on CPU and CUDA.
        vertical = torch.linspace(-1, 1, size[0], device=fine.device)
        horizontal = torch.linspace(-1 / ratio, 1 / ratio, size[1], device=fine.device)
        yy, xx = torch.meshgrid(vertical, horizontal, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        sampled = F.grid_sample(
            encoded.float(),
            grid.float(),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.to(encoded.dtype)

    def _condition(
        self,
        decoded: Tensor,
        native: Tensor,
        seed_patches: Tensor | None,
        relative_depth: Tensor | None,
        relative_dx_m: Tensor | None,
        seed_mask: Tensor | None,
        seed_patch_valid: Tensor | None,
    ) -> Tensor:
        batch, _, depth, traces = decoded.shape
        empty = decoded.new_zeros((batch, self.embedding_dim + 6, depth, traces))
        if not self.conditioned or seed_patches is None:
            return empty
        if seed_patches.ndim != 5 or seed_patches.shape[0] != batch or seed_patches.shape[2] != 1:
            raise ValueError("Expected seed_patches [batch, seed, 1, patch_depth, patch_trace]")
        if seed_patches.shape[1] == 0:
            return empty
        seeds, patch_depth, patch_traces = (
            seed_patches.shape[1], seed_patches.shape[3], seed_patches.shape[4]
        )
        if patch_depth % 2 != 1 or patch_traces % 2 != 1:
            raise ValueError("Seed patches must have odd dimensions with the observation at center")
        if seed_mask is None:
            seed_mask = torch.ones((batch, seeds), dtype=torch.bool, device=decoded.device)
        elif seed_mask.shape != (batch, seeds):
            raise ValueError("seed_mask must have shape [batch, seed]")
        seed_mask = seed_mask.to(device=decoded.device, dtype=torch.bool)
        if relative_depth is None or relative_dx_m is None:
            raise ValueError("Conditioned seed patches require relative_depth and relative_dx_m")
        if relative_depth.shape not in ((batch, seeds, depth, traces), (batch, seeds, depth, 1)):
            raise ValueError("relative_depth must have shape [batch, seed, depth, trace or 1]")
        if relative_dx_m.shape != (batch, seeds, traces):
            raise ValueError("relative_dx_m must have shape [batch, seed, trace]")

        if seed_patch_valid is None:
            patch_valid = torch.ones_like(seed_patches, dtype=torch.bool)
        elif seed_patch_valid.shape != seed_patches.shape:
            raise ValueError("seed_patch_valid must have the same shape as seed_patches")
        else:
            patch_valid = seed_patch_valid.to(device=seed_patches.device, dtype=torch.bool)
        # A valid patch neighborhood cannot make a missing observation valid.
        seed_mask = seed_mask & patch_valid[:, :, 0, patch_depth // 2, patch_traces // 2]
        patch_valid = patch_valid & seed_mask[:, :, None, None, None]
        patches = seed_patches.reshape(batch * seeds, 1, patch_depth, patch_traces)
        patch_valid = patch_valid.reshape_as(patches)
        encoded = self.stem(self._with_validity(patches, patch_valid, "seed_patch_valid"))
        center = encoded[:, :, patch_depth // 2, patch_traces // 2]
        context = encoded.mean(dim=(-2, -1))
        embeddings = F.normalize(self.seed_embedding(torch.cat((center, context), dim=1)), dim=1)
        embeddings = embeddings.reshape(batch, seeds, self.embedding_dim)
        query = F.normalize(self.query_embedding(torch.cat((decoded, native), dim=1)), dim=1)
        similarity = torch.einsum("bse,bedw->bsdw", embeddings, query)
        # Signed log transforms keep metres/tens-of-metres offsets represented
        # without numerical saturation. They impose no preferred reflector path.
        delta_depth = torch.where(seed_mask[:, :, None, None], relative_depth, 0).float()
        delta_x = torch.where(seed_mask[:, :, None], relative_dx_m, 0).float()
        delta_depth = torch.sign(delta_depth) * torch.log1p(delta_depth.abs() / 32.0)
        delta_x = torch.sign(delta_x) * torch.log1p(delta_x.abs() / 25.0)
        delta_depth = delta_depth.expand(-1, -1, -1, traces)
        delta_x = delta_x[:, :, None, :].expand(-1, -1, depth, -1)
        pair = torch.stack(
            (similarity, delta_depth, delta_depth.abs(), delta_x, delta_x.abs()), dim=2
        ).to(decoded.dtype)
        attention_logits = self.seed_attention(pair.reshape(batch * seeds, 5, depth, traces))
        attention_logits = attention_logits.reshape(batch, seeds, depth, traces).float()
        attention_logits = attention_logits.masked_fill(~seed_mask[:, :, None, None], -1e4)
        weights = attention_logits.softmax(dim=1) * seed_mask[:, :, None, None]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weights = weights.to(decoded.dtype)
        attended_embedding = torch.einsum("bsdw,bse->bedw", weights, embeddings.to(decoded.dtype))
        attended_pair = (weights[:, :, None] * pair).sum(dim=1)
        present = seed_mask.any(dim=1).to(decoded.dtype)[:, None, None, None]
        present = present.expand(-1, 1, depth, traces)
        return torch.cat((attended_embedding, attended_pair, present), dim=1)

    def forward(
        self,
        fine: Tensor,
        coarse: Tensor | None = None,
        seed_patches: Tensor | None = None,
        relative_depth: Tensor | None = None,
        relative_dx_m: Tensor | None = None,
        seed_mask: Tensor | None = None,
        layer: Tensor | None = None,
        *,
        coarse_span_ratio: float = 4.0,
        previous_probability: Tensor | None = None,
        fine_valid: Tensor | None = None,
        coarse_valid: Tensor | None = None,
        seed_patch_valid: Tensor | None = None,
    ) -> Tensor:
        if fine.ndim != 4 or fine.shape[1] != 1 or min(fine.shape[2:]) < 4:
            raise ValueError("fine must have shape [batch, 1, depth >= 4, trace >= 4]")
        batch, _, depth, traces = fine.shape
        native = self.stem(self._with_validity(fine, fine_valid, "fine_valid"))
        level1 = self.encoder1(F.avg_pool2d(native, 2, ceil_mode=True))
        level2 = self.encoder2(F.avg_pool2d(level1, 2, ceil_mode=True))
        context = self._aligned_context(
            fine, coarse, level2.shape[-2:], coarse_span_ratio, coarse_valid
        )
        decoded = self.bridge(torch.cat((level2, context), dim=1))
        decoded = F.interpolate(
            decoded, size=level1.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder1(torch.cat((decoded, level1), dim=1))
        decoded = F.interpolate(
            decoded, size=native.shape[-2:], mode="bilinear", align_corners=False
        )
        decoded = self.decoder0(torch.cat((decoded, native), dim=1))
        conditioned = self._condition(
            decoded,
            native,
            seed_patches,
            relative_depth,
            relative_dx_m,
            seed_mask,
            seed_patch_valid,
        )
        if layer is None:
            layer = torch.zeros(batch, device=fine.device, dtype=torch.long)
        if layer.shape != (batch,):
            raise ValueError("layer must have shape [batch] with semantic indices 0..num_layers")
        layer_features = self.layer_embedding(layer.long())[:, :, None, None]
        layer_features = layer_features.expand(-1, -1, depth, traces)
        if previous_probability is None:
            previous_probability = fine.new_zeros((batch, depth, traces))
        elif previous_probability.shape != (batch, depth, traces):
            raise ValueError("previous_probability must have shape [batch, depth, trace]")
        features = torch.cat(
            (decoded, conditioned, layer_features, previous_probability[:, None]), dim=1
        )
        logits = self.readout(features)[:, 0]
        if fine_valid is not None:
            output_valid = fine_valid[:, 0].to(device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(~output_valid, -1e4)
        return logits


def masked_depth_cross_entropy(
    logits: Tensor,
    target_depth: Tensor,
    valid: Tensor,
    *,
    support_mask: Tensor | None = None,
) -> Tensor:
    """Depth classification only at known, non-support target-interface traces.

    ``target_depth`` contains native output-grid sample coordinates ``[B,W]``;
    finite in-range values are rounded to the nearest sample. This is a loss
    discretization only: original fractional references must remain unchanged
    for native-time/signed-lobe evaluation. Unreviewed traces, supplied support
    traces, and targets outside this depth crop contribute exactly zero loss and
    gradient. No target width or visibility/absence label is manufactured.
    """
    if logits.ndim != 3 or target_depth.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("Expected logits [batch, depth, trace] and target_depth [batch, trace]")
    if valid.shape != target_depth.shape:
        raise ValueError("valid must have the same shape as target_depth")
    if support_mask is not None and support_mask.shape != target_depth.shape:
        raise ValueError("support_mask must have the same shape as target_depth")
    known = valid.bool() & torch.isfinite(target_depth)
    known = known & (target_depth >= 0) & (target_depth <= logits.shape[1] - 1)
    if support_mask is not None:
        known = known & ~support_mask.bool()
    safe_target = torch.where(known, target_depth, 0).round().long()
    safe_logits = torch.where(known[:, None, :], logits.float(), 0)
    point_loss = F.cross_entropy(safe_logits, safe_target, reduction="none")
    # Multiply before reduction so an all-unknown episode remains differentiable
    # with zero gradient, without a Python branch or a CUDA host synchronization.
    return (point_loss * known).sum() / known.sum().clamp_min(1)
