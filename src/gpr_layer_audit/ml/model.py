"""Compact conditional U-Net, inspired by IRHMapNet's radar-horizon architecture.

Independent PyTorch implementation; no TensorFlow or upstream training scripts.
Import only through optional ML commands/inference.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, incoming, outgoing):
        super().__init__(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(4, outgoing),
            nn.SiLU(),
        )


class SeedUNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                Block(6, width),
                Block(width, width * 2),
                Block(width * 2, width * 4),
                Block(width * 4, width * 8),
            ]
        )
        self.bridge = Block(width * 8, width * 16)
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose2d(width * 16, width * 8, 2, stride=2),
                nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2),
                nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2),
                nn.ConvTranspose2d(width * 2, width, 2, stride=2),
            ]
        )
        self.decoder = nn.ModuleList(
            [
                Block(width * 16, width * 8),
                Block(width * 8, width * 4),
                Block(width * 4, width * 2),
                Block(width * 2, width),
            ]
        )
        self.boundary = nn.Conv2d(width, 1, 1)
        self.visibility = nn.Conv2d(width, 1, 1)

    def forward(self, values):
        original = values.shape[-2:]
        values = F.pad(values, (0, (-original[1]) % 16, 0, (-original[0]) % 16))
        skips = []
        for block in self.encoder:
            values = block(values)
            skips.append(values)
            values = F.max_pool2d(values, 2)
        values = self.bridge(values)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips), strict=True):
            values = block(torch.cat([up(values), skip], dim=1))
        values = values[..., : original[0], : original[1]]
        return self.boundary(values)[:, 0], self.visibility(values)[:, 0].mean(dim=-1)


def masked_loss(logits, visibility_logits, target, valid, visible, observed):
    valid, observed = valid.float(), observed.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum()
    bce = bce / valid.sum().clamp_min(1)
    probability = logits.sigmoid() * valid
    truth = target * valid
    dice = 1 - (2 * (probability * truth).sum() + 1) / (probability.sum() + truth.sum() + 1)
    visibility = (
        F.binary_cross_entropy_with_logits(visibility_logits, visible, reduction="none") * observed
    ).sum()
    return bce + dice + 0.25 * visibility / observed.sum().clamp_min(1)
