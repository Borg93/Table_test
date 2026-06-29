#!/usr/bin/env python3
"""Multi-scale projector: plain-ViT tokens -> a P3..P6 feature pyramid.

A ViT is single-scale (everything at stride /14). DETR decoders want a pyramid.
This is RF-DETR's `MultiScaleProjector` / ViTDet "simple feature pyramid" idea,
written from scratch: take several tapped ViT layers (each [B, C, gh, gw] at /14)
and resample them to 4 levels with a learned conv block per level, all emitting
`hidden_dim` channels. Upper levels are upsampled (transpose-conv), lower levels
downsampled (stride conv), so the deformable decoder sees coarse-to-fine context.

We tap 4 layers and map tap i -> level i; with fewer taps the last tap is reused.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# scale relative to the /14 token grid: >1 upsamples, <1 downsamples.
DEFAULT_SCALES = (2.0, 1.0, 0.5, 0.25)


def _resampler(in_ch: int, out_ch: int, scale: float) -> nn.Module:
    """One pyramid level: resample by `scale`, then conv-norm-conv to out_ch."""
    layers: list[nn.Module] = []
    if scale == 2.0:
        layers += [nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2),
                   nn.GroupNorm(32, in_ch), nn.GELU()]
    elif scale == 1.0:
        pass
    elif scale == 0.5:
        layers += [nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1),
                   nn.GroupNorm(32, in_ch), nn.GELU()]
    elif scale == 0.25:
        layers += [nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1),
                   nn.GroupNorm(32, in_ch), nn.GELU(),
                   nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1),
                   nn.GroupNorm(32, in_ch), nn.GELU()]
    else:
        raise ValueError(f"unsupported scale {scale}")
    layers += [nn.Conv2d(in_ch, out_ch, 1), nn.GroupNorm(32, out_ch),
               nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(32, out_ch)]
    return nn.Sequential(*layers)


class MultiScaleProjector(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int = 256,
                 scales: tuple[float, ...] = DEFAULT_SCALES):
        super().__init__()
        self.scales = scales
        # one input feature map (tap) per output level; reuse last tap if too few.
        self.tap_for_level = [min(i, len(in_channels) - 1) for i in range(len(scales))]
        self.blocks = nn.ModuleList([
            _resampler(in_channels[self.tap_for_level[i]], out_channels, scales[i])
            for i in range(len(scales))
        ])
        self.num_channels = [out_channels] * len(scales)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """feats: list of [B, C, gh, gw] tapped ViT grids -> list of P3..P6."""
        return [self.blocks[i](feats[self.tap_for_level[i]]) for i in range(len(self.scales))]
