#!/usr/bin/env python3
"""TipsV2Backbone — drop-in replacement for RF-DETR's DinoV2 backbone (Exp B / B1).

Mirrors the contract RF-DETR's `MultiScaleProjector` expects from the encoder:
  - forward(pixel_values) -> list[Tensor] of [B, C, H/14, W/14], one per tap layer
  - .num_channels -> list[int], the channel dim of each returned feature map

Wraps the verified TIPS loader in ../models/tips_encoder.py. Integrate per
INTEGRATION.md (branch in models/backbone/backbone.py on name 'tipsv2*').

Runs on the H100 box with torch + transformers + the TIPS weights. Not runnable
in CI (no GPU). Windowed attention for high-res is TODO — run tiled first.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from tips_encoder import TipsEncoder  # noqa: E402

# ViT depth per TIPS variant (for resolving negative tap indexes to absolute).
_DEPTH = {"B": 12, "L": 24, "So": 27, "g": 40}


class TipsV2Backbone(nn.Module):
    def __init__(self, size: str = "L", image_size: int = 1568,
                 out_feature_indexes: list[int] | None = None, freeze: bool = True,
                 npz_vision_ckpt: str | None = None):
        super().__init__()
        taps = tuple(out_feature_indexes or [5, 11, 17, 23])
        self.encoder = TipsEncoder(variant=size, image_size=image_size, freeze=freeze,
                                   npz_vision_ckpt=npz_vision_ckpt, taps=taps)
        dim = self.encoder.embed_dim
        # MultiScaleProjector wants one in-channel value per returned feature map.
        self.num_channels = [dim] * len(taps)
        self._n_taps = len(taps)

    def forward(self, pixel_values):
        feats = self.encoder(pixel_values)        # list[[B, C, gh, gw]]
        assert len(feats) == self._n_taps, (len(feats), self._n_taps)
        return feats


if __name__ == "__main__":
    print("TipsV2Backbone: integrate into rfdetr per exp_b_rfdetr/INTEGRATION.md (B1).")
    print("Contract: forward -> list of [B,C,H/14,W/14]; .num_channels -> list[int].")
