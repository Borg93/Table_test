#!/usr/bin/env python3
"""TIPS v2 encoder wrapper for the "build our own" track (Exp B).

Grounded in the two TIPS code paths you provided:
  - HF DPT path   (TIPS Feature Explorer Space): AutoModel -> backbone.vision_encoder
  - npz ckpt path (TIPS foreground-seg notebook): image_encoder.vit_* + get_intermediate_layers

It exposes frozen multi-layer patch features (B, C, H/14, W/14) ready to feed a
detection head (DETR) or a dense head. A plain ViT is single-scale; build a
multi-scale pyramid from several layers with a ViT-Adapter / ViTDet simple FPN
before the detection decoder (see build_simple_pyramid() note).

Runs on a GPU box with `torch`, `transformers`, and the `tips` repo / weights
installed (see requirements.txt). Not executable in CI (no GPU here).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF

PATCH_SIZE = 14
# TIPS preprocessing from the Feature Explorer: mean 0, std 1, square resize.
IMAGE_MEAN = (0.0, 0.0, 0.0)
IMAGE_STD = (1.0, 1.0, 1.0)

HF_VARIANTS = {
    "B": "google/tipsv2-b14-dpt",
    "L": "google/tipsv2-l14-dpt",
    "So400m": "google/tipsv2-so400m14-dpt",
    "g": "google/tipsv2-g14-dpt",
}
NPZ_CONSTRUCTOR = {"B": "vit_base", "L": "vit_large", "So": "vit_so", "g": "vit_giant2"}
NPZ_NUM_LAYERS = {"B": 12, "L": 24, "So": 27, "g": 40}


def preprocess(img, size: int) -> torch.Tensor:
    """PIL -> normalized CHW tensor, square-resized to a multiple of PATCH_SIZE."""
    size = (size // PATCH_SIZE) * PATCH_SIZE
    t = TF.to_tensor(TF.resize(img.convert("RGB"), (size, size)))
    return TF.normalize(t, mean=IMAGE_MEAN, std=IMAGE_STD)


class TipsEncoder(nn.Module):
    """Frozen TIPS v2 vision encoder returning multi-layer patch-token grids."""

    def __init__(self, variant: str = "L", image_size: int = 1372, freeze: bool = True,
                 npz_vision_ckpt: str | None = None, taps: tuple[int, ...] = (-1,)):
        super().__init__()
        self.image_size = (image_size // PATCH_SIZE) * PATCH_SIZE
        self.taps = taps

        if npz_vision_ckpt:
            # ---- npz path (foreground-seg notebook) ----
            import numpy as np
            from tips.pytorch import image_encoder
            weights = {k: torch.tensor(v) for k, v in dict(np.load(npz_vision_ckpt)).items()}
            ffn = "swiglu" if variant == "g" else "mlp"
            ctor = getattr(image_encoder, NPZ_CONSTRUCTOR[variant])
            self.vision = ctor(img_size=self.image_size, patch_size=PATCH_SIZE, ffn_layer=ffn,
                               block_chunks=0, init_values=1.0, interpolate_antialias=True,
                               interpolate_offset=0.0)
            self.vision.load_state_dict(weights)
            self.n_layers = NPZ_NUM_LAYERS[variant]
            self._mode = "npz"
        else:
            # ---- HF DPT path (Feature Explorer) ----
            from transformers import AutoModel
            dpt = AutoModel.from_pretrained(HF_VARIANTS[variant], trust_remote_code=True)
            dpt._get_backbone()
            self.vision = dpt._backbone.vision_encoder
            self.n_layers = len(self.vision.blocks)
            self._mode = "hf"

        if freeze:
            self.vision.eval()
            for p in self.vision.parameters():
                p.requires_grad_(False)
        self.embed_dim = getattr(self.vision, "embed_dim", None)

    @property
    def grid(self) -> int:
        return self.image_size // PATCH_SIZE

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B,3,H,W) -> list of (B, C, gh, gw) token grids, one per tap layer."""
        layers = [l if l >= 0 else self.n_layers + l for l in self.taps]
        feats = self.vision.get_intermediate_layers(
            x, n=sorted(set(layers)), reshape=True, norm=True
        )
        return list(feats)


def build_simple_pyramid(grid_feats: torch.Tensor) -> dict[str, torch.Tensor]:
    """ViTDet-style simple feature pyramid from a single /14 token grid.

    Strides ~ {4,8,16,32} via transpose-conv / pooling. Feed these to a
    Deformable-DETR / Mask2Former decoder. (Stub: swap in a learned ViT-Adapter
    for best results; this is the minimal interpolation-based version.)
    """
    b, c, h, w = grid_feats.shape
    return {
        "p2": F.interpolate(grid_feats, scale_factor=4.0, mode="bilinear", align_corners=False),
        "p3": F.interpolate(grid_feats, scale_factor=2.0, mode="bilinear", align_corners=False),
        "p4": grid_feats,
        "p5": F.max_pool2d(grid_feats, kernel_size=2),
    }


if __name__ == "__main__":
    print("TipsEncoder module — load on a GPU box, e.g.:")
    print("  enc = TipsEncoder(variant='L', image_size=1372, taps=(-1,-7,-13))")
    print("  feats = enc(preprocess(pil_img, 1372).unsqueeze(0).cuda())")
    print("Then: pyr = build_simple_pyramid(feats[-1]) -> Deformable-DETR / Mask2Former head.")
