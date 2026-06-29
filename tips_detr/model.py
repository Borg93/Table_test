#!/usr/bin/env python3
"""TipsDETR — frozen TIPS v2 ViT backbone + multi-scale deformable DETR head.

The "build our own" detector (Exp B), inspired by RF-DETR but **not** a fork and
fully Apache-clean: every module here is ours (`box_ops`, `ms_deform_attn`,
`projector`, `deformable_transformer`). The backbone is the frozen TIPS v2 encoder
(`models/tips_encoder.py`); only the projector + transformer + heads train.

Because inference is offline/batch, we keep the full ViT (no windowed attention)
and a heavier 4-level deformable decoder — accuracy over latency.

Output classes (TATR target from data/page_to_targets.py --coco-mode tatr):
    0 = table row, 1 = table column, 2 = table spanning cell
Intersect detected row-bands x column-bands to recover the logical cell grid;
spanning-cell boxes fix merges. (Switch to per-cell with --coco-mode cells.)
"""
from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from .deformable_transformer import DeformableTransformer, inverse_sigmoid
from .projector import MultiScaleProjector


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers):
        super().__init__()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x).relu() if i < len(self.layers) - 1 else layer(x)
        return x


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #
class StubBackbone(nn.Module):
    """CPU-runnable stand-in for TIPS: a patch-14 conv stem, repeated as taps.

    Lets the transformer/heads/criterion be smoke-tested with no GPU or weights.
    """

    def __init__(self, embed_dim=256, n_taps=4, patch=14, freeze=False):
        super().__init__()
        self.stem = nn.Conv2d(3, embed_dim, patch, stride=patch)
        self.embed_dim, self.n_taps = embed_dim, n_taps
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        f = self.stem(x)
        return [f for _ in range(self.n_taps)]


class TipsBackbone(nn.Module):
    """TIPS v2 encoder, exposing `n_taps` tapped /14 grids. Frozen or fine-tunable."""

    def __init__(self, variant="L", image_size=1568, taps=(5, 11, 17, 23),
                 npz_vision_ckpt=None, freeze=True):
        super().__init__()
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
        from tips_encoder import TipsEncoder  # noqa: E402
        self.encoder = TipsEncoder(variant=variant, image_size=image_size, freeze=freeze,
                                   npz_vision_ckpt=npz_vision_ckpt, taps=tuple(taps))
        embed_dim = self.encoder.embed_dim
        assert embed_dim is not None, "TIPS encoder exposed no embed_dim"
        self.embed_dim: int = embed_dim
        self.n_taps = len(taps)

    def forward(self, x):
        return self.encoder(x)


# --------------------------------------------------------------------------- #
# Config + model
# --------------------------------------------------------------------------- #
@dataclass
class TipsDETRConfig:
    num_classes: int = 3                 # tatr: row / column / spanning-cell
    hidden_dim: int = 256
    num_queries: int = 900               # dense pages: 100s of rows+cols+spans
    enc_layers: int = 4
    dec_layers: int = 6
    n_heads: int = 8
    n_points: int = 4
    d_ffn: int = 1024
    dropout: float = 0.1
    scales: tuple = (2.0, 1.0, 0.5, 0.25)
    # backbone
    backbone: str = "tips"               # "tips" | "stub"
    tips_variant: str = "L"
    image_size: int = 1568
    tips_taps: tuple = (5, 11, 17, 23)
    npz_vision_ckpt: str | None = None
    freeze_backbone: bool = True         # set False to fine-tune the ViT (RF-DETR-style)
    aux_loss: bool = True


class TipsDETR(nn.Module):
    def __init__(self, cfg: TipsDETRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone: StubBackbone | TipsBackbone
        if cfg.backbone == "stub":
            self.backbone = StubBackbone(cfg.hidden_dim, len(cfg.tips_taps),
                                         freeze=cfg.freeze_backbone)
        else:
            self.backbone = TipsBackbone(cfg.tips_variant, cfg.image_size,
                                         cfg.tips_taps, cfg.npz_vision_ckpt,
                                         freeze=cfg.freeze_backbone)
        in_ch = [self.backbone.embed_dim] * self.backbone.n_taps
        self.projector = MultiScaleProjector(in_ch, cfg.hidden_dim, cfg.scales)
        self.transformer = DeformableTransformer(
            d_model=cfg.hidden_dim, n_heads=cfg.n_heads, n_levels=len(cfg.scales),
            n_points=cfg.n_points, enc_layers=cfg.enc_layers, dec_layers=cfg.dec_layers,
            d_ffn=cfg.d_ffn, dropout=cfg.dropout, num_queries=cfg.num_queries)

        # one classification + box head per decoder layer (box-refine: not shared)
        n_pred = cfg.dec_layers
        class_embed = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        bbox_embed = MLP(cfg.hidden_dim, cfg.hidden_dim, 4, 3)
        prior = math.log((1 - 0.01) / 0.01)                     # focal bias prior (float)
        nn.init.constant_(class_embed.bias, -prior)
        bbox_last = cast(nn.Linear, bbox_embed.layers[-1])
        nn.init.constant_(bbox_last.weight, 0.0)
        nn.init.constant_(bbox_last.bias, 0.0)
        self.class_embed = nn.ModuleList([copy.deepcopy(class_embed) for _ in range(n_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(bbox_embed) for _ in range(n_pred)])
        self.transformer.bbox_embed = self.bbox_embed           # enables iterative refine

    def forward(self, images):
        feats = self.backbone(images)                # list[[B,C,gh,gw]]
        srcs = self.projector(feats)                 # list of P3..P6
        hs, init_ref, inter_refs = self.transformer(srcs)

        classes, coords = [], []
        for lvl in range(hs.shape[0]):
            reference = init_ref if lvl == 0 else inter_refs[lvl - 1]
            reference = inverse_sigmoid(reference)
            classes.append(self.class_embed[lvl](hs[lvl]))
            coords.append((self.bbox_embed[lvl](hs[lvl]) + reference).sigmoid())

        out = {"pred_logits": classes[-1], "pred_boxes": coords[-1]}
        if self.cfg.aux_loss:
            out["aux_outputs"] = [{"pred_logits": c, "pred_boxes": b}
                                  for c, b in zip(classes[:-1], coords[:-1])]
        return out


@torch.no_grad()
def post_process(outputs, target_sizes, topk=None):
    """outputs -> per-image dict(scores, labels, boxes[xyxy abs]). target_sizes (B,2)=(h,w).

    One label per query (argmax over classes), so a single box is never emitted under
    two labels (which would double-count a band as both a row and a column). topk
    defaults to num_queries (no hard cap) so dense pages aren't truncated.
    """
    from .box_ops import box_cxcywh_to_xyxy
    logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
    prob = logits.sigmoid()
    bs, nq, nc = prob.shape
    scores_q, labels_q = prob.max(-1)                      # (bs, nq) best class per query
    k = min(topk or nq, nq)
    scores, qidx = scores_q.topk(k, dim=1)
    labels = labels_q.gather(1, qidx)
    boxes = box_cxcywh_to_xyxy(boxes)
    boxes = torch.gather(boxes, 1, qidx.unsqueeze(-1).expand(-1, -1, 4))
    h, w = target_sizes.unbind(1)
    scale = torch.stack([w, h, w, h], 1).unsqueeze(1)
    boxes = boxes * scale
    return [{"scores": scores[i], "labels": labels[i], "boxes": boxes[i]} for i in range(bs)]
