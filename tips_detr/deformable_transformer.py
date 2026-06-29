#!/usr/bin/env python3
"""Deformable transformer (encoder + decoder) — the DETR core, our own build.

Architecture follows Deformable-DETR / RF-DETR's LW-DETR core: a deformable
encoder refines the multi-scale memory, a decoder attends a fixed set of object
queries into that memory with **iterative box refinement** (each layer predicts a
delta to its reference box). Single-stage (learned queries), which is plenty for
the bounded object count of a table page (rows/cols/cells), and simpler than the
two-stage proposal machinery.

We assume fixed-size square inputs (offline batch resizes every page the same),
so there is no padding mask and `valid_ratios == 1` everywhere — the plumbing is
kept general but trivially satisfied.
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from .ms_deform_attn import MSDeformAttn


def _clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class PositionEmbeddingSine(nn.Module):
    """2-D sine positional embedding (DETR), no learned params."""

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    def forward(self, b, h, w, device, dtype):
        y = torch.arange(1, h + 1, device=device, dtype=dtype).view(1, h, 1).expand(b, h, w)
        x = torch.arange(1, w + 1, device=device, dtype=dtype).view(1, 1, w).expand(b, h, w)
        y = y / (h + 1e-6) * 2 * math.pi
        x = x / (w + 1e-6) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x[..., None] / dim_t
        pos_y = y[..., None] / dim_t
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=4).flatten(3)
        return torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (B, 2*npf, H, W)


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_levels, n_heads, n_points, dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, src, pos, reference_points, spatial_shapes):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes)
        src = self.norm1(src + self.dropout(src2))
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = self.norm2(src + self.dropout(src2))
        return src


class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, n_levels, n_heads, n_points, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, tgt, query_pos, reference_points, src, spatial_shapes):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = self.norm1(tgt + self.dropout(tgt2))
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, src, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        tgt2 = self.linear2(self.dropout(self.act(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4,
                 enc_layers=4, dec_layers=6, d_ffn=1024, dropout=0.1, num_queries=900):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.encoder = _clones(
            DeformableEncoderLayer(d_model, d_ffn, n_levels, n_heads, n_points, dropout), enc_layers)
        self.decoder = _clones(
            DeformableDecoderLayer(d_model, d_ffn, n_levels, n_heads, n_points, dropout), dec_layers)
        self.pos_embed = PositionEmbeddingSine(d_model // 2)
        self.level_embed = nn.Parameter(torch.empty(n_levels, d_model))
        self.query_embed = nn.Embedding(num_queries, 2 * d_model)   # -> (query_pos, tgt)
        self.reference_points = nn.Linear(d_model, 4)               # initial reference boxes
        self.bbox_embed = None                                      # set by the model (box refine)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.level_embed)
        nn.init.xavier_uniform_(self.reference_points.weight, gain=1.0)
        nn.init.constant_(self.reference_points.bias, 0.0)

    @staticmethod
    def _enc_reference_points(spatial_shapes, device, dtype):
        refs = []
        for (h, w) in spatial_shapes:
            ry = torch.linspace(0.5, h - 0.5, h, device=device, dtype=dtype) / h
            rx = torch.linspace(0.5, w - 0.5, w, device=device, dtype=dtype) / w
            ry, rx = torch.meshgrid(ry, rx, indexing="ij")
            refs.append(torch.stack([rx.reshape(-1), ry.reshape(-1)], -1))
        ref = torch.cat(refs, 0)                          # (S, 2)
        return ref[None, :, None].repeat(1, 1, len(spatial_shapes), 1)  # (1, S, L, 2)

    def forward(self, srcs: list[torch.Tensor]):
        """srcs: list of [B, C, H, W] pyramid maps -> (hs, init_reference, inter_references)."""
        b = srcs[0].shape[0]
        src_flat, pos_flat, spatial_shapes = [], [], []
        for lvl, src in enumerate(srcs):
            _, c, h, w = src.shape
            spatial_shapes.append((h, w))
            pos = self.pos_embed(b, h, w, src.device, src.dtype).flatten(2).transpose(1, 2)
            src_flat.append(src.flatten(2).transpose(1, 2))
            pos_flat.append(pos + self.level_embed[lvl].view(1, 1, -1))
        src_flat = torch.cat(src_flat, 1)                 # (B, S, C)
        pos_flat = torch.cat(pos_flat, 1)

        # --- encoder ---
        enc_ref = self._enc_reference_points(spatial_shapes, src_flat.device, src_flat.dtype).repeat(b, 1, 1, 1)
        memory = src_flat
        for layer in self.encoder:
            memory = layer(memory, pos_flat, enc_ref, spatial_shapes)

        # --- decoder setup (single stage: learned queries) ---
        query_pos, tgt = torch.split(self.query_embed.weight, self.d_model, dim=1)
        query_pos = query_pos[None].expand(b, -1, -1)
        tgt = tgt[None].expand(b, -1, -1)
        reference = self.reference_points(query_pos).sigmoid()    # (B, Q, 4) boxes in [0,1]
        init_reference = reference

        # --- decoder with iterative box refinement ---
        hs, inter_refs = [], []
        for lid, layer in enumerate(self.decoder):
            ref_input = reference[:, :, None].expand(-1, -1, self.n_levels, -1)   # (B,Q,L,4)
            tgt = layer(tgt, query_pos, ref_input, memory, spatial_shapes)
            if self.bbox_embed is not None:
                delta = self.bbox_embed[lid](tgt)
                reference = (delta + inverse_sigmoid(reference)).sigmoid().detach()
            hs.append(tgt)
            inter_refs.append(reference)
        return torch.stack(hs), init_reference, torch.stack(inter_refs)
