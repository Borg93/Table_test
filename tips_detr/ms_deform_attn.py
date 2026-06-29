#!/usr/bin/env python3
"""Multi-scale deformable attention — pure PyTorch (no custom CUDA kernel).

This is the attention used by Deformable-DETR / RF-DETR's LW-DETR core. We
deliberately use the `grid_sample` reference path instead of the fused CUDA
kernel: we run **offline / batch** (the user's constraint), so we trade ~throughput
for portability — it builds on any torch, needs no `MultiScaleDeformableAttention`
extension, and is numerically identical.

Each query predicts, per head / per feature level / per sampling point, a 2-D
offset from a reference point and a scalar attention weight; values are bilinearly
sampled at those locations and summed. Cost is linear in #queries (unlike dense
attention), which is what makes high-res, many-token document pages tractable.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """Reference deformable-attention sampling.

    value:              (N, S, M, D)        flattened multi-level values, M heads, D head-dim
    value_spatial_shapes: list[(H_l, W_l)]  per level; sum(H_l*W_l) == S
    sampling_locations: (N, Lq, M, L, P, 2) normalized [0,1] sample coords
    attention_weights:  (N, Lq, M, L, P)    softmaxed over (L*P)
    returns:            (N, Lq, M*D)
    """
    N, S, M, D = value.shape
    _, Lq, _, L, P, _ = sampling_locations.shape
    split = [H * W for (H, W) in value_spatial_shapes]
    value_list = value.split(split, dim=1)
    sampling_grids = 2 * sampling_locations - 1          # [0,1] -> [-1,1] for grid_sample
    sampled = []
    for lid, (H, W) in enumerate(value_spatial_shapes):
        # (N, H*W, M, D) -> (N*M, D, H, W)
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(N * M, D, H, W)
        # (N, Lq, M, P, 2) -> (N*M, Lq, P, 2)
        g = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    # stack levels: (N*M, D, Lq, L, P) -> (N*M, D, Lq, L*P)
    out = torch.stack(sampled, dim=-2).flatten(-2)
    attn = attention_weights.transpose(1, 2).reshape(N * M, 1, Lq, L * P)
    out = (out * attn).sum(-1).view(N, M * D, Lq)
    return out.transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_levels, self.n_heads, self.n_points = d_model, n_levels, n_heads, n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        # initialise sampling offsets to point outward in a ring per head (DefDETR init)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid = grid.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid.view(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        """query (N,Lq,C); reference_points (N,Lq,L,2) or (N,Lq,L,4); value (N,S,C)."""
        N, Lq, _ = query.shape
        S = value.shape[1]
        v = self.value_proj(value).view(N, S, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        weights = self.attention_weights(query).view(
            N, Lq, self.n_heads, self.n_levels * self.n_points)
        weights = weights.softmax(-1).view(
            N, Lq, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_norm = torch.as_tensor(
                [[w, h] for (h, w) in value_spatial_shapes],
                dtype=query.dtype, device=query.device)
            loc = reference_points[:, :, None, :, None, :] \
                + offsets / offset_norm[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            loc = reference_points[:, :, None, :, None, :2] \
                + offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        out = ms_deform_attn_core_pytorch(v, value_spatial_shapes, loc, weights)
        return self.output_proj(out)
