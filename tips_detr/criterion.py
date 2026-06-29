#!/usr/bin/env python3
"""Set criterion — sigmoid-focal classification + L1/GIoU box, with aux losses.

Same loss family as Deformable-DETR. `num_classes` is the count of real classes
(no background): with the TATR target that is 3 = {table row, table column, table
spanning cell}. Classification uses sigmoid focal loss over those classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(logits, targets, num_boxes, alpha=0.25, gamma=2.0):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.alpha, self.gamma = alpha, gamma

    def _idx(self, indices):
        batch = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src = torch.cat([src for (src, _) in indices])
        return batch, src

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]                      # (B, Q, C)
        idx = self._idx(indices)
        tgt_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target = torch.zeros_like(src_logits)
        if tgt_classes_o.numel():
            target[idx[0], idx[1], tgt_classes_o] = 1.0
        loss = sigmoid_focal_loss(src_logits, target, num_boxes, self.alpha, self.gamma) \
            * src_logits.shape[1]
        return {"loss_ce": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], 0)
        if tgt_boxes.numel() == 0:
            z = src_boxes.sum() * 0.0
            return {"loss_bbox": z, "loss_giou": z}
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
        giou = generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes))
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(self, outputs, targets):
        out_main = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(out_main, targets)
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=outputs["pred_logits"].device).item()

        losses = {}
        losses.update(self.loss_labels(out_main, targets, indices, num_boxes))
        losses.update(self.loss_boxes(out_main, targets, indices, num_boxes))

        # auxiliary losses on every intermediate decoder layer
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            ind = self.matcher(aux, targets)
            for lf in (self.loss_labels, self.loss_boxes):
                for k, v in lf(aux, targets, ind, num_boxes).items():
                    losses[f"{k}_{i}"] = v
        return losses

    def total(self, losses):
        return sum(losses[k] * self.weight_dict[k.rsplit("_", 1)[0] if k[-1].isdigit() else k]
                   for k in losses if (k.rsplit("_", 1)[0] if k[-1].isdigit() else k) in self.weight_dict)
