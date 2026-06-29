#!/usr/bin/env python3
"""CPU smoke test — verify the model/criterion wiring end-to-end, no GPU/weights.

Builds TipsDETR with the stub backbone, runs forward on random images + random
targets, computes the loss, backprops, and checks shapes. Run on any box with
torch + scipy:  python -m tips_detr.smoke
"""
from __future__ import annotations

import torch

from .criterion import SetCriterion
from .matcher import HungarianMatcher
from .model import TipsDETR, TipsDETRConfig, post_process


def main():
    torch.manual_seed(0)
    cfg = TipsDETRConfig(backbone="stub", num_classes=3, num_queries=100,
                         enc_layers=2, dec_layers=3, hidden_dim=128, image_size=14 * 16)
    model = TipsDETR(cfg)
    matcher = HungarianMatcher()
    criterion = SetCriterion(cfg.num_classes, matcher,
                             {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0})

    b, hw = 2, cfg.image_size
    images = torch.randn(b, 3, hw, hw)
    targets = []
    for _ in range(b):
        n = torch.randint(3, 8, (1,)).item()
        cxcy = torch.rand(n, 2) * 0.6 + 0.2
        wh = torch.rand(n, 2) * 0.2 + 0.05
        targets.append({"boxes": torch.cat([cxcy, wh], 1),
                        "labels": torch.randint(0, cfg.num_classes, (n,))})

    out = model(images)
    assert out["pred_logits"].shape == (b, cfg.num_queries, cfg.num_classes), out["pred_logits"].shape
    assert out["pred_boxes"].shape == (b, cfg.num_queries, 4), out["pred_boxes"].shape
    assert len(out["aux_outputs"]) == cfg.dec_layers - 1

    losses = criterion(out, targets)
    total = criterion.total(losses)
    total.backward()

    grad = sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None)
    assert torch.isfinite(total) and grad > 0, (total, grad)

    sizes = torch.as_tensor([[1000, 800], [1200, 900]])
    res = post_process(out, sizes, topk=20)
    assert res[0]["boxes"].shape == (20, 4)

    print(f"OK  forward+loss+backward  loss={total.item():.3f}  "
          f"terms={ {k: round(v.item(), 3) for k, v in losses.items() if not k[-1].isdigit()} }")
    print(f"    pred_logits={tuple(out['pred_logits'].shape)}  "
          f"pred_boxes={tuple(out['pred_boxes'].shape)}  postproc boxes={tuple(res[0]['boxes'].shape)}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
