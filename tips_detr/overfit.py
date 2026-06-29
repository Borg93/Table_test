#!/usr/bin/env python3
"""Overfit-one-batch test — the canonical "can this model actually be trained?" check.

If the model can drive the loss toward ~0 on a single fixed batch, then the
architecture + loss + matcher + gradients + optimizer are correct and it *learns*
(not merely "runs"). Run it on the GPU box with the REAL TIPS backbone to verify the
full pipeline before committing to a long training run:

    python -m tips_detr.overfit --coco dataset/train/_annotations.coco.json \
        --images dataset/train --backbone tips --npz-vision-ckpt tips_L_vision.npz \
        --image-size 1568 --steps 200

With --backbone stub it runs on CPU and is the verification we use in CI.
"""
from __future__ import annotations

import argparse

import torch

from .criterion import SetCriterion
from .dataset import CocoTableDataset, collate_fn
from .matcher import HungarianMatcher
from .model import TipsDETR, TipsDETRConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--backbone", default="stub", choices=["tips", "stub"])
    ap.add_argument("--npz-vision-ckpt", default=None)
    ap.add_argument("--tips-variant", default="L")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-queries", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="final loss below this = PASS (it can learn)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CocoTableDataset(args.coco, args.images, args.image_size)
    batch = [ds[i] for i in range(min(args.batch, len(ds)))]
    images, targets = collate_fn(batch)
    images = images.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    cfg = TipsDETRConfig(
        backbone=args.backbone, num_classes=ds.num_classes, num_queries=args.num_queries,
        image_size=args.image_size, tips_variant=args.tips_variant,
        npz_vision_ckpt=args.npz_vision_ckpt, freeze_backbone=False)
    model = TipsDETR(cfg).to(device).train()
    crit = SetCriterion(ds.num_classes, HungarianMatcher(),
                        {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"backbone={args.backbone} classes={ds.num_classes} batch={len(batch)} device={device.type}")
    print("step   loss")
    loss = torch.tensor(float("inf"))
    for step in range(args.steps + 1):
        loss = crit.total(crit(model(images), targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, args.steps // 6) == 0:
            print(f"  {step:4d}  {loss.item():8.3f}")

    ok = loss.item() < args.threshold
    print(f"\nVERDICT: {'PASS — model learns (loss collapsed)' if ok else 'FAIL — loss did not collapse'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
