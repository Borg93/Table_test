#!/usr/bin/env python3
"""Train TipsDETR on the COCO json from data/page_to_targets.py (--coco-mode tatr).

Single-node; the frozen TIPS backbone means only the projector + transformer +
heads update. Offline/batch target, so AMP + large effective batch over the 8×H100
box (wrap with `torchrun --nproc_per_node=8` and the DDP block below). Run the
shape/loss smoke test first:  python -m tips_detr.smoke

Example:
    python -m tips_detr.train \
        --train-coco dataset/train/_annotations.coco.json --train-images dataset/train \
        --val-coco   dataset/valid/_annotations.coco.json --val-images   dataset/valid \
        --backbone tips --tips-variant L --image-size 1568 \
        --epochs 100 --batch-size 2 --lr 2e-4 --out runs/tips_detr
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .criterion import SetCriterion
from .dataset import CocoTableDataset, collate_fn
from .matcher import HungarianMatcher
from .model import TipsDETR, TipsDETRConfig


def move(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def build(args, num_classes):
    cfg = TipsDETRConfig(
        num_classes=num_classes, backbone=args.backbone, tips_variant=args.tips_variant,
        image_size=args.image_size, num_queries=args.num_queries)
    model = TipsDETR(cfg)
    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    weight_dict = {"loss_ce": 2.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = SetCriterion(num_classes, matcher, weight_dict)
    return model, criterion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-coco", required=True)
    ap.add_argument("--train-images", required=True)
    ap.add_argument("--val-coco")
    ap.add_argument("--val-images")
    ap.add_argument("--backbone", default="tips", choices=["tips", "stub"])
    ap.add_argument("--tips-variant", default="L")
    ap.add_argument("--image-size", type=int, default=1568)
    ap.add_argument("--num-queries", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/tips_detr")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    # ---- optional DDP (torchrun sets these) ----
    ddp = "RANK" in os.environ
    if ddp:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    train_ds = CocoTableDataset(args.train_coco, args.train_images, args.image_size)
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if ddp else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None,
                          sampler=sampler, num_workers=args.workers, collate_fn=collate_fn,
                          pin_memory=True, drop_last=True)

    model, criterion = build(args, train_ds.num_classes)
    model.to(device)
    criterion.to(device)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    if local_rank == 0:
        print(f"trainable params: {n_train/1e6:.1f}M  (backbone frozen)")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        for it, (images, targets) in enumerate(train_dl):
            images = images.to(device, non_blocking=True)
            targets = move(targets, device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss = criterion.total(losses)
            opt.zero_grad()
            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if local_rank == 0 and it % 20 == 0:
                terms = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items()
                                 if not k[-1].isdigit())
                print(f"ep{epoch} it{it}/{len(train_dl)} loss={loss.item():.3f} {terms}")
        if local_rank == 0:
            ckpt = {"model": (model.module if ddp else model).state_dict(),
                    "epoch": epoch, "cfg": vars(args)}
            torch.save(ckpt, out_dir / "last.pth")
            if (epoch + 1) % 10 == 0:
                torch.save(ckpt, out_dir / f"epoch{epoch+1}.pth")
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
