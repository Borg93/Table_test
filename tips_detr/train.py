#!/usr/bin/env python3
"""Train TipsDETR on the COCO json from data/page_to_targets.py (--coco-mode tatr).

Single-node. By default the TIPS backbone is frozen (only projector + transformer +
heads update); pass --no-freeze-backbone to fine-tune the ViT too, with a separate
--lr-encoder and ViT layer-wise LR decay (RF-DETR's recipe) — recommended for the
large PubTables pretrain, keep frozen for tiny fine-tunes. Offline/batch target, so
AMP + large effective batch over the 8×H100 box (wrap with `torchrun
--nproc_per_node=8`). Run the shape/loss smoke test first:  python -m tips_detr.smoke

Example:
    python -m tips_detr.train \
        --train-coco dataset/train/_annotations.coco.json --train-images dataset/train \
        --val-coco   dataset/valid/_annotations.coco.json --val-images   dataset/valid \
        --backbone tips --tips-variant L --image-size 1568 \
        --epochs 100 --batch-size 2 --lr 2e-4 --out runs/tips_detr
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .criterion import SetCriterion
from .dataset import CocoTableDataset, collate_fn
from .evaluate import evaluate_teds
from .matcher import HungarianMatcher
from .model import TipsDETR, TipsDETRConfig


def load_init_weights(model, path):
    """Load a pretrained checkpoint for fine-tuning; skip shape-mismatched keys
    (e.g. the class head when the class count changes). Returns (kept, skipped)."""
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model", sd)
    msd = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    skipped = [k for k in sd if k not in keep]
    model.load_state_dict(keep, strict=False)
    return len(keep), skipped


def move(targets, device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def _vit_layer_lr(name, base_lr, decay, num_layers):
    """RF-DETR-style ViT layer-wise LR decay: earlier blocks move less."""
    m = re.search(r"blocks\.(\d+)\.", name)
    if m is None:
        return base_lr                      # patch-embed / norm: full encoder lr
    return base_lr * (decay ** (num_layers - 1 - int(m.group(1))))


def build_param_groups(model, args, num_bb_layers):
    """Separate LR for the (optionally unfrozen) backbone, with layer-wise decay."""
    groups, others = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            groups.append({"params": [p],
                           "lr": _vit_layer_lr(n, args.lr_encoder, args.lr_vit_layer_decay, num_bb_layers)})
        else:
            others.append(p)
    groups.append({"params": others, "lr": args.lr})
    return groups


def build(args, num_classes):
    cfg = TipsDETRConfig(
        num_classes=num_classes, backbone=args.backbone, tips_variant=args.tips_variant,
        image_size=args.image_size, num_queries=args.num_queries,
        npz_vision_ckpt=args.npz_vision_ckpt, freeze_backbone=args.freeze_backbone)
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
    ap.add_argument("--npz-vision-ckpt", default=None,
                    help="Path to TIPS v2 *_vision.npz (from download_checkpoints.sh) "
                         "for --backbone tips")
    ap.add_argument("--image-size", type=int, default=1568)
    ap.add_argument("--num-queries", type=int, default=900)
    ap.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-freeze-backbone to fine-tune the ViT (recommended for "
                         "the large PubTables pretrain; keep frozen for tiny fine-tunes)")
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4,
                    help="LR for the backbone when unfrozen (RF-DETR uses 1.5e-4)")
    ap.add_argument("--lr-vit-layer-decay", type=float, default=0.8,
                    help="Layer-wise LR decay for the ViT blocks (RF-DETR uses 0.8)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/tips_detr")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--init-weights", default=None,
                    help="Pretrained checkpoint to fine-tune from (shape-mismatched "
                         "heads, e.g. a different class count, are reset)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="Run validation TEDS every N epochs (needs --val-coco + apted)")
    ap.add_argument("--score-thresh", type=float, default=0.5)
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

    val_dl = None
    val_class_names = None
    if args.val_coco and args.val_images:
        val_ds = CocoTableDataset(args.val_coco, args.val_images, args.image_size)
        val_class_names = val_ds.class_names
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers,
                            collate_fn=collate_fn)

    model, criterion = build(args, train_ds.num_classes)
    model.to(device)
    criterion.to(device)
    if args.init_weights:
        kept, skipped = load_init_weights(model, args.init_weights)
        if local_rank == 0:
            print(f"init from {args.init_weights}: loaded {kept} tensors, "
                  f"reset {len(skipped)} (e.g. class head on class-count change)")
    # `model` always refers to the unwrapped TipsDETR (for .state_dict());
    # `train_model` is what we call forward on (DDP-wrapped under torchrun).
    train_model = (torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
                   if ddp else model)

    num_bb_layers = getattr(getattr(model.backbone, "encoder", None), "n_layers", 12)
    param_groups = build_param_groups(model, args, num_bb_layers)
    trainable = [p for g in param_groups for p in g["params"]]
    n_train = sum(p.numel() for p in trainable)
    n_bb = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and n.startswith("backbone."))
    if local_rank == 0:
        state = "frozen" if args.freeze_backbone else f"fine-tuned, lr_encoder={args.lr_encoder}"
        print(f"trainable params: {n_train/1e6:.1f}M  (backbone {n_bb/1e6:.1f}M {state})")
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_teds = -1.0

    for epoch in range(args.epochs):
        train_model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        for it, (images, targets) in enumerate(train_dl):
            images = images.to(device, non_blocking=True)
            targets = move(targets, device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = train_model(images)
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
            ckpt = {"model": model.state_dict(),
                    "epoch": epoch, "cfg": vars(args)}
            torch.save(ckpt, out_dir / "last.pth")
            if (epoch + 1) % 10 == 0:
                torch.save(ckpt, out_dir / f"epoch{epoch+1}.pth")
            if val_dl is not None and (epoch + 1) % args.eval_every == 0:
                try:
                    teds, n = evaluate_teds(model, val_dl, val_class_names,
                                        device, args.score_thresh)
                    print(f"ep{epoch} val TEDS={teds:.4f} (structure-only, {n} imgs)")
                    if teds > best_teds:
                        best_teds = teds
                        torch.save(ckpt, out_dir / "best.pth")
                        print(f"ep{epoch} new best TEDS={teds:.4f} -> best.pth")
                except SystemExit as e:        # apted missing -> eval_teds exits
                    print(f"ep{epoch} TEDS skipped: {e}")
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
