#!/usr/bin/env python3
"""TEDS evaluation for TipsDETR — the structure-quality number.

For each image: decode the predicted row/column/spanning-cell boxes into HTML, do the
same for the ground-truth boxes, and score them with structure-only TEDS
(eval/eval_teds.py). Mean TEDS over the set is the metric that says "is it good".

Used two ways:
  * inside training (per-epoch validation, see train.py)
  * standalone on a checkpoint:
      python -m tips_detr.evaluate --coco val.json --images val_dir \
          --ckpt runs/tips_detr/last.pth --backbone tips --npz-vision-ckpt tips_L_vision.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .box_ops import box_cxcywh_to_xyxy
from .dataset import CocoTableDataset, collate_fn
from .decode import detections_to_html, postprocess_to_html
from .model import TipsDETR, TipsDETRConfig, post_process

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from eval_teds import teds  # noqa: E402


def gt_to_html(target, class_names: list[str]) -> str:
    """Ground-truth boxes (normalized cxcywh) + labels -> structure HTML."""
    boxes = box_cxcywh_to_xyxy(target["boxes"]).tolist()
    labels = [class_names[i] for i in target["labels"].tolist()]
    return detections_to_html(labels, [tuple(b) for b in boxes])


@torch.no_grad()
def evaluate_teds(model, loader, class_names, device, score_thresh: float = 0.5):
    """Mean structure-only TEDS over the loader. Returns (mean_teds, n_images)."""
    was_training = model.training
    model.eval()
    scores = []
    for images, targets in loader:
        images = images.to(device)
        out = model(images)
        sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
        results = post_process(out, sizes)
        for res, tgt in zip(results, targets):
            pred_html = postprocess_to_html(res, class_names, score_thresh)
            gt_html = gt_to_html(tgt, class_names)
            scores.append(teds(pred_html, gt_html, structure_only=True))
    if was_training:
        model.train()
    return (sum(scores) / len(scores) if scores else 0.0), len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", help="checkpoint .pth (state_dict under 'model')")
    ap.add_argument("--backbone", default="stub", choices=["tips", "stub"])
    ap.add_argument("--npz-vision-ckpt", default=None)
    ap.add_argument("--tips-variant", default="L")
    ap.add_argument("--image-size", type=int, default=1568)
    ap.add_argument("--num-queries", type=int, default=900)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CocoTableDataset(args.coco, args.images, args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)

    cfg = TipsDETRConfig(
        backbone=args.backbone, num_classes=ds.num_classes, num_queries=args.num_queries,
        image_size=args.image_size, tips_variant=args.tips_variant,
        npz_vision_ckpt=args.npz_vision_ckpt)
    model = TipsDETR(cfg).to(device)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=device)
        sd = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded {args.ckpt}  (missing={len(missing)} unexpected={len(unexpected)})")

    mean_teds, n = evaluate_teds(model, dl, ds.class_names, device, args.score_thresh)
    print(f"TEDS (structure-only) = {mean_teds:.4f}  over {n} images")


if __name__ == "__main__":
    main()
