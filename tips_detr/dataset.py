#!/usr/bin/env python3
"""COCO dataset for TipsDETR — reads the json from data/page_to_targets.py.

Use `--coco-mode tatr` output (categories: table row / column / spanning-cell).
Pages are stretched to a square `image_size` (offline batch: every page the same
size, so no padding mask is needed downstream) and normalized with TIPS stats
(mean 0, std 1). Boxes become normalized (cx, cy, w, h) in [0,1]; labels are
0-indexed (category_id - 1) for the sigmoid-focal head.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PATCH_SIZE = 14
IMAGE_MEAN = (0.0, 0.0, 0.0)   # TIPS preprocessing
IMAGE_STD = (1.0, 1.0, 1.0)


def _to_normalized_chw(img: Image.Image, size: int) -> torch.Tensor:
    """PIL RGB -> normalized CHW float tensor, square-resized (torchvision-free)."""
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (arr - mean) / std


class CocoTableDataset(Dataset):
    def __init__(self, ann_json: str, images_root: str, image_size: int = 1568):
        self.images_root = Path(images_root)
        self.image_size = (image_size // PATCH_SIZE) * PATCH_SIZE
        coco = json.loads(Path(ann_json).read_text())
        self.images = coco["images"]
        self.cats = sorted(c["id"] for c in coco["categories"])
        self.cat2label = {cid: i for i, cid in enumerate(self.cats)}
        self.num_classes = len(self.cats)
        id2name = {c["id"]: c["name"] for c in coco["categories"]}
        # class name per 0-indexed label (the order the model's logits use)
        self.class_names = [id2name[cid] for cid in self.cats]
        by_img = defaultdict(list)
        for a in coco["annotations"]:
            by_img[a["image_id"]].append(a)
        self.anns_by_img = by_img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        img = Image.open(self.images_root / info["file_name"]).convert("RGB")
        w0, h0 = img.width, img.height
        x = _to_normalized_chw(img, self.image_size)

        boxes, labels = [], []
        for a in self.anns_by_img.get(info["id"], []):
            bx, by, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx, cy = (bx + bw / 2) / w0, (by + bh / 2) / h0
            boxes.append([cx, cy, bw / w0, bh / h0])
            labels.append(self.cat2label[a["category_id"]])
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "orig_size": torch.as_tensor([h0, w0]),
            "image_id": torch.as_tensor(info["id"]),
        }
        return x, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], 0)   # same square size -> stackable
    targets = [b[1] for b in batch]
    return images, targets
