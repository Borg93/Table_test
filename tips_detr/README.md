# TipsDETR — TIPS v2 backbone + multi-scale deformable DETR

Our own detector for handwritten table structure (Exp B). **Inspired by RF-DETR,
not a fork** — every module here is original and Apache-clean; the only external
weights are TIPS v2 (CC-BY-4.0, attribute Google).

> **Why this shape?** We run **offline / batch**, never real-time. That removes
> the entire reason RF-DETR adds windowed attention, NMS-free speed tricks, and a
> light decoder. So we keep the **full** frozen ViT and spend the budget on a
> heavier 4-level **deformable** decoder — accuracy over latency.

## How TIPS v2 plugs into a DETR

A plain ViT is single-scale: every token is at stride /14. A DETR decoder wants a
feature *pyramid* and a way to attend sparsely over many tokens. Three pieces bridge that:

```
page image (square, e.g. 1568 = 14·112)
      │
      ▼
┌─────────────────────────┐   TipsBackbone (frozen)         models/tips_encoder.py
│  TIPS v2 ViT-L, patch-14│   get_intermediate_layers at taps (5,11,17,23)
└─────────────────────────┘   → 4 grids  [B, 1024, 112, 112]
      │
      ▼
┌─────────────────────────┐   MultiScaleProjector            projector.py
│  ViTDet "simple FPN"    │   tap i → level i, resample ×2 / ×1 / ÷2 / ÷4
└─────────────────────────┘   → P3..P6  [B, 256, {224,112,56,28}²]
      │
      ▼
┌─────────────────────────┐   DeformableTransformer          deformable_transformer.py
│  4× deformable encoder  │   multi-scale deformable self-attn (linear in #tokens)
│  6× deformable decoder  │   900 queries, iterative box refinement
└─────────────────────────┘   → hidden states per layer
      │
      ▼   class head (sigmoid-focal) + box head (cxcywh)   model.py
   {table row, table column, table spanning cell}
```

The frozen TIPS ViT contributes the pretrained visual representation; **only the
projector + transformer + heads train** (~40M params, printed at startup). Swapping
TIPS → DINOv3 is a one-line backbone change (both patch-14 register ViTs).

## From boxes to logical structure (TATR)

Train on `data/page_to_targets.py --coco-mode tatr` (3 classes). At inference:
**intersect** detected row-bands × column-bands → the logical cell grid with
(row, col) indices; **spanning-cell** boxes mark the merges. No custom head, works
even when gridlines are faint and cells are empty (the bands are geometric, not
text-driven). If row×col intersection can't resolve some spans, add a per-query
logical head (`--coco-mode cells` already emits `logic_axis`; head is a 2-line add
in `model.py`, mirroring TableCenterNet/LORE).

## Files

| File | What |
|------|------|
| `model.py` | `TipsDETR` + `TipsDETRConfig` + `post_process`; backbones (`TipsBackbone`, CPU `StubBackbone`) |
| `deformable_transformer.py` | deformable encoder + decoder, sine pos-embed, iterative box refine |
| `ms_deform_attn.py` | multi-scale deformable attention, **pure-PyTorch** (grid_sample, no CUDA ext) |
| `projector.py` | ViT tokens → P3..P6 pyramid (ViTDet simple FPN) |
| `matcher.py` / `criterion.py` | Hungarian matcher / sigmoid-focal + L1 + GIoU + aux losses |
| `dataset.py` | reads the converter's COCO; square-resize + TIPS normalize; torchvision-free |
| `train.py` | single-node + optional `torchrun` DDP; frozen backbone, AMP, cosine |
| `smoke.py` | CPU end-to-end check (stub backbone, random data) — **verified passing** |

## Run

```bash
python -m tips_detr.smoke                       # CPU wiring check, no weights/GPU

# data (Exp B target): row / column / spanning-cell boxes
python data/page_to_targets.py train/*.xml --coco dataset/train/_annotations.coco.json --coco-mode tatr
python data/page_to_targets.py val/*.xml   --coco dataset/valid/_annotations.coco.json --coco-mode tatr

# train (8×H100)
torchrun --nproc_per_node=8 -m tips_detr.train \
    --train-coco dataset/train/_annotations.coco.json --train-images dataset/train \
    --val-coco   dataset/valid/_annotations.coco.json --val-images   dataset/valid \
    --backbone tips --tips-variant L --image-size 1568 \
    --epochs 100 --batch-size 2 --lr 2e-4 --amp --out runs/tips_detr
```

## Resolution / tiling

Square resolution must be a multiple of the patch size (14). The frozen ViT's
self-attention is O(tokens²): at 1568 px that's 112² ≈ 12.5k tokens per page — fine
offline on an H100, but for ~4000 px scans **tile** to e.g. 1568-px crops (overlap a
row/col) and merge detections, or raise `--image-size` and eat the memory. The
*decoder* is deformable (linear in queries), so it is never the bottleneck.

## Status

- [x] Architecture + losses implemented; `smoke.py` passes (forward/loss/backward, post-process).
- [x] Data path verified: converter COCO → `CocoTableDataset` → train step (synthetic PAGE).
- [ ] Run on the H100 box with real TIPS v2 weights + real Transkribus PAGE export.
- [ ] Encoder linear-probe (TIPS vs DINOv3) before committing the backbone.
- [ ] Evaluate with `eval/eval_teds.py` (structure-only TEDS), not box IoU.
