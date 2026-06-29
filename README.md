# Handwritten table-structure recognition

Logical table recognition for **handwritten historical tables** (e.g. 1820s Swedish
registers): identify **coherent key–value structure over cells / rows / columns**,
robust to **faint or missing gridlines** and **cells that span multiple rows/cols**.

Structure-first: cells have geometry and logical row/col **even when no text is
written**. Text transcription is a later, optional stage.

- **Ground truth:** Transkribus **PAGE XML** (use PAGE, not ALTO/ABBYY — only PAGE
  encodes cell `row/col/rowSpan/colSpan` + polygons for empty cells).
- **Compute:** 8× H100 / Blackwell available.

## Two experiments (we run both)

We are **not** using Surya. The two tracks are:

### Exp A — fine-tune LocateAnything (grounded VLM)
NVlabs/Eagle `Embodied/` = **LocateAnything**: MoonViT encoder + Qwen2.5/Qwen3 LLM
+ MLP projector, with **Parallel Box Decoding** (atomic per-box decoding — key for
dense pages with 100s of cells). Already does dense detection + document layout/OCR.
- **Why:** inherits vision-language alignment + grounding pretraining; promptable;
  extends to reading the text later with no rearchitecture.
- **Data:** `data/page_to_targets.py` emits the ShareGPT JSONL it consumes
  (detection boxes in `[0,1000]` + OTSL/HTML structure).
- **Train:** `Embodied/eaglevl/train/locany_finetune_magi_stream.py` (LoRA on
  vision+LLM, Magi Attention on H100/Blackwell). Note: LocateAnything **weights are
  under NVIDIA license**.

### Exp B — build our own: RF-DETR (+ optional TIPS v2 backbone)
Base it on **RF-DETR** (Roboflow, **Apache 2.0**) = LW-DETR + Deformable-attention on
a pretrained **DINOv2-with-registers** ViT, patch-14, COCO-trained, easy to fine-tune.
It already solves the two hard parts of "ViT + DETR" (`MultiScaleProjector` ViT→P3–P6;
`dinov2_with_windowed_attn` for high-res). Full recipe in
[`exp_b_rfdetr/INTEGRATION.md`](exp_b_rfdetr/INTEGRATION.md). **Staged — fork only when justified:**

- **B0 (zero fork):** stock RF-DETR on our COCO. `--coco-mode tatr` emits
  `row / column / spanning-cell` boxes → intersect rows×cols for the logical grid (no
  custom head, no dataloader edit). `--coco-mode cells` gives one box/cell for the
  reconstruction route.
- **B1 (small fork):** swap DINOv2→TIPS v2 — both patch-14 register ViTs, so projector/
  transformer/heads are unchanged; only a `TipsV2Backbone` (`exp_b_rfdetr/tips_backbone.py`,
  wrapping `models/tips_encoder.py`) + an encoder-enum branch. DINOv3 is a drop-in alt.
- **B2 (additive):** per-cell `logic_embed` head + `ConvertCoco` `logic_axis` parse +
  L1 loss — only if row×col intersection can't resolve spans.

**Why:** no vision-language alignment cost (a detection head needs none), pixel-accurate
boxes, single-node; also an **auto-labeler / teacher** for A. Settle the encoder choice
(TIPS vs DINOv3 vs MoonViT) with the cheap linear probe first. Note the resolution
constraint: square res must be divisible by `patch_size * num_windows` (tile the ~4000px pages).

## Repo

| Path | What | Status |
|------|------|--------|
| `data/page_to_targets.py` | PAGE XML → ShareGPT (A) + COCO `--coco` cells/`tatr` (B) + OTSL/HTML; both PAGE flavors; empty cells kept | **done, self-test passes** |
| `models/tips_encoder.py` | Frozen TIPS v2 encoder (HF DPT + npz paths) | scaffold (needs GPU) |
| `exp_b_rfdetr/INTEGRATION.md` | Staged RF-DETR recipe (B0 stock → B1 TIPS swap → B2 logic head), real symbols | guide |
| `exp_b_rfdetr/tips_backbone.py` | `TipsV2Backbone` matching RF-DETR's DinoV2 contract | scaffold (needs GPU) |
| `eval/eval_teds.py` | TEDS structure metric (logical correctness, not box IoU) | scaffold (`pip install apted`) |

```bash
python data/page_to_targets.py --self-test                 # verify (no deps)
python data/page_to_targets.py *.xml --jsonl out.jsonl     # Exp A: LocateAnything ShareGPT
python data/page_to_targets.py *.xml --coco train.json     # Exp B: RF-DETR/TIPS COCO
python eval/eval_teds.py --self-test                       # needs apted
```

## Status / next

- [x] PAGE→targets converter, verified on synthetic Transkribus + PRImA XML.
- [x] TIPS v2 encoder wrapper (from the provided code) + TEDS metric.
- [ ] **Validate converter on one real Transkribus PAGE export** (the earlier file
      was the archive's ABBYY OCR, not annotation — need a real PAGE sample).
- [ ] Exp A: LocateAnything LoRA fine-tune config on the H100 box.
- [ ] Exp B: TIPS+DETR head + training loop; encoder linear-probe first.
- [ ] Evaluate both with TEDS (structure-only) on a held-out set.

Evaluate on the **logical** metric (cell→row/col→key-value, span accuracy), not box IoU.
