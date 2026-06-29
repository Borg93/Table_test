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

### Exp B — build our own: RF-DETR with a TIPS v2 backbone
Base it on **RF-DETR** (Roboflow, **Apache 2.0**) = LW-DETR + Deformable-attention on
a pretrained **DINOv2-with-registers** ViT, patch-14, COCO-trained, easy to fine-tune.
It already solves the two hard parts of "ViT + DETR":
- `MultiScaleProjector` (ViT layers → P3–P6 pyramid) — the real version of our FPN stub.
- `dinov2_with_windowed_attn` — runs the ViT at high res efficiently (most blocks
  windowed, few global) — needed for our ~4000px dense pages.

**Adaptation (small, well-scoped):**
1. Write a `TipsV2` backbone mirroring RF-DETR's `DinoV2` interface (both are patch-14
   register ViTs, so the projector/transformer/heads are unchanged). DINOv3 is a
   drop-in alternative. Encoder uses `models/tips_encoder.py` (the TIPS API you gave).
2. Add `logic_embed = MLP(d, d, 4, 3)` to `DetectionHead` → regress
   `(start_col,end_col,start_row,end_row)`; add the L1 term in `criterion.py`.
3. Targets: `python data/page_to_targets.py *.xml --coco train.json` — COCO boxes +
   `logic_axis` per cell. Warm-start from RF-DETR COCO weights; tile/raise resolution
   for thin columns.

- **Why:** no vision-language alignment cost (a detection head needs none),
  pixel-accurate boxes, trains on a single node; also an **auto-labeler / teacher** for A.
- **Encoder choice (TIPS vs DINOv3 vs MoonViT):** settle with the cheap linear probe
  (foreground-seg recipe) before committing.
- **Caveat:** swapping the encoder *inside a VLM* (vs. inside this detector) would mean
  redoing alignment pretraining — expensive. Here, inside RF-DETR, the swap is free.

## Repo

| Path | What | Status |
|------|------|--------|
| `data/page_to_targets.py` | PAGE XML → ShareGPT (Exp A) **+ COCO `--coco` (Exp B)** + OTSL/HTML; both PAGE flavors; empty cells kept | **done, self-test passes** |
| `models/tips_encoder.py` | Frozen TIPS v2 encoder (HF DPT + npz paths); drop into RF-DETR's backbone slot (use its `MultiScaleProjector`) | scaffold (needs GPU) |
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
