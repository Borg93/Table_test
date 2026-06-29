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

### Exp B — build our own: **TipsDETR** (TIPS v2 backbone + deformable DETR)
Our own detector, **inspired by RF-DETR but not a fork** (Apache-clean): a frozen
**TIPS v2** ViT-L (patch-14) → a ViTDet `MultiScaleProjector` → a 4-level
**deformable-DETR** encoder/decoder with iterative box refinement → focal class +
box heads. Implemented and **smoke-tested** in [`tips_detr/`](tips_detr/README.md);
multi-scale deformable attention is pure-PyTorch (no CUDA ext).

> Because we run **offline / batch**, we drop RF-DETR's real-time machinery
> (windowed attention, light decoder) and instead run the **full** frozen ViT with a
> heavier deformable decoder — accuracy over latency.

```bash
python -m tips_detr.smoke                         # CPU end-to-end wiring check (passes)
torchrun --nproc_per_node=8 -m tips_detr.train \
    --train-coco dataset/train/_annotations.coco.json --train-images dataset/train \
    --backbone tips --tips-variant L --image-size 1568 --amp --out runs/tips_detr
```

Train on `--coco-mode tatr` (classes `table row / table column / table spanning cell`);
at inference **intersect** row-bands × column-bands → logical cell grid, spanning-cell
boxes fix merges (works with faint gridlines + empty cells — bands are geometric).
Swapping TIPS → DINOv3 is a one-line backbone change. If spans can't be resolved by
intersection, add a per-query `logic_embed` head (`--coco-mode cells` already emits
`logic_axis`). Resolution must be a multiple of 14; **tile** ~4000px scans.

**Stock-RF-DETR fallback (zero code):** if you'd rather not run our model, the same
`--coco-mode tatr` json trains stock RF-DETR unchanged — staged notes (B0 stock → B1
TIPS swap → B2 logic head) in [`exp_b_rfdetr/INTEGRATION.md`](exp_b_rfdetr/INTEGRATION.md).

**Why Exp B at all:** no vision-language alignment cost, pixel-accurate boxes,
single-node, permissive license; also an **auto-labeler / teacher** for A. Settle the
encoder choice (TIPS vs DINOv3) with the cheap linear probe first.

## Repo

| Path | What | Status |
|------|------|--------|
| `data/page_to_targets.py` | PAGE XML → ShareGPT (A) + COCO `--coco` cells/`tatr` (B) + OTSL/HTML; both PAGE flavors; empty cells kept | **done, self-test passes** |
| `tips_detr/` | **Our own TIPS v2 + deformable-DETR detector** (model, transformer, deformable attn, matcher, criterion, dataset, train, smoke) | **done, smoke + data path verified** |
| `models/tips_encoder.py` | Frozen TIPS v2 encoder (HF DPT + npz paths), wrapped by `tips_detr` | scaffold (needs GPU) |
| `exp_b_rfdetr/INTEGRATION.md` | Fallback: staged **stock** RF-DETR recipe (B0 → B1 → B2) | guide |
| `eval/eval_teds.py` | TEDS structure metric (logical correctness, not box IoU) | scaffold (`pip install apted`) |

```bash
python data/page_to_targets.py --self-test                       # verify (no deps)
python data/page_to_targets.py *.xml --coco train.json --coco-mode tatr  # Exp B: TipsDETR COCO
python -m tips_detr.smoke                                         # Exp B: model wiring (torch+scipy)
python data/page_to_targets.py *.xml --jsonl out.jsonl           # Exp A: LocateAnything ShareGPT
python eval/eval_teds.py --self-test                             # needs apted
```

## Status / next

- [x] PAGE→targets converter, verified on synthetic Transkribus + PRImA XML.
- [x] TIPS v2 encoder wrapper (from the provided code) + TEDS metric.
- [x] **Exp B model built & verified**: `tips_detr/` (TIPS backbone + deformable DETR);
      `smoke.py` passes (forward/loss/backward) and the COCO→dataset→train-step path runs.
- [ ] **Validate converter on one real Transkribus PAGE export** (the earlier file
      was the archive's ABBYY OCR, not annotation — need a real PAGE sample).
- [ ] Exp B: run on the H100 box with real TIPS v2 weights; encoder linear-probe first.
- [ ] Exp A: LocateAnything LoRA fine-tune config on the H100 box.
- [ ] Evaluate both with TEDS (structure-only) on a held-out set.

Evaluate on the **logical** metric (cell→row/col→key-value, span accuracy), not box IoU.
