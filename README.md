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

### Exp B — build our own (TIPS v2 encoder + detection head)
Frozen **TIPS v2** encoder (`models/tips_encoder.py`, wired to the exact API from
the TIPS Feature Explorer + foreground-seg notebook) → ViT-Adapter / simple FPN →
**Deformable-DETR / Mask2Former** decoder with a per-cell **logical-coordinate head**
(`start_col,end_col,start_row,end_row`). DINOv3 is a drop-in alternative encoder.
- **Why:** no alignment cost (a detection head needs none), pixel-accurate quads,
  trains on a single node; also serves as **auto-labeler / distillation teacher** for A.
- **Encoder choice (TIPS vs DINOv3 vs MoonViT):** settle empirically with the cheap
  linear probe (the foreground-seg recipe) before committing.
- **Build-your-own VLM caveat:** swapping the encoder *inside a VLM* means redoing
  vision-language alignment pretraining — expensive. Only go there if a probe shows a
  clear win and fine-tuning A plateaus.

## Repo

| Path | What | Status |
|------|------|--------|
| `data/page_to_targets.py` | PAGE XML → detection-box + OTSL + HTML targets (both Transkribus & PRImA flavors; empty cells kept) | **done, self-test passes** |
| `models/tips_encoder.py` | Frozen TIPS v2 encoder (HF DPT + npz paths) → multi-layer features for a DETR/Mask2Former head | scaffold (needs GPU) |
| `eval/eval_teds.py` | TEDS structure metric (logical correctness, not box IoU) | scaffold (`pip install apted`) |

```bash
python data/page_to_targets.py --self-test          # verify (no deps)
python data/page_to_targets.py PAGE.xml --jsonl out.jsonl
python eval/eval_teds.py --self-test                # needs apted
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
