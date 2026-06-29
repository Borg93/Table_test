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
dense pages with 100s of cells). We frame table structure as **grounding/detection**:
prompt with categories, model outputs boxes — no text transcription needed.
Full recipe in [`exp_a_locateanything/FINETUNE.md`](exp_a_locateanything/FINETUNE.md).

- **Why:** inherits vision-language grounding pretraining; promptable; extends to
  reading the text later with no rearchitecture.
- **Data:** `data/page_to_targets.py --jsonl … --jsonl-mode tatr` emits the **exact**
  ShareGPT format it consumes — verified against Eagle's `train/tools.py`
  `_BOX_RE = <box><(\d+)><(\d+)><(\d+)><(\d+)></box>`: coordinate **tokens** in
  `[0,1000]`, `<ref>label</ref>` per instance, categories joined by `</c>`. `--recipe`
  also writes the `--meta_path` recipe JSON.
- **Train:** `exp_a_locateanything/finetune_lora.sh` (LLM LoRA r=64, frozen MoonViT,
  `--block_size 6 --causal_attn False` for PBD, Magi Attention on H100/Blackwell);
  `finetune_sft.sh` for full-param. Flags verified against `eaglevl/train/arguments.py`.
  Note: LocateAnything **weights are NVIDIA non-commercial** (code is Apache).

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
| `data/page_to_targets.py` | PAGE XML → LocateAnything ShareGPT (A, `--jsonl`) + COCO (B, `--coco` cells/`tatr`) + OTSL/HTML; both PAGE flavors; empty cells kept | **done, self-test passes** |
| `exp_a_locateanything/` | **LocateAnything fine-tune** (recipe.json + `finetune_lora.sh`/`finetune_sft.sh` + FINETUNE.md), format verified vs Eagle source | **done, data format verified** |
| `tips_detr/` | **Our own TIPS v2 + deformable-DETR detector** (model, transformer, deformable attn, matcher, criterion, dataset, train, smoke) | **done, smoke + data path verified** |
| `models/tips_encoder.py` | Frozen TIPS v2 encoder (HF DPT + npz paths), wrapped by `tips_detr` | scaffold (needs GPU) |
| `exp_b_rfdetr/INTEGRATION.md` | Fallback: staged **stock** RF-DETR recipe (B0 → B1 → B2) | guide |
| `eval/eval_teds.py` | TEDS structure metric (logical correctness, not box IoU) | scaffold (`pip install apted`) |

```bash
python data/page_to_targets.py --self-test                                       # verify (no deps)
python data/page_to_targets.py *.xml --jsonl tables.jsonl --recipe recipe.json   # Exp A: LocateAnything
python data/page_to_targets.py *.xml --coco train.json --coco-mode tatr          # Exp B: TipsDETR COCO
python -m tips_detr.smoke                                                         # Exp B: model wiring
python eval/eval_teds.py --self-test                                             # needs apted
pyrefly check                                                                     # static types: 0 errors
```

Verified two ways: it **runs** (the self-test / smoke / integration above all execute
and pass) and it **type-checks** (`pyrefly check` → 0 errors; config in `pyrefly.toml`).

## Status / next

- [x] PAGE→targets converter, verified on synthetic Transkribus + PRImA XML.
- [x] **Exp A data + train configs**: `--jsonl`/`--recipe` emit LocateAnything's exact
      grounding format (verified vs Eagle `tools.py`/`arguments.py`); LoRA + SFT scripts.
- [x] **Exp B model built & verified**: `tips_detr/` (TIPS backbone + deformable DETR);
      `smoke.py` passes (forward/loss/backward) and the COCO→dataset→train-step path runs.
- [ ] **Validate converter on one real Transkribus PAGE export** (the earlier file
      was the archive's ABBYY OCR, not annotation — need a real PAGE sample).
- [ ] Exp A: run `finetune_lora.sh` on the H100/Blackwell box (Magi Attention).
- [ ] Exp B: run on the H100 box with real TIPS v2 weights; encoder linear-probe first.
- [ ] Evaluate both with TEDS (structure-only) on a held-out set.

Evaluate on the **logical** metric (cell→row/col→key-value, span accuracy), not box IoU.
