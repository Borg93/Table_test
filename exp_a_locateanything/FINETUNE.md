# Exp A — fine-tune LocateAnything for handwritten table structure

Treat table structure as a **grounding / detection** task for the LocateAnything VLM
(NVlabs/Eagle `Embodied/` = MoonViT + Qwen2.5/3 + MLP projector, Parallel Box
Decoding). We don't transcribe text — we ground the geometry of rows, columns and
spanning cells, which exists even when cells are empty or gridlines are faint.

> **License:** LocateAnything **weights are NVIDIA non-commercial**. The training
> code is Apache-clean; our data/converter is ours. Use accordingly (research/eval).
> If you need a permissive deployable model, that's Exp B (`tips_detr/`).

## Data format (already produced by our converter)

LocateAnything consumes ShareGPT JSONL + a recipe JSON (`--meta_path`). Coordinates
are normalized integer **tokens** in `[0,1000]`; boxes are
`<box><x1><y1><x2><y2></box>`; labels are wrapped in `<ref>…</ref>`; detection
categories in the prompt are joined by `</c>`. (Verified against Eagle's
`train/tools.py` `_BOX_RE = <box><(\d+)><(\d+)><(\d+)><(\d+)></box>` and
`document/DATA_PREPARATION.md`.)

`data/page_to_targets.py` emits exactly this:

```bash
# TATR target (recommended): row / column / spanning-cell categories
python data/page_to_targets.py train/*.xml \
    --jsonl data/annotations/tables_tatr.jsonl --jsonl-mode tatr \
    --image-root /data/page_images \
    --recipe exp_a_locateanything/recipe.json
```

Produces one sample per page, e.g.:

```json
{"image": "page_0001.jpg", "conversations": [
  {"from": "human", "value": "<image-1>\nLocate all the instances that matches the following description: table row</c>table column</c>table spanning cell."},
  {"from": "gpt",   "value": "<ref>table row</ref><box><0><0><1000><200></box><ref>table row</ref><box><0><200><1000><400></box><ref>table column</ref><box><0><0><500><400></box>...<ref>table spanning cell</ref><box><0><0><500><200></box>"}
]}
```

`--jsonl-mode cells` instead emits one box per cell with the logical position carried
in the ref (`<ref>row 0 col 1 rowspan 1 colspan 2</ref><box>…</box>`) — use it if you
want the model to name cells directly rather than reconstructing the grid from bands.

## Setup

```bash
git clone https://github.com/NVlabs/Eagle.git && cd Eagle/Embodied
pip install -e .                                   # eagle_vl + deps
# Magi Attention (Hopper/Blackwell only; needed for 16K+ dense pages):
#   https://sandai-org.github.io/MagiAttention  (build ~10-20 min)
#   Blackwell: export MAGI_ATTENTION_PREBUILD_FFA=0 ; pip install --no-build-isolation . ; export MAGI_ATTENTION_FA4_BACKEND=1
# On A100/L40 (no Magi): use --attn_implementation sdpa and MAX_SEQ<=4096.
```

Put the recipe where the script expects it and point it at the JSONL:

```bash
mkdir -p locany_recipe && cp /path/to/Table_test/exp_a_locateanything/recipe.json \
    locany_recipe/handwritten_tables.json
# edit "annotation" -> your tables_tatr.jsonl, "root" -> your image dir
```

## Train

```bash
# LoRA (cheap; frozen MoonViT, LLM LoRA r=64, trainable MLP) — start here
META_PATH=./locany_recipe/handwritten_tables.json \
OUTPUT_DIR=work_dirs/locany_tables_lora GPUS=8 \
  bash /path/to/Table_test/exp_a_locateanything/finetune_lora.sh

# Full-param SFT if LoRA underfits the domain
bash /path/to/Table_test/exp_a_locateanything/finetune_sft.sh
```

Both scripts set `--block_size 6 --causal_attn False` for Parallel Box Decoding and
default `--attn_implementation magi`. Flags verified against `eaglevl/train/arguments.py`
(`use_llm_lora`, `use_backbone_lora`, `freeze_backbone/llm/mlp`, `meta_path`,
`block_size`, `attn_implementation`).

## Inference → logical grid

The model outputs the same grounding string. Parse it with the box regex above, then:
**intersect** `table row` bands × `table column` bands to get the logical cell grid
with (row, col) indices; `table spanning cell` boxes mark merges. This is robust to
empty cells and faint gridlines because the bands are geometric, not text-driven.
Score structure (not box IoU) with `eval/eval_teds.py` (TEDS, structure-only).

## Why two experiments

A (this) inherits vision-language grounding pretraining and is promptable / extensible
to reading text later; B (`tips_detr/`) is permissive, single-node, pixel-accurate, and
serves as an auto-labeler/teacher. Compare both on held-out TEDS.
