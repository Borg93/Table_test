#!/usr/bin/env bash
# LoRA fine-tune LocateAnything for handwritten table structure (Exp A).
#
# Run from inside the cloned NVlabs/Eagle "Embodied" dir (after `pip install -e .`),
# with this repo's recipe + JSONL prepared (see FINETUNE.md). LoRA on the LLM,
# frozen MoonViT backbone, trainable MLP connector — cheap, single-node 8xH100.
#
# Arch note: --attn_implementation magi needs Hopper/Blackwell. On A100/L40 use
# sdpa and keep --max_seq_length <= 4096 (dense pages may need tiling).
set -euo pipefail

# ---- paths (edit) ----------------------------------------------------------
BASE_MODEL=${BASE_MODEL:-nvidia/LocateAnything-3B}   # HF id or local checkpoint
META_PATH=${META_PATH:-./locany_recipe/handwritten_tables.json}   # the recipe.json
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/locany_tables_lora}
GPUS=${GPUS:-8}
ATTN=${ATTN:-magi}                                   # magi (Hopper/Blackwell) | sdpa
MAX_SEQ=${MAX_SEQ:-16384}                            # drop to 4096 for sdpa

mkdir -p "$OUTPUT_DIR"

torchrun \
    --nnodes=1 \
    --nproc_per_node="$GPUS" \
    --master_port=29500 \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "$BASE_MODEL" \
  --meta_path "$META_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir False \
  --attn_implementation "$ATTN" \
  --block_size 6 \
  --causal_attn False \
  --use_llm_lora 64 \
  --freeze_llm False \
  --freeze_backbone True \
  --freeze_mlp False \
  --vision_select_layer -1 \
  --mlp_connector_layers 2 \
  --bf16 True \
  --grad_checkpoint True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --warmup_steps 200 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --max_seq_length "$MAX_SEQ" \
  --max_num_tokens_per_sample "$MAX_SEQ" \
  --max_num_tokens 25600 \
  --packing_buffer_size 32 \
  --dataloader_num_workers 4 \
  --save_strategy steps \
  --save_steps 200 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --do_train True \
  --group_by_length False \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --report_to tensorboard \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
