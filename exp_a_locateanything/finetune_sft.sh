#!/usr/bin/env bash
# Full-parameter SFT of LocateAnything for handwritten table structure (Exp A).
#
# Heavier than finetune_lora.sh (all of LLM + MLP train; backbone optionally too).
# Use when LoRA underfits the new domain. Mirrors Embodied/document/TRAINING.md.
set -euo pipefail

BASE_MODEL=${BASE_MODEL:-nvidia/LocateAnything-3B}
META_PATH=${META_PATH:-./locany_recipe/handwritten_tables.json}
OUTPUT_DIR=${OUTPUT_DIR:-work_dirs/locany_tables_sft}
GPUS=${GPUS:-8}
ATTN=${ATTN:-magi}
MAX_SEQ=${MAX_SEQ:-16384}

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
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --mlp_connector_layers 2 \
  --bf16 True \
  --grad_checkpoint True \
  --num_train_epochs 1 \
  --max_steps 25000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_steps 500 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --max_seq_length "$MAX_SEQ" \
  --max_num_tokens_per_sample "$MAX_SEQ" \
  --max_num_tokens 25600 \
  --packing_buffer_size 32 \
  --dataloader_num_workers 4 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --do_train True \
  --group_by_length False \
  --deepspeed deepspeed_configs/zero_stage2_config.json \
  --report_to tensorboard \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
