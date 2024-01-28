#!/bin/bash

BASE_MODEL="HuggingFaceH4/zephyr-7b-beta"
MODEL_NAME="/scratch1/redditqa/ws23/zephyr_sft_askhistorians_bf16/final_checkpoint_merged"
ADAPTER_PATH="/scratch1/redditqa/ws23/zephyr_dpo_askhistorians_bf16/checkpoint-500"
DATASET_NAME="askhistorians"
WANDB_PROJECT="reddit-qa-ws24"
OUTPUT_DIR=""
TASKS=("knowledge-token")
# "base" "gpt4-preference" "create-human-comp"

# Stop on error
set -e

# Run SFT
python3 -m redditqa.evaluation.run_eval \
    --tasks "${TASKS[@]}" \
    --baseline_model_name=${BASE_MODEL} \
    --model_name=${MODEL_NAME} \
    --adapter_name=${ADAPTER_PATH} \
    --dataset_name=${DATASET_NAME} \
    --wandb_project=${WANDB_PROJECT} \
    --output_dir=${OUTPUT_DIR} \
    --split=test \
    --n_questions=100
