export CUDA_VISIBLE_DEVICES=0

python3 -m redditqa.training.dpo_training \
    --model_name_or_path="/scratch1/redditqa/ws23/zephyr_sft_filtered_dataset/checkpoint-2000_merged" \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --max_length=1024 \
    --max_steps=10000 \
    --output_dir="/scratch1/redditqa/ws23/zephyr_dpo_filtered_dataset"
