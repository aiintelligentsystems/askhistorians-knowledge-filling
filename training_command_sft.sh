python3 -m redditqa.training.supervised_finetuning \
    --model_path="HuggingFaceH4/zephyr-7b-beta" \
    --batch_size=1 \
    --gradient_accumulation_steps=32 \
    --seq_length=512 \
    --output_dir="/scratch1/redditqa/ws23/zephyr_sft_filtered_dataset"
