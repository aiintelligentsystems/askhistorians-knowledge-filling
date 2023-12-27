BASE_MODEL="HuggingFaceH4/zephyr-7b-beta"
OUT_MODEL_PATH_SFT="/scratch1/redditqa/paper_eli5/zephyr_sft"
OUT_MODEL_PATH_DPO="/scratch1/redditqa/paper_eli5/zephyr_dpo"

# Stop on error
set -e

# Run SFT
python3 -m redditqa.training.run_sft \
    --model_path=${BASE_MODEL} \
    --learning_rate=2.0e-05 \
    --max_seq_length=2048 \
    --batch_size=2 \
    --gradient_accumulation_steps=32 \
    --num_train_epochs=1 \
    --output_dir=${OUT_MODEL_PATH_SFT} 

# Merge SFT model
python3 -m redditqa.scripts.merge_peft_adapter \
    --adapter_model_name=${OUT_MODEL_PATH_SFT}/final_checkpoint \
    --base_model_name=${BASE_MODEL} 

# Run DPO
python3 -m redditqa.training.run_dpo \
    --model_name_or_path=${OUT_MODEL_PATH_SFT}/final_checkpoint_merged \
    --beta=0.1 \
    --learning_rate=5.0e-7 \
    --max_length=1024 \
    --max_prompt_length=512 \
    --gradient_accumulation_steps=512 \
    --batch_size=1 \
    --max_steps=10000 \
    --output_dir=${OUT_MODEL_PATH_DPO} 

# Merge DPO model
python3 -m redditqa.scripts.merge_peft_adapter \
    --adapter_model_name=${OUT_MODEL_PATH_DPO}/final_checkpoint \
    --base_model_name=${OUT_MODEL_PATH_SFT}/final_checkpoint_merged

echo "Final model saved at ${OUT_MODEL_PATH_DPO}/final_checkpoint_merged"