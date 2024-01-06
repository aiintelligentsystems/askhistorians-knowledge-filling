BASE_MODEL="lmsys/vicuna-7b-v1.5-16k"
OUT_MODEL_PATH_SFT="/scratch1/redditqa/paper_eli5/vicuna1.5_sft"
OUT_MODEL_PATH_RM="/scratch1/redditqa/paper_eli5/vicuna1.5_rm_margin10"

# Stop on error
set -e

# # Run SFT
# python3 -m redditqa.training.run_sft \
#     --model_path=${BASE_MODEL} \
#     --learning_rate=2.0e-05 \
#     --max_seq_length=2048 \
#     --batch_size=2 \
#     --gradient_accumulation_steps=32 \
#     --num_train_epochs=1 \
#     --output_dir=${OUT_MODEL_PATH_SFT} 

# # Merge SFT model
# python3 -m redditqa.scripts.merge_peft_adapter \
#     --adapter_model_name=${OUT_MODEL_PATH_SFT}/final_checkpoint \
#     --base_model_name=${BASE_MODEL}

# Run RM
python3 -m redditqa.training.run_rm \
    --model_name=${OUT_MODEL_PATH_SFT}/final_checkpoint_merged \
    --num_train_epochs=5 \
    --max_length=2048 \
    --batch_size=2 \
    --gradient_accumulation_steps=8 \
    --score_margin=10 \
    --output_dir=${OUT_MODEL_PATH_RM} 