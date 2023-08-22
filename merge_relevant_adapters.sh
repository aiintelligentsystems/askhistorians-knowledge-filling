#python3 ~/reddit_qa/redditqa/scripts/merge_peft_adapter.py \
#    --adapter_model_name=/scratch1/jhoff/checkpoints/finetuned_llama-2-7b-hf/checkpoint-2000 \
#    --base_model_name=meta-llama/Llama-2-7b-hf 
#
#python3 ~/reddit_qa/redditqa/scripts/merge_peft_adapter.py \
#    --adapter_model_name=/scratch1/jhoff/checkpoints/finetuned_open-llama-3b-v2/checkpoint-2500 \
#    --base_model_name=openlm-research/open_llama_3b_v2 
#
#python3 redditqa/scripts/merge_peft_adapter.py \
#    --adapter_model_name=/scratch1/jhoff/checkpoints/ppo_llama-2-7b_sft-checkpoint-2500/final \
#    --base_model_name=/scratch1/jhoff/checkpoints/finetuned_llama-2-7b-chat-hf/checkpoint-2500_merged 

python3 redditqa/scripts/merge_peft_adapter.py \
    --adapter_model_name=/scratch1/jhoff/checkpoints/ppo_llama-2-7b-nochat_sft-checkpoint-2000/final \
    --base_model_name=/scratch1/jhoff/checkpoints/finetuned_llama-2-7b-hf/checkpoint-2000_merged
