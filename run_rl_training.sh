sh ./run_in_docker.sh 0 'python3 -m redditqa.training.rl_training.py --model_name=EleutherAI/pythia-2.8b --tokenizer_name=meta-llama/Llama-2-7b-chat-hf --reward_model_name=/scratch1/jhoff/checkpoints/reward_llama-2-7b-chat-hf/checkpoint-3000_merged --output_max_length 128 --output_dir=/scratch1/jhoff/checkpoints/ppo_llama-2-7b-chat-hf --save_freq=1000 --steps=20000 --adafactor=True'