set -eux

CUDA_DEVICE=$1
COMMAND=$2

echo "Running command '$COMMAND' on GPU $CUDA_DEVICE"

docker build . -t redditqa

docker run \
    --gpus device=$CUDA_DEVICE \
    --mount type=bind,source=/scratch1/jhoff,target=/scratch1/jhoff \
    --mount type=bind,source=/home/jhoffbauer/.cache/huggingface,target=/hf_cache \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    -it redditqa \
    $COMMAND 2>&1 | tee -a logs/output_$(date +"%d-%m-%Y").txt
