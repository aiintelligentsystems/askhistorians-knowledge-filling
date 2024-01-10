set -eux

CUDA_DEVICE=$1
COMMAND=$2

echo "Running command '$COMMAND' on GPU $CUDA_DEVICE"

echo "Building container"
docker build . -t redditqa --build-arg HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

echo "Starting container"
docker run \
    --gpus device=$CUDA_DEVICE \
    --shm-size=8G \
    --mount type=bind,source=/scratch/tbuz,target=/scratch1/redditqa \
    --mount type=bind,source=/scratch/tbuz/hf_cache,target=/hf_cache \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -it redditqa \
    $COMMAND
