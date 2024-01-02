set -eux

CUDA_DEVICE=$1
COMMAND=$2

echo "Running command '$COMMAND' on GPU $CUDA_DEVICE"

echo "Building container"
docker build . -t redditqa --build-arg HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

echo "Starting container"
docker run \
    --gpus device=$CUDA_DEVICE \
    --mount type=bind,source=/scratch/tbuz/eli5,target=/scratch1/redditqa/data/eli5 \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    -it redditqa \
    $COMMAND
