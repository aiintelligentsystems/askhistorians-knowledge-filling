set -eux

CUDA_DEVICE=$1
COMMAND=$2
SCRATCH_PATH=$3

echo "Running command '$COMMAND' on GPU $CUDA_DEVICE"

docker build . -t redditqa --build-arg HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

docker run \
    --gpus device=$CUDA_DEVICE \
    --mount type=bind,source=$SCRATCH_PATH,target=/scratch1/jhoff \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    -it redditqa \
    $COMMAND 2>&1 | tee -a logs/output_$(date +"%d-%m-%Y").txt
