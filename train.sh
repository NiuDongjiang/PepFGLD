#!/bin/bash
########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"


########## parsing yaml configs ##########
if [ -z $1 ]; then
    echo "Config missing. Usage example: GPU=0,1 bash $0 <config> [optional arguments]"
    exit 1;
fi


########## setup  distributed training ##########
GPU="${GPU:--1}" # default using CPU
MASTER_ADDR="${ADDR:-localhost}"
MASTER_PORT="${PORT:-9901}"
echo "Using GPUs: $GPU"
echo "Master address: ${MASTER_ADDR}, Master port: ${MASTER_PORT}"

export CUDA_VISIBLE_DEVICES=$GPU
GPU_ARR=(`echo $GPU | tr ',' ' '`)

if [ ${#GPU_ARR[@]} -gt 1 ]; then
    export OMP_NUM_THREADS=2
	PREFIX="torchrun --nproc_per_node=${#GPU_ARR[@]} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} --nnodes=1"
else
    PREFIX="python"
fi


########## start training ##########
cd $CODE_DIR
${PREFIX} train.py --gpus "${!GPU_ARR[@]}" --config $@
