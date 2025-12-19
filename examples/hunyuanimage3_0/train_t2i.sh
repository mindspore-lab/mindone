#!/bin/bash
export TOKENIZERS_PARALLELISM=False
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${WORLD_SIZE:-8}

# Model configuration
MODEL_ID=${MODEL_ID:-"HunyuanImage-3/"}  # Using HuggingFace model ID

# Training entry point
entry_file="run_image_train.py"

# Output configuration
output_dir="output/train"

# Input argument (To be filled)
dataset_path="datasets/pokemon-blip-captions"
deepspeed="scripts/zero3.json"
learning_rate=1e-5
num_train_epochs=1
seed=0
save_strategy="no"
do_eval="False"

# Launch inference
msrun --worker_num=${NPROC_PER_NODE} \
    --local_worker_num=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --log_dir="logs/train" \
    --join=True \
    ${entry_file} \
    --dataset_path "${dataset_path}" \
    --deepspeed "${deepspeed}" \
    --model_path "${MODEL_ID}" \
    --output_dir "${output_dir}" \
    --num_train_epochs "${num_train_epochs}" \
    --learning_rate "${learning_rate}" \
    --seed "${seed}" \
    --save_strategy "${save_strategy}" \
    --do_eval "${do_eval}" \
    --bf16
