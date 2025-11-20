#!/bin/bash
export TOKENIZERS_PARALLELISM=False
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${WORLD_SIZE:-8}

# Model configuration
llm="/mnt/disk3/dxw/hunyuanimage3/"  # Using HuggingFace model ID

# Training entry point
entry_file="run_image_gen.py"

# Checkpoint configuration
# lora_dir=./output

# Input argument (To be filled)
image_path="image_repro.png"
prompt="A brown and white dog is running on the grass"
seed=0
verbose=1
enable_amp="True"
image_size="832x1216"

# Launch inference
msrun --worker_num=${NPROC_PER_NODE} \
    --local_worker_num=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --log_dir="logs/infer" \
    --join=True \
    ${entry_file} \
    --model-id "${llm}" \
    --save "${image_path}" \
    --prompt "${prompt}" \
    --seed "${seed}" \
    --verbose "${verbose}" \
    --enable-ms-amp "${enable_amp}"\
    --image-size "${image_size}"\
    --reproduce
