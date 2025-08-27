#!/bin/bash
export TOKENIZERS_PARALLELISM=False

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${WORLD_SIZE:-4}

# Model configuration
llm=Qwen/Qwen2.5-VL-72B-Instruct  # Using HuggingFace model ID

# Training entry point
entry_file=qwenvl/infer/infer_qwen_zero3_lora.py

# Checkpoint configuration
lora_dir=./output

# Input argument (To be filled)
image_path="IMAGE_PATH"
prompt="What was the final event ?"

# Inference arguments
args="
    --model_name_or_path ${llm} \
    --lora_root ${lora_dir} \
    --image_path ${image_path} \
    --prompt \"${prompt}\""

# Launch inference
msrun --worker_num=${NPROC_PER_NODE} \
    --local_worker_num=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --log_dir="logs/infer" \
    --join=True \
    ${entry_file} ${args}
