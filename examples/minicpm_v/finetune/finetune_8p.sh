#!/bin/bash

MODEL="openbmb/MiniCPM-V-2_6"
# or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/data3/wcr/mindone/examples/minicpm/finetune/finetune.json"
#EVAL_DATA="path/to/test_data"
LLM_TYPE="qwen2" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3"
MODEL_MAX_Length=2048 # if conduct multi-images sft, please set MODEL_MAX_Length=4096

export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

msrun --worker_num=4 --local_worker_num=4 --master_port=8118 --log_dir=pynative_logs --join=True --cluster_time_out=300 finetune.py  \
    --model_name_or_path $MODEL \
    --llm_type $LLM_TYPE \
    --data_path $DATA \
    --remove_unused_columns false \
    --label_names "labels" \
    --prediction_loss_only false \
    --bf16 false \
    --bf16_full_eval false \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --tune_vision true \
    --tune_llm false \
    --model_max_length $MODEL_MAX_Length \
    --max_slice_nums 9 \
    --max_steps 10000 \
    --output_dir output/output_minicpmv26 \
    --logging_dir output/output_minicpmv26 \
    --logging_strategy "steps" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-6 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --distributed true \
    > pynative_logs/train_vision.log 2>&1 &
