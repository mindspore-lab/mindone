#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # TO REPLACE
NPUS=8                                  # TO REPLACE
MASTER_PORT=9000                        # TO REPLACE
DATAPATH="your data path (json file)"   # TO REPLACE
EXP_NAME="Emu3-VQA-SFT-Trial"           # TO REPLACE
LOG_DIR=outputs/parallel_logs/${EXP_NAME}

msrun --bind_core=True --worker_num=${NPUS} --local_worker_num=${NPUS} --master_port=${MASTER_PORT} --log_dir=${LOG_DIR} \
python emu3/train/train_seq_parallel.py \
    --model_name_or_path BAAI/Emu3-Stage1 \
    --mode 1 \
    --pynative_debug False \
    --fp16 True \
    --sequence_parallel_shards ${NPUS} \
    --ms_zero_stage 3 \
    --optim adamw_mindspore \
    --jit_level O1 \
    --is_distribute True \
    --train_data_path ${DATAPATH} \
    --dataloader_num_workers 1 \
    --null_prompt_prob 0.05 \
    --apply_loss_on_only_vision False \
    --apply_loss_on_only_text True \
    --image_area 147456 \
    --max_position_embeddings 2560 \
    --output_dir ${LOG_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --min_learning_rate 1e-6 \
    --weight_decay 0.1 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --warmup_steps 30 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --run_name ${EXP_NAME} \
    --loss_scaler_type dynamic \
    --max_device_memory "59GB"
