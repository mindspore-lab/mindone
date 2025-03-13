#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # TO REPLACE
NPUS=8                                  # TO REPLACE
MASTER_PORT=9000                        # TO REPLACE
DATAPATH="your data path (json file)"   # TO REPLACE
EXP_NAME="Emu3-T2I-SFT-Trial"           # TO REPLACE
LOG_DIR=outputs/parallel_logs/${EXP_NAME}

msrun --bind_core=True --worker_num=${NPUS} --local_worker_num=${NPUS} --master_port=${MASTER_PORT} --log_dir=${LOG_DIR} \
python emu3/train/train.py \
    --model_name_or_path BAAI/Emu3-Gen \
    --bf16 True \
    --optim adamw_zero2_mindspore \
    --is_distribute True \
    --train_data_path ${DATAPATH} \
    --dataloader_num_workers 4 \
    --null_prompt_prob 0.05 \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_text False \
    --image_area 518400 \
    --max_position_embeddings 10240 \
    --output_dir ${LOG_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --min_learning_rate 1e-6 \
    --weight_decay 0.1 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --warmup_steps 30 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --save_strategy steps \
    --run_name ${EXP_NAME}
