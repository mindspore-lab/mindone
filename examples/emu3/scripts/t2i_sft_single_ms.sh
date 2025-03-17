export DEVICE_ID=0  # TO REPLACE
NPUS=8                                  # TO REPLACE
MASTER_PORT=9000                        # TO REPLACE
DATAPATH="your data path (json file)"   # TO REPLACE
EXP_NAME="Emu3-T2I-SFT-Trial"           # TO REPLACE
LOG_DIR=outputs/${EXP_NAME}

python emu3/train/train_seq_parallel.py \
    --model_name_or_path BAAI/Emu3-Stage1 \
    --mode 1 \
    --debug False \
    --fp16 True \
    --sequence_parallel_shards 1 \
    --ms_zero_stage 3 \
    --optim adamw_mindspore \
    --jit_level O1 \
    --is_distribute False \
    --train_data_path ${DATAPATH} \
    --dataloader_num_workers 1 \
    --null_prompt_prob 0.05 \
    --apply_loss_on_only_vision True \
    --apply_loss_on_only_text False \
    --image_area 262144 \
    --max_position_embeddings 1024 \
    --trainable_hidden_layers 16 \
    --output_dir ${LOG_DIR} \
    --num_train_epochs 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_steps 1 \
    --save_strategy epoch \
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
    --logging_steps 1 \
    --gradient_checkpointing False \
    --save_strategy epoch \
    --run_name ${EXP_NAME} \
    --loss_scaler_type dynamic \
    --max_device_memory "59GB"
