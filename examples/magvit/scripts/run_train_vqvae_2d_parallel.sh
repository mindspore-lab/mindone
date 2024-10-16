#!/bin/bash

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
# export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
# export MS_MEMORY_STATISTIC=0

# export MS_DATASET_SINK_QUEUE=4

# operation/graph fusion for dynamic shape
# export MS_DEV_ENABLE_KERNEL_PACKET=on

# enable kbk : 1
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

# log level
export GLOG_v=2

output_dir=outputs/vqvae_2d/

msrun --bind_core=True --master_port=8090 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
    python scripts/train_vqvae.py \
    --model_class vqvae-2d \
    --use_discriminator False \
    --use_parallel True \
    --use_ema False \
    --dataset_name image \
    --data_path ./datasets/ImageNet/train/ \
    --size 128 \
    --crop_size 128 \
    --num_parallel_workers 1 \
    --drop_overflow_update True \
    --batch_size 1 \
    --epochs 2 \
    --log_interval 400 \
    --ckpt_save_interval 1 \
    --gradient_accumulation_steps 8 \
    --clip_grad True \
    --max_grad_norm 1.0 \
    --optim adamw \
    --betas 0.99 \
    --weight_decay 0.01 \
    --warmup_steps 1000 \
    --base_learning_rate 2.0e-05 \
    --end_learning_rate 1.0e-07 \
    --scale_lr False \
    --init_loss_scale 1024 \
    --loss_scaler_type dynamic \
    --scale_window 1000 \
    --dtype fp32 \
    --global_bf16 True \
    --mode 0 \
    --seed 1234 \
    --output_path outputs/vqvae_2d/
    > train_2d.log 2>&1 &
