#!/bin/bash

export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1
export DEVICE_ID=7

python scripts/train_vqvae.py \
  --model_class vqvae-3d \
  --pretrained ./model_weights/vqvae2d-lfq-128-init.ckpt \
  --use_discriminator True \
  --use_ema True \
  --dataset_name video \
  --data_path ./datasets/ucf101/train/ \
  --num_frames 17 \
  --frame_stride 1 \
  --size 128 \
  --crop_size 128 \
  --num_parallel_workers 1 \
  --drop_overflow_update True \
  --batch_size 1 \
  --epochs 60 \
  --log_interval 400 \
  --ckpt_save_interval 1 \
  --gradient_accumulation_steps 16 \
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
  --scale_window 50000 \
  --dtype fp32 \
  --global_bf16 True \
  --mode 0 \
  --debug False \
  --seed 1234 \
  --output_path outputs/vqvae_3d/
  > train_3d.log 2>&1 &
