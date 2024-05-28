#!/bin/bash

export DEVICE_ID=7

nohup python train_vqvae.py \
  --use_discriminator False \
  --use_ema False \
  --dataset_name video \
  --data_path /disk3/katekong/magvit/datasets/ucf101/rec_trainset/ \
  --num_frames 17 \
  --crop_size 256 \
  --num_parallel_workers 1 \
  --drop_overflow_update True \
  --batch_size 1 \
  --epochs 30 \
  --log_interval 64 \
  --gradient_accumulation_steps 256 \
  --clip_grad True \
  --optim adamw \
  --base_learning_rate 5.0e-05 \
  --scale_lr False \
  --init_loss_scale 1024 \
  --loss_scaler_type dynamic \
  --dtype bf16 \
  --mode 0 \
  --output_path train_vqvae_17x256x256 \
  > train.log 2>&1 &