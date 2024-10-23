#!/bin/bash

export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1
export DEVICE_ID=7

python scripts/eval_vqvae.py \
    --model_class vqvae-3d \
    --data_path ./datasets/ucf101/test/ \
    --output_path vqvae_eval \
    --size 128 \
    --crop_size 128 \
    --num_frames 17 \
    --frame_stride 1 \
    --batch_size 16 \
    --mode 1 \
    --ckpt_path outputs/vqvae_3d/ckpt/vqvae.ckpt \
    --dtype fp32 \
    --eval_loss True \
