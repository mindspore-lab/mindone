#!/bin/bash

export DEVICE_ID=7

python eval_vqvae.py \
    --data_path /disk3/katekong/magvit/datasets/ucf101/minibatch/ \
    --output_path vqvae_eval \
    --size 128 \
    --crop_size 128 \
    --mode 0 \
    --ckpt_path outputs/vae_celeba_train/ckpt/vae_kl_f8-e22.ckpt \
    --eval_loss True \