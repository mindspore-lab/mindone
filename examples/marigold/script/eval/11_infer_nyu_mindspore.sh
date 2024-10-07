#!/usr/bin/env bash
set -e
set -x

# Use specified checkpoint path, otherwise, default value
use_fp16=$1
ckpt=${2:-"marigold-checkpoint/marigold-vkitti.ckpt"}
subfolder=${3:-"eval"}
base_data_dir=${BASE_DATA_DIR:-"marigold-data"}

python infer.py \
    $use_fp16 \
    --ms_ckpt \
    --checkpoint $ckpt \
    --seed 1234 \
    --base_data_dir $base_data_dir \
    --denoise_steps 50 \
    --ensemble_size 5 \
    --processing_res 0 \
    --dataset_config config/dataset/data_nyu_test.yaml \
    --output_dir output/${subfolder}/nyu_test/prediction
