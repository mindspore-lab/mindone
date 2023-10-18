#!bin/bash

device_id=4
export RANK_SIZE=1
export DEVICE_ID=$device_id

python train.py \
    --data_path=data.json \
    --train_config=configs/train_config_v.json \
    --output_path="output/exp" \
    --use_parallel=False \
    --pretrained_model_path="." \
    --pretrained_model_file="tango_full_ft_audiocaps-fa8f707f.ckpt" \
    --train_batch_size=1
