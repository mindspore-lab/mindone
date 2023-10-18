#!bin/bash

device_id=7
export RANK_SIZE=1
export DEVICE_ID=$device_id

python evaluate.py \
  --config_path="configs" \
  --ckpt="../../../tango_full_ft_audiocaps-fa8f707f.ckpt" \
  --num_steps 200 \
  --guidance 3 \
  --num_samples 1
