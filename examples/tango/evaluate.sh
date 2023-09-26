#!bin/bash

device_id=7
export RANK_SIZE=1
export DEVICE_ID=$device_id

python evaluate.py \
  --config_path="configs" \
  --ckpt="../../../tango_ms_full.ckpt" \
  --num_steps 200 \
  --guidance 3 \
  --num_samples 1
