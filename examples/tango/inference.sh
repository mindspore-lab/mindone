#!bin/bash

device_id=7
export RANK_SIZE=1
export DEVICE_ID=$device_id

python inference.py \
  --original_args="configs/config.json" \
  --model="pytorch_model_2.bin" --num_steps 200 --guidance 3 --num_samples 1
