#!/bin/bash

if [ $# != 5 ]
then
  echo "For Multiple Devices In Single/Multiple Machine"
  echo "Usage Help: bash run_distribute.sh [RANK_TABLE_FILE] [RANK_START] [RANK_END] [RANK_SIZE] [DATASET_PATH]"
  echo "Example as: bash run_distribute.sh hccl_8p.json 0 8 8 /PATH TO/YOUR DATASET/"
  exit 1
fi

RANK_TABLE_FILE=$1
START_DEVICE=$2
END_DEVICE=$3
RANK_SIZE=$4
DATASET_PATH=$5

export RANK_TABLE_FILE=$RANK_TABLE_FILE
export RANK_SIZE=$RANK_SIZE
export DEVICE_NUM=$(($END_DEVICE - $START_DEVICE))

test -d ./logs_for_distribute || mkdir ./logs_for_distribute
env > logs_for_distribute/env.log

for((i=${START_DEVICE}; i<${END_DEVICE}; i++))
do
  export RANK_ID=${i}
  export DEVICE_ID=$((i-START_DEVICE))
  echo "start training for rank $RANK_ID, device $DEVICE_ID"
  python train.py \
    --config configs/training/sd_xl_base_finetune_910b.yaml \
    --weight checkpoints/sd_xl_base_1.0_ms.ckpt \
    --data_path $DATASET_PATH \
    --save_path_with_time False \
    --max_device_memory "59GB" \
    --is_parallel True > logs_for_distribute/log_$i.txt 2>&1 &
done
