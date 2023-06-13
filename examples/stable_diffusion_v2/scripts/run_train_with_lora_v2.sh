#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

export GLOG_v=3
export HCCL_CONNECT_TIMEOUT=600
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

export SD_VERSION="2.0" # TODO: parse by args. or fix to 2.0 later

device_id=0

output_path=output
task_name=train_with_lora_1148
data_path=/home/mindocr/congw/data/pokemon_raw/
pretrained_model_path=/home/mindocr/congw/project/stablediffusionv2/models/
pretrained_model_file=stablediffusionv2_512.ckpt
train_config_file=configs/train_config_v2.json
image_size=512 # v2-base 512, v2.1 768

rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export RANK_SIZE=1;export DEVICE_ID=$device_id;export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}; \
# nohup 
python -u train_with_lora.py \
    --data_path=$data_path \
    --train_config=$train_config_file \
    --output_path=$output_path/$task_name \
    --use_parallel=False \
    --pretrained_model_path=$pretrained_model_path \
    --pretrained_model_file=$pretrained_model_file \
    --image_size=$image_size \
    --epochs=10 \
    --save_lora=True \
    --optim=adamw \
    --start_learning_rate=1e-4 \
    # > $output_path/$task_name/log_train 2>&1 &