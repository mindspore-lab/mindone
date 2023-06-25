
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
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export DEVICE_ID=4 

export SD_VERSOIN="2.0"

python text_to_image.py \
    --prompt "A wolfie in winter" \
    --config configs/v2-inference.yaml \
    --output_path ./output/ \
    --seed 42 \
    --n_iter 4 \
    --n_samples 1 \
    --W 512 \
    --H 512 \
    --ddim_steps 50
