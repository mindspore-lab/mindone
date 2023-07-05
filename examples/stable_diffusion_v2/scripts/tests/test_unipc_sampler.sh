
#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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
python text_to_image.py \
    --prompt "A Van Gogh style oil painting of sunflower" \
    --config configs/v2-inference.yaml \
    --output_path ./output/ \
    --seed 42 \
    --uni_pc \
    --n_iter 8 \
    --n_samples 1 \
    --W 512 \
    --H 512 \
    --ddim_steps 20
