# Copyright 2023-present the HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/peft
# with modifications to run peft on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mindone.peft.utils import register_peft_method

from .config import EvaConfig, LoftQConfig, LoraConfig, LoraRuntimeConfig
from .layer import Conv2d, Conv3d, Linear, LoraLayer
from .model import LoraModel

__all__ = [
    "Conv2d",
    "Conv3d",
    "EvaConfig",
    "Linear",
    "LoftQConfig",
    "LoraConfig",
    "LoraLayer",
    "LoraModel",
    "LoraRuntimeConfig",
]

register_peft_method(name="lora", config_cls=LoraConfig, model_cls=LoraModel, is_mixed_compatible=True)
