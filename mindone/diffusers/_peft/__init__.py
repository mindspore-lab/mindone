# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
# Hacked together by / Copyright 2024 Genius Patrick @ MindSpore Team.
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

__version__ = "0.8.2"

from .config import PeftConfig, PromptLearningConfig
from .mapping import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    PEFT_TYPE_TO_CONFIG_MAPPING,
    get_peft_config,
    get_peft_model,
    inject_adapter_in_model,
)
from .peft_model import PeftModel
from .tuners import LoftQConfig, LoraConfig, LoraModel
from .utils import PeftType, TaskType, get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
