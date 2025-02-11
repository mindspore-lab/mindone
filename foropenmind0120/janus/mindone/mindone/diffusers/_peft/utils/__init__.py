# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

from .constants import (
    CONFIG_NAME,
    EMBEDDING_LAYER_NAMES,
    INCLUDE_LINEAR_LAYERS_SHORTHAND,
    SAFETENSORS_WEIGHTS_NAME,
    TOKENIZER_CONFIG_NAME,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    WEIGHTS_NAME,
)
from .other import (
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_batch_size,
    _get_submodules,
    _is_valid_match,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    transpose,
)
from .peft_types import PeftType, TaskType
from .save_and_load import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
