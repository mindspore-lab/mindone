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

from .other import (
    CONFIG_NAME,
    INCLUDE_LINEAR_LAYERS_SHORTHAND,
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    WEIGHTS_NAME,
    AuxiliaryTrainingWrapper,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_batch_size,
    _get_input_embeddings_name,
    _get_subcell,
    _get_submodules,
    _is_valid_match,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    cast_mixed_precision_params,
    prepare_model_for_kbit_training,
    refresh_parameter_name_of_model,
    shift_tokens_right,
    transpose,
)
from .peft_types import PeftType, TaskType, register_peft_method
from .save_and_load import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict

__all__ = [
    "CONFIG_NAME",
    "INCLUDE_LINEAR_LAYERS_SHORTHAND",
    "SAFETENSORS_WEIGHTS_NAME",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING",
    "TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING",
    "WEIGHTS_NAME",
    "AuxiliaryTrainingWrapper",
    "ModulesToSaveWrapper",
    "PeftType",
    "TaskType",
    "_freeze_adapter",
    "_get_batch_size",
    "_get_input_embeddings_name",
    "_get_subcell",
    "_get_submodules",
    "_is_valid_match",
    "_prepare_prompt_learning_config",
    "_set_adapter",
    "_set_trainable",
    "cast_mixed_precision_params",
    "get_peft_model_state_dict",
    "load_peft_weights",
    "prepare_model_for_kbit_training",
    "register_peft_method",
    "refresh_parameter_name_of_model",
    "set_peft_model_state_dict",
    "shift_tokens_right",
    "transpose",
]
