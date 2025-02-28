#!/usr/bin/env python
# coding=utf-8

# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

from .. import __version__

from .doc import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    copy_func,
    replace_return_docstrings,
)

from .generic import (
    ContextManagers,
    ExplicitEnum,
    ModelOutput,
    PaddingStrategy,
    TensorType,
    add_model_info_to_auto_map,
    is_jax_tensor,
    is_numpy_array,
    is_tensor,
    is_tf_tensor,
    is_torch_device,
    is_torch_dtype,
    is_torch_tensor,
    to_numpy,
    to_py_obj,
)

from .hub import (
    HF_MODULES_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    PushToHubMixin,
    cached_file,
    download_url,
    extract_commit_hash,
    has_file,
    is_offline_mode,
    is_remote_url,
    try_to_load_from_cache,
)

from .import_utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_sentencepiece_available,
    is_torch_available,
    is_tf_available,
    is_flax_available,
    is_optimum_available,
    is_safetensors_available,
    is_torch_tpu_available,
    is_torch_fx_proxy,
    is_tokenizers_available,
    requires_backends,
)

WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF2_WEIGHTS_INDEX_NAME = "tf_model.h5.index.json"
TF_WEIGHTS_NAME = "model.ckpt"
FLAX_WEIGHTS_NAME = "flax_model.msgpack"
FLAX_WEIGHTS_INDEX_NAME = "flax_model.msgpack.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_NAME = "config.json"
FEATURE_EXTRACTOR_NAME = "preprocessor_config.json"
IMAGE_PROCESSOR_NAME = FEATURE_EXTRACTOR_NAME
GENERATION_CONFIG_NAME = "generation_config.json"
MODEL_CARD_NAME = "modelcard.json"

SENTENCEPIECE_UNDERLINE = "▁"
SPIECE_UNDERLINE = SENTENCEPIECE_UNDERLINE  # Kept for backward compatibility

MULTIPLE_CHOICE_DUMMY_INPUTS = [
    [[0, 1, 0, 1], [1, 0, 0, 1]]
] * 2  # Needs to have 0s and 1s only since XLM uses it for langs too.
DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
