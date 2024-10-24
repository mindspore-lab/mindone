# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import mindspore as ms

from mindone.diffusers import StableCascadeUNet
from mindone.diffusers.utils import logging
from mindone.diffusers.utils.testing_utils import enable_full_determinism, slow

logger = logging.get_logger(__name__)

enable_full_determinism()


@slow
class TestStableCascadeUNetSingleFileTest:
    def test_single_file_components_stage_b(self):
        model_single_file = StableCascadeUNet.from_single_file(
            "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_bf16.safetensors",
            mindspore_dtype=ms.bfloat16,
        )
        model = StableCascadeUNet.from_pretrained(
            "stabilityai/stable-cascade", variant="bf16", subfolder="decoder", use_safetensors=True
        )

        PARAMS_TO_IGNORE = ["mindspore_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

    def test_single_file_components_stage_b_lite(self):
        model_single_file = StableCascadeUNet.from_single_file(
            "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite_bf16.safetensors",
            mindspore_dtype=ms.bfloat16,
        )
        model = StableCascadeUNet.from_pretrained(
            "stabilityai/stable-cascade", variant="bf16", subfolder="decoder_lite"
        )

        PARAMS_TO_IGNORE = ["mindspore_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

    def test_single_file_components_stage_c(self):
        model_single_file = StableCascadeUNet.from_single_file(
            "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors",
            mindspore_dtype=ms.bfloat16,
        )
        model = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", subfolder="prior")

        PARAMS_TO_IGNORE = ["mindspore_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"

    def test_single_file_components_stage_c_lite(self):
        model_single_file = StableCascadeUNet.from_single_file(
            "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_lite_bf16.safetensors",
            mindspore_dtype=ms.bfloat16,
        )
        model = StableCascadeUNet.from_pretrained(
            "stabilityai/stable-cascade-prior", variant="bf16", subfolder="prior_lite"
        )

        PARAMS_TO_IGNORE = ["mindspore_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between single file loading and pretrained loading"
