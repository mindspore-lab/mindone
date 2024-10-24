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

import numpy as np

import mindspore as ms

from mindone.diffusers import AutoencoderKL
from mindone.diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_downloaded_numpy_from_hf_hub,
    numpy_cosine_similarity_distance,
    slow,
)

enable_full_determinism()


@slow
class TestAutoencoderKLSingleFile:
    model_class = AutoencoderKL
    ckpt_path = (
        "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
    )
    repo_id = "stabilityai/sd-vae-ft-mse"
    main_input_name = "sample"
    base_precision = 1e-2

    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = ms.float16 if fp16 else ms.float32
        image = load_downloaded_numpy_from_hf_hub(
            repo_id="fusing/diffusers-testing", filename=self.get_file_format(seed, shape)
        )
        image = ms.Tensor.from_numpy(image).to(dtype)
        return image

    def test_single_file_inference_same_as_pretrained(self):
        model_1 = self.model_class.from_pretrained(self.repo_id)
        model_2 = self.model_class.from_single_file(self.ckpt_path, config=self.repo_id)

        image = self.get_sd_image(33)

        generator = np.random.default_rng(0)
        sample_1 = model_1(image, generator=generator)[0]

        generator = np.random.default_rng(0)
        sample_2 = model_2(image, generator=generator)[0]

        assert sample_1.shape == sample_2.shape

        output_slice_1 = sample_1.flatten().float().numpy()
        output_slice_2 = sample_2.flatten().float().numpy()

        assert numpy_cosine_similarity_distance(output_slice_1, output_slice_2) < 1e-4

    def test_single_file_components(self):
        model = self.model_class.from_pretrained(self.repo_id)
        model_single_file = self.model_class.from_single_file(self.ckpt_path, config=self.repo_id)

        PARAMS_TO_IGNORE = ["mindspore_dtype", "_name_or_path", "_use_default_values", "_diffusers_version"]
        for param_name, param_value in model_single_file.config.items():
            if param_name in PARAMS_TO_IGNORE:
                continue
            assert (
                model.config[param_name] == param_value
            ), f"{param_name} differs between pretrained loading and single file loading"

    def test_single_file_arguments(self):
        model_default = self.model_class.from_single_file(self.ckpt_path, config=self.repo_id)

        assert model_default.config.scaling_factor == 0.18215
        assert model_default.config.sample_size == 256
        assert model_default.dtype == ms.float32

        scaling_factor = 2.0
        sample_size = 512
        mindspore_dtype = ms.float16

        model = self.model_class.from_single_file(
            self.ckpt_path,
            config=self.repo_id,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            mindspore_dtype=mindspore_dtype,
        )
        assert model.config.scaling_factor == scaling_factor
        assert model.config.sample_size == sample_size
        assert model.dtype == mindspore_dtype
