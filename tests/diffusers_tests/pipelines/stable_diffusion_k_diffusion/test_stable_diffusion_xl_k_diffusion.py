# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin, get_module

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@slow
@ddt
class StableDiffusionXLPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 9.0,
            "height": 512,
            "width": 512,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_stable_diffusion_xl_lms(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module(
            "mindone.diffusers.pipelines.stable_diffusion_k_diffusion.StableDiffusionXLKDiffusionPipeline"
        )
        sd_pipe = pipe_cls.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms_dtype)

        sd_pipe.set_scheduler("sample_lms")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()

        torch.manual_seed(0)
        image = sd_pipe(**inputs)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"sdxl_lms_{dtype}.npy",
            subfolder="stable_diffusion_k_diffusion",
        )

        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
