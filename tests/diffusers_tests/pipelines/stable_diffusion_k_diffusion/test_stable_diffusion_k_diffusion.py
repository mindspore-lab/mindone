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

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin, get_module

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@slow
@ddt
class StableDiffusionPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self):
        inputs = {
            "prompt": "a photo of an astronaut riding a horse on mars",
            "num_inference_steps": 20,
            "guidance_scale": 9.0,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_stable_diffusion_1_5_lms(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module(
            "mindone.diffusers.pipelines.stable_diffusion_k_diffusion.StableDiffusionKDiffusionPipeline"
        )
        sd_pipe = pipe_cls.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms_dtype)

        sd_pipe.set_scheduler("sample_lms")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()

        torch.manual_seed(0)
        image = sd_pipe(**inputs)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"sd_1_5_lms_{dtype}.npy",
            subfolder="stable_diffusion_k_diffusion",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
