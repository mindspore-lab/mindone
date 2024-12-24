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

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class DDPMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d.UNet2DModel",
            "mindone.diffusers.models.unets.unet_2d.UNet2DModel",
            dict(
                block_out_channels=(4, 8),
                layers_per_block=1,
                norm_num_groups=4,
                sample_size=8,
                in_channels=3,
                out_channels=3,
                down_block_types=("DownBlock2D", "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_fast_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.ddpm.DDPMPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.ddpm.DDPMPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        if dtype == "float32":
            torch.manual_seed(0)
            pt_image = pt_pipe(**inputs)
            pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        else:
            # torch.flot16 requires CUDA
            pt_image_slice = np.array([[1.0, 1.0, 0.00244009], [0.04179019, 1.0, 0.00200841], [1.0, 1.0, 0.0052653]])

        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class DDPMPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "google/ddpm-cat-256"

        pipe_cls = get_module("mindone.diffusers.pipelines.ddpm.DDPMPipeline")
        ddpm = pipe_cls.from_pretrained(model_id, mindspore_dtype=ms_dtype)
        ddpm.set_progress_bar_config(disable=None)

        torch.manual_seed(0)
        image = ddpm()[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"ddpm_{dtype}.npy",
            subfolder="ddpm",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
