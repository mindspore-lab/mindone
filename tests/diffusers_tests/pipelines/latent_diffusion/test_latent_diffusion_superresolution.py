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

import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils import PIL_INTERPOLATION
from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
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
class LDMSuperResolutionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d.UNet2DModel",
            "mindone.diffusers.models.unets.unet_2d.UNet2DModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=6,
                out_channels=3,
                down_block_types=("DownBlock2D", "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "UpBlock2D"),
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(),
        ],
        [
            "vqvae",
            "diffusers.models.autoencoders.vq_model.VQModel",
            "mindone.diffusers.models.autoencoders.vq_model.VQModel",
            dict(
                block_out_channels=[32, 64],
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=3,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vqvae",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        pt_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "image": pt_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        ms_inputs = {
            "image": ms_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference_superresolution(self, mode, dtype):
        # If strict mode is disabled, this pipeline will randomly fail.
        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        else:
            ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.latent_diffusion.LDMSuperResolutionPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.latent_diffusion.LDMSuperResolutionPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class LDMSuperResolutionPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference_superresolution(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        init_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "teddy_bear_pool.png",
            subfolder="vq_diffusion",
        )
        init_image = init_image.resize((64, 64), resample=PIL_INTERPOLATION["lanczos"])

        pipe_cls = get_module("mindone.diffusers.pipelines.latent_diffusion.LDMSuperResolutionPipeline")
        ldm = pipe_cls.from_pretrained(
            "CompVis/ldm-super-resolution-4x-openimages", variant="fp16", mindspore_dtype=ms_dtype
        )
        ldm.set_progress_bar_config(disable=None)

        torch.manual_seed(0)
        image = ldm(image=init_image, num_inference_steps=20)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"superresolution_{dtype}.npy",
            subfolder="latent_diffusion",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
