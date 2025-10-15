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
import pytest
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import KolorsPAGPipeline
from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
]


ms.runtime.launch_blocking()


@ddt
class KolorsPAGPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(2, 4),
                layers_per_block=2,
                time_cond_proj_dim=None,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                # specific config below
                attention_head_dim=(2, 4),
                use_linear_projection=True,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                transformer_layers_per_block=(1, 2),
                projection_class_embeddings_input_dim=56,
                cross_attention_dim=8,
                norm_num_groups=1,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                steps_offset=1,
                beta_schedule="scaled_linear",
                timestep_spacing="leading",
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[32, 64],
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
                sample_size=128,
            ),
        ],
        [
            "text_encoder",
            "diffusers.pipelines.kolors.text_encoder.ChatGLMModel",
            "mindone.diffusers.pipelines.kolors.text_encoder.ChatGLMModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-chatglm3-6b",
            ),
        ],
        [
            "tokenizer",
            "diffusers.pipelines.kolors.tokenizer.ChatGLMTokenizer",
            "mindone.diffusers.pipelines.kolors.tokenizer.ChatGLMTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-chatglm3-6b",
            ),
        ],
    ]

    # Copied from tests.pipelines.kolors.test_kolors.KolorsPipelineFastTests.get_dummy_components
    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "image_encoder",
                "feature_extractor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "pag_scale": 0.9,
            "output_type": "np",
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_pag_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.pag.pipeline_pag_kolors.KolorsPAGPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.pag.pipeline_pag_kolors.KolorsPAGPipeline")

        pt_pipe_pag = pt_pipe_cls(**pt_components, pag_applied_layers=["mid", "up", "down"])
        ms_pipe_pag = ms_pipe_cls(**ms_components, pag_applied_layers=["mid", "up", "down"])

        pt_pipe_pag.set_progress_bar_config(disable=None)
        ms_pipe_pag.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe_pag = pt_pipe_pag.to(pt_dtype)
        ms_pipe_pag = ms_pipe_pag.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe_pag(**inputs).images
        torch.manual_seed(0)
        ms_image = ms_pipe_pag(**inputs)[0]

        pt_image_slice = pt_image[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class KolorsPAGPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_pag_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline only has the weight of fp16")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = KolorsPAGPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers",
            variant="fp16",
            mindspore_dtype=ms_dtype,
            enable_pag=True,
            pag_applied_layers=["down_blocks.2.attentions.1", "up_blocks.0.attentions.1"],
        )

        prompt = "A photo of a ladybug, macro, zoom, high quality, film, holding a wooden sign with the text 'KOLORS'"
        torch.manual_seed(0)
        image = pipe(prompt, guidance_scale=5.5, pag_scale=1.5)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"kolors_{dtype}.npy",
            subfolder="pag",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
