# Copyright 2025 The HuggingFace Team.
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

from mindone.diffusers import LTXLatentUpsamplePipeline
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
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class LTXLatentUpsamplePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=8,
                block_out_channels=(8, 8, 8, 8),
                decoder_block_out_channels=(8, 8, 8, 8),
                layers_per_block=(1, 1, 1, 1, 1),
                decoder_layers_per_block=(1, 1, 1, 1, 1),
                spatio_temporal_scaling=(True, True, False, False),
                decoder_spatio_temporal_scaling=(True, True, False, False),
                decoder_inject_noise=(False, False, False, False, False),
                upsample_residual=(False, False, False, False),
                upsample_factor=(1, 1, 1, 1),
                timestep_conditioning=False,
                patch_size=1,
                patch_size_t=1,
                encoder_causal=True,
                decoder_causal=False,
            ),
        ],
        [
            "latent_upsampler",
            "diffusers.pipelines.ltx.modeling_latent_upsampler.LTXLatentUpsamplerModel",
            "mindone.diffusers.pipelines.ltx.modeling_latent_upsampler.LTXLatentUpsamplerModel",
            dict(
                in_channels=8,
                mid_channels=32,
                num_blocks_per_stage=1,
                dims=3,
                spatial_upsample=True,
                temporal_upsample=False,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "vae",
                "latent_upsampler",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        pt_components["vae"].use_framewise_encoding = False
        ms_components["vae"].use_framewise_encoding = False
        pt_components["vae"].use_framewise_decoding = False
        ms_components["vae"].use_framewise_decoding = False

        return pt_components, ms_components

    def get_dummy_inputs(self):
        pt_video = torch.randn((5, 3, 32, 32))
        ms_video = ms.tensor(pt_video.numpy())

        pt_inputs = {
            "video": pt_video,
            "height": 16,
            "width": 16,
            "output_type": "np",
        }

        ms_inputs = {
            "video": ms_video,
            "height": 16,
            "width": 16,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.ltx.LTXLatentUpsamplePipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.ltx.LTXLatentUpsamplePipeline")

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

        pt_image_slice = pt_image.frames[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class LTXLatentUpsamplePipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline will OOM in float32")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
            "Lightricks/ltxv-spatial-upscaler-0.9.7", mindspore_dtype=ms_dtype
        )

        latents = ms.tensor(
            load_numpy_from_local_file(
                "mindone-testing-arrays",
                f"latent_upsample_latents_{dtype}.npy",
                subfolder="ltx",
            )
        )

        image = pipe_upsample(
            latents=latents,
            height=1024,
            width=1536,
        )[0][
            0
        ][1]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"latent_upsample_{dtype}.npy",
            subfolder="ltx",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
