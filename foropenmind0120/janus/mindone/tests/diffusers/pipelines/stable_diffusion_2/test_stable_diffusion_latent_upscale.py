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
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

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
class StableDiffusionLatentUpscalePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                act_fn="gelu",
                attention_head_dim=8,
                norm_num_groups=None,
                block_out_channels=[32, 32, 64, 64],
                time_cond_proj_dim=160,
                conv_in_kernel=1,
                conv_out_kernel=1,
                cross_attention_dim=32,
                down_block_types=(
                    "KDownBlock2D",
                    "KCrossAttnDownBlock2D",
                    "KCrossAttnDownBlock2D",
                    "KCrossAttnDownBlock2D",
                ),
                in_channels=8,
                mid_block_type=None,
                only_cross_attention=False,
                out_channels=5,
                resnet_time_scale_shift="scale_shift",
                time_embedding_type="fourier",
                timestep_post_act="gelu",
                up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[32, 32, 64, 64],
                in_channels=3,
                out_channels=3,
                down_block_types=[
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            dict(
                prediction_type="sample",
            ),
        ],
        [
            "text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    pad_token_id=1,
                    vocab_size=1000,
                    hidden_act="quick_gelu",
                    projection_dim=512,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        pt_components["unet"] = pt_components["unet"].eval()
        pt_components["vae"] = pt_components["vae"].eval()
        ms_components["unet"] = ms_components["unet"].set_train(False)
        ms_components["vae"] = ms_components["vae"].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        pt_image = floats_tensor((1, 4, 16, 16), rng=random.Random(0))
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": pt_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": ms_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_diffusion.StableDiffusionLatentUpscalePipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.stable_diffusion.StableDiffusionLatentUpscalePipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        if dtype == "float32":
            torch.manual_seed(0)
            pt_image = pt_pipe(**pt_inputs)
            pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        else:
            # torch.flot16 requires CUDA
            pt_image_slice = np.array(
                [
                    [0.19506836, 0.42089844, 0.51171875],
                    [0.33984375, 0.49487305, 0.55615234],
                    [0.48486328, 0.5761719, 0.4645996],
                ]
            )

        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class StableDiffusionLatentUpscalePipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_latent_upscaler(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.stable_diffusion.StableDiffusionPipeline")
        pipe = pipe_cls.from_pretrained("CompVis/stable-diffusion-v1-4", mindspore_dtype=ms_dtype)

        upscaler_cls = get_module("mindone.diffusers.pipelines.stable_diffusion.StableDiffusionLatentUpscalePipeline")
        upscaler = upscaler_cls.from_pretrained("stabilityai/sd-x2-latent-upscaler", mindspore_dtype=ms_dtype)

        prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"

        torch.manual_seed(0)
        low_res_latents = pipe(prompt, output_type="latent")[0]

        torch.manual_seed(0)
        image = upscaler(
            prompt=prompt,
            image=low_res_latents,
            num_inference_steps=20,
            guidance_scale=0,
        )[
            0
        ][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"latent_upscale_{dtype}.npy",
            subfolder="stable_diffusion_2",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
