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
class ControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "controlnet",
            "diffusers.models.controlnet.ControlNetModel",
            "mindone.diffusers.models.controlnet.ControlNetModel",
            dict(
                block_out_channels=(4, 8),
                layers_per_block=2,
                in_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                cross_attention_dim=32,
                conditioning_embedding_out_channels=(16, 32),
                norm_num_groups=1,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[4, 8],
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
                norm_num_groups=2,
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

    def get_unet_config(self, time_cond_proj_dim=None):
        return [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(4, 8),
                layers_per_block=2,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
                norm_num_groups=1,
                time_cond_proj_dim=time_cond_proj_dim,
            ),
        ]

    def get_dummy_components(self, time_cond_proj_dim=None):
        components = {
            key: None
            for key in [
                "unet",
                "controlnet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "safety_checker",
                "feature_extractor",
                "image_encoder",
            ]
        }
        unet = self.get_unet_config(time_cond_proj_dim)
        pipeline_config = [unet] + self.pipeline_config

        return get_pipeline_components(components, pipeline_config)

    def get_dummy_inputs(self, seed=0):
        controlnet_embedder_scale_factor = 2
        pt_image = floats_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            rng=random.Random(seed),
        )
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": pt_image,
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": ms_image,
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_controlnet_lcm(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components(time_cond_proj_dim=256)
        pt_pipe_cls = get_module("diffusers.pipelines.controlnet.StableDiffusionControlNetPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionControlNetPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_scheduler_cls = get_module("diffusers.schedulers.scheduling_lcm.LCMScheduler")
        ms_scheduler_cls = get_module("mindone.diffusers.schedulers.scheduling_lcm.LCMScheduler")
        pt_pipe.scheduler = pt_scheduler_cls.from_config(pt_pipe.scheduler.config)
        ms_pipe.scheduler = ms_scheduler_cls.from_config(ms_pipe.scheduler.config)

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
class ControlNetPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_canny(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet_cls = get_module("mindone.diffusers.models.controlnet.ControlNetModel")
        controlnet = controlnet_cls.from_pretrained("lllyasviel/sd-controlnet-canny", mindspore_dtype=ms_dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionControlNetPipeline")
        pipe = pipe_cls.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            safety_checker=None,
            controlnet=controlnet,
            mindspore_dtype=ms_dtype,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "bird"
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "bird_canny.png",
            subfolder="sd_controlnet",
        )

        torch.manual_seed(0)
        output = pipe(prompt, image, num_inference_steps=3)

        image = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"t2i_canny_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

    @data(*test_cases)
    @unpack
    def test_depth(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet_cls = get_module("mindone.diffusers.models.controlnet.ControlNetModel")
        controlnet = controlnet_cls.from_pretrained("lllyasviel/sd-controlnet-depth", mindspore_dtype=ms_dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionControlNetPipeline")
        pipe = pipe_cls.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            safety_checker=None,
            controlnet=controlnet,
            mindspore_dtype=ms_dtype,
        )
        pipe.set_progress_bar_config(disable=None)

        prompt = "Stormtrooper's lecture"
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "stormtrooper_depth.png",
            subfolder="sd_controlnet",
        )

        torch.manual_seed(0)
        output = pipe(prompt, image, num_inference_steps=3)

        image = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"t2i_depth_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
