# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


class AdapterTests:
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
                time_cond_proj_dim=None,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_pndm.PNDMScheduler",
            "mindone.diffusers.schedulers.scheduling_pndm.PNDMScheduler",
            dict(
                skip_prk_steps=True,
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

    def get_adapter_config(self, adapter_type):
        return [
            "adapter",
            "diffusers.models.adapter.T2IAdapter",
            "mindone.diffusers.models.adapter.T2IAdapter",
            dict(
                in_channels=3,
                channels=[32, 64],
                num_res_blocks=2,
                downscale_factor=2,
                adapter_type=adapter_type,
            ),
        ]

    def get_dummy_components(self, adapter_type, time_cond_proj_dim=None):
        components = {
            key: None
            for key in [
                "adapter",
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "safety_checker",
                "feature_extractor",
            ]
        }

        if adapter_type == "full_adapter" or adapter_type == "light_adapter":
            adapter = self.get_adapter_config(adapter_type)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}, must be 'full_adapter' or 'light_adapter'")

        pipeline_config = self.pipeline_config + [adapter]

        return get_pipeline_components(components, pipeline_config)

    def get_dummy_inputs(self, seed=0, height=64, width=64):
        pt_image = floats_tensor((1, 3, height, width), rng=random.Random(seed))
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": pt_image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": ms_image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs


@ddt
class StableDiffusionFullAdapterPipelineFastTests(PipelineTesterMixin, AdapterTests, unittest.TestCase):
    def get_dummy_components(self, time_cond_proj_dim=None):
        return super().get_dummy_components("full_adapter", time_cond_proj_dim=time_cond_proj_dim)

    @data(*test_cases)
    @unpack
    def test_stable_diffusion_adapter_default_case(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.t2i_adapter.StableDiffusionAdapterPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.t2i_adapter.StableDiffusionAdapterPipeline")

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


@ddt
class StableDiffusionLightAdapterPipelineFastTests(PipelineTesterMixin, AdapterTests, unittest.TestCase):
    def get_dummy_components(self, time_cond_proj_dim=None):
        return super().get_dummy_components("light_adapter", time_cond_proj_dim=time_cond_proj_dim)

    @data(*test_cases)
    @unpack
    def test_stable_diffusion_adapter_default_case(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.t2i_adapter.StableDiffusionAdapterPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.t2i_adapter.StableDiffusionAdapterPipeline")

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
class StableDiffusionAdapterPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_stable_diffusion_adapter_color(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        adapter_model = "TencentARC/t2iadapter_color_sd14v1"
        sd_model = "CompVis/stable-diffusion-v1-4"
        prompt = "snail"
        input_channels = 3

        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "color.png",
            subfolder="t2i_adapter",
        )
        if input_channels == 1:
            image = image.convert("L")

        adapter = get_module("mindone.diffusers.models.adapter.T2IAdapter")
        adapter = adapter.from_pretrained(adapter_model, revision="refs/pr/3", mindspore_dtype=ms_dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.t2i_adapter.StableDiffusionAdapterPipeline")
        pipe = pipe_cls.from_pretrained(sd_model, adapter=adapter, safety_checker=None, mindspore_dtype=ms_dtype)
        pipe.set_progress_bar_config(disable=None)

        torch.manual_seed(0)
        image = pipe(prompt=prompt, image=image, num_inference_steps=2)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"adapter_color_{dtype}.npy",
            subfolder="stable_diffusion_adapter",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
