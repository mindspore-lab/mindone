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
from PIL import Image
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
class StableDiffusionDiffEditPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
                # SD2-specific config below
                attention_head_dim=(2, 4),
                use_linear_projection=True,
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
            "inverse_scheduler",
            "diffusers.schedulers.scheduling_ddim_inverse.DDIMInverseScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim_inverse.DDIMInverseScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_zero=False,
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
                    # SD2-specific config below
                    hidden_act="gelu",
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
                "inverse_scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "safety_checker",
                "feature_extractor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        pt_mask = floats_tensor((1, 16, 16), rng=random.Random(seed))
        pt_latents = floats_tensor((1, 2, 4, 16, 16), rng=random.Random(seed))
        ms_mask = ms.Tensor(pt_mask.numpy())
        ms_latents = ms.Tensor(pt_latents.numpy())

        pt_inputs = {
            "prompt": "a dog and a newt",
            "mask_image": pt_mask,
            "image_latents": pt_latents,
            "num_inference_steps": 2,
            "inpaint_strength": 1.0,
            "guidance_scale": 6.0,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "a dog and a newt",
            "mask_image": ms_mask,
            "image_latents": ms_latents,
            "num_inference_steps": 2,
            "inpaint_strength": 1.0,
            "guidance_scale": 6.0,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    def get_dummy_mask_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")

        inputs = {
            "image": image,
            "source_prompt": "a cat and a frog",
            "target_prompt": "a dog and a newt",
            "num_inference_steps": 2,
            "num_maps_per_mask": 2,
            "mask_encode_strength": 1.0,
            "guidance_scale": 6.0,
            "output_type": "np",
        }

        return inputs

    def get_dummy_inversion_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")

        inputs = {
            "image": image,
            "prompt": "a cat and a frog",
            "num_inference_steps": 2,
            "inpaint_strength": 1.0,
            "guidance_scale": 6.0,
            "decode_latents": True,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_mask(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_diffusion_diffedit.StableDiffusionDiffEditPipeline")
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.stable_diffusion_diffedit.StableDiffusionDiffEditPipeline"
        )

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_mask_inputs()

        torch.manual_seed(0)
        pt_mask = pt_pipe.generate_mask(**inputs)
        torch.manual_seed(0)
        ms_mask = ms_pipe.generate_mask(**inputs)

        pt_mask_slice = pt_mask[0, -3:, -3:]
        ms_mask_slice = ms_mask[0, -3:, -3:]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.abs(pt_mask_slice - ms_mask_slice).max() < threshold

    @data(*test_cases)
    @unpack
    def test_inversion(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_diffusion_diffedit.StableDiffusionDiffEditPipeline")
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.stable_diffusion_diffedit.StableDiffusionDiffEditPipeline"
        )

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inversion_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe.invert(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe.invert(**inputs)

        pt_image_slice = pt_image.images[0, -1, -3:, -3:]
        ms_image_slice = ms_image[1][0, -1, -3:, -3:]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class StableDiffusionDiffEditPipelineNightlyTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_stable_diffusion_diffedit_ddim(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.stable_diffusion_diffedit.StableDiffusionDiffEditPipeline")
        pipe = pipe_cls.from_pretrained(
            "stabilityai/stable-diffusion-2-1", safety_checker=None, mindspore_dtype=ms_dtype
        )
        scheduler = get_module("mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler")
        inverse_scheduler = get_module("mindone.diffusers.schedulers.scheduling_ddim_inverse.DDIMInverseScheduler")
        pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = inverse_scheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)

        source_prompt = "a bowl of fruit"
        target_prompt = "a bowl of pears"
        raw_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "fruit.png",
            subfolder="diffedit",
        )
        raw_image = raw_image.convert("RGB").resize((512, 512))

        torch.manual_seed(0)
        mask_image = pipe.generate_mask(
            image=raw_image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
        )

        torch.manual_seed(0)
        inv_latents = pipe.invert(
            prompt=source_prompt,
            image=raw_image,
            inpaint_strength=0.7,
            num_inference_steps=25,
        )[0]

        torch.manual_seed(0)
        image = pipe(
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            negative_prompt=source_prompt,
            inpaint_strength=0.7,
            num_inference_steps=25,
        )[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"diffedit_ddim_{dtype}.npy",
            subfolder="stable_diffusion_2",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
