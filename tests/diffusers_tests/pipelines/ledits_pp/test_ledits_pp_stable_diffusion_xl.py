# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
from transformers import CLIPTextConfig, CLIPVisionConfig

import mindspore as ms

from mindone.diffusers import LEditsPPPipelineStableDiffusionXL
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
class LEditsPPPipelineStableDiffusionXLFastTests(PipelineTesterMixin, unittest.TestCase):
    time_cond_proj_dim = None
    skip_first_text_encoder = None
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
                time_cond_proj_dim=time_cond_proj_dim,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                # SD2-specific config below
                attention_head_dim=(2, 4),
                use_linear_projection=True,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                transformer_layers_per_block=(1, 2),
                projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
                cross_attention_dim=64 if not skip_first_text_encoder else 32,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            "mindone.diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            dict(
                algorithm_type="sde-dpmsolver++",
                solver_order=2,
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
            "image_encoder",
            "transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            dict(
                config=CLIPVisionConfig(
                    hidden_size=32,
                    image_size=224,
                    projection_dim=32,
                    intermediate_size=37,
                    num_attention_heads=4,
                    num_channels=3,
                    num_hidden_layers=5,
                    patch_size=14,
                ),
            ),
        ],
        [
            "feature_extractor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(
                crop_size=224,
                do_center_crop=True,
                do_normalize=True,
                do_resize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                resample=3,
                size=224,
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
                    projection_dim=32,
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
        [
            "text_encoder_2",
            "transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
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
                    projection_dim=32,
                ),
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
    ]

    def get_dummy_components(self, skip_first_text_encoder=False, time_cond_proj_dim=None):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "text_encoder_2",
                "tokenizer_2",
                "image_encoder",
                "feature_extractor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        if skip_first_text_encoder:
            pt_components["text_encoder"] = None
            pt_components["tokenizer"] = None
            ms_components["text_encoder"] = None
            ms_components["tokenizer"] = None

        return pt_components, ms_components

    def get_dummy_inputs(self):
        inputs = {
            "editing_prompt": ["wearing glasses", "sunshine"],
            "reverse_editing_direction": [False, True],
            "edit_guidance_scale": [10.0, 5.0],
            "output_type": "np",
        }
        return inputs

    def get_dummy_inversion_inputs(self):
        images = floats_tensor((2, 3, 32, 32), rng=random.Random(0)).permute(0, 2, 3, 1)
        images = 255 * images
        image_1 = Image.fromarray(np.uint8(images[0])).convert("RGB")
        image_2 = Image.fromarray(np.uint8(images[1])).convert("RGB")

        inputs = {
            "image": [image_1, image_2],
            "source_prompt": "",
            "source_guidance_scale": 3.5,
            "num_inversion_steps": 2,
            "skip": 0.15,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_ledits_pp_sdxl(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion_xl.LEditsPPPipelineStableDiffusionXL"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion_xl.LEditsPPPipelineStableDiffusionXL"
        )

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        invert_inputs = self.get_dummy_inversion_inputs()
        invert_inputs["image"] = invert_inputs["image"][0]
        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        _ = pt_pipe.invert(**invert_inputs)
        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)

        torch.manual_seed(0)
        _ = ms_pipe.invert(**invert_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -1, -3:, -3:]
        ms_image_slice = ms_image[0][0, -1, -3:, -3:]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class LEditsPPPipelineStableDiffusionXLSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_ledits_pp_edit(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            safety_checker=None,
            add_watermarker=None,
            mindspore_dtype=ms_dtype,
        )
        pipe.set_progress_bar_config(disable=None)

        raw_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat_6.png",
            subfolder="pix2pix",
        )
        raw_image = raw_image.convert("RGB").resize((512, 512))

        torch.manual_seed(0)
        _ = pipe.invert(image=raw_image, num_zero_noise_steps=0)
        inputs = {
            "editing_prompt": ["cat", "dog"],
            "reverse_editing_direction": [True, False],
            "edit_guidance_scale": [2.0, 4.0],
            "edit_threshold": [0.8, 0.8],
        }
        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"sdxl_{dtype}.npy",
            subfolder="ledits_pp",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
