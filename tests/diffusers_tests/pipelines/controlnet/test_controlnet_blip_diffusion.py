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
from PIL import Image
from transformers import Blip2Config, CLIPTextConfig

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
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
class BlipDiffusionControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "text_encoder",
            "diffusers.pipelines.blip_diffusion.ContextCLIPTextModel",
            "mindone.diffusers.pipelines.blip_diffusion.ContextCLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    vocab_size=1000,
                    hidden_size=16,
                    intermediate_size=16,
                    projection_dim=16,
                    num_hidden_layers=1,
                    num_attention_heads=1,
                    max_position_embeddings=77,
                ),
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                in_channels=4,
                out_channels=4,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                block_out_channels=(32,),
                layers_per_block=1,
                act_fn="silu",
                latent_channels=4,
                norm_num_groups=16,
                sample_size=16,
            ),
        ],
        [
            "qformer",
            "diffusers.pipelines.blip_diffusion.modeling_blip2.Blip2QFormerModel",
            "mindone.diffusers.pipelines.blip_diffusion.modeling_blip2.Blip2QFormerModel",
            dict(
                config=Blip2Config(
                    vision_config={
                        "hidden_size": 16,
                        "intermediate_size": 16,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "image_size": 224,
                        "patch_size": 14,
                        "hidden_act": "quick_gelu",
                    },
                    qformer_config={
                        "vocab_size": 1000,
                        "hidden_size": 16,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 1,
                        "intermediate_size": 16,
                        "max_position_embeddings": 512,
                        "cross_attention_frequency": 1,
                        "encoder_hidden_size": 16,
                    },
                    num_query_tokens=16,
                    tokenizer="hf-internal-testing/tiny-random-bert",
                ),
            ),
        ],
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(4, 16),
                layers_per_block=1,
                norm_num_groups=4,
                sample_size=16,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=16,
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
            "scheduler",
            "diffusers.schedulers.scheduling_pndm.PNDMScheduler",
            "mindone.diffusers.schedulers.scheduling_pndm.PNDMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                set_alpha_to_one=False,
                skip_prk_steps=True,
            ),
        ],
        [
            "controlnet",
            "diffusers.models.controlnet.ControlNetModel",
            "mindone.diffusers.models.controlnet.ControlNetModel",
            dict(
                block_out_channels=(4, 16),
                layers_per_block=1,
                in_channels=4,
                norm_num_groups=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                cross_attention_dim=16,
                conditioning_embedding_out_channels=(8, 16),
            ),
        ],
        [
            "image_processor",
            "diffusers.pipelines.blip_diffusion.BlipImageProcessor",
            "mindone.diffusers.pipelines.blip_diffusion.BlipImageProcessor",
            dict(),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "text_encoder",
                "vae",
                "qformer",
                "unet",
                "tokenizer",
                "scheduler",
                "controlnet",
                "image_processor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        eval_components = ["vae", "qformer", "text_encoder"]

        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        np.random.seed(seed)
        reference_image = np.random.rand(32, 32, 3) * 255
        reference_image = Image.fromarray(reference_image.astype("uint8")).convert("RGBA")
        cond_image = np.random.rand(32, 32, 3) * 255
        cond_image = Image.fromarray(cond_image.astype("uint8")).convert("RGBA")

        inputs = {
            "prompt": "swimming underwater",
            "reference_image": reference_image,
            "condtioning_image": cond_image,
            "source_subject_category": "dog",
            "target_subject_category": "dog",
            "height": 32,
            "width": 32,
            "guidance_scale": 7.5,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_blipdiffusion_controlnet(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.controlnet.BlipDiffusionControlNetPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.BlipDiffusionControlNetPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, 0]
        ms_image_slice = ms_image[0][0, -3:, -3:, 0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold
