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
import pytest
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPTextConfig, DPTConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import (
    fast,
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


@fast
@ddt
class StableDiffusionDepth2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    backbone_config = {
        "global_padding": "same",
        "layer_type": "bottleneck",
        "depths": [3, 4, 9],
        "out_features": ["stage1", "stage2", "stage3"],
        "embedding_dynamic_padding": True,
        "hidden_sizes": [96, 192, 384, 768],
        "num_groups": 2,
    }

    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=5,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
                attention_head_dim=(2, 4),
                use_linear_projection=True,
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
            "depth_estimator",
            "transformers.models.dpt.modeling_dpt.DPTForDepthEstimation",
            "mindone.transformers.models.dpt.modeling_dpt.DPTForDepthEstimation",
            dict(
                config=DPTConfig(
                    image_size=32,
                    patch_size=16,
                    num_channels=3,
                    hidden_size=32,
                    num_hidden_layers=4,
                    backbone_out_indices=(0, 1, 2, 3),
                    num_attention_heads=4,
                    intermediate_size=37,
                    hidden_act="gelu",
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    is_decoder=False,
                    initializer_range=0.02,
                    is_hybrid=True,
                    backbone_config=backbone_config,
                    backbone_featmap_shape=[1, 384, 24, 24],
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
            "feature_extractor",
            "transformers.models.dpt.feature_extraction_dpt.DPTFeatureExtractor",
            "transformers.models.dpt.feature_extraction_dpt.DPTFeatureExtractor",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-DPTForDepthEstimation",
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
                "depth_estimator",
                "feature_extractor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        pt_components["depth_estimator"] = pt_components["depth_estimator"].eval()
        ms_components["depth_estimator"] = ms_components["depth_estimator"].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB").resize((32, 32))

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_stable_diffusion_depth2img_default_case(self, mode, dtype):
        if mode == ms.GRAPH_MODE:
            pytest.skip("add graph mode after ops.interpolate is fixed")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_diffusion.StableDiffusionDepth2ImgPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.stable_diffusion.StableDiffusionDepth2ImgPipeline")

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

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class StableDiffusionDepth2ImgPipelineNightlyTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_depth2img_pndm(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.stable_diffusion.StableDiffusionDepth2ImgPipeline")
        pipe = pipe_cls.from_pretrained("stabilityai/stable-diffusion-2-depth", mindspore_dtype=ms_dtype)
        pipe.set_progress_bar_config(disable=None)

        init_image = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "depth2img_input.jpg",
            subfolder="stable_diffusion_2",
        )
        prompt = "two tigers"
        n_propmt = "bad, deformed, ugly, bad anotomy"

        torch.manual_seed(0)
        image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.7)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"depth2img_pndm_{dtype}.npy",
            subfolder="stable_diffusion_2",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
