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

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
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


class Dummies:
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            {
                "in_channels": 4,
                # Out channels is double in channels because predicts mean and variance
                "out_channels": 8,
                "addition_embed_type": "image",
                "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
                "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
                "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
                "block_out_channels": (32, 64),
                "layers_per_block": 1,
                "encoder_hid_dim": 32,
                "encoder_hid_dim_type": "image_proj",
                "cross_attention_dim": 32,
                "attention_head_dim": 4,
                "resnet_time_scale_shift": "scale_shift",
                "class_embed_type": None,
            },
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                num_train_timesteps=1000,
                beta_schedule="linear",
                beta_start=0.00085,
                beta_end=0.012,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type="epsilon",
                thresholding=False,
            ),
        ],
        [
            "movq",
            "diffusers.models.autoencoders.vq_model.VQModel",
            "mindone.diffusers.models.autoencoders.vq_model.VQModel",
            {
                "block_out_channels": [32, 64],
                "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
                "in_channels": 3,
                "latent_channels": 4,
                "layers_per_block": 1,
                "norm_num_groups": 8,
                "norm_type": "spatial",
                "num_vq_embeddings": 12,
                "out_channels": 3,
                "up_block_types": [
                    "AttnUpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ],
                "vq_embed_dim": 4,
            },
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "movq",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        pt_image_embeds = floats_tensor((1, 32), rng=random.Random(seed))
        pt_negative_image_embeds = floats_tensor((1, 32), rng=random.Random(seed + 1))
        ms_image_embeds = ms.Tensor(pt_image_embeds.numpy())
        ms_negative_image_embeds = ms.Tensor(pt_negative_image_embeds.numpy())

        pt_inputs = {
            "image_embeds": pt_image_embeds,
            "negative_image_embeds": pt_negative_image_embeds,
            "height": 64,
            "width": 64,
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        ms_inputs = {
            "image_embeds": ms_image_embeds,
            "negative_image_embeds": ms_negative_image_embeds,
            "height": 64,
            "width": 64,
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs


@ddt
class KandinskyV22PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        dummies = Dummies()
        return dummies.get_dummy_components()

    def get_dummy_inputs(self, seed=0):
        dummies = Dummies()
        return dummies.get_dummy_inputs(seed=seed)

    @data(*test_cases)
    @unpack
    def test_kandinsky(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky2_2.KandinskyV22Pipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky2_2.KandinskyV22Pipeline")

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
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold
