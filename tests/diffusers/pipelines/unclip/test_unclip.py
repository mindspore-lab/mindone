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
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

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
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class UnCLIPPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "prior",
            "diffusers.models.transformers.prior_transformer.PriorTransformer",
            "mindone.diffusers.models.transformers.prior_transformer.PriorTransformer",
            {
                "num_attention_heads": 2,
                "attention_head_dim": 12,
                "embedding_dim": 32,
                "num_layers": 1,
            },
        ],
        [
            "decoder",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            {
                "sample_size": 32,
                # RGB in channels
                "in_channels": 3,
                # Out channels is double in channels because predicts mean and variance
                "out_channels": 6,
                "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
                "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
                "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
                "block_out_channels": (32, 64),
                "layers_per_block": 1,
                "cross_attention_dim": 100,
                "attention_head_dim": 4,
                "resnet_time_scale_shift": "scale_shift",
                "class_embed_type": "identity",
            },
        ],
        [
            "text_proj",
            "diffusers.pipelines.unclip.UnCLIPTextProjModel",
            "mindone.diffusers.pipelines.unclip.UnCLIPTextProjModel",
            {
                "clip_embeddings_dim": 32,
                "time_embed_dim": 128,
                "cross_attention_dim": 100,
            },
        ],
        [
            "text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=32,
                    projection_dim=32,
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
        [
            "super_res_first",
            "diffusers.models.unets.unet_2d.UNet2DModel",
            "mindone.diffusers.models.unets.unet_2d.UNet2DModel",
            {
                "sample_size": 64,
                "layers_per_block": 1,
                "down_block_types": ("ResnetDownsampleBlock2D", "ResnetDownsampleBlock2D"),
                "up_block_types": ("ResnetUpsampleBlock2D", "ResnetUpsampleBlock2D"),
                "block_out_channels": (32, 64),
                "in_channels": 6,
                "out_channels": 3,
            },
        ],
        [
            "super_res_last",
            "diffusers.models.unets.unet_2d.UNet2DModel",
            "mindone.diffusers.models.unets.unet_2d.UNet2DModel",
            {
                "sample_size": 64,
                "layers_per_block": 1,
                "down_block_types": ("ResnetDownsampleBlock2D", "ResnetDownsampleBlock2D"),
                "up_block_types": ("ResnetUpsampleBlock2D", "ResnetUpsampleBlock2D"),
                "block_out_channels": (32, 64),
                "in_channels": 6,
                "out_channels": 3,
            },
        ],
        [
            "prior_scheduler",
            "diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            "mindone.diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            dict(
                variance_type="fixed_small_log",
                prediction_type="sample",
                num_train_timesteps=1000,
                clip_sample_range=5.0,
            ),
        ],
        [
            "decoder_scheduler",
            "diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            "mindone.diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            dict(
                variance_type="learned_range",
                prediction_type="epsilon",
                num_train_timesteps=1000,
            ),
        ],
        [
            "super_res_scheduler",
            "diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            "mindone.diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            dict(
                variance_type="fixed_small_log",
                prediction_type="epsilon",
                num_train_timesteps=1000,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "prior",
                "decoder",
                "text_proj",
                "text_encoder",
                "tokenizer",
                "super_res_first",
                "super_res_last",
                "prior_scheduler",
                "decoder_scheduler",
                "super_res_scheduler",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "horse",
            "prior_num_inference_steps": 2,
            "decoder_num_inference_steps": 2,
            "super_res_num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_unclip(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.unclip.UnCLIPPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.unclip.UnCLIPPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        if dtype == "float32":
            torch.manual_seed(0)
            pt_image = pt_pipe(**inputs)
            pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        else:
            # torch.flot16 requires CUDA
            # expected value depends on the version of transformers
            pt_image_slice = np.array(
                [
                    [9.9902344e-01, 2.4414062e-04, 7.0800781e-03],
                    [5.1269531e-03, 3.6621094e-03, 1.0],
                    [5.1269531e-03, 2.4414062e-04, 8.3007812e-03],
                ]
            )

        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class UnCLIPPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_unclip_karlo(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.unclip.UnCLIPPipeline")
        pipeline = pipe_cls.from_pretrained("kakaobrain/karlo-v1-alpha", mindspore_dtype=ms_dtype)
        pipeline.set_progress_bar_config(disable=None)

        torch.manual_seed(0)
        output = pipeline("horse")
        image = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"unclip_karlo_{dtype}.npy",
            subfolder="unclip",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
