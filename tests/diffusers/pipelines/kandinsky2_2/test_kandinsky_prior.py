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
from transformers import CLIPTextConfig, CLIPVisionConfig

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


class Dummies:
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
            "image_processor",
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
            "scheduler",
            "diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            "mindone.diffusers.schedulers.scheduling_unclip.UnCLIPScheduler",
            dict(
                variance_type="fixed_small_log",
                prediction_type="sample",
                num_train_timesteps=1000,
                clip_sample=True,
                clip_sample_range=10.0,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "prior",
                "image_encoder",
                "text_encoder",
                "tokenizer",
                "scheduler",
                "image_processor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        # clip_std and clip_mean is initialized to be 0 so PriorTransformer.post_process_latents will always return 0
        # set clip_std to be 1 so it won't return 0
        pt_components["prior"].clip_std = torch.nn.Parameter(torch.ones(pt_components["prior"].clip_std.shape))
        ms_components["prior"].clip_std = ms.Parameter(
            ms.ops.ones(ms_components["prior"].clip_std.shape), name="clip_std"
        )

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "horse",
            "guidance_scale": 4.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs


@ddt
class KandinskyV22PriorPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        dummies = Dummies()
        return dummies.get_dummy_components()

    def get_dummy_inputs(self, seed=0):
        dummies = Dummies()
        return dummies.get_dummy_inputs(seed=seed)

    @data(*test_cases)
    @unpack
    def test_kandinsky_prior(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky2_2.KandinskyV22PriorPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky2_2.KandinskyV22PriorPipeline")

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

        pt_image_slice = pt_image.image_embeds[0, -10:]
        ms_image_slice = ms_image[0][0, -10:]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold
