# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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
import pytest
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

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
class StableCascadeDecoderPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "decoder",
            "diffusers.models.unets.unet_stable_cascade.StableCascadeUNet",
            "mindone.diffusers.models.unets.unet_stable_cascade.StableCascadeUNet",
            {
                "in_channels": 4,
                "out_channels": 4,
                "conditioning_dim": 128,
                "block_out_channels": [16, 32, 64, 128],
                "num_attention_heads": [-1, -1, 1, 2],
                "down_num_layers_per_block": [1, 1, 1, 1],
                "up_num_layers_per_block": [1, 1, 1, 1],
                "down_blocks_repeat_mappers": [1, 1, 1, 1],
                "up_blocks_repeat_mappers": [3, 3, 2, 2],
                "block_types_per_layer": [
                    ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
                    ["SDCascadeResBlock", "SDCascadeTimestepBlock"],
                    ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
                    ["SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"],
                ],
                "switch_level": None,
                "clip_text_pooled_in_channels": 32,
                "dropout": [0.1, 0.1, 0.1, 0.1],
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
                    projection_dim=32,
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
        [
            "vqgan",
            "diffusers.pipelines.wuerstchen.PaellaVQModel",
            "mindone.diffusers.pipelines.wuerstchen.PaellaVQModel",
            {
                "bottleneck_blocks": 1,
                "num_vq_embeddings": 2,
            },
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm_wuerstchen.DDPMWuerstchenScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm_wuerstchen.DDPMWuerstchenScheduler",
            dict(),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "decoder",
                "vqgan",
                "text_encoder",
                "tokenizer",
                "scheduler",
                "latent_dim_scale",
            ]
        }
        components["latent_dim_scale"] = (4.0, 4.0)

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        eval_components = ["text_encoder", "decoder", "vqgan"]

        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        pt_image_embeddings = torch.ones((1, 4, 4, 4))
        ms_image_embeddings = ms.Tensor(pt_image_embeddings.numpy())

        pt_inputs = {
            "image_embeddings": pt_image_embeddings,
            "prompt": "horse",
            "guidance_scale": 2.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        ms_inputs = {
            "image_embeddings": ms_image_embeddings,
            "prompt": "horse",
            "guidance_scale": 2.0,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_stable_cascade_decoder(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_cascade.StableCascadeDecoderPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.stable_cascade.StableCascadeDecoderPipeline")

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
            pt_image_slice = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)
        ms_image_slice = ms_image[0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class StableCascadeDecoderPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_stable_cascade_decoder(self, mode, dtype):
        if dtype == "float16":
            pytest.skip("Skipping this case since this pipeline has precision issue in float16")

        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, max_call_depth=2000)
        else:
            ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        prior_pipe_cls = get_module("mindone.diffusers.pipelines.stable_cascade.StableCascadePriorPipeline")
        prior_pipe = prior_pipe_cls.from_pretrained("stabilityai/stable-cascade-prior", mindspore_dtype=ms_dtype)
        gen_pipe_cls = get_module("mindone.diffusers.pipelines.stable_cascade.StableCascadeDecoderPipeline")
        gen_pipe = gen_pipe_cls.from_pretrained("stabilityai/stable-cascade", mindspore_dtype=ms_dtype)

        prompt = "an image of a shiba inu, donning a spacesuit and helmet"

        torch.manual_seed(0)
        prior_output = prior_pipe(prompt)
        torch.manual_seed(0)
        image = gen_pipe(prior_output[0], prompt=prompt)[0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"stable_cascade_decoder_{dtype}.npy",
            subfolder="stable_cascade",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
