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
import pytest
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
class WuerstchenCombinedPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "prior_prior",
            "diffusers.pipelines.wuerstchen.WuerstchenPrior",
            "mindone.diffusers.pipelines.wuerstchen.WuerstchenPrior",
            {"c_in": 2, "c": 8, "depth": 2, "c_cond": 32, "c_r": 8, "nhead": 2},
        ],
        [
            "prior_text_encoder",
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
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm_wuerstchen.DDPMWuerstchenScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm_wuerstchen.DDPMWuerstchenScheduler",
            dict(),
        ],
        [
            "prior_scheduler",
            "diffusers.schedulers.scheduling_ddpm_wuerstchen.DDPMWuerstchenScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm_wuerstchen.DDPMWuerstchenScheduler",
            dict(),
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
            "prior_tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
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
            "decoder",
            "diffusers.pipelines.wuerstchen.WuerstchenDiffNeXt",
            "mindone.diffusers.pipelines.wuerstchen.WuerstchenDiffNeXt",
            {
                "c_cond": 32,
                "c_hidden": [320],
                "nhead": [-1],
                "blocks": [4],
                "level_config": ["CT"],
                "clip_embd": 32,
                "inject_effnet": [False],
            },
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
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "tokenizer",
                "text_encoder",
                "decoder",
                "vqgan",
                "scheduler",
                "prior_prior",
                "prior_text_encoder",
                "prior_tokenizer",
                "prior_scheduler",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        eval_components = ["text_encoder", "decoder", "vqgan", "prior_prior", "prior_text_encoder"]

        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "horse",
            "prior_guidance_scale": 4.0,
            "decoder_guidance_scale": 4.0,
            "num_inference_steps": 2,
            "prior_num_inference_steps": 2,
            "output_type": "np",
            "height": 128,
            "width": 128,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_wuerstchen(self, mode, dtype):
        # If strict mode is disabled, this pipeline will throw an error.
        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        else:
            ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.wuerstchen.WuerstchenCombinedPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.wuerstchen.WuerstchenCombinedPipeline")

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
            pt_image_slice = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.49951172], [0.0, 0.0, 1.0]])

        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)
        ms_image_slice = ms_image[0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class WuerstchenCombinedPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_wuerstchen(self, mode, dtype):
        if dtype == "float16":
            pytest.skip("Skipping this case since this pipeline has precision issue in float16")

        # If strict mode is disabled, this pipeline will throw an error.
        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        else:
            ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.wuerstchen.WuerstchenCombinedPipeline")
        pipe = pipe_cls.from_pretrained("warp-ai/wuerstchen", mindspore_dtype=ms_dtype)

        prompt = "an image of a shiba inu, donning a spacesuit and helmet"

        torch.manual_seed(0)
        image = pipe(prompt=prompt)[0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"wuerstchen_combined_{dtype}.npy",
            subfolder="wuerstchen",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
