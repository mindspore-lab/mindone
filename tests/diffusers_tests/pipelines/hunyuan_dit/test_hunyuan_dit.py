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
class HunyuanDiTPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel",
            "mindone.diffusers.models.transformers.hunyuan_transformer_2d.HunyuanDiT2DModel",
            dict(
                sample_size=16,
                num_layers=2,
                patch_size=2,
                attention_head_dim=8,
                num_attention_heads=3,
                in_channels=4,
                cross_attention_dim=32,
                cross_attention_dim_t5=32,
                pooled_projection_dim=16,
                hidden_size=24,
                activation_fn="gelu-approximate",
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.bert.modeling_bert.BertModel",
            "mindone.transformers.models.bert.modeling_bert.BertModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-BertModel",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-BertModel",
            ),
        ],
        [
            "text_encoder_2",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "text_encoder",
                "tokenizer",
                "text_encoder_2",
                "tokenizer_2",
                "safety_checker",
                "feature_extractor",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        eval_components = ["transformer", "vae"]

        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "use_resolution_binning": False,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        # If strict mode is disabled, this pipeline will have precision issue.
        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        else:
            ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.hunyuandit.HunyuanDiTPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.hunyuandit.HunyuanDiTPipeline")

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
class HunyuanDiTPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    prompt = "一个宇航员在骑马"

    @data(*test_cases)
    @unpack
    def test_hunyuan_dit_1024(self, mode, dtype):
        # If strict mode is disabled, this pipeline will have precision issue.
        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        else:
            ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.hunyuandit.HunyuanDiTPipeline")
        pipe = pipe_cls.from_pretrained("Tencent-Hunyuan/HunyuanDiT-Diffusers", mindspore_dtype=ms_dtype)
        prompt = self.prompt

        torch.manual_seed(0)
        image = pipe(prompt=prompt, height=1024, width=1024, num_inference_steps=2)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"t2i_1024_{dtype}.npy",
            subfolder="hunyuan_dit",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
