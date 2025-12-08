# Copyright 2024 Bria AI and The HuggingFace Team. All rights reserved.
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
from transformers import SmolLM3Config

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": 1, "dtype": "bfloat16"},
    {"mode": 1, "dtype": "float16"},
    {"mode": 0, "dtype": "bfloat16"},
    {"mode": 0, "dtype": "float16"},
]


@ddt
class BriaFiboPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_bria_fibo.BriaFiboTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_bria_fibo.BriaFiboTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=16,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=8,
                num_attention_heads=2,
                joint_attention_dim=64,
                text_encoder_dim=32,
                pooled_projection_dim=None,
                axes_dims_rope=[0, 4, 4],
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan",
            dict(
                base_dim=160,
                decoder_base_dim=256,
                num_res_blocks=2,
                out_channels=12,
                patch_size=2,
                scale_factor_spatial=16,
                scale_factor_temporal=4,
                temperal_downsample=[False, True, True],
                z_dim=16,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.smollm3.modeling_smollm3.SmolLM3ForCausalLM",
            "mindone.transformers.models.smollm3.modeling_smollm3.SmolLM3ForCausalLM",
            dict(config=SmolLM3Config(hidden_size=32)),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "vae",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "{'text': 'A painting of a squirrel eating a burger'}",
            "negative_prompt": "bad, ugly",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        """Test basic inference in animation mode."""
        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.bria_fibo.BriaFiboPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.bria_fibo.BriaFiboPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        if mode == 0:
            ms_pipe.transformer.construct = ms.jit(ms_pipe.transformer.construct)

        torch.manual_seed(0)
        pt_output = pt_pipe(**inputs).images[0]
        torch.manual_seed(0)
        ms_output = ms_pipe(**inputs).images[0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_output - ms_output) / np.linalg.norm(pt_output) < threshold
