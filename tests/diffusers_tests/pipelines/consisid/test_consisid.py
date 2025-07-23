# Copyright 2025 The HuggingFace Team.
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
import torch
from ddt import data, ddt, unpack
from PIL import Image

import mindspore as ms
from mindspore import ops

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class ConsisIDPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.ConsisIDTransformer3DModel",
            "mindone.diffusers.models.transformers.ConsisIDTransformer3DModel",
            dict(
                num_attention_heads=2,
                attention_head_dim=16,
                in_channels=8,
                out_channels=4,
                time_embed_dim=2,
                text_embed_dim=32,
                num_layers=1,
                sample_width=2,
                sample_height=2,
                sample_frames=9,
                patch_size=2,
                temporal_compression_ratio=4,
                max_text_seq_length=16,
                use_rotary_positional_embeddings=True,
                use_learned_positional_embeddings=True,
                cross_attn_interval=1,
                is_kps=False,
                is_train_face=True,
                cross_attn_dim_head=1,
                cross_attn_num_heads=1,
                LFE_id_dim=2,
                LFE_vit_dim=2,
                LFE_depth=5,
                LFE_dim_head=8,
                LFE_num_heads=2,
                LFE_num_id_token=1,
                LFE_num_querie=1,
                LFE_output_dim=21,
                LFE_ff_mult=1,
                LFE_num_scale=1,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.AutoencoderKLCogVideoX",
            "mindone.diffusers.models.autoencoders.AutoencoderKLCogVideoX",
            dict(
                in_channels=3,
                out_channels=3,
                down_block_types=(
                    "CogVideoXDownBlock3D",
                    "CogVideoXDownBlock3D",
                    "CogVideoXDownBlock3D",
                    "CogVideoXDownBlock3D",
                ),
                up_block_types=(
                    "CogVideoXUpBlock3D",
                    "CogVideoXUpBlock3D",
                    "CogVideoXUpBlock3D",
                    "CogVideoXUpBlock3D",
                ),
                block_out_channels=(8, 8, 8, 8),
                latent_channels=4,
                layers_per_block=1,
                norm_num_groups=2,
                temporal_compression_ratio=4,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.DDIMScheduler",
            "mindone.diffusers.schedulers.DDIMScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
            ),
        ],
        [
            "tokenizer",
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
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, dtype):
        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)

        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))

        pt_inputs = {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": image_height,
            "width": image_width,
            "num_frames": 8,
            "max_sequence_length": 16,
            "id_vit_hidden": [torch.ones([1, 2, 2], dtype=pt_dtype)] * 1,
            "id_cond": torch.ones(1, 2, dtype=pt_dtype),
            "output_type": "np",
        }

        ms_inputs = {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": image_height,
            "width": image_width,
            "num_frames": 8,
            "max_sequence_length": 16,
            "id_vit_hidden": [ops.ones([1, 2, 2], dtype=ms_dtype)] * 1,
            "id_cond": ops.ones((1, 2), dtype=ms_dtype),
            "output_type": "np",
        }
        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.ConsisIDPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.ConsisIDPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs(dtype=dtype)

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs).frames
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)[0]

        pt_generated_image = pt_image[0]
        ms_generated_image = ms_image[0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert (
            np.max(np.linalg.norm(pt_generated_image - ms_generated_image) / np.linalg.norm(pt_generated_image))
            < threshold
        )
