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
import pytest
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig, LlamaConfig

import mindspore as ms

from mindone.diffusers import HunyuanVideoPipeline
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
]


@ddt
class HunyuanVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_hunyuan_video.HunyuanVideoTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_hunyuan_video.HunyuanVideoTransformer3DModel",
            dict(
                in_channels=4,
                out_channels=4,
                num_attention_heads=2,
                attention_head_dim=10,
                num_layers=1,
                num_single_layers=1,
                num_refiner_layers=1,
                patch_size=1,
                patch_size_t=1,
                guidance_embeds=True,
                text_embed_dim=16,
                pooled_projection_dim=8,
                rope_axes_dim=(2, 4, 4),
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_hunyuan_video.AutoencoderKLHunyuanVideo",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_hunyuan_video.AutoencoderKLHunyuanVideo",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                down_block_types=(
                    "HunyuanVideoDownBlock3D",
                    "HunyuanVideoDownBlock3D",
                    "HunyuanVideoDownBlock3D",
                    "HunyuanVideoDownBlock3D",
                ),
                up_block_types=(
                    "HunyuanVideoUpBlock3D",
                    "HunyuanVideoUpBlock3D",
                    "HunyuanVideoUpBlock3D",
                    "HunyuanVideoUpBlock3D",
                ),
                block_out_channels=(8, 8, 8, 8),
                layers_per_block=1,
                act_fn="silu",
                norm_num_groups=4,
                scaling_factor=0.476986,
                spatial_compression_ratio=8,
                temporal_compression_ratio=4,
                mid_block_add_attention=True,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(shift=7.0),
        ],
        [
            "text_encoder",
            "transformers.models.llama.modeling_llama.LlamaModel",
            "mindone.transformers.models.llama.modeling_llama.LlamaModel",
            dict(
                config=LlamaConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=16,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    pad_token_id=1,
                    vocab_size=1000,
                    hidden_act="gelu",
                    projection_dim=32,
                    _attn_implementation="eager",
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.llama.tokenization_llama.LlamaTokenizer",
            "transformers.models.llama.tokenization_llama.LlamaTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
            ),
        ],
        [
            "text_encoder_2",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=8,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    pad_token_id=1,
                    vocab_size=1000,
                    hidden_act="gelu",
                    projection_dim=32,
                ),
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
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
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        # Cannot test with dummy prompt because tokenizers are not configured correctly.
        # TODO(aryan): create dummy tokenizers and using from hub
        inputs = {
            "prompt": "",
            "prompt_template": {
                "template": "{}",
                "crop_start": 0,
            },
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": 16,
            "width": 16,
            # 4 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip(" FP32 is not supported since HunyuanVideoPipeline contains nn.Conv3d")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.hunyuan_video.HunyuanVideoPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.hunyuan_video.HunyuanVideoPipeline")

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
            pt_image_slice = pt_image.frames[0][0, -3:, -3:, -1]
        else:
            # torch.float16 requires CUDA
            pt_image_slice = np.array(
                [
                    [0.55126953, 0.5209961, 0.5390625],
                    [0.4765625, 0.4404297, 0.43408203],
                    [0.48754883, 0.47485352, 0.46069336],
                ]
            )

        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)
        ms_image_slice = ms_image[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class HunyuanVideoPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip(" FP32 is not supported since HunyuanVideoPipeline contains nn.Conv3d")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "hunyuanvideo-community/HunyuanVideo"
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, mindspore_dtype=ms_dtype)
        # transformer_hunyuan_video only support bf16
        pipe.transformer.to(ms.bfloat16)
        pipe.vae.enable_tiling()

        torch.manual_seed(0)
        image = pipe(
            prompt="A cat walks on the grass, realistic",
            height=320,
            width=512,
            num_frames=61,
            num_inference_steps=30,
        )[0][0][1]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"hunyuan_video_t2v_{dtype}.npy",
            subfolder="hunyuan_video",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
