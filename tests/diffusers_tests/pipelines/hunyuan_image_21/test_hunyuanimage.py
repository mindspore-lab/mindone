# Copyright 2025 The HuggingFace Team.
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
from transformers import Qwen2_5_VLConfig, T5Config

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": 0, "dtype": "bfloat16"},
    {"mode": 1, "dtype": "bfloat16"},
]


@ddt
class HunyuanImagePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_hunyuanimage.HunyuanImageTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_hunyuanimage.HunyuanImageTransformer2DModel",
            dict(
                in_channels=4,
                out_channels=4,
                num_attention_heads=4,
                attention_head_dim=8,
                num_layers=1,
                num_single_layers=1,
                num_refiner_layers=1,
                patch_size=(1, 1),
                guidance_embeds=False,
                text_embed_dim=32,
                text_embed_2_dim=32,
                rope_axes_dim=(4, 4),
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_hunyuanimage.AutoencoderKLHunyuanImage",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_hunyuanimage.AutoencoderKLHunyuanImage",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                block_out_channels=(32, 64, 64, 64),
                layers_per_block=1,
                scaling_factor=0.476986,
                spatial_compression_ratio=8,
                sample_size=128,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(shift=7.0),
        ],
        [
            "guider",
            "diffusers.guiders.adaptive_projected_guidance_mix.AdaptiveProjectedMixGuidance",
            "mindone.diffusers.guiders.adaptive_projected_guidance_mix.AdaptiveProjectedMixGuidance",
            dict(adaptive_projected_guidance_start_step=2),
        ],
        [
            "ocr_guider",
            "diffusers.guiders.adaptive_projected_guidance_mix.AdaptiveProjectedMixGuidance",
            "mindone.diffusers.guiders.adaptive_projected_guidance_mix.AdaptiveProjectedMixGuidance",
            dict(adaptive_projected_guidance_start_step=3),
        ],
        [
            "text_encoder",
            "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration",
            "mindone.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration",
            dict(
                config=Qwen2_5_VLConfig(
                    text_config={
                        "hidden_size": 32,
                        "intermediate_size": 32,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 2,
                        "rope_scaling": {
                            "mrope_section": [2, 2, 4],
                            "rope_type": "default",
                            "type": "default",
                        },
                        "rope_theta": 1000000.0,
                    },
                    vision_config={
                        "depth": 2,
                        "hidden_size": 32,
                        "intermediate_size": 32,
                        "num_heads": 2,
                        "out_hidden_size": 32,
                    },
                    hidden_size=32,
                    vocab_size=152064,
                    vision_end_token_id=151653,
                    vision_start_token_id=151652,
                    vision_token_id=151654,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration",
            ),
        ],
        [
            "text_encoder_2",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                config=T5Config(
                    d_model=32,
                    d_kv=4,
                    d_ff=16,
                    num_layers=2,
                    num_heads=2,
                    relative_attention_num_buckets=8,
                    relative_attention_max_distance=32,
                    vocab_size=256,
                    feed_forward_proj="gated-gelu",
                    dense_act_fn="gelu_new",
                    is_encoder_decoder=False,
                    use_cache=False,
                    tie_word_embeddings=False,
                ),
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.byt5.tokenization_byt5.ByT5Tokenizer",
            "transformers.models.byt5.tokenization_byt5.ByT5Tokenizer",
            dict(),
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
                "guider",
                "ocr_guider",
            ]
        }
        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 5,
            "height": 16,
            "width": 16,
            "output_type": "np",
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage.HunyuanImagePipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage.HunyuanImagePipeline")

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
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:].flatten()
        ms_image_slice = ms_image[0][0, -3:, -3:].flatten()

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold
