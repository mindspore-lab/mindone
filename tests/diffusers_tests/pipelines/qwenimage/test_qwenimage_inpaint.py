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

import random
import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from transformers import Qwen2_5_VLConfig

import mindspore as ms

from mindone.diffusers import QwenImageInpaintPipeline
from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class QwenImageInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_qwenimage.QwenImageTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_qwenimage.QwenImageTransformer2DModel",
            dict(
                patch_size=2,
                in_channels=16,
                out_channels=4,
                num_layers=2,
                attention_head_dim=16,
                num_attention_heads=3,
                joint_attention_dim=16,
                guidance_embeds=False,
                axes_dims_rope=(8, 4, 4),
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_qwenimage.AutoencoderKLQwenImage",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_qwenimage.AutoencoderKLQwenImage",
            dict(
                base_dim=4 * 6,
                z_dim=4,
                dim_mult=[1, 2, 4],
                num_res_blocks=1,
                temperal_downsample=[False, True],
                # fmt: off
                latents_mean=[0.0] * 4,
                latents_std=[1.0] * 4,
                # fmt: on
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
            "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration",
            "mindone.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration",
            dict(
                config=Qwen2_5_VLConfig(
                    text_config={
                        "hidden_size": 16,
                        "intermediate_size": 16,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 2,
                        "rope_scaling": {
                            "mrope_section": [1, 1, 2],
                            "rope_type": "default",
                            "type": "default",
                        },
                        "rope_theta": 1000000.0,
                    },
                    vision_config={
                        "depth": 2,
                        "hidden_size": 16,
                        "intermediate_size": 16,
                        "num_heads": 2,
                        "out_hidden_size": 16,
                    },
                    attention_dropout=0.0,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    num_key_value_heads=2,
                    rms_norm_eps=1e-06,
                    max_position_embeddings=128000,
                    hidden_size=16,
                    hidden_act="silu",
                    intermediate_size=16,
                    initializer_range=0.02,
                    vocab_size=152064,
                    vision_end_token_id=151653,
                    vision_start_token_id=151652,
                    vision_token_id=151654,
                    sliding_window=32768,  # None
                    use_sliding_window=False,
                    use_cache=True,
                    attn_implementation="eager",
                    rope_scaling={
                        "mrope_section": [1, 1, 2],
                        "rope_type": "default",
                        "type": "default",
                    },
                    rope_theta=1000000.0,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
            "transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration",
                trust_remote_code=True,
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

    def get_dummy_inputs(self, seed=0):
        pt_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        pt_mask_image = torch.ones((1, 1, 32, 32))
        ms_image = ms.Tensor(pt_image.numpy())
        ms_mask_image = ms.mint.ones((1, 1, 32, 32))

        pt_inputs = {
            "image": pt_image,
            "mask_image": pt_mask_image,
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "true_cfg_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        ms_inputs = {
            "image": ms_image,
            "mask_image": ms_mask_image,
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "true_cfg_scale": 1.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.qwenimage.QwenImageInpaintPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.qwenimage.QwenImageInpaintPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs).images
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)[0]

        pt_generated_image = pt_image[0]
        ms_generated_image = ms_image[0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert (
            np.max(np.linalg.norm(pt_generated_image - ms_generated_image) / np.linalg.norm(pt_generated_image))
            < threshold
        )


@slow
@ddt
class QwenImageInpaintPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline will OOM in float32")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "Qwen/Qwen-Image"
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(0))  # load given image
        mask_image = ms.mint.ones((1, 1, 32, 32))

        pipe = QwenImageInpaintPipeline.from_pretrained(model_id, mindspore_dtype=ms_dtype)

        pipe.vae.enable_tiling()

        torch.manual_seed(0)
        image = pipe(
            image=ms.Tensor(image.numpy()),
            mask_image=mask_image,
            prompt="dance monkey",
            negative_prompt="bad quality",
        )[0][0]

        # The text_coder causes deviations between ms and pt versions. However, the deviation\
        # is within THRESHOLD_PIXEL when using the same intermediate results of text_encoder.
        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"qwenimage_inpaint_{dtype}.npy",
            subfolder="qwenimage",
        )

        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
