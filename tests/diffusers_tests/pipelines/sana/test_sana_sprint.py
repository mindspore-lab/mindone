# Copyright 2024 The HuggingFace Team.
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
from transformers import Gemma2Config

import mindspore as ms

from mindone.diffusers import SanaSprintPipeline
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
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "bfloat16"},
]


@ddt
class SanaSprintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.sana_transformer.SanaTransformer2DModel",
            "mindone.diffusers.models.transformers.sana_transformer.SanaTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                num_attention_heads=2,
                attention_head_dim=4,
                num_cross_attention_heads=2,
                cross_attention_head_dim=4,
                cross_attention_dim=8,
                caption_channels=8,
                sample_size=32,
                qk_norm="rms_norm_across_heads",
                guidance_embeds=True,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_dc.AutoencoderDC",
            "mindone.diffusers.models.autoencoders.autoencoder_dc.AutoencoderDC",
            dict(
                in_channels=3,
                latent_channels=4,
                attention_head_dim=2,
                encoder_block_types=(
                    "ResBlock",
                    "EfficientViTBlock",
                ),
                decoder_block_types=(
                    "ResBlock",
                    "EfficientViTBlock",
                ),
                encoder_block_out_channels=(8, 8),
                decoder_block_out_channels=(8, 8),
                encoder_qkv_multiscales=((), (5,)),
                decoder_qkv_multiscales=((), (5,)),
                encoder_layers_per_block=(1, 1),
                decoder_layers_per_block=[1, 1],
                downsample_block_type="conv",
                upsample_block_type="interpolate",
                decoder_norm_types="rms_norm",
                decoder_act_fns="silu",
                scaling_factor=0.41407,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_scm.SCMScheduler",
            "mindone.diffusers.schedulers.scheduling_scm.SCMScheduler",
            dict(),
        ],
        [
            "text_encoder",
            "transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            "mindone.transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            dict(
                config=Gemma2Config(
                    head_dim=16,
                    hidden_size=8,
                    initializer_range=0.02,
                    intermediate_size=64,
                    max_position_embeddings=8192,
                    model_type="gemma2",
                    num_attention_heads=2,
                    num_hidden_layers=1,
                    num_key_value_heads=2,
                    vocab_size=8,
                    attn_implementation="eager",
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer",
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/dummy-gemma",
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

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
            "complex_human_instruction": None,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.sana.pipeline_sana_sprint.SanaSprintPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.sana.pipeline_sana_sprint.SanaSprintPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(42)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(42)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class SanaSprintPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_sana_sprint(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = SanaSprintPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers", mindspore_dtype=ms_dtype
        )

        torch.manual_seed(42)
        image = pipe(prompt="a tiny astronaut hatching from an egg on the moon")[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"sprint_{dtype}.npy",
            subfolder="sana",
        )
        assert np.linalg.norm(expected_image - image) / np.linalg.norm(expected_image) < THRESHOLD_PIXEL
