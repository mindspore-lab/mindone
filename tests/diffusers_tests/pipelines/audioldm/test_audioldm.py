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

import diffusers
import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from packaging.version import Version
from transformers import ClapTextConfig, SpeechT5HifiGanConfig

import mindspore as ms

from mindone.diffusers import AudioLDMPipeline
from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    AUDIO_THRESHOLD_FP16,
    AUDIO_THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@ddt
class AudioLDMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(8, 16),
                layers_per_block=1,
                norm_num_groups=8,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=(8, 16),
                class_embed_type="simple_projection",
                projection_class_embeddings_input_dim=8,
                class_embeddings_concat=True,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[8, 16],
                in_channels=1,
                out_channels=1,
                norm_num_groups=8,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
            ),
        ],
        [
            "text_encoder",
            "transformers.models.clap.modeling_clap.ClapTextModelWithProjection",
            "mindone.transformers.models.clap.modeling_clap.ClapTextModelWithProjection",
            dict(
                config=ClapTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=8,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=1,
                    num_hidden_layers=1,
                    pad_token_id=1,
                    vocab_size=1000,
                    projection_dim=8,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.roberta.tokenization_roberta.RobertaTokenizer",
            "transformers.models.roberta.tokenization_roberta.RobertaTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-roberta",
                model_max_length=77,
            ),
        ],
        [
            "vocoder",
            "transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan",
            "mindone.transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan",
            dict(
                config=SpeechT5HifiGanConfig(
                    model_in_dim=8,
                    sampling_rate=16000,
                    upsample_initial_channel=16,
                    upsample_rates=[2, 2],
                    upsample_kernel_sizes=[4, 4],
                    resblock_kernel_sizes=[3, 7],
                    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
                    normalize_before=False,
                ),
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "vocoder",
            ]
        }
        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_audioldm(self, mode, dtype):
        last_supported_version = Version("0.33.1")
        current_version = Version(diffusers.__version__)
        if current_version > last_supported_version:
            pytest.skip(f"AudioLDMPipeline is not supported in diffusers version {current_version}")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.audioldm.pipeline_audioldm.AudioLDMPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_audio = pt_pipe(**inputs)[0][0]
        torch.manual_seed(0)
        ms_audio = ms_pipe(**inputs)[0][0]

        threshold = AUDIO_THRESHOLD_FP32 if dtype == "float32" else AUDIO_THRESHOLD_FP16
        assert np.linalg.norm(pt_audio - ms_audio) / np.linalg.norm(pt_audio) < threshold


@slow
@ddt
class AudioLDMPipelineNightlyTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_audioldm(self, mode, dtype):
        last_supported_version = Version("0.33.1")
        current_version = Version(diffusers.__version__)
        if current_version > last_supported_version:
            pytest.skip(f"AudioLDMPipeline is not supported in diffusers version {current_version}")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        audioldm_pipe = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", mindspore_dtype=ms_dtype)
        audioldm_pipe.set_progress_bar_config(disable=None)

        prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
        torch.manual_seed(0)
        audio = audioldm_pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0)[0][0]

        expected_audio = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"t2a_{dtype}.npy",
            subfolder="audioldm",
        )
        threshold = AUDIO_THRESHOLD_FP32 if dtype == "float32" else AUDIO_THRESHOLD_FP16
        assert np.linalg.norm(expected_audio - audio) / np.linalg.norm(expected_audio) < threshold
