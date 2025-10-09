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

import mindspore as ms

from mindone.diffusers import LTXImageToVideoPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

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
class LTXImageToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_ltx.LTXVideoTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_ltx.LTXVideoTransformer3DModel",
            dict(
                in_channels=8,
                out_channels=8,
                patch_size=1,
                patch_size_t=1,
                num_attention_heads=4,
                attention_head_dim=8,
                cross_attention_dim=32,
                num_layers=1,
                caption_channels=32,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=8,
                block_out_channels=(8, 8, 8, 8),
                decoder_block_out_channels=(8, 8, 8, 8),
                layers_per_block=(1, 1, 1, 1, 1),
                decoder_layers_per_block=(1, 1, 1, 1, 1),
                spatio_temporal_scaling=(True, True, False, False),
                decoder_spatio_temporal_scaling=(True, True, False, False),
                decoder_inject_noise=(False, False, False, False, False),
                upsample_residual=(False, False, False, False),
                upsample_factor=(1, 1, 1, 1),
                timestep_conditioning=False,
                patch_size=1,
                patch_size_t=1,
                encoder_causal=True,
                decoder_causal=False,
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

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        pt_components["vae"].use_framewise_encoding = False
        ms_components["vae"].use_framewise_encoding = False
        pt_components["vae"].use_framewise_decoding = False
        ms_components["vae"].use_framewise_decoding = False

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        pt_image = torch.randn((1, 3, 32, 32))
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "image": pt_image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            # 8 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        ms_inputs = {
            "image": ms_image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            # 8 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip(" FP32 is not supported since LTXImageToVideoPipeline contains nn.Conv3d")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.ltx.LTXImageToVideoPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.ltx.LTXImageToVideoPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.frames[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@ddt
@slow
class LTXImageToVideoPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self):
        image = load_downloaded_image_from_hf_hub(
            "a-r-r-o-w/tiny-meme-dataset-captioned",
            filename="8.png",
            subfolder="images",
            repo_type="dataset",
        )

        inputs = {
            "image": image,
            "prompt": "A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background. Flames engulf the structure, with smoke billowing into the air. Firefighters in protective gear rush to the scene, a fire truck labeled '38' visible behind them. The girl's neutral expression contrasts sharply with the chaos of the fire, creating a poignant and emotionally charged scene.",  # noqa: E501
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "width": 704,
            "height": 480,
            "num_frames": 161,
            "num_inference_steps": 50,
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", mindspore_dtype=ms_dtype)
        inputs = self.get_inputs()
        torch.manual_seed(0)
        image = pipe(**inputs)[0][0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"ltx_image2video_{dtype}.npy",
            subfolder="ltx",
        )

        assert (
            np.mean(np.abs(np.array(image, dtype=np.float32) - np.array(expected_image, dtype=np.float32)))
            < THRESHOLD_PIXEL
        )
