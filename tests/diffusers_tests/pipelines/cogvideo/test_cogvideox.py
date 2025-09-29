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

import mindspore as ms

from mindone.diffusers import CogVideoXPipeline
from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

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
class CogVideoXPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel",
            "mindone.diffusers.models.transformers.cogvideox_transformer_3d.CogVideoXTransformer3DModel",
            dict(
                # Product of num_attention_heads * attention_head_dim must be divisible by 16 for 3D positional embeddings
                # But, since we are using tiny-random-t5 here, we need the internal dim of CogVideoXTransformer3DModel
                # to be 32. The internal dim is product of num_attention_heads and attention_head_dim
                num_attention_heads=4,
                attention_head_dim=8,
                in_channels=4,
                out_channels=4,
                time_embed_dim=2,
                text_embed_dim=32,  # Must match with tiny-random-t5
                num_layers=1,
                sample_width=2,  # latent width: 2 -> final width: 16
                sample_height=2,  # latent height: 2 -> final height: 16
                sample_frames=9,  # latent frames: (9 - 1) / 4 + 1 = 3 -> final frames: 9
                patch_size=2,
                temporal_compression_ratio=4,
                max_text_seq_length=16,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_cogvideox.AutoencoderKLCogVideoX",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_cogvideox.AutoencoderKLCogVideoX",
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
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
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

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            # Cannot reduce because convolution kernel becomes bigger than sample
            "height": 16,
            "width": 16,
            "num_frames": 8,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_video = pt_pipe(**inputs).frames
        torch.manual_seed(0)
        ms_video = ms_pipe(**inputs)[0]

        pt_generated_video = pt_video[0]
        ms_generated_video = ms_video[0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert (
            np.max(np.linalg.norm(pt_generated_video - ms_generated_video) / np.linalg.norm(pt_generated_video))
            < threshold
        )


@slow
@ddt
class CogVideoXPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_cogvideox(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", mindspore_dtype=ms_dtype)

        prompt = "A painting of a squirrel eating a burger."

        torch.manual_seed(0)
        video = pipe(
            prompt=prompt,
            height=480,
            width=720,
            num_frames=16,
            num_inference_steps=2,
            output_type="np",
        )[
            0
        ][0]

        expected_video = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"cogvideo_t2v_{dtype}.npy",
            subfolder="cogvideo",
        )
        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(expected_video - video) / np.linalg.norm(expected_video)) < threshold
