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
from transformers import T5Config

import mindspore as ms

from mindone.diffusers import AllegroPipeline
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
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@ddt
class AllegroPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_allegro.AllegroTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_allegro.AllegroTransformer3DModel",
            dict(
                num_attention_heads=2,
                attention_head_dim=12,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                cross_attention_dim=24,
                sample_width=8,
                sample_height=8,
                sample_frames=8,
                caption_channels=24,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_allegro.AutoencoderKLAllegro",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_allegro.AutoencoderKLAllegro",
            dict(
                in_channels=3,
                out_channels=3,
                down_block_types=(
                    "AllegroDownBlock3D",
                    "AllegroDownBlock3D",
                    "AllegroDownBlock3D",
                    "AllegroDownBlock3D",
                ),
                up_block_types=(
                    "AllegroUpBlock3D",
                    "AllegroUpBlock3D",
                    "AllegroUpBlock3D",
                    "AllegroUpBlock3D",
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
                config=T5Config(
                    d_ff=37,
                    d_kv=8,
                    d_model=24,
                    num_decoder_layers=2,
                    num_heads=4,
                    num_layers=2,
                    relative_attention_num_buckets=8,
                    vocab_size=1103,
                ),
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
        components = {key: None for key in ["transformer", "vae", "scheduler", "text_encoder", "tokenizer"]}

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
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
        pt_pipe_cls = get_module("diffusers.pipelines.allegro.AllegroPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.allegro.AllegroPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_pipe.enable_vae_tiling()
        ms_pipe.enable_vae_tiling()

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
class AllegroPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_allegro(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = AllegroPipeline.from_pretrained("rhymes-ai/Allegro", mindspore_dtype=ms_dtype)
        pipe.vae.enable_tiling()

        prompt = "A painting of a squirrel eating a burger."
        torch.manual_seed(0)
        video = pipe(
            prompt=prompt,
            height=720,
            width=1280,
            num_frames=88,
            num_inference_steps=2,
        )[
            0
        ][0]

        expected_video = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"t2v_{dtype}.npy",
            subfolder="allegro",
        )
        assert np.linalg.norm(expected_video - video) / np.linalg.norm(expected_video) < THRESHOLD_PIXEL
