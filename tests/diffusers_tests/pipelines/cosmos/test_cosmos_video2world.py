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
import PIL.Image
import pytest
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import CosmosVideoToWorldPipeline
from mindone.diffusers.utils.testing_utils import (  # noqa F401
    load_image_from_local_file,
    load_numpy_from_local_file,
    slow,
)

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)
from .cosmos_guardrail import DummyCosmosSafetyChecker, MsDummyCosmosSafetyChecker

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class CosmosVideoToWorldPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_cosmos.CosmosTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_cosmos.CosmosTransformer3DModel",
            dict(
                in_channels=4 + 1,
                out_channels=4,
                num_attention_heads=2,
                attention_head_dim=16,
                num_layers=2,
                mlp_ratio=2,
                text_embed_dim=32,
                adaln_lora_dim=4,
                max_size=(4, 32, 32),
                patch_size=(1, 2, 2),
                rope_scale=(2.0, 1.0, 1.0),
                concat_padding_mask=True,
                extra_pos_embed_type="learnable",
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.AutoencoderKLCosmos",
            "mindone.diffusers.models.autoencoders.AutoencoderKLCosmos",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                encoder_block_out_channels=(8, 8, 8, 8),
                decode_block_out_channels=(8, 8, 8, 8),
                attention_resolutions=(8,),
                resolution=64,
                num_layers=2,
                patch_size=4,
                patch_type="haar",
                scaling_factor=1.0,
                spatial_compression_ratio=4,
                temporal_compression_ratio=4,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler",
            "mindone.diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler",
            dict(
                sigma_min=0.002,
                sigma_max=80,
                sigma_data=0.5,
                sigma_schedule="karras",
                num_train_timesteps=1000,
                prediction_type="epsilon",
                rho=7.0,
                final_sigmas_type="sigma_min",
            ),
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
        pt_components["safety_checker"] = DummyCosmosSafetyChecker()
        ms_components["safety_checker"] = MsDummyCosmosSafetyChecker()
        return pt_components, ms_components

    def get_dummy_inputs(self):
        image_height = 32
        image_width = 32
        image = PIL.Image.new("RGB", (image_width, image_height))

        inputs = {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "bad quality",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": image_height,
            "width": image_width,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.cosmos.pipeline_cosmos_video2world.CosmosVideoToWorldPipeline")
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.cosmos.pipeline_cosmos_video2world.CosmosVideoToWorldPipeline"
        )

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
class CosmosVideoToWorldPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float16":
            pytest.skip("FP16 runs black results on both torch and mindspore")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "nvidia/Cosmos-1.0-Diffusion-7B-Video2World"
        prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day."  # noqa E501
        image = load_image_from_local_file(
            "mindone-testing-arrays",
            "cosmos-video2world-input.jpg",
            subfolder="cosmos",
        )
        pipe = CosmosVideoToWorldPipeline.from_pretrained(model_id, mindspore_dtype=ms_dtype)

        torch.manual_seed(0)
        frame = pipe(prompt=prompt, image=image)[0][0][0]

        expected_frame = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"cosmos_v2w_{dtype}.npy",
            subfolder="cosmos",
        )
        assert np.mean(np.abs(np.array(frame, dtype=np.float32) - expected_frame)) < THRESHOLD_PIXEL
