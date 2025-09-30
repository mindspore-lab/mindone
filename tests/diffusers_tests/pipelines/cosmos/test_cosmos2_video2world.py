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

from mindone.diffusers import Cosmos2VideoToWorldPipeline
from mindone.diffusers.utils.testing_utils import load_image_from_local_file, load_numpy_from_local_file, slow

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
class Cosmos2VideoToWorldPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_cosmos.CosmosTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_cosmos.CosmosTransformer3DModel",
            dict(
                in_channels=16 + 1,
                out_channels=16,
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
            "diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_wan.AutoencoderKLWan",
            dict(
                base_dim=3,
                z_dim=16,
                dim_mult=[1, 1, 1, 1],
                num_res_blocks=1,
                temperal_downsample=[False, True, True],
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(use_karras_sigmas=True),
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
        pt_pipe_cls = get_module("diffusers.pipelines.cosmos.pipeline_cosmos2_video2world.Cosmos2VideoToWorldPipeline")
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.cosmos.pipeline_cosmos2_video2world.Cosmos2VideoToWorldPipeline"
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
class Cosmos2VideoToWorldPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float16":
            pytest.skip("FP16 runs black results on both torch and mindspore")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "nvidia/Cosmos-Predict2-2B-Video2World"
        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, mindspore_dtype=ms_dtype)

        torch.manual_seed(1)
        image = load_image_from_local_file(
            "mindone-testing-arrays",
            "yellow-scrubber.png",
            subfolder="cosmos",
        )
        pipe(
            image=image,
            prompt="A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess.",  # noqa E501
            negative_prompt="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.",  # noqa E501
        )[0][0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"cosmos2_v2w_{dtype}.npy",
            subfolder="cosmos",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
