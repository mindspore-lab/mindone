# Copyright 2024 Bria AI and The HuggingFace Team. All rights reserved.
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

import mindspore as ms

from mindone.diffusers import BriaPipeline
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
class BriaPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_bria.BriaTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_bria.BriaTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=16,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=8,
                num_attention_heads=2,
                joint_attention_dim=32,
                pooled_projection_dim=None,
                axes_dims_rope=[0, 4, 4],
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                act_fn="silu",
                block_out_channels=(32,),
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D"],
                latent_channels=4,
                sample_size=32,
                shift_factor=0,
                scaling_factor=0.13025,
                use_post_quant_conv=True,
                use_quant_conv=True,
                force_upcast=False,
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
            "transformers.models.t5.tokenization_t5_fast.T5TokenizerFast",
            "transformers.models.t5.tokenization_t5_fast.T5TokenizerFast",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "vae",
                "image_encoder",
                "feature_extractor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "negative_prompt": "bad, ugly",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_bria(self, mode, dtype):
        required_version = Version("0.35.2")
        current_version = Version(diffusers.__version__)
        if current_version <= required_version:
            pytest.skip(f"BriaPipeline is not supported in diffusers version {current_version}")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.bria.pipeline_bria.BriaPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.bria.pipeline_bria.BriaPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(1)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(1)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, 0]
        ms_image_slice = ms_image[0][0, -3:, -3:, 0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class BriaPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = BriaPipeline.from_pretrained("briaai/BRIA-3.2", mindspore_dtype=ms_dtype)

        prompt = "Photorealistic food photography of a stack of fluffy pancakes on a white plate, with maple syrup being poured over them. On top of the pancakes are the words 'BRIA 3.2' in bold, yellow, 3D letters. The background is dark and out of focus."  # noqa
        torch.manual_seed(0)
        image = pipe(prompt=prompt)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"bria_{dtype}.npy",
            subfolder="bria",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
