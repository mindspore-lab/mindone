# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class DanceDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_1d.UNet1DModel",
            "mindone.diffusers.models.unets.unet_1d.UNet1DModel",
            dict(
                block_out_channels=(32, 32, 64),
                extra_in_channels=16,
                sample_size=512,
                sample_rate=16_000,
                in_channels=2,
                out_channels=2,
                flip_sin_to_cos=True,
                use_timestep_embedding=False,
                time_embedding_type="fourier",
                mid_block_type="UNetMidBlock1D",
                down_block_types=("DownBlock1DNoSkip", "DownBlock1D", "AttnDownBlock1D"),
                up_block_types=("AttnUpBlock1D", "UpBlock1D", "UpBlock1DNoSkip"),
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ipndm.IPNDMScheduler",
            "mindone.diffusers.schedulers.scheduling_ipndm.IPNDMScheduler",
            dict(),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        inputs = {
            "batch_size": 1,
            "num_inference_steps": 4,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_dance_diffusion(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.dance_diffusion.DanceDiffusionPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.dance_diffusion.DanceDiffusionPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        if dtype == "float32":
            torch.manual_seed(0)
            pt_audio = pt_pipe(**inputs)
            pt_audio_slice = pt_audio.audios[0, -3:, -3:]
        else:
            # torch.flot16 requires CUDA
            pt_audio_slice = np.array([[-0.9033203, 0.5449219, 1.0], [-0.49975586, 1.0, 0.6455078]])

        torch.manual_seed(0)
        ms_audio = ms_pipe(**inputs)
        ms_audio_slice = ms_audio[0][0, -3:, -3:]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_audio_slice - ms_audio_slice) / np.linalg.norm(pt_audio_slice) < threshold


@slow
@ddt
class DanceDiffusionPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_dance_diffusion(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.dance_diffusion.DanceDiffusionPipeline")
        pipe = pipe_cls.from_pretrained("harmonai/maestro-150k", revision="refs/pr/3", mindspore_dtype=ms_dtype)
        pipe.set_progress_bar_config(disable=None)

        torch.manual_seed(0)
        audio = pipe(num_inference_steps=100, audio_length_in_s=4.096)[0]

        expected_audio = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"dance_diffusion_{dtype}.npy",
            subfolder="dance_diffusion",
        )
        # The pipeline uses a larger threshold, and accuracy can be ensured at this threshold.
        threshold = 5e-2 if dtype == "float32" else 3e-1
        assert np.linalg.norm(expected_audio - audio) / np.linalg.norm(expected_audio) < threshold
