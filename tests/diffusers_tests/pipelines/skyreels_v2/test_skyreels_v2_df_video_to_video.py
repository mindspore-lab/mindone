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
import pytest
import torch
from ddt import data, ddt, unpack
from PIL import Image

import mindspore as ms

from mindone.diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline
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
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "bfloat16"},
]


@ddt
class SkyReelsV2DiffusionForcingVideoToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
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
            "diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler",
            "mindone.diffusers.schedulers.scheduling_unipc_multistep.UniPCMultistepScheduler",
            dict(
                flow_shift=5.0,
                use_flow_sigmas=True,
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
        [
            "transformer",
            "diffusers.models.transformers.transformer_skyreels_v2.SkyReelsV2Transformer3DModel",
            "mindone.diffusers.models.transformers.transformer_skyreels_v2.SkyReelsV2Transformer3DModel",
            dict(
                patch_size=(1, 2, 2),
                num_attention_heads=2,
                attention_head_dim=12,
                in_channels=16,
                out_channels=16,
                text_dim=32,
                freq_dim=256,
                ffn_dim=32,
                num_layers=2,
                cross_attn_norm=True,
                qk_norm="rms_norm_across_heads",
                rope_max_seq_len=32,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        video = [Image.new("RGB", (16, 16))] * 7
        inputs = {
            "video": video,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "num_inference_steps": 4,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "max_sequence_length": 16,
            "output_type": "np",
            "overlap_history": 3,
            "num_frames": 17,
            "base_num_frames": 5,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.skyreels_v2.pipeline_skyreels_v2_diffusion_forcing_v2v.SkyReelsV2DiffusionForcingVideoToVideoPipeline"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.skyreels_v2.pipeline_skyreels_v2_diffusion_forcing_v2v.SkyReelsV2DiffusionForcingVideoToVideoPipeline"
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
        pt_frame = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_frame = ms_pipe(**inputs)

        pt_frame_slice = pt_frame.frames[0][-1, -3:, -3:, -1]
        ms_frame_slice = ms_frame[0][0][-1, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_frame_slice - ms_frame_slice) / np.linalg.norm(pt_frame_slice) < threshold
