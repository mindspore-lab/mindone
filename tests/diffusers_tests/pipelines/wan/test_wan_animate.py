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
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPVisionConfig

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": 1, "dtype": "float16"},
    {"mode": 1, "dtype": "bfloat16"},
    {"mode": 0, "dtype": "float16"},
    {"mode": 0, "dtype": "bfloat16"},
]


@ddt
class WanAnimatePipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(shift=7.0),
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
            "diffusers.models.transformers.transformer_wan_animate.WanAnimateTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_wan_animate.WanAnimateTransformer3DModel",
            dict(
                patch_size=(1, 2, 2),
                num_attention_heads=2,
                attention_head_dim=12,
                in_channels=36,
                latent_channels=16,
                out_channels=16,
                text_dim=32,
                freq_dim=256,
                ffn_dim=32,
                num_layers=2,
                cross_attn_norm=True,
                qk_norm="rms_norm_across_heads",
                image_dim=4,
                rope_max_seq_len=32,
                motion_encoder_channel_sizes={"4": 16, "8": 16, "16": 16},
                motion_encoder_size=16,
                motion_style_dim=8,
                motion_dim=4,
                motion_encoder_dim=16,
                face_encoder_hidden_dim=16,
                face_encoder_num_heads=2,
                inject_face_latents_blocks=2,
            ),
        ],
        [
            "image_encoder",
            "transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            dict(
                config=CLIPVisionConfig(
                    hidden_size=4,
                    projection_dim=4,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                    image_size=4,
                    intermediate_size=16,
                    patch_size=1,
                ),
            ),
        ],
        [
            "image_processor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(
                crop_size=4,
                size=4,
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
                "image_encoder",
                "image_processor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        num_frames = 17
        height = 16
        width = 16
        face_height = 16
        face_width = 16

        image = Image.new("RGB", (height, width))
        pose_video = [Image.new("RGB", (height, width))] * num_frames
        face_video = [Image.new("RGB", (face_height, face_width))] * num_frames

        inputs = {
            "image": image,
            "pose_video": pose_video,
            "face_video": face_video,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "height": height,
            "width": width,
            "segment_frame_length": 77,  # TODO: can we set this to num_frames?
            "num_inference_steps": 2,
            "mode": "animate",
            "prev_segment_conditioning_frames": 1,
            "guidance_scale": 1.0,
            "output_type": "pt",
            "max_sequence_length": 16,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        """Test basic inference in animation mode."""
        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.wan.WanAnimatePipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.wan.WanAnimatePipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        if mode == 0:
            ms_pipe.transformer.construct = ms.jit(ms_pipe.transformer.construct)

        torch.manual_seed(0)
        pt_frame = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_frame = ms_pipe(**inputs)

        pt_frame_slice = pt_frame.frames[0][0, -3:, -3:, -1]
        ms_frame_slice = ms_frame[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_frame_slice - ms_frame_slice) / np.linalg.norm(pt_frame_slice) < threshold
