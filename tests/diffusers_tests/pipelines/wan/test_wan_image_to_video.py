# Copyright 2024 The HuggingFace Team.
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
from transformers import CLIPVisionConfig

import mindspore as ms

from mindone.diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)
from mindone.transformers import CLIPVisionModel

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
class WanImageToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
            "diffusers.models.transformers.transformer_wan.WanTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_wan.WanTransformer3DModel",
            dict(
                patch_size=(1, 2, 2),
                num_attention_heads=2,
                attention_head_dim=12,
                in_channels=36,
                out_channels=16,
                text_dim=32,
                freq_dim=256,
                ffn_dim=32,
                num_layers=2,
                cross_attn_norm=True,
                qk_norm="rms_norm_across_heads",
                rope_max_seq_len=32,
                image_dim=4,
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
                    image_size=32,
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
                crop_size=32,
                size=32,
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
                "image_encoder",
                "image_processor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        image_height = 16
        image_width = 16
        image = Image.new("RGB", (image_width, image_height))
        inputs = {
            "image": image,
            "prompt": "dance monkey",
            "negative_prompt": "negative",  # TODO
            "height": image_height,
            "width": image_width,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        # TODO: diffusers link: https://github.com/huggingface/diffusers/issues/11351
        if dtype == "float16" or dtype == "bfloat16":
            pytest.skip("Skipping this case since torch will throw an issue when dtype of `vae` is not float32")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.wan.WanImageToVideoPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.wan.WanImageToVideoPipeline")

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

        pt_frame_slice = pt_frame.frames[0][0, -3:, -3:, -1]
        ms_frame_slice = ms_frame[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_frame_slice - ms_frame_slice) / np.linalg.norm(pt_frame_slice) < threshold


@slow
@ddt
class WanImageToVideoPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32" or dtype == "float16":
            pytest.skip("Skipping this case since this pipeline will OOM in float32 and float16")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", mindspore_dtype=ms.float32)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", mindspore_dtype=ms.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id, vae=vae, image_encoder=image_encoder, mindspore_dtype=ms_dtype
        )

        image = load_downloaded_image_from_hf_hub(
            "huggingface/documentation-images",
            "astronaut.jpg",
            subfolder="diffusers",
        )
        max_area = 480 * 832
        aspect_ratio = image.height / image.width
        mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        prompt = (
            "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "  # noqa: E501
            "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
        )
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"  # noqa: E501

        torch.manual_seed(0)
        image = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=21,
            guidance_scale=5.0,
        )[0][0][1]
        image = Image.fromarray((image * 255).astype("uint8"))

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"wan_i2v_{dtype}.npy",
            subfolder="wan",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
