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
from PIL import Image
from transformers import CLIPTextConfig, LlamaConfig, SiglipImageProcessor

import mindspore as ms

from mindone.diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow
from mindone.transformers import SiglipVisionModel

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
]


@ddt
class HunyuanVideoFramepackPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_hunyuan_video.AutoencoderKLHunyuanVideo",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_hunyuan_video.AutoencoderKLHunyuanVideo",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=4,
                down_block_types=(
                    "HunyuanVideoDownBlock3D",
                    "HunyuanVideoDownBlock3D",
                    "HunyuanVideoDownBlock3D",
                    "HunyuanVideoDownBlock3D",
                ),
                up_block_types=(
                    "HunyuanVideoUpBlock3D",
                    "HunyuanVideoUpBlock3D",
                    "HunyuanVideoUpBlock3D",
                    "HunyuanVideoUpBlock3D",
                ),
                block_out_channels=(8, 8, 8, 8),
                layers_per_block=1,
                act_fn="silu",
                norm_num_groups=4,
                scaling_factor=0.476986,
                spatial_compression_ratio=8,
                temporal_compression_ratio=4,
                mid_block_add_attention=True,
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
            "transformers.models.llama.modeling_llama.LlamaModel",
            "mindone.transformers.models.llama.modeling_llama.LlamaModel",
            dict(
                config=LlamaConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=16,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    pad_token_id=1,
                    vocab_size=1000,
                    hidden_act="gelu",
                    projection_dim=32,
                    _attn_implementation="eager",
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.llama.tokenization_llama.LlamaTokenizer",
            "transformers.models.llama.tokenization_llama.LlamaTokenizer",
            dict(
                pretrained_model_name_or_path="finetrainers/dummy-hunyaunvideo",
                subfolder="tokenizer",
            ),
        ],
        [
            "text_encoder_2",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=8,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    pad_token_id=1,
                    vocab_size=1000,
                    hidden_act="gelu",
                    projection_dim=32,
                ),
            ),
        ],
        [
            "tokenizer_2",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
        [
            "feature_extractor",
            "transformers.models.siglip.image_processing_siglip.SiglipImageProcessor",
            "transformers.models.siglip.image_processing_siglip.SiglipImageProcessor",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-SiglipVisionModel",
                size={"height": 30, "width": 30},
            ),
        ],
        [
            "image_encoder",
            "transformers.models.siglip.modeling_siglip.SiglipVisionModel",
            "mindone.transformers.models.siglip.modeling_siglip.SiglipVisionModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-SiglipVisionModel",
            ),
        ],
    ]

    def get_transformer_config(self, num_layers: int = 1, num_single_layers: int = 1):
        return [
            "transformer",
            "diffusers.models.transformers.transformer_hunyuan_video_framepack.HunyuanVideoFramepackTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_hunyuan_video_framepack.HunyuanVideoFramepackTransformer3DModel",
            dict(
                in_channels=4,
                out_channels=4,
                num_attention_heads=2,
                attention_head_dim=10,
                num_layers=num_layers,
                num_single_layers=num_single_layers,
                num_refiner_layers=1,
                patch_size=2,
                patch_size_t=1,
                guidance_embeds=True,
                text_embed_dim=16,
                pooled_projection_dim=8,
                rope_axes_dim=(2, 4, 4),
                image_condition_type=None,
                has_image_proj=True,
                image_proj_dim=32,
                has_clean_x_embedder=True,
            ),
        ]

    def get_dummy_components(self, num_layers: int = 1, num_single_layers: int = 1):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "text_encoder",
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
                "feature_extractor",
                "image_encoder",
            ]
        }

        transformer = self.get_transformer_config(num_layers, num_single_layers)
        pipeline_config = [transformer] + self.pipeline_config

        return get_pipeline_components(components, pipeline_config)

    def get_dummy_inputs(self):
        image_height = 32
        image_width = 32
        image = Image.new("RGB", (image_width, image_height))
        inputs = {
            "image": image,
            "prompt": "dance monkey",
            "prompt_template": {
                "template": "{}",
                "crop_start": 0,
            },
            "num_inference_steps": 2,
            "guidance_scale": 4.5,
            "height": image_height,
            "width": image_width,
            "num_frames": 9,
            "latent_window_size": 3,
            "max_sequence_length": 256,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.hunyuan_video.HunyuanVideoFramepackPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.hunyuan_video.HunyuanVideoFramepackPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        if dtype == "float16":
            # torch.float16 requires CUDA
            pt_image_slice = np.array(
                [
                    [0.5019531, 0.4963379, 0.4716797],
                    [0.50439453, 0.50341797, 0.46801758],
                    [0.5083008, 0.5029297, 0.4716797],
                ]
            )
        else:
            torch.manual_seed(0)
            pt_image = pt_pipe(**inputs)
            pt_image_slice = pt_image.frames[0][0, -3:, -3:, -1]

        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)
        ms_image_slice = ms_image[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class HunyuanVideoFramepackPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline has oom issue in float32")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
            "lllyasviel/FramePackI2V_HY", mindspore_dtype=ms.bfloat16
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            "lllyasviel/flux_redux_bfl", subfolder="image_encoder", mindspore_dtype=ms_dtype
        )
        pipe = HunyuanVideoFramepackPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            transformer=transformer,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            mindspore_dtype=ms_dtype,
        )
        pipe.vae.enable_tiling()

        image = load_downloaded_image_from_hf_hub(
            "huggingface/documentation-images",
            "penguin.png",
            subfolder="diffusers",
        )

        torch.manual_seed(0)
        image = pipe(
            image=image,
            prompt="A penguin dancing in the snow",
            height=832,
            width=480,
            num_frames=91,
            num_inference_steps=30,
            guidance_scale=9.0,
            sampling_type="inverted_anti_drifting",
        )[0][0][1]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"hunyuan_video_framepack_{dtype}.npy",
            subfolder="hunyuan_video",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
