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

import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig, CLIPVisionConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    floats_tensor,
    get_module,
    get_pipeline_components,
)

# FP32 is not supported since I2VGenXLPipeline contains nn.Conv3d
test_cases = [
    # {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    # {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class I2VGenXLPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
        ],
        [
            "unet",
            "diffusers.models.unets.unet_i2vgen_xl.I2VGenXLUNet",
            "mindone.diffusers.models.unets.unet_i2vgen_xl.I2VGenXLUNet",
            dict(
                block_out_channels=(4, 8),
                layers_per_block=1,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("CrossAttnDownBlock3D", "DownBlock3D"),
                up_block_types=("UpBlock3D", "CrossAttnUpBlock3D"),
                cross_attention_dim=4,
                attention_head_dim=4,
                num_attention_heads=None,
                norm_num_groups=2,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=(8,),
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D"],
                latent_channels=4,
                sample_size=32,
                norm_num_groups=2,
            ),
        ],
        [
            "text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=4,
                    intermediate_size=16,
                    layer_norm_eps=1e-05,
                    num_attention_heads=2,
                    num_hidden_layers=2,
                    pad_token_id=1,
                    vocab_size=1000,
                    hidden_act="gelu",
                    projection_dim=32,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
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
            "feature_extractor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(crop_size=32, size=32),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "image_encoder",
                "tokenizer",
                "feature_extractor",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        pt_input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        ms_input_image = ms.Tensor(pt_input_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": pt_input_image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "pt",
            "num_frames": 4,
            "width": 32,
            "height": 32,
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": ms_input_image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "pt",
            "num_frames": 4,
            "width": 32,
            "height": 32,
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_text_to_video_default_case(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.i2vgen_xl.I2VGenXLPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.i2vgen_xl.I2VGenXLPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()
        pt_inputs["output_type"] = "np"
        ms_inputs["output_type"] = "np"

        torch.manual_seed(0)
        pt_frame = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_frame = ms_pipe(**ms_inputs)

        pt_image_slice = pt_frame.frames[0][0][-3:, -3:, -1]
        ms_image_slice = ms_frame[0][0][0][-3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class I2VGenXLPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_i2vgen_xl(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.i2vgen_xl.I2VGenXLPipeline")
        pipe = pipe_cls.from_pretrained("ali-vilab/i2vgen-xl", mindspore_dtype=ms_dtype, variant="fp16")
        pipe.set_progress_bar_config(disable=None)

        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat_6.png",
            subfolder="pix2pix",
        )

        num_frames = 3

        torch.manual_seed(0)
        output = pipe(
            image=image,
            prompt="my cat",
            num_frames=num_frames,
            num_inference_steps=3,
            output_type="np",
        )

        image = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"i2vgen_xl_{dtype}.npy",
            subfolder="i2vgen_xl",
        )
        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(expected_image - image) / np.linalg.norm(expected_image) < threshold
