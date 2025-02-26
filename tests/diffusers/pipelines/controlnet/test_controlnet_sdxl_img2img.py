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
from PIL import Image
from transformers import CLIPTextConfig

import mindspore as ms
from mindspore import ops

from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
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
class ControlNetPipelineSDXLImg2ImgFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "controlnet",
            "diffusers.models.controlnet.ControlNetModel",
            "mindone.diffusers.models.controlnet.ControlNetModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                in_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                conditioning_embedding_out_channels=(16, 32),
                # SD2-specific config below
                attention_head_dim=(2, 4),
                use_linear_projection=True,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                transformer_layers_per_block=(1, 2),
                projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
                cross_attention_dim=64,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                steps_offset=1,
                beta_schedule="scaled_linear",
                timestep_spacing="leading",
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[32, 64],
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
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
                    hidden_size=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    pad_token_id=1,
                    vocab_size=1000,
                    # SD2-specific config below
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
            "text_encoder_2",
            "transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModelWithProjection",
            dict(
                config=CLIPTextConfig(
                    bos_token_id=0,
                    eos_token_id=2,
                    hidden_size=32,
                    intermediate_size=37,
                    layer_norm_eps=1e-05,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    pad_token_id=1,
                    vocab_size=1000,
                    # SD2-specific config below
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
    ]

    def get_unet_config(self, skip_first_text_encoder=False):
        return [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                # SD2-specific config below
                attention_head_dim=(2, 4),
                use_linear_projection=True,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                transformer_layers_per_block=(1, 2),
                projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
                cross_attention_dim=64 if not skip_first_text_encoder else 32,
            ),
        ]

    def get_dummy_components(self, skip_first_text_encoder=False):
        components = {
            key: None
            for key in [
                "unet",
                "controlnet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "text_encoder_2",
                "tokenizer_2",
                "image_encoder",
                "feature_extractor",
            ]
        }
        unet = self.get_unet_config(skip_first_text_encoder)
        pipeline_config = [unet] + self.pipeline_config

        pt_components, ms_components = get_pipeline_components(components, pipeline_config)

        if skip_first_text_encoder:
            pt_components["text_encoder"] = None
            pt_components["tokenizer"] = None
            ms_components["text_encoder"] = None
            ms_components["tokenizer"] = None

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        controlnet_embedder_scale_factor = 2
        pt_image = floats_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            rng=random.Random(seed),
        )
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": pt_image,
            "control_image": pt_image,
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": ms_image,
            "control_image": ms_image,
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_stable_diffusion_xl_controlnet_img2img(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.controlnet.StableDiffusionXLControlNetImg2ImgPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionXLControlNetImg2ImgPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class ControlNetPipelineSDXLImg2ImgIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def get_depth_map(self, image, feature_extractor, depth_estimator):
        image = ms.Tensor(feature_extractor(images=image, return_tensors="np").pixel_values)
        depth_estimator.set_train(False)
        depth_map = depth_estimator(image)[0]

        depth_map = ops.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = ops.amin(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_max = ops.amax(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = ops.cat([depth_map] * 3, axis=1)
        image = image.permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    @data(*test_cases)
    @unpack
    def test_stable_diffusion_xl_controlnet_img2img(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        depth_estimator_cls = get_module("mindone.transformers.models.dpt.modeling_dpt.DPTForDepthEstimation")
        depth_estimator = depth_estimator_cls.from_pretrained("Intel/dpt-hybrid-midas", revision="refs/pr/7")
        feature_extractor_cls = get_module("transformers.models.dpt.feature_extraction_dpt.DPTFeatureExtractor")
        feature_extractor = feature_extractor_cls.from_pretrained("Intel/dpt-hybrid-midas", revision="refs/pr/7")
        controlnet_cls = get_module("mindone.diffusers.models.controlnet.ControlNetModel")
        controlnet = controlnet_cls.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small",
            variant="fp16",
            use_safetensors=True,
            mindspore_dtype=ms_dtype,
        )
        vae_cls = get_module("mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL")
        vae = vae_cls.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms_dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionXLControlNetImg2ImgPipeline")
        pipe = pipe_cls.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            mindspore_dtype=ms_dtype,
        )

        prompt = "A robot, 4k photo"
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat.png",
            subfolder="kandinsky",
        ).resize((1024, 1024))
        controlnet_conditioning_scale = 0.5  # recommended for good generalization
        depth_image = self.get_depth_map(image, feature_extractor, depth_estimator)

        torch.manual_seed(0)
        image = pipe(
            prompt,
            image=image,
            control_image=depth_image,
            strength=0.99,
            num_inference_steps=50,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"img2img_sdxl_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
