# coding=utf-8
# Copyright 2024 Harutatsu Akiyama, Jinbin Bai, and HuggingFace Inc.
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

import cv2
import numpy as np
import torch
from ddt import data, ddt, unpack
from PIL import Image
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
class ControlNetPipelineSDXLFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
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
                cross_attention_dim=64,
            ),
        ],
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
        [
            "image_encoder",
            "transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            dict(
                config=CLIPVisionConfig(
                    hidden_size=32,
                    image_size=224,
                    projection_dim=32,
                    intermediate_size=37,
                    num_attention_heads=4,
                    num_channels=3,
                    num_hidden_layers=5,
                    patch_size=14,
                ),
            ),
        ],
        [
            "feature_extractor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(
                crop_size=224,
                do_center_crop=True,
                do_normalize=True,
                do_resize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                resample=3,
                size=224,
            ),
        ],
    ]

    def get_dummy_components(self):
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

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0, img_res=64):
        # Get random floats in [0, 1] as image
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        mask_image = torch.ones_like(image)
        controlnet_embedder_scale_factor = 2
        control_image = floats_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            rng=random.Random(seed),
        ).cpu()
        control_image = control_image.cpu().permute(0, 2, 3, 1)[0]
        # Convert image and mask_image to [0, 255]
        image = 255 * image
        mask_image = 255 * mask_image
        control_image = 255 * control_image
        # Convert to PIL image
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((img_res, img_res))
        mask_image = Image.fromarray(np.uint8(mask_image)).convert("L").resize((img_res, img_res))
        control_image = Image.fromarray(np.uint8(control_image)).convert("RGB").resize((img_res, img_res))

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "image": init_image,
            "mask_image": mask_image,
            "control_image": control_image,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.controlnet.StableDiffusionXLControlNetInpaintPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionXLControlNetInpaintPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class ControlNetPipelineSDXLIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def make_canny_condition(self, image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        return image

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        init_image = load_downloaded_image_from_hf_hub(
            "diffusers/test-arrays",
            "boy.png",
            subfolder="stable_diffusion_inpaint",
        )
        init_image = init_image.resize((1024, 1024))

        mask_image = load_downloaded_image_from_hf_hub(
            "diffusers/test-arrays",
            "boy_mask.png",
            subfolder="stable_diffusion_inpaint",
        )
        mask_image = mask_image.resize((1024, 1024))

        control_image = self.make_canny_condition(init_image)

        controlnet_cls = get_module("mindone.diffusers.models.controlnet.ControlNetModel")
        controlnet = controlnet_cls.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", mindspore_dtype=ms_dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.controlnet.StableDiffusionXLControlNetInpaintPipeline")
        pipe = pipe_cls.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, mindspore_dtype=ms_dtype
        )

        # generate image
        torch.manual_seed(0)
        image = pipe(
            "a handsome man with ray-ban sunglasses",
            num_inference_steps=20,
            eta=1.0,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
        )[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"inpaint_sdxl_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
