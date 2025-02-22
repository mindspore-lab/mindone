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

import mindspore as ms

from mindone.diffusers.pipelines.kandinsky.text_encoder import MCLIPConfig
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


class Dummies:
    pipeline_config = [
        [
            "text_encoder",
            "diffusers.pipelines.kandinsky.MultilingualCLIP",
            "mindone.diffusers.pipelines.kandinsky.MultilingualCLIP",
            dict(
                config=MCLIPConfig(
                    numDims=32,
                    transformerDimensions=32,
                    hidden_size=32,
                    intermediate_size=37,
                    num_attention_heads=4,
                    num_hidden_layers=5,
                    vocab_size=1005,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast",
            "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast",
            dict(
                pretrained_model_name_or_path="YiYiXu/tiny-random-mclip-base",
            ),
        ],
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            {
                "in_channels": 9,
                # Out channels is double in channels because predicts mean and variance
                "out_channels": 8,
                "addition_embed_type": "text_image",
                "down_block_types": ("ResnetDownsampleBlock2D", "SimpleCrossAttnDownBlock2D"),
                "up_block_types": ("SimpleCrossAttnUpBlock2D", "ResnetUpsampleBlock2D"),
                "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
                "block_out_channels": (32, 64),
                "layers_per_block": 1,
                "encoder_hid_dim": 32,
                "encoder_hid_dim_type": "text_image_proj",
                "cross_attention_dim": 32,
                "attention_head_dim": 4,
                "resnet_time_scale_shift": "scale_shift",
                "class_embed_type": None,
            },
        ],
        [
            "movq",
            "diffusers.models.autoencoders.vq_model.VQModel",
            "mindone.diffusers.models.autoencoders.vq_model.VQModel",
            {
                "block_out_channels": [32, 64],
                "down_block_types": ["DownEncoderBlock2D", "AttnDownEncoderBlock2D"],
                "in_channels": 3,
                "latent_channels": 4,
                "layers_per_block": 1,
                "norm_num_groups": 8,
                "norm_type": "spatial",
                "num_vq_embeddings": 12,
                "out_channels": 3,
                "up_block_types": [
                    "AttnUpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ],
                "vq_embed_dim": 4,
            },
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                num_train_timesteps=1000,
                beta_schedule="linear",
                beta_start=0.00085,
                beta_end=0.012,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type="epsilon",
                thresholding=False,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "text_encoder",
                "tokenizer",
                "unet",
                "scheduler",
                "movq",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        pt_components["text_encoder"] = pt_components["text_encoder"].eval()
        ms_components["text_encoder"] = ms_components["text_encoder"].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        pt_image_embeds = floats_tensor((1, 32), rng=random.Random(seed))
        pt_negative_image_embeds = floats_tensor((1, 32), rng=random.Random(seed + 1))
        ms_image_embeds = ms.Tensor(pt_image_embeds.numpy())
        ms_negative_image_embeds = ms.Tensor(pt_negative_image_embeds.numpy())
        # create init_image
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed))
        image = image.cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((256, 256))
        # create mask
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[:32, :32] = 1

        pt_inputs = {
            "prompt": "horse",
            "image": init_image,
            "mask_image": mask,
            "image_embeds": pt_image_embeds,
            "negative_image_embeds": pt_negative_image_embeds,
            "height": 64,
            "width": 64,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "horse",
            "image": init_image,
            "mask_image": mask,
            "image_embeds": ms_image_embeds,
            "negative_image_embeds": ms_negative_image_embeds,
            "height": 64,
            "width": 64,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "output_type": "np",
        }
        return pt_inputs, ms_inputs


@ddt
class KandinskyInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        dummies = Dummies()
        return dummies.get_dummy_components()

    def get_dummy_inputs(self, seed=0):
        dummies = Dummies()
        return dummies.get_dummy_inputs(seed=seed)

    @data(*test_cases)
    @unpack
    def test_kandinsky_inpaint(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky.KandinskyInpaintPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyInpaintPipeline")

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
class KandinskyInpaintPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_kandinsky_inpaint(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        init_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat.png",
            subfolder="kandinsky",
        )
        mask = np.zeros((768, 768), dtype=np.float32)
        mask[:250, 250:-250] = 1

        prompt = "a hat"

        pipe_prior_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyPriorPipeline")
        pipe_prior = pipe_prior_cls.from_pretrained("kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms_dtype)

        pipeline_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyInpaintPipeline")
        pipeline = pipeline_cls.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", mindspore_dtype=ms_dtype)
        pipeline.set_progress_bar_config(disable=None)

        torch.manual_seed(0)
        image_emb, zero_image_emb = pipe_prior(
            prompt,
            num_inference_steps=5,
            negative_prompt="",
        )

        torch.manual_seed(0)
        output = pipeline(
            prompt,
            image=init_image,
            mask_image=mask,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            num_inference_steps=100,
            height=768,
            width=768,
        )

        image = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"inpaint_{dtype}.npy",
            subfolder="kandinsky",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
