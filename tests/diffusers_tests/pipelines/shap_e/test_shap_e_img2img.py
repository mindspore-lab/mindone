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
import pytest
import torch
from ddt import data, ddt, unpack
from transformers import CLIPVisionConfig

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
class ShapEImg2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "prior",
            "diffusers.models.transformers.prior_transformer.PriorTransformer",
            "mindone.diffusers.models.transformers.prior_transformer.PriorTransformer",
            {
                "num_attention_heads": 2,
                "attention_head_dim": 16,
                "embedding_dim": 16,
                "num_embeddings": 32,
                "embedding_proj_dim": 16,
                "time_embed_dim": 64,
                "num_layers": 1,
                "clip_embed_dim": 32,
                "additional_embeddings": 0,
                "time_embed_act_fn": "gelu",
                "norm_in_type": "layer",
                "embedding_proj_norm_type": "layer",
                "encoder_hid_proj_type": None,
                "added_emb_type": None,
            },
        ],
        [
            "image_encoder",
            "transformers.models.clip.modeling_clip.CLIPVisionModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPVisionModel",
            dict(
                config=CLIPVisionConfig(
                    hidden_size=16,
                    image_size=32,
                    projection_dim=16,
                    intermediate_size=24,
                    num_attention_heads=2,
                    num_channels=3,
                    num_hidden_layers=5,
                    patch_size=1,
                ),
            ),
        ],
        [
            "image_processor",
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
        [
            "shap_e_renderer",
            "diffusers.pipelines.shap_e.ShapERenderer",
            "mindone.diffusers.pipelines.shap_e.ShapERenderer",
            {
                "param_shapes": (
                    (8, 93),
                    (8, 8),
                    (8, 8),
                    (8, 8),
                ),
                "d_latent": 16,
                "d_hidden": 8,
                "n_output": 12,
                "background": (
                    0.1,
                    0.1,
                    0.1,
                ),
            },
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler",
            dict(
                beta_schedule="exp",
                num_train_timesteps=1024,
                prediction_type="sample",
                use_karras_sigmas=True,
                clip_sample=True,
                clip_sample_range=1.0,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "prior",
                "image_encoder",
                "image_processor",
                "shap_e_renderer",
                "scheduler",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        pt_input_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        ms_input_image = ms.Tensor(pt_input_image.numpy())

        pt_inputs = {
            "image": pt_input_image,
            "num_inference_steps": 1,
            "frame_size": 32,
            "output_type": "latent",
        }

        ms_inputs = {
            "image": ms_input_image,
            "num_inference_steps": 1,
            "frame_size": 32,
            "output_type": "latent",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_shap_e(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.shap_e.ShapEImg2ImgPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.shap_e.ShapEImg2ImgPipeline")

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

        pt_image_slice = pt_image.images[0, -3:, -3:].numpy()
        ms_image_slice = ms_image[0][0, -3:, -3:].asnumpy()

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class ShapEImg2ImgPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_shap_e_img2img(self, mode, dtype):
        pytest.skip("Skipping this case since the pretrained model only has .bin file")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.shap_e.ShapEImg2ImgPipeline")
        pipe = pipe_cls.from_pretrained("openai/shap-e-img2img", variant="fp16", mindspore_dtype=ms_dtype)
        pipe.set_progress_bar_config(disable=None)

        input_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "corgi.png",
            subfolder="shap_e",
        )

        torch.manual_seed(0)
        image = pipe(
            input_image,
            guidance_scale=3.0,
            num_inference_steps=64,
            frame_size=64,
        )[0][
            0
        ][1]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"shap_e_img2img_{dtype}.npy",
            subfolder="shap_e",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
