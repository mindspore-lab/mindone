# Copyright 2024 Marigold authors, PRS ETH Zurich. All rights reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# --------------------------------------------------------------------------
# More information and citation instructions are available on the
# Marigold project website: https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig

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
class MarigoldDepthPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "scheduler",
            "diffusers.schedulers.scheduling_lcm.LCMScheduler",
            "mindone.diffusers.schedulers.scheduling_lcm.LCMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                prediction_type="v_prediction",
                set_alpha_to_one=False,
                steps_offset=1,
                beta_schedule="scaled_linear",
                clip_sample=False,
                thresholding=False,
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
    ]

    def get_unet_config(self, time_cond_proj_dim=None):
        return [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                time_cond_proj_dim=time_cond_proj_dim,
                sample_size=32,
                in_channels=8,
                out_channels=4,
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
            ),
        ]

    def get_dummy_components(self, time_cond_proj_dim=None):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "text_encoder",
                "tokenizer",
                "prediction_type",
                "scale_invariant",
                "shift_invariant",
            ]
        }
        components["prediction_type"] = ("depth", "depth")
        components["scale_invariant"] = (True, True)
        components["shift_invariant"] = (True, True)

        unet = self.get_unet_config(time_cond_proj_dim)
        pipeline_config = self.pipeline_config + [unet]

        return get_pipeline_components(components, pipeline_config)

    def get_dummy_inputs(self, seed=0):
        pt_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        pt_image = pt_image / 2 + 0.5
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "image": pt_image,
            "num_inference_steps": 1,
            "processing_resolution": 0,
            "output_type": "np",
        }

        ms_inputs = {
            "image": ms_image,
            "num_inference_steps": 1,
            "processing_resolution": 0,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    def _test_marigold_depth(self, generator_seed, mode, dtype, **pipe_kwargs):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.marigold.MarigoldDepthPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.marigold.MarigoldDepthPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs(seed=generator_seed)
        pt_inputs.update(**pipe_kwargs)
        ms_inputs.update(**pipe_kwargs)

        if dtype == "float32":
            torch.manual_seed(0)
            pt_prediction = pt_pipe(**pt_inputs)
            pt_prediction_slice = pt_prediction.prediction[0, -3:, -3:, -1]
        else:
            # torch.flot16 requires CUDA
            pt_prediction_slice = np.array(
                [
                    [0.44702148, 0.4453125, 0.4645996],
                    [0.4465332, 0.43017578, 0.44140625],
                    [0.53027344, 0.50683594, 0.48242188],
                ]
            )

        torch.manual_seed(0)
        ms_prediction = ms_pipe(**ms_inputs)
        ms_prediction_slice = ms_prediction[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert (
            np.linalg.norm(pt_prediction_slice - ms_prediction_slice) / np.linalg.norm(pt_prediction_slice) < threshold
        )

    @data(*test_cases)
    @unpack
    def test_marigold_depth_dummy_defaults(self, mode, dtype):
        self._test_marigold_depth(generator_seed=0, mode=mode, dtype=dtype)


@slow
@ddt
class MarigoldDepthPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_marigold_depth(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.marigold.MarigoldDepthPipeline")
        pipe = pipe_cls.from_pretrained("prs-eth/marigold-depth-lcm-v1-0", variant="fp16", mindspore_dtype=ms_dtype)

        image = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "marigold_input.jpg",
            subfolder="marigold",
        )
        depth = pipe(image)

        image = pipe.image_processor.visualize_depth(depth[0])[0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"marigold_depth_{dtype}.npy",
            subfolder="marigold",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
