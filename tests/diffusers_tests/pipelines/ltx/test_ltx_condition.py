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
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.pipelines.ltx.pipeline_ltx_condition import LTXConditionPipeline, LTXVideoCondition
from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    load_downloaded_video_from_hf_hub,
    slow,
)

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
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class LTXConditionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_ltx.LTXVideoTransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_ltx.LTXVideoTransformer3DModel",
            dict(
                in_channels=8,
                out_channels=8,
                patch_size=1,
                patch_size_t=1,
                num_attention_heads=4,
                attention_head_dim=8,
                cross_attention_dim=32,
                num_layers=1,
                caption_channels=32,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_ltx.AutoencoderKLLTXVideo",
            dict(
                in_channels=3,
                out_channels=3,
                latent_channels=8,
                block_out_channels=(8, 8, 8, 8),
                decoder_block_out_channels=(8, 8, 8, 8),
                layers_per_block=(1, 1, 1, 1, 1),
                decoder_layers_per_block=(1, 1, 1, 1, 1),
                spatio_temporal_scaling=(True, True, False, False),
                decoder_spatio_temporal_scaling=(True, True, False, False),
                decoder_inject_noise=(False, False, False, False, False),
                upsample_residual=(False, False, False, False),
                upsample_factor=(1, 1, 1, 1),
                timestep_conditioning=False,
                patch_size=1,
                patch_size_t=1,
                encoder_causal=True,
                decoder_causal=False,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(),
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
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        pt_components["vae"].use_framewise_encoding = False
        ms_components["vae"].use_framewise_encoding = False
        pt_components["vae"].use_framewise_decoding = False
        ms_components["vae"].use_framewise_decoding = False

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0, use_conditions=False):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pt_image = torch.randn((1, 3, 32, 32), generator=generator)
        ms_image = ms.tensor(pt_image.numpy())
        if use_conditions:
            pt_conditions = LTXVideoCondition(
                image=pt_image,
            )
            ms_conditions = LTXVideoCondition(
                image=ms_image,
            )
        else:
            pt_conditions = None
            ms_conditions = None

        pt_inputs = {
            "conditions": pt_conditions,
            "image": None if use_conditions else pt_image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            # 8 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        ms_inputs = {
            "conditions": ms_conditions,
            "image": None if use_conditions else ms_image,
            "prompt": "dance monkey",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "height": 32,
            "width": 32,
            # 8 * k + 1 is the recommendation
            "num_frames": 9,
            "max_sequence_length": 16,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.ltx.LTXConditionPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.ltx.LTXConditionPipeline")

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

        pt_image_slice = pt_image.frames[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@ddt
@slow
class LTXConditionPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = LTXConditionPipeline.from_pretrained("Lightricks/LTX-Video-0.9.5", mindspore_dtype=ms_dtype)

        video = load_downloaded_video_from_hf_hub(
            "huggingface/documentation-images",
            filename="cosmos-video2world-input-vid.mp4",
            subfolder="diffusers/cosmos",
            repo_type="dataset",
        )
        image = load_downloaded_image_from_hf_hub(
            "huggingface/documentation-images",
            filename="cosmos-video2world-input.jpg",
            subfolder="diffusers/cosmos",
            repo_type="dataset",
        )

        condition1 = LTXVideoCondition(
            image=image,
            frame_index=0,
        )
        condition2 = LTXVideoCondition(
            video=video,
            frame_index=80,
        )

        prompt = "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."  # noqa
        negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

        torch.manual_seed(0)
        image = pipe(
            conditions=[condition1, condition2],
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=768,
            height=512,
            num_frames=161,
            num_inference_steps=40,
        )[0][0][1]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"ltx_condition_{dtype}.npy",
            subfolder="ltx",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
