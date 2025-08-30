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
from PIL import Image

import mindspore as ms

from mindone.diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

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
class WanVACEPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
            "diffusers.models.transformers.transformer_wan_vace.WanVACETransformer3DModel",
            "mindone.diffusers.models.transformers.transformer_wan_vace.WanVACETransformer3DModel",
            dict(
                patch_size=(1, 2, 2),
                num_attention_heads=2,
                attention_head_dim=12,
                in_channels=16,
                out_channels=16,
                text_dim=32,
                freq_dim=256,
                ffn_dim=32,
                num_layers=3,
                cross_attn_norm=True,
                qk_norm="rms_norm_across_heads",
                rope_max_seq_len=32,
                vace_layers=[0, 2],
                vace_in_channels=96,
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
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        num_frames = 17
        height = 16
        width = 16

        video = [Image.new("RGB", (height, width))] * num_frames
        mask = [Image.new("L", (height, width), 0)] * num_frames

        inputs = {
            "video": video,
            "mask": mask,
            "prompt": "dance monkey",
            "negative_prompt": "negative",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "height": 16,
            "width": 16,
            "num_frames": num_frames,
            "max_sequence_length": 16,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.wan.WanVACEPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.wan.WanVACEPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_pipe.vae.to(torch.float32)
        ms_pipe.vae.to(ms.float32)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_frame = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_frame = ms_pipe(**inputs)

        pt_frame_slice = pt_frame.frames[0][0, -3:, -3:, -1]
        ms_frame_slice = ms_frame[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_frame_slice - ms_frame_slice) / np.linalg.norm(pt_frame_slice) < threshold

    @data(*test_cases)
    @unpack
    def test_inference_with_single_reference_image(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.wan.WanVACEPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.wan.WanVACEPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_pipe.vae.to(torch.float32)
        ms_pipe.vae.to(ms.float32)

        inputs = self.get_dummy_inputs()
        inputs["reference_images"] = Image.new("RGB", (16, 16))

        torch.manual_seed(0)
        pt_frame = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_frame = ms_pipe(**inputs)

        pt_frame_slice = pt_frame.frames[0][0, -3:, -3:, -1]
        ms_frame_slice = ms_frame[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_frame_slice - ms_frame_slice) / np.linalg.norm(pt_frame_slice) < threshold

    @data(*test_cases)
    @unpack
    def test_inference_with_multiple_reference_image(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.wan.WanVACEPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.wan.WanVACEPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_pipe.vae.to(torch.float32)
        ms_pipe.vae.to(ms.float32)

        inputs = self.get_dummy_inputs()
        inputs["reference_images"] = [[Image.new("RGB", (16, 16))] * 2]

        torch.manual_seed(0)
        pt_frame = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_frame = ms_pipe(**inputs)

        pt_frame_slice = pt_frame.frames[0][0, -3:, -3:, -1]
        ms_frame_slice = ms_frame[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_frame_slice - ms_frame_slice) / np.linalg.norm(pt_frame_slice) < threshold


def prepare_video_and_mask(first_img: Image.Image, last_img: Image.Image, height: int, width: int, num_frames: int):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = [first_img]
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = Image.new("L", (width, height), 0)
    mask_white = Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask


@slow
@ddt
class WanVACEPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", mindspore_dtype=ms_dtype)
        pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, mindspore_dtype=ms_dtype)
        flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

        prompt = (
            "CG animation style, a small blue bird takes off from the ground, flapping its wings. The bird's "
            "feathers are delicate, with a unique pattern on its chest. The background shows a blue sky with "
            "white clouds under bright sunshine. The camera follows the bird upward, capturing its flight and "
            "the vastness of the sky from a close-up, low-angle perspective."
        )
        negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
            "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
            "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
            "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in"
            " the background, walking backwards"
        )
        first_frame = load_downloaded_image_from_hf_hub(
            "huggingface/documentation-images",
            "flf2v_input_first_frame.png",
            subfolder="diffusers",
        )
        last_frame = load_downloaded_image_from_hf_hub(
            "huggingface/documentation-images",
            "flf2v_input_last_frame.png",
            subfolder="diffusers",
        )

        height = 512
        width = 512
        num_frames = 81
        video, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)

        torch.manual_seed(0)
        output = pipe(
            video=video,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=30,
            guidance_scale=5.0,
        )[0][0]

        expected_video = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"wan_vace_{dtype}.npy",
            subfolder="wan_vace",
        )
        assert np.mean(np.abs(np.array(output, dtype=np.float32) - expected_video)) < THRESHOLD_PIXEL
