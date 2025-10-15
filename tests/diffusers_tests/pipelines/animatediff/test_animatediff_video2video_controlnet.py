"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/animatediff/test_animatediff_video2video_controlnet.py."""

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers import (
    AnimateDiffVideoToVideoControlNetPipeline,
    AutoencoderKL,
    ControlNetModel,
    LCMScheduler,
    MotionAdapter,
)
from mindone.diffusers.utils.testing_utils import load_downloaded_video_from_hf_hub, load_numpy_from_local_file, slow

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
]


@ddt
class AnimateDiffVideoToVideoControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    cross_attention_dim = 8
    block_out_channels = (8, 8)

    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=block_out_channels,
                layers_per_block=2,
                sample_size=8,
                in_channels=4,
                out_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=cross_attention_dim,
                norm_num_groups=2,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="linear",
                clip_sample=False,
            ),
        ],
        [
            "controlnet",
            "diffusers.models.controlnets.controlnet.ControlNetModel",
            "mindone.diffusers.models.controlnets.controlnet.ControlNetModel",
            dict(
                block_out_channels=block_out_channels,
                layers_per_block=2,
                in_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                cross_attention_dim=cross_attention_dim,
                conditioning_embedding_out_channels=(8, 8),
                norm_num_groups=1,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=block_out_channels,
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
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
                    hidden_size=cross_attention_dim,
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
        [
            "motion_adapter",
            "diffusers.models.unets.unet_motion_model.MotionAdapter",
            "mindone.diffusers.models.unets.unet_motion_model.MotionAdapter",
            dict(
                block_out_channels=block_out_channels,
                motion_layers_per_block=2,
                motion_norm_num_groups=2,
                motion_num_attention_heads=4,
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
                "motion_adapter",
                "text_encoder",
                "tokenizer",
                "feature_extractor",
                "image_encoder",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, num_frames: int = 2):
        video_height = 32
        video_width = 32
        video = [Image.new("RGB", (video_width, video_height))] * num_frames

        video_height = 32
        video_width = 32
        conditioning_frames = [Image.new("RGB", (video_width, video_height))] * num_frames

        inputs = {
            "video": video,
            "conditioning_frames": conditioning_frames,
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.animatediff.pipeline_animatediff_video2video_controlnet.AnimateDiffVideoToVideoControlNetPipeline"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.animatediff.pipeline_animatediff_video2video_controlnet.AnimateDiffVideoToVideoControlNetPipeline"
        )

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

        pt_image_slice = pt_frame.frames[0][0, -3:, -3:, -1]
        ms_image_slice = ms_frame[0][0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class AnimateDiffVideoToVideoControlNetPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", mindspore_dtype=ms_dtype)
        motion_adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", mindspore_dtype=ms_dtype)
        pipe = AnimateDiffVideoToVideoControlNetPipeline.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE",
            motion_adapter=motion_adapter,
            controlnet=controlnet,
            vae=vae,
        ).to(dtype=ms_dtype)

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")
        pipe.load_lora_weights(
            "wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora"
        )
        pipe.set_adapters(["lcm-lora"], [0.8])

        video = load_downloaded_video_from_hf_hub(
            "huggingface/documentation-images",
            "dance.gif",
            subfolder="diffusers",
        )
        video = [frame.convert("RGB") for frame in video]

        prompt = "astronaut in space, dancing"
        negative_prompt = "bad quality, worst quality, jpeg artifacts, ugly"

        conditioning_videos = load_numpy_from_local_file(
            "mindone-testing-arrays",
            "v2v_controlnet_conditioning_frames.npy",
            subfolder="animatediff",
        )
        conditioning_frames = []
        for frame in conditioning_videos:
            conditioning_frames.append(Image.fromarray(frame))

        strength = 0.8
        torch.manual_seed(0)
        video = pipe(
            video=video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=10,
            guidance_scale=2.0,
            controlnet_conditioning_scale=0.75,
            conditioning_frames=conditioning_frames,
            strength=strength,
        )[0][0]

        expected_video = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"v2v_controlnet_{dtype}.npy",
            subfolder="animatediff",
        )
        assert np.mean(np.abs(np.array(video, dtype=np.float32) - expected_video)) < THRESHOLD_PIXEL
