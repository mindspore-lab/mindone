"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/animatediff/test_animatediff_sparsectrl.py."""

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers import (
    AnimateDiffSparseControlNetPipeline,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    MotionAdapter,
    SparseControlNetModel,
)
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
]


@ddt
class AnimateDiffSparseControlNetPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            "mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel",
            dict(
                block_out_channels=(8, 8),
                layers_per_block=2,
                sample_size=8,
                in_channels=4,
                out_channels=4,
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=8,
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
            "diffusers.models.controlnets.controlnet_sparsectrl.SparseControlNetModel",
            "mindone.diffusers.models.controlnets.controlnet_sparsectrl.SparseControlNetModel",
            dict(
                block_out_channels=(8, 8),
                layers_per_block=2,
                in_channels=4,
                conditioning_channels=3,
                down_block_types=("CrossAttnDownBlockMotion", "DownBlockMotion"),
                cross_attention_dim=8,
                conditioning_embedding_out_channels=(8, 8),
                norm_num_groups=1,
                use_simplified_condition_embedding=False,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=(8, 8),
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
                    hidden_size=8,
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
                block_out_channels=(8, 8),
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
        conditioning_frames = [Image.new("RGB", (video_width, video_height))] * num_frames

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "conditioning_frames": conditioning_frames,
            "controlnet_frame_indices": list(range(num_frames)),
            "num_inference_steps": 2,
            "num_frames": num_frames,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.animatediff.AnimateDiffSparseControlNetPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.animatediff.AnimateDiffSparseControlNetPipeline")

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
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class AnimateDiffSparseControlNetPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
        controlnet_id = "guoyww/animatediff-sparsectrl-scribble"
        lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
        vae_id = "stabilityai/sd-vae-ft-mse"

        motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, mindspore_dtype=ms_dtype)
        controlnet = SparseControlNetModel.from_pretrained(controlnet_id, mindspore_dtype=ms_dtype)
        vae = AutoencoderKL.from_pretrained(vae_id, mindspore_dtype=ms_dtype)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            beta_schedule="linear",
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(
            model_id,
            motion_adapter=motion_adapter,
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            mindspore_dtype=ms_dtype,
        )
        pipe.load_lora_weights(
            lora_adapter_id, adapter_name="motion_lora", weight_name="diffusion_pytorch_model.safetensors"
        )
        pipe.fuse_lora(lora_scale=1.0)

        prompt = "an aerial view of a cyberpunk city, night time, neon lights, masterpiece, high quality"
        negative_prompt = "low quality, worst quality, letterboxed"

        image_files = [
            ["huggingface/documentation-images", "animatediff-scribble-1.png", "diffusers"],
            ["huggingface/documentation-images", "animatediff-scribble-2.png", "diffusers"],
            ["huggingface/documentation-images", "animatediff-scribble-3.png", "diffusers"],
        ]
        condition_frame_indices = [0, 8, 15]
        conditioning_frames = [load_downloaded_image_from_hf_hub(*img_file) for img_file in image_files]

        torch.manual_seed(0)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            conditioning_frames=conditioning_frames,
            controlnet_conditioning_scale=1.0,
            controlnet_frame_indices=condition_frame_indices,
        )[0][0][1]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"sparsecontrolnet_{dtype}.npy",
            subfolder="animatediff",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
