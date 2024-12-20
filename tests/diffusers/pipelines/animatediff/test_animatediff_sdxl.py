import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
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
class AnimateDiffPipelineSDXLFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
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
                sample_size=128,
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
            "motion_adapter",
            "diffusers.models.unets.unet_motion_model.MotionAdapter",
            "mindone.diffusers.models.unets.unet_motion_model.MotionAdapter",
            dict(
                block_out_channels=(32, 64, 128),
                motion_layers_per_block=2,
                motion_norm_num_groups=2,
                motion_num_attention_heads=4,
                use_motion_mid_block=False,
            ),
        ],
    ]

    def get_unet_config(self, time_cond_proj_dim=None):
        return [
            "unet",
            "diffusers.models.unets.unet_motion_model.UNetMotionModel",
            "mindone.diffusers.models.unets.unet_motion_model.UNetMotionModel",
            dict(
                block_out_channels=(32, 64, 128),
                layers_per_block=2,
                time_cond_proj_dim=time_cond_proj_dim,
                sample_size=32,
                in_channels=4,
                out_channels=4,
                down_block_types=("DownBlockMotion", "CrossAttnDownBlockMotion", "CrossAttnDownBlockMotion"),
                up_block_types=("CrossAttnUpBlockMotion", "CrossAttnUpBlockMotion", "UpBlockMotion"),
                # SD2-specific config below
                num_attention_heads=(2, 4, 8),
                use_linear_projection=True,
                addition_embed_type="text_time",
                addition_time_embed_dim=8,
                transformer_layers_per_block=(1, 2, 4),
                projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
                cross_attention_dim=64,
                norm_num_groups=1,
            ),
        ]

    def get_dummy_components(self, time_cond_proj_dim=None):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "motion_adapter",
                "text_encoder",
                "tokenizer",
                "text_encoder_2",
                "tokenizer_2",
                "feature_extractor",
                "image_encoder",
            ]
        }
        unet = self.get_unet_config(time_cond_proj_dim)
        pipeline_config = [unet] + self.pipeline_config

        return get_pipeline_components(components, pipeline_config)

    def get_dummy_inputs(self, seed=0):
        inputs = {
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
        pt_pipe_cls = get_module("diffusers.pipelines.animatediff.AnimateDiffSDXLPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.animatediff.AnimateDiffSDXLPipeline")

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
class AnimateDiffPipelineSDXLIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline will OOM in float32")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        adapter_cls = get_module("mindone.diffusers.models.unets.unet_motion_model.MotionAdapter")
        adapter = adapter_cls.from_pretrained(
            "a-r-r-o-w/animatediff-motion-adapter-sdxl-beta", mindspore_dtype=ms_dtype
        )

        vae_cls = get_module("mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL")
        vae = vae_cls.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms_dtype)

        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        scheduler_cls = get_module("mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler")
        scheduler = scheduler_cls.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipe_cls = get_module("mindone.diffusers.pipelines.animatediff.AnimateDiffSDXLPipeline")
        pipe = pipe_cls.from_pretrained(
            model_id,
            vae=vae,
            motion_adapter=adapter,
            scheduler=scheduler,
            mindspore_dtype=ms_dtype,
            variant="fp16",
        )

        torch.manual_seed(0)
        output = pipe(
            prompt="a panda surfing in the ocean, realistic, high quality",
            negative_prompt="low quality, worst quality",
            num_inference_steps=20,
            guidance_scale=8,
            width=768,
            height=768,
            num_frames=8,
            output_type="np",
        )
        frames = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"t2v_sdxl_{dtype}.npy",
            subfolder="animatediff",
        )
        # The pipeline uses a specific threshold, since the random initialization of two parameters in diffusers, which
        # causes the randomness to be uncontrollable.
        assert np.linalg.norm(expected_image - frames) / np.linalg.norm(expected_image) < 2.0
