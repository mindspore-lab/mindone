import random
import unittest

import numpy as np
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

# float32 in StableVideoDiffusion is not supported
test_cases = [
    # {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    # {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class StableVideoDiffusionPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "unet",
            "diffusers.models.unets.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel",
            "mindone.diffusers.models.unets.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel",
            dict(
                block_out_channels=(32, 64),
                layers_per_block=2,
                sample_size=32,
                in_channels=8,
                out_channels=4,
                down_block_types=(
                    "CrossAttnDownBlockSpatioTemporal",
                    "DownBlockSpatioTemporal",
                ),
                up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
                cross_attention_dim=32,
                num_attention_heads=8,
                projection_class_embeddings_input_dim=96,
                addition_time_embed_dim=32,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                interpolation_type="linear",
                num_train_timesteps=1000,
                prediction_type="v_prediction",
                sigma_max=700.0,
                sigma_min=0.002,
                steps_offset=1,
                timestep_spacing="leading",
                timestep_type="continuous",
                trained_betas=None,
                use_karras_sigmas=True,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl_temporal_decoder.AutoencoderKLTemporalDecoder",
            "mindone.diffusers.models.autoencoders.autoencoder_kl_temporal_decoder.AutoencoderKLTemporalDecoder",
            dict(
                block_out_channels=[32, 64],
                in_channels=3,
                out_channels=3,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                latent_channels=4,
            ),
        ],
        [
            "image_encoder",
            "transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            dict(
                config=CLIPVisionConfig(
                    hidden_size=32,
                    projection_dim=32,
                    num_hidden_layers=5,
                    num_attention_heads=4,
                    image_size=32,
                    intermediate_size=37,
                    patch_size=1,
                ),
            ),
        ],
        [
            "feature_extractor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(
                crop_size=32,
                size=32,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "vae",
                "feature_extractor",
                "image_encoder",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        pt_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "image": pt_image,
            "num_inference_steps": 2,
            "output_type": "np",
            "min_guidance_scale": 1.0,
            "max_guidance_scale": 2.5,
            "num_frames": 2,
            "height": 32,
            "width": 32,
        }

        ms_inputs = {
            "image": ms_image,
            "num_inference_steps": 2,
            "output_type": "np",
            "min_guidance_scale": 1.0,
            "max_guidance_scale": 2.5,
            "num_frames": 2,
            "height": 32,
            "width": 32,
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_stable_video_diffusion_default_case(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.stable_video_diffusion.StableVideoDiffusionPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.stable_video_diffusion.StableVideoDiffusionPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_output = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_output = ms_pipe(**ms_inputs)

        pt_output_slice = pt_output.frames[0, -1, -3:, -3:, -1]
        ms_output_slice = ms_output[0, -1, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_output_slice - ms_output_slice) / np.linalg.norm(pt_output_slice) < threshold


@slow
@ddt
class StableVideoDiffusionPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_sd_video(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.stable_video_diffusion.StableVideoDiffusionPipeline")
        pipe = pipe_cls.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            variant="fp16",
            mindspore_dtype=ms_dtype,
        )
        pipe.set_progress_bar_config(disable=None)

        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat_6.png",
            subfolder="pix2pix",
        )
        num_frames = 3

        torch.manual_seed(0)
        output = pipe(
            image=image,
            num_frames=num_frames,
            num_inference_steps=3,
        )

        image = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"sd_video_{dtype}.npy",
            subfolder="stable_video_diffusion",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
