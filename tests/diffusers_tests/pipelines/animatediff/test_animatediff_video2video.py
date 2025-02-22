import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from PIL import Image
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import (
    load_downloaded_numpy_from_hf_hub,
    load_downloaded_video_from_hf_hub,
    slow,
)

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
class AnimateDiffVideoToVideoPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
                down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=32,
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
        [
            "motion_adapter",
            "diffusers.models.unets.unet_motion_model.MotionAdapter",
            "mindone.diffusers.models.unets.unet_motion_model.MotionAdapter",
            dict(
                block_out_channels=(32, 64),
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

    def get_dummy_inputs(self, seed=0):
        video_height = 32
        video_width = 32
        video_num_frames = 2
        video = [Image.new("RGB", (video_width, video_height))] * video_num_frames

        inputs = {
            "video": video,
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
        pt_pipe_cls = get_module("diffusers.pipelines.animatediff.AnimateDiffVideoToVideoPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.animatediff.AnimateDiffVideoToVideoPipeline")

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
class AnimateDiffVideoToVideoPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline will OOM in float32")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        adapter_cls = get_module("mindone.diffusers.models.unets.unet_motion_model.MotionAdapter")
        adapter = adapter_cls.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", mindspore_dtype=ms_dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.animatediff.AnimateDiffVideoToVideoPipeline")
        pipe = pipe_cls.from_pretrained(
            "SG161222/Realistic_Vision_V5.1_noVAE", motion_adapter=adapter, mindspore_dtype=ms_dtype
        )
        scheduler_cls = get_module("mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler")
        pipe.scheduler = scheduler_cls(
            beta_schedule="linear", steps_offset=1, clip_sample=False, timestep_spacing="linspace"
        )

        video = load_downloaded_video_from_hf_hub(
            "huggingface/documentation-images",
            "animatediff-vid2vid-input-1.gif",
            subfolder="diffusers",
        )

        torch.manual_seed(0)
        output = pipe(
            video=video,
            prompt="panda playing a guitar, on a boat, in the ocean, high quality",
            strength=0.5,
            output_type="np",
        )
        frames = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"v2v_{dtype}.npy",
            subfolder="animatediff",
        )
        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(expected_image - frames) / np.linalg.norm(expected_image) < threshold
