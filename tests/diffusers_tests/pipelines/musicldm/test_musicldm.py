"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/musicldm/test_musicldm.py."""

import unittest

import diffusers
import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from packaging.version import Version
from transformers.models.clap.configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
from transformers.models.speecht5.configuration_speecht5 import SpeechT5HifiGanConfig

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

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
]


@ddt
class MusicLDMPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
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
                down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
                up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
                cross_attention_dim=(32, 64),
                class_embed_type="simple_projection",
                projection_class_embeddings_input_dim=32,
                class_embeddings_concat=True,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            "mindone.diffusers.schedulers.scheduling_ddim.DDIMScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                block_out_channels=[32, 64],
                in_channels=1,
                out_channels=1,
                down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
                latent_channels=4,
            ),
        ],
        [
            "tokenizer",
            "transformers.models.roberta.tokenization_roberta.RobertaTokenizer",
            "transformers.models.roberta.tokenization_roberta.RobertaTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-roberta",
                model_max_length=77,
            ),
        ],
        [
            "feature_extractor",
            "transformers.models.clap.feature_extraction_clap.ClapFeatureExtractor",
            "mindone.transformers.models.clap.feature_extraction_clap.ClapFeatureExtractor",
            dict(pretrained_model_name_or_path="hf-internal-testing/tiny-random-ClapModel", hop_length=7900),
        ],
        [
            "text_encoder",
            "transformers.models.clap.modeling_clap.ClapModel",
            "mindone.transformers.models.clap.modeling_clap.ClapModel",
            dict(
                config=ClapConfig.from_text_audio_configs(
                    text_config=ClapTextConfig(
                        bos_token_id=0,
                        eos_token_id=2,
                        hidden_size=16,
                        intermediate_size=37,
                        layer_norm_eps=1e-05,
                        num_attention_heads=2,
                        num_hidden_layers=2,
                        pad_token_id=1,
                        vocab_size=1000,
                    ),
                    audio_config=ClapAudioConfig(
                        spec_size=64,
                        window_size=4,
                        num_mel_bins=64,
                        intermediate_size=37,
                        layer_norm_eps=1e-05,
                        depths=[2, 2],
                        num_attention_heads=[2, 2],
                        num_hidden_layers=2,
                        hidden_size=192,
                        patch_size=2,
                        patch_stride=2,
                        patch_embed_input_channels=4,
                    ),
                    projection_dim=32,
                )
            ),
        ],
        [
            "vocoder",
            "transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan",
            "mindone.transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan",
            dict(
                config=SpeechT5HifiGanConfig(
                    model_in_dim=8,
                    sampling_rate=16000,
                    upsample_initial_channel=16,
                    upsample_rates=[2, 2],
                    upsample_kernel_sizes=[4, 4],
                    resblock_kernel_sizes=[3, 7],
                    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
                    normalize_before=False,
                )
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "unet",
                "scheduler",
                "text_encoder",
                "vae",
                "tokenizer",
                "feature_extractor",
                "vocoder",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        required_version = Version("0.33.1")
        current_version = Version(diffusers.__version__)
        if current_version > required_version:
            pytest.skip(f"MusicLDMPipeline is not supported in diffusers version {current_version}")

        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.musicldm.pipeline_musicldm.MusicLDMPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.musicldm.pipeline_musicldm.MusicLDMPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        latents = np.random.RandomState(0).standard_normal((1, 4, 32, 4))
        pt_audio = pt_pipe(**inputs, latents=torch.from_numpy(latents).to(device="cpu", dtype=pt_dtype)).audios
        torch.manual_seed(0)
        ms_audio = ms_pipe(**inputs, latents=ms.Tensor(latents).to(ms_dtype)).audios

        pt_generated_audio = pt_audio[0][:10]
        ms_generated_audio = ms_audio[0][:10]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_generated_audio - ms_generated_audio) / np.linalg.norm(pt_generated_audio) < threshold


@slow
@ddt
class MusicLDMPipelineNightlyTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.musicldm.pipeline_musicldm.MusicLDMPipeline")
        pipe = pipe_cls.from_pretrained("ucsd-reach/musicldm", mindspore_dtype=ms_dtype)
        torch.manual_seed(0)
        latents = np.random.RandomState(0).standard_normal((1, 8, 128, 16))
        latents = ms.Tensor(latents).to(ms_dtype)
        prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
        audios = pipe(
            prompt=prompt,
            latents=latents,
            num_inference_steps=10,
            audio_length_in_s=5.0,
        ).audios
        audio = audios[0]

        expected_audio = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"musicldm_{dtype}.npy",
            subfolder="musicldm",
        )

        threshold = 5e-2 if dtype == "float32" else 3e-1
        assert np.linalg.norm(audio - expected_audio) / np.linalg.norm(expected_audio) < threshold
