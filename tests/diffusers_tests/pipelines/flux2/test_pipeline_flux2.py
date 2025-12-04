"""Adapted from https://github.com/huggingface/diffusers/blob/main/tests/pipelines/flux2/test_pipeline_flux2.py."""

import unittest

import diffusers
import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from packaging.version import Version
from transformers import Mistral3Config

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [{"mode": 1, "dtype": "bfloat16"}, {"mode": 0, "dtype": "bfloat16"}]


@ddt
class Flux2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.Flux2Transformer2DModel",
            "mindone.diffusers.Flux2Transformer2DModel",
            dict(
                patch_size=1,
                in_channels=4,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=16,
                num_attention_heads=2,
                joint_attention_dim=16,
                timestep_guidance_channels=256,  # Hardcoded in original code
                axes_dims_rope=[4, 4, 4, 4],
            ),
        ],
        [
            "text_encoder",
            "transformers.Mistral3ForConditionalGeneration",
            "mindone.transformers.Mistral3ForConditionalGeneration",
            dict(
                config=Mistral3Config(
                    text_config={
                        "model_type": "mistral",
                        "vocab_size": 32000,
                        "hidden_size": 16,
                        "intermediate_size": 37,
                        "max_position_embeddings": 512,
                        "num_attention_heads": 4,
                        "num_hidden_layers": 1,
                        "num_key_value_heads": 2,
                        "rms_norm_eps": 1e-05,
                        "rope_theta": 1000000000.0,
                        "sliding_window": None,
                        "bos_token_id": 2,
                        "eos_token_id": 3,
                        "pad_token_id": 4,
                    },
                    vision_config={
                        "model_type": "pixtral",
                        "hidden_size": 16,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "intermediate_size": 37,
                        "image_size": 30,
                        "patch_size": 6,
                        "num_channels": 3,
                    },
                    bos_token_id=2,
                    eos_token_id=3,
                    pad_token_id=4,
                    model_dtype="mistral3",
                    image_seq_length=4,
                    vision_feature_layer=-1,
                    image_token_index=1,
                ),
            ),
        ],
        [
            "tokenizer",
            "transformers.AutoProcessor",
            "transformers.AutoProcessor",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/Mistral-Small-3.1-24B-Instruct-2503-only-processor",
            ),
        ],
        [
            "vae",
            "diffusers.AutoencoderKLFlux2",
            "mindone.diffusers.AutoencoderKLFlux2",
            dict(
                sample_size=32,
                in_channels=3,
                out_channels=3,
                down_block_types=("DownEncoderBlock2D",),
                up_block_types=("UpDecoderBlock2D",),
                block_out_channels=(4,),
                layers_per_block=1,
                latent_channels=1,
                norm_num_groups=1,
                use_quant_conv=False,
                use_post_quant_conv=False,
            ),
        ],
        [
            "scheduler",
            "diffusers.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.FlowMatchEulerDiscreteScheduler",
            dict(),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "scheduler",
                "text_encoder",
                "tokenizer",
                "transformer",
                "vae",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "a dog is dancing",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 8,
            "output_type": "np",
            "text_encoder_out_layers": (1,),
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        required_version = Version("0.35.2")
        current_version = Version(diffusers.__version__)
        if current_version <= required_version:
            pytest.skip(f"Flux2Pipeline is not supported in diffusers version {current_version}")

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        if mode == 0:
            ms_pipe.transformer.construct = ms.jit(ms_pipe.transformer.construct)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold
