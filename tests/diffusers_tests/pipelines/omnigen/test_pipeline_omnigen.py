"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/omnigen/test_pipeline_omnigen.py."""

import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_numpy_from_local_file, slow

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
class OmniGenPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_omnigen.OmniGenTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_omnigen.OmniGenTransformer2DModel",
            dict(
                hidden_size=16,
                num_attention_heads=4,
                num_key_value_heads=4,
                intermediate_size=32,
                num_layers=1,
                in_channels=4,
                time_step_dim=4,
                rope_scaling={"long_factor": list(range(1, 3)), "short_factor": list(range(1, 3))},
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                sample_size=32,
                in_channels=3,
                out_channels=3,
                block_out_channels=(4, 4, 4, 4),
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=1,
                up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(invert_sigmas=True, num_train_timesteps=1),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(pretrained_model_name_or_path="hf-internal-testing/llama-tokenizer"),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "transformer",
                "vae",
                "scheduler",
                "tokenizer",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 1,
            "guidance_scale": 3.0,
            "output_type": "np",
            "height": 16,
            "width": 16,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.omnigen.pipeline_omnigen.OmniGenPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.omnigen.pipeline_omnigen.OmniGenPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class OmniGenPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_omnigen_inference(self, mode, dtype):
        if dtype == "float16":
            pytest.skip(
                "Skipping this case since the pipeline generates black pic in float16 in both pytorch and mindspore"
            )
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.omnigen.pipeline_omnigen.OmniGenPipeline")
        pipe = pipe_cls.from_pretrained("shitao/OmniGen-v1-diffusers", mindspore_dtype=ms_dtype)

        prompt = "A photo of a cat"
        torch.manual_seed(0)
        image = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=2.5)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays", f"omnigen_{dtype}.npy", subfolder="omnigen"
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
