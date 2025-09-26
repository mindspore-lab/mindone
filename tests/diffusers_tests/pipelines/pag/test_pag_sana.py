"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/pag/test_pag_sana.py."""

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers.models.gemma2 import Gemma2Config

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
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@ddt
class SanaPAGPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.sana_transformer.SanaTransformer2DModel",
            "mindone.diffusers.models.transformers.sana_transformer.SanaTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=4,
                out_channels=4,
                num_layers=2,
                num_attention_heads=2,
                attention_head_dim=4,
                num_cross_attention_heads=2,
                cross_attention_head_dim=4,
                cross_attention_dim=8,
                caption_channels=8,
                sample_size=32,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_dc.AutoencoderDC",
            "mindone.diffusers.models.autoencoders.autoencoder_dc.AutoencoderDC",
            dict(
                in_channels=3,
                latent_channels=4,
                attention_head_dim=2,
                encoder_block_types=(
                    "ResBlock",
                    "EfficientViTBlock",
                ),
                decoder_block_types=(
                    "ResBlock",
                    "EfficientViTBlock",
                ),
                encoder_block_out_channels=(8, 8),
                decoder_block_out_channels=(8, 8),
                encoder_qkv_multiscales=((), (5,)),
                decoder_qkv_multiscales=((), (5,)),
                encoder_layers_per_block=(1, 1),
                decoder_layers_per_block=[1, 1],
                downsample_block_type="conv",
                upsample_block_type="interpolate",
                decoder_norm_types="rms_norm",
                decoder_act_fns="silu",
                scaling_factor=0.41407,
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(shift=7.0),
        ],
        [
            "tokenizer",
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer",
            "transformers.models.gemma.tokenization_gemma.GemmaTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/dummy-gemma",
            ),
        ],
        [
            "text_encoder",
            "transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            "mindone.transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            dict(
                config=Gemma2Config(
                    head_dim=16,
                    hidden_size=8,
                    initializer_range=0.02,
                    intermediate_size=64,
                    max_position_embeddings=8192,
                    model_type="gemma2",
                    num_attention_heads=2,
                    num_hidden_layers=1,
                    num_key_value_heads=2,
                    attn_implementation="eager",
                )
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
                "tokenizer",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "",
            "negative_prompt": "",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "pag_scale": 3.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 16,
            "output_type": "np",
            "complex_human_instruction": None,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.pag.pipeline_pag_sana.SanaPAGPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.pag.pipeline_pag_sana.SanaPAGPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs).images[0, -3:, -3:, -1]
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image - ms_image) / np.linalg.norm(pt_image)) < threshold


@slow
@ddt
class SanaPAGPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.pag.pipeline_pag_sana.SanaPAGPipeline")
        pipe = pipe_cls.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            pag_applied_layers=["transformer_blocks.8"],
            mindspore_dtype=ms_dtype,
        )

        torch.manual_seed(0)
        pipe.text_encoder.to(ms.bfloat16)
        pipe.transformer = pipe.transformer.to(ms.bfloat16)
        prompt = 'a cyberpunk cat with a neon sign that says "Sana"'
        image = pipe(prompt=prompt)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"pag_sana_{dtype}.npy",
            subfolder="pag",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
