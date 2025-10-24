import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig, T5GemmaModuleConfig

import mindspore as ms

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class PRXPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.PRXTransformer2DModel",
            "mindone.diffusers.PRXTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=4,
                context_in_dim=8,
                hidden_size=8,
                mlp_ratio=2.0,
                num_heads=2,
                depth=1,
                axes_dim=[2, 2],
            ),
        ],
        [
            "vae",
            "diffusers.AutoencoderKL",
            "mindone.diffusers.AutoencoderKL",
            dict(
                sample_size=32,
                in_channels=3,
                out_channels=3,
                block_out_channels=(4,),
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=1,
                use_quant_conv=False,
                use_post_quant_conv=False,
                shift_factor=0.0,
                scaling_factor=1.0,
            ),
        ],
        [
            "scheduler",
            "diffusers.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.FlowMatchEulerDiscreteScheduler",
            dict(),
        ],
        [
            "tokenizer",
            "transformers.AutoTokenizer",
            "transformers.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/dummy-gemma",
                model_max_length=64,
            ),
        ],
        [
            "text_encoder",
            "transformers.models.t5gemma.modeling_t5gemma.T5GemmaEncoder",
            "mindone.transformers.models.t5gemma.modeling_t5gemma.T5GemmaEncoder",
            dict(
                config=T5GemmaConfig(
                    encoder=T5GemmaModuleConfig(
                        vocab_size=8,
                        hidden_size=8,
                        intermediate_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=4,
                        max_position_embeddings=64,
                        layer_types=["full_attention"],
                        attention_bias=False,
                        attention_dropout=0.0,
                        dropout_rate=0.0,
                        hidden_activation="gelu_pytorch_tanh",
                        rms_norm_eps=1e-06,
                        attn_logit_softcapping=50.0,
                        final_logit_softcapping=30.0,
                        query_pre_attn_scalar=4,
                        rope_theta=10000.0,
                        sliding_window=4096,
                    ),
                    is_encoder_decoder=False,
                    vocab_size=8,
                    hidden_size=8,
                    intermediate_size=16,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=4,
                    max_position_embeddings=64,
                    layer_types=["full_attention"],
                    attention_bias=False,
                    attention_dropout=0.0,
                    dropout_rate=0.0,
                    hidden_activation="gelu_pytorch_tanh",
                    rms_norm_eps=1e-06,
                    attn_logit_softcapping=50.0,
                    final_logit_softcapping=30.0,
                    query_pre_attn_scalar=4,
                    rope_theta=10000.0,
                    sliding_window=4096,
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
                "text_encoder",
                "tokenizer",
            ]
        }

        pt_components, ms_components = get_pipeline_components(components, self.pipeline_config)
        pt_components["vae"] = pt_components["vae"].eval()
        ms_components["vae"] = ms_components["vae"].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self):
        return {
            "prompt": "",
            "negative_prompt": "",
            "num_inference_steps": 2,
            "guidance_scale": 1.0,
            "height": 32,
            "width": 32,
            "output_type": "pt",
            "use_resolution_binning": False,
        }

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.PRXPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.PRXPipeline")

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
