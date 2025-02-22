import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import GemmaConfig

import mindspore as ms

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
class LuminaText2ImgPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.lumina_nextdit2d.LuminaNextDiT2DModel",
            "mindone.diffusers.models.transformers.lumina_nextdit2d.LuminaNextDiT2DModel",
            dict(
                sample_size=4,
                patch_size=2,
                in_channels=4,
                hidden_size=4,
                num_layers=2,
                num_attention_heads=1,
                num_kv_heads=1,
                multiple_of=16,
                ffn_dim_multiplier=None,
                norm_eps=1e-5,
                learn_sigma=True,
                qk_norm=True,
                cross_attention_dim=8,
                scaling_factor=1.0,
            ),
        ],
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            "mindone.diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler",
            dict(),
        ],
        [
            "tokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/dummy-gemma",
            ),
        ],
        [
            "text_encoder",
            "transformers.models.gemma.modeling_gemma.GemmaForCausalLM",
            "mindone.transformers.models.gemma.modeling_gemma.GemmaForCausalLM",
            dict(
                config=GemmaConfig(
                    head_dim=2,
                    hidden_size=8,
                    intermediate_size=37,
                    num_attention_heads=4,
                    num_hidden_layers=2,
                    num_key_value_heads=4,
                ),
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
        eval_components = ["transformer", "vae", "text_encoder"]

        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_lumina_prompt_embeds(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.lumina.pipeline_lumina.LuminaText2ImgPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.lumina.pipeline_lumina.LuminaText2ImgPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_output_with_prompt = pt_pipe(**inputs).images[0]
        torch.manual_seed(0)
        ms_output_with_prompt = ms_pipe(**inputs)[0][0]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert (
            np.linalg.norm(pt_output_with_prompt - ms_output_with_prompt) / np.linalg.norm(pt_output_with_prompt)
            < threshold
        )
