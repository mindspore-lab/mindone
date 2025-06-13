import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import Gemma2Config

import mindspore as ms

from mindone.diffusers import Lumina2Pipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

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
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "bfloat16"},
]


@ddt
class Lumina2PipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_lumina2.Lumina2Transformer2DModel",
            "mindone.diffusers.models.transformers.transformer_lumina2.Lumina2Transformer2DModel",
            dict(
                sample_size=4,
                patch_size=2,
                in_channels=4,
                hidden_size=8,
                num_layers=2,
                num_attention_heads=1,
                num_kv_heads=1,
                multiple_of=16,
                ffn_dim_multiplier=None,
                norm_eps=1e-5,
                scaling_factor=1.0,
                axes_dim_rope=[4, 2, 2],
                cap_feat_dim=8,
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
                block_out_channels=(4,),
                layers_per_block=1,
                latent_channels=4,
                norm_num_groups=1,
                use_quant_conv=False,
                use_post_quant_conv=False,
                shift_factor=0.0609,
                scaling_factor=1.5035,
            ),
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
            "transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            "mindone.transformers.models.gemma2.modeling_gemma2.Gemma2Model",
            dict(
                config=Gemma2Config(
                    head_dim=4,
                    hidden_size=8,
                    intermediate_size=8,
                    num_attention_heads=2,
                    num_hidden_layers=2,
                    num_key_value_heads=2,
                    sliding_window=2,
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
        eval_components = ["vae"]

        for component in eval_components:
            pt_components[component] = pt_components[component].eval()
            ms_components[component] = ms_components[component].set_train(False)

        return pt_components, ms_components

    def get_dummy_inputs(self):
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_lumina_prompt_embeds(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.lumina2.pipeline_lumina2.Lumina2Pipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.lumina2.pipeline_lumina2.Lumina2Pipeline")

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


@slow
@ddt
class Lumina2PipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_lumina_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = Lumina2Pipeline.from_pretrained("Alpha-VLLM/Lumina-Image-2.0", mindspore_dtype=ms_dtype)

        prompt = "Upper body of a young woman in a Victorian-era outfit with brass goggles and leather straps. Background shows an industrial revolution cityscape with smoky skies and tall, metal structures"  # noqa
        torch.manual_seed(0)
        image = pipe(prompt=prompt)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"t2i_{dtype}.npy",
            subfolder="lumina2",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
