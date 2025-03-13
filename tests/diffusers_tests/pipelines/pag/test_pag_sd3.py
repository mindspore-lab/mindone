import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig

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
class StableDiffusion3PAGPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel",
            "mindone.diffusers.models.transformers.transformer_sd3.SD3Transformer2DModel",
            dict(
                sample_size=32,
                patch_size=1,
                in_channels=4,
                num_layers=2,
                attention_head_dim=8,
                num_attention_heads=4,
                caption_projection_dim=32,
                joint_attention_dim=32,
                pooled_projection_dim=64,
                out_channels=4,
            ),
        ],
        [
            "text_encoder",
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
                    hidden_act="gelu",
                    projection_dim=32,
                ),
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
                    hidden_act="gelu",
                    projection_dim=32,
                ),
            ),
        ],
        [
            "text_encoder_3",
            "transformers.models.t5.modeling_t5.T5EncoderModel",
            "mindone.transformers.models.t5.modeling_t5.T5EncoderModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
                revision="refs/pr/1",
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
            "tokenizer_2",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
        [
            "tokenizer_3",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5",
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
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "scheduler",
                "text_encoder",
                "text_encoder_2",
                "text_encoder_3",
                "tokenizer",
                "tokenizer_2",
                "tokenizer_3",
                "transformer",
                "vae",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "pag_scale": 0.0,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_pag_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.pag.pipeline_pag_sd_3.StableDiffusion3PAGPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.pag.pipeline_pag_sd_3.StableDiffusion3PAGPipeline")

        pt_pipe_pag = pt_pipe_cls(**pt_components, pag_applied_layers=["blocks.0"])
        ms_pipe_pag = ms_pipe_cls(**ms_components, pag_applied_layers=["blocks.0"])

        pt_pipe_pag.set_progress_bar_config(disable=None)
        ms_pipe_pag.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe_pag = pt_pipe_pag.to(pt_dtype)
        ms_pipe_pag = ms_pipe_pag.to(ms_dtype)

        inputs = self.get_dummy_inputs()
        torch.manual_seed(0)
        pt_image = pt_pipe_pag(**inputs).images
        torch.manual_seed(0)
        ms_image = ms_pipe_pag(**inputs)[0]

        pt_image_slice = pt_image[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold
