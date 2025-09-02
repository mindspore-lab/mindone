"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/flux/test_pipeline_flux_controlnet.py."""

import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers import FluxControlNetModel, FluxControlNetPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
]


@ddt
class FluxControlNetInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_flux.FluxTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_flux.FluxTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=16,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=16,
                num_attention_heads=2,
                joint_attention_dim=32,
                pooled_projection_dim=32,
                axes_dims_rope=[4, 4, 8],
            ),
        ],
        [
            "controlnet",
            "diffusers.models.controlnets.controlnet_flux.FluxControlNetModel",
            "mindone.diffusers.models.controlnets.controlnet_flux.FluxControlNetModel",
            dict(
                patch_size=1,
                in_channels=16,
                num_layers=1,
                num_single_layers=1,
                attention_head_dim=16,
                num_attention_heads=2,
                joint_attention_dim=32,
                pooled_projection_dim=32,
                axes_dims_rope=[4, 4, 8],
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
                    hidden_act="gelu",
                    projection_dim=32,
                ),
            ),
        ],
        [
            "text_encoder_2",
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
            dict(pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip"),
        ],
        [
            "tokenizer_2",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            "transformers.models.auto.tokenization_auto.AutoTokenizer",
            dict(pretrained_model_name_or_path="hf-internal-testing/tiny-random-t5"),
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
                "tokenizer",
                "tokenizer_2",
                "transformer",
                "vae",
                "controlnet",
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        generator = torch.manual_seed(seed)

        pt_control_image = randn_tensor(
            (1, 3, 32, 32),
            generator=generator,
        )
        ms_control_image = ms.tensor(pt_control_image.numpy())

        controlnet_conditioning_scale = 0.5

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 3.5,
            "output_type": "np",
            "control_image": pt_control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "num_inference_steps": 2,
            "guidance_scale": 3.5,
            "output_type": "np",
            "control_image": ms_control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }
        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.flux.pipeline_flux_controlnet.FluxControlNetPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.flux.pipeline_flux_controlnet.FluxControlNetPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@slow
@ddt
class FluxControlNetPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self):
        control_image = load_downloaded_image_from_hf_hub(
            "InstantX/FLUX.1-dev-Controlnet-Canny",
            "canny.jpg",
            subfolder=None,
            repo_type="model",
        )
        inputs = {
            "prompt": "A girl in city, 25 years old, cool, futuristic",
            "control_image": control_image,
            "control_guidance_start": 0.2,
            "control_guidance_end": 0.8,
            "controlnet_conditioning_scale": 1.0,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_canny(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline has precision issue in float32.")

        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-controlnet-canny", mindspore_dtype=ms_dtype
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", controlnet=controlnet, mindspore_dtype=ms_dtype
        )
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"flux_controlnet_{dtype}.npy",
            subfolder="flux",
        )

        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
