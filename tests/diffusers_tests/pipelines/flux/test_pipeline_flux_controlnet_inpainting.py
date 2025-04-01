import random
import unittest

import numpy as np
import pytest
import torch
from ddt import data, ddt, unpack
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers import FluxControlNetInpaintPipeline
from mindone.diffusers.models import FluxControlNetModel
from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
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
class FluxControlNetInpaintPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_flux.FluxTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_flux.FluxTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=8,
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
                latent_channels=2,
                norm_num_groups=1,
                use_quant_conv=False,
                use_post_quant_conv=False,
                shift_factor=0.0609,
                scaling_factor=1.5035,
            ),
        ],
        [
            "controlnet",
            "diffusers.models.controlnets.controlnet_flux.FluxControlNetModel",
            "mindone.diffusers.models.controlnets.controlnet_flux.FluxControlNetModel",
            dict(
                patch_size=1,
                in_channels=8,
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
        pt_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        pt_mask_image = torch.ones((1, 1, 32, 32))
        pt_control_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))

        ms_image = ms.tensor(pt_image.numpy())
        ms_mask_image = ms.tensor(pt_mask_image.numpy())
        ms_control_image = ms.tensor(pt_control_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": pt_image,
            "mask_image": pt_mask_image,
            "control_image": pt_control_image,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            "strength": 0.8,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": ms_image,
            "mask_image": ms_mask_image,
            "control_image": ms_control_image,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            "strength": 0.8,
            "output_type": "np",
        }
        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module(
            "diffusers.pipelines.flux.pipeline_flux_controlnet_inpainting.FluxControlNetInpaintPipeline"
        )
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.flux.pipeline_flux_controlnet_inpainting.FluxControlNetInpaintPipeline"
        )

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


@ddt
@slow
class FluxControlNetInpaintPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self):
        # control_image = load_image(
        #     "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg"
        # )
        control_image = load_downloaded_image_from_hf_hub(
            "InstantX/FLUX.1-dev-Controlnet-Canny",
            "canny.jpg",
            subfolder=None,
            repo_type="model",
        )
        init_image = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "inpaint_input.png",
            subfolder="stable_diffusion_xl",
            repo_type="dataset",
        )
        mask_image = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "inpaint_mask.png",
            subfolder="stable_diffusion_xl",
            repo_type="dataset",
        )

        inputs = {
            "prompt": "A girl holding a sign that says InstantX",
            "image": init_image,
            "mask_image": mask_image,
            "control_image": control_image,
            "control_guidance_start": 0.2,
            "control_guidance_end": 0.8,
            "controlnet_conditioning_scale": 0.7,
            "strength": 0.7,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        if dtype == "float32":
            pytest.skip("Skipping this case since this pipeline has precision issue in float32.")

        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-controlnet-canny", mindspore_dtype=ms_dtype
        )
        pipe = FluxControlNetInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", controlnet=controlnet, mindspore_dtype=ms_dtype
        )

        inputs = self.get_inputs()
        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"flux_controlnet_inpainting_{dtype}.npy",
            subfolder="flux",
        )

        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
