"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/flux/test_pipeline_flux_control_inpaint.py."""

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from image_gen_aux import DepthPreprocessor
from PIL import Image
from transformers import CLIPTextConfig

import mindspore as ms

from mindone.diffusers import FluxControlInpaintPipeline
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
class FluxControlPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "transformer",
            "diffusers.models.transformers.transformer_flux.FluxTransformer2DModel",
            "mindone.diffusers.models.transformers.transformer_flux.FluxTransformer2DModel",
            dict(
                patch_size=1,
                in_channels=8,
                out_channels=4,
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
            dict(
                pretrained_model_name_or_path="hf-internal-testing/tiny-random-clip",
            ),
        ],
        [
            "tokenizer_2",
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
                latent_channels=1,
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
            ]
        }

        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self):
        image = Image.new("RGB", (8, 8), 0)
        control_image = Image.new("RGB", (8, 8), 0)
        mask_image = Image.new("RGB", (8, 8), 255)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "control_image": control_image,
            "image": image,
            "mask_image": mask_image,
            "strength": 0.8,
            "num_inference_steps": 2,
            "guidance_scale": 30.0,
            "height": 8,
            "width": 8,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.flux.pipeline_flux_control_inpaint.FluxControlInpaintPipeline")
        ms_pipe_cls = get_module(
            "mindone.diffusers.pipelines.flux.pipeline_flux_control_inpaint.FluxControlInpaintPipeline"
        )

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
class FluxControlInpaintPipelineslowTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self):
        image = load_downloaded_image_from_hf_hub(
            "huggingface/documentation-images", filename="robot.png", subfolder=None, repo_type="dataset"
        )

        head_mask = np.zeros_like(image)
        head_mask[65:580, 300:642] = 255
        mask_image = Image.fromarray(head_mask)

        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        control_image = processor(image)[0].convert("RGB")

        inputs = {
            "prompt": "a blue robot singing opera with human-like expressions",
            "image": image,
            "control_image": control_image,
            "mask_image": mask_image,
            "num_inference_steps": 30,
            "strength": 0.9,
            "guidance_scale": 10.0,
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = FluxControlInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Depth-dev",
            mindspore_dtype=ms_dtype,
        )

        inputs = self.get_inputs()
        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"flux_inpaint_control_{dtype}.npy",
            subfolder="flux",
        )

        assert (
            np.mean(np.abs(np.array(image, dtype=np.float32) - np.array(expected_image, dtype=np.float32)))
            < THRESHOLD_PIXEL
        )
