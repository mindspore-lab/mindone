"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/controlnet/test_controlnet_union_sdxl_img2img.py."""

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import AutoencoderKL, ControlNetUnionModel, StableDiffusionXLControlNetUnionImg2ImgPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@slow
@ddt
class ControlNetUnionPipelineSDXLImg2ImgIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        # If strict mode is disabled, this pipeline will have performance issues.
        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        else:
            ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet = ControlNetUnionModel.from_pretrained(
            "brad-twinkl/controlnet-union-sdxl-1.0-promax", mindspore_dtype=ms_dtype
        )
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms_dtype)
        pipe = StableDiffusionXLControlNetUnionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            mindspore_dtype=ms_dtype,
            variant="fp16",
        )

        prompt = "A cat"
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat.png",
            subfolder="kandinsky",
        )

        torch.manual_seed(0)
        image = pipe(
            prompt=prompt,
            image=image,
            control_image=[image],
            control_mode=[6],
            num_inference_steps=30,
        )[
            0
        ][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"controlnet_union_sdxl_img2img_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
