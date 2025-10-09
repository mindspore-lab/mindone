"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/controlnet/test_controlnet_union_sdxl.py."""

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import AutoencoderKL, ControlNetUnionModel, StableDiffusionXLControlNetUnionPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@slow
@ddt
class ControlNetUnionPipelineSDXLIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        # If strict mode is disabled, this pipeline will have performance issues.
        if mode == ms.GRAPH_MODE:
            ms.set_context(mode=mode, jit_syntax_level=ms.STRICT)
        else:
            ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet = ControlNetUnionModel.from_pretrained("xinsir/controlnet-union-sdxl-1.0", mindspore_dtype=ms_dtype)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", mindspore_dtype=ms_dtype)
        pipe = StableDiffusionXLControlNetUnionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            mindspore_dtype=ms_dtype,
            variant="fp16",
        )

        prompt = "A cat"
        controlnet_img = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "controlnet_img.png",
            subfolder="controlnet",
        )

        torch.manual_seed(0)
        image = pipe(prompt, control_image=[controlnet_img], control_mode=[3], height=1024, width=1024)[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"controlnet_union_sdxl_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
