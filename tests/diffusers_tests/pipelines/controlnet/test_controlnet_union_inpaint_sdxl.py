"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/controlnet/test_controlnet_union_inpaint_sdxl.py."""

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from PIL import Image

import mindspore as ms

from mindone.diffusers import AutoencoderKL, ControlNetUnionModel, StableDiffusionXLControlNetUnionInpaintPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@slow
@ddt
class ControlNetUnionPipelineSDXLInpaintIntegrationTests(PipelineTesterMixin, unittest.TestCase):
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
        pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            mindspore_dtype=ms_dtype,
            variant="fp16",
        )

        prompt = "A cat"
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "overture-creations-5sI6fQgYIuo.png",
            subfolder="in_paint",
        ).resize((1024, 1024))
        mask = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "overture-creations-5sI6fQgYIuo_mask.png",
            subfolder="in_paint",
        ).resize((1024, 1024))

        controlnet_img = image.copy()
        controlnet_img_np = np.array(controlnet_img)
        mask_np = np.array(mask)
        controlnet_img_np[mask_np > 0] = 0
        controlnet_img = Image.fromarray(controlnet_img_np)

        torch.manual_seed(0)
        image = pipe(prompt, image=image, mask_image=mask, control_image=[controlnet_img], control_mode=[7])[0][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"controlnet_union_inpaint_sdxl_{dtype}.npy",
            subfolder="controlnet",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
