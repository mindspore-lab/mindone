import random
import unittest

import cv2
from PIL import Image
import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import (
    StableDiffusionXLControlNetPAGPipeline,
    ControlNetModel,
    AutoencoderKL,
)

from ..pipeline_test_utils import (
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
)

from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@slow
@ddt
class StableDiffusionXLControlNetPAGPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):

    def get_inputs(self):
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "hf-logo.png",
            subfolder="sd_controlnet",
        )
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        inputs = {
            "prompt": "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
            "controlnet_conditioning_scale": 0.5,
            "image": canny_image,
            "pag_scale": 0.3,
        }

        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            mindspore_dtype=ms_dtype
        )

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            mindspore_dtype=ms_dtype
        )

        pipeline = StableDiffusionXLControlNetPAGPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            mindspore_dtype=ms_dtype,
            enable_pag=True
        )

        inputs = self.get_inputs()
        torch.manual_seed(0)
        image = pipeline(**inputs)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f'stable_diffusion_3_inpaint_{dtype}.npy',
            subfolder="flux",
        )

        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
