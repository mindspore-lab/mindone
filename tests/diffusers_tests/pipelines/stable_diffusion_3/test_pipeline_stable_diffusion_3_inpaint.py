import random
import unittest

import cv2
from PIL import Image
import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import (
    StableDiffusion3InpaintPipeline,
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
        # img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        # mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
        #
        # from mindone.diffusers.utils.testing_utils import load_image
        # source = load_image(img_url)
        # mask = load_image(mask_url)

        source = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "inpaint_input.png",
            subfolder="stable_diffusion_xl",
            repo_type="dataset",
        )
        mask = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "inpaint_mask.png",
            subfolder="stable_diffusion_xl",
            repo_type="dataset",
        )

        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": source,
            "mask_image": mask,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = StableDiffusion3InpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            mindspore_dtype=ms_dtype
        )

        inputs = self.get_inputs()
        torch.manual_seed(0)
        image = pipe(**inputs)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f'stable_diffusion_3_inpaint_{dtype}.npy',
            subfolder="flux",
        )

        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
