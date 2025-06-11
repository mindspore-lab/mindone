import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import FluxPipeline, FluxPriorReduxPipeline
from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "bfloat16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@slow
@ddt
class FluxReduxSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_flux_redux_inference(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        repo_redux = "black-forest-labs/FLUX.1-Redux-dev"
        repo_base = "black-forest-labs/FLUX.1-dev"
        pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(repo_redux, mindspore_dtype=ms_dtype)
        pipe = FluxPipeline.from_pretrained(repo_base, text_encoder=None, text_encoder_2=None, mindspore_dtype=ms_dtype)

        image = load_downloaded_image_from_hf_hub(
            "YiYiXu/testing-images",
            "img5.png",
            subfolder="style_ziggy",
        )
        prompt_embeds, pooled_prompt_embeds = pipe_prior_redux(image)

        torch.manual_seed(0)
        image = pipe(
            num_inference_steps=2,
            guidance_scale=2.0,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"redux_{dtype}.npy",
            subfolder="flux",
        )
        assert np.linalg.norm(expected_image - image) / np.linalg.norm(expected_image) < THRESHOLD_PIXEL
