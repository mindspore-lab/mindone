import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import DDIMScheduler, TextToVideoZeroPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
]


@ddt
@slow
class StableTextToVideoZeroPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_text_to_video_zero(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe = TextToVideoZeroPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms_dtype
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        prompt = "A bear is playing a guitar on Times Square"
        torch.manual_seed(42)
        image = pipe(prompt=prompt).images[0] * 255

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"t2v_synth_zero_{dtype}.npy",
            subfolder="text_to_video_synthesis",
        )

        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
