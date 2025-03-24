import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import StableAudioPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
]


@ddt
@slow
class StableAudioPipelineIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def get_inputs(self, dtype=ms.float32, seed=0):
        latents = np.random.RandomState(seed).standard_normal((1, 64, 1024))
        latents = ms.Tensor(latents, dtype=dtype)
        inputs = {
            "prompt": "A hammer hitting a wooden surface",
            "latents": latents,
            "num_inference_steps": 3,
            "audio_end_in_s": 30,
            "guidance_scale": 2.5,
        }
        return inputs

    @data(*test_cases)
    @unpack
    def test_stable_audio(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        stable_audio_pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0", mindspore_dtype=ms_dtype
        )
        stable_audio_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 25
        torch.manual_seed(0)
        audio = stable_audio_pipe(**inputs).audios[0]

        expected_audio = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"stable_audio_{dtype}.npy",
            subfolder="stable_audio",
        )

        assert np.mean(np.abs(np.array(audio, dtype=np.float32) - expected_audio.T)) < THRESHOLD_PIXEL
