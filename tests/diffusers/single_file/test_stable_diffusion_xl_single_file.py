import numpy as np

import mindspore as ms

from mindone.diffusers import StableDiffusionXLPipeline
from mindone.diffusers.utils.testing_utils import enable_full_determinism, slow

from .single_file_testing_utils import SDXLSingleFileTesterMixin

enable_full_determinism()


@slow
class TestStableDiffusionXLPipelineSingleFileSlow(SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )

    def get_inputs(self, dtype=ms.float32, seed=0):
        generator = np.random.default_rng(seed)
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
