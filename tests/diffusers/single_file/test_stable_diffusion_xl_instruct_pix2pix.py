import numpy as np

import mindspore as ms

from mindone.diffusers import StableDiffusionXLInstructPix2PixPipeline
from mindone.diffusers.utils.testing_utils import enable_full_determinism, slow

enable_full_determinism()


@slow
class TestStableDiffusionXLInstructPix2PixPipeline:
    pipeline_class = StableDiffusionXLInstructPix2PixPipeline
    ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
    original_config = None
    repo_id = "diffusers/sdxl-instructpix2pix-768"

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

    def test_single_file_setting_cosxl_edit(self):
        # Default is PNDM for this checkpoint
        pipe = self.pipeline_class.from_single_file(self.ckpt_path, config=self.repo_id, is_cosxl_edit=True)
        assert pipe.is_cosxl_edit is True
