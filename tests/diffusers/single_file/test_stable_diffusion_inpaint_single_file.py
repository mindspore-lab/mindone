import numpy as np
import pytest

import mindspore as ms

from mindone.diffusers import StableDiffusionInpaintPipeline
from mindone.diffusers.utils.testing_utils import enable_full_determinism, load_downloaded_image_from_hf_hub, slow

from .single_file_testing_utils import SDSingleFileTesterMixin

enable_full_determinism()


@slow
@pytest.mark.skip("No safetensors files to use in this repo.")
class TestStableDiffusionInpaintPipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionInpaintPipeline
    ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt"
    original_config = "https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inpainting-inference.yaml"
    repo_id = "runwayml/stable-diffusion-inpainting"

    def get_inputs(self, dtype=ms.float32, seed=0):
        generator = np.random.default_rng(seed)
        init_image = load_downloaded_image_from_hf_hub(
            repo_id="diffusers/test-arrays",
            subfolder="stable_diffusion_inpaint",
            filename="input_bench_image.png",
        )
        mask_image = load_downloaded_image_from_hf_hub(
            repo_id="diffusers/test-arrays",
            subfolder="stable_diffusion_inpaint",
            filename="input_bench_mask.png",
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)

    def test_single_file_loading_4_channel_unet(self):
        # Test loading single file inpaint with a 4 channel UNet
        ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
        pipe = self.pipeline_class.from_single_file(ckpt_path)

        assert pipe.unet.config.in_channels == 4


@slow
class TestStableDiffusion21InpaintPipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionInpaintPipeline
    ckpt_path = (
        "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.safetensors"
    )
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inpainting-inference.yaml"
    repo_id = "stabilityai/stable-diffusion-2-inpainting"

    def get_inputs(self, dtype=ms.float32, seed=0):
        generator = np.random.default_rng(seed)
        init_image = load_downloaded_image_from_hf_hub(
            repo_id="diffusers/test-arrays",
            subfolder="stable_diffusion_inpaint",
            filename="input_bench_image.png",
        )
        mask_image = load_downloaded_image_from_hf_hub(
            repo_id="diffusers/test-arrays",
            subfolder="stable_diffusion_inpaint",
            filename="input_bench_mask.png",
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
