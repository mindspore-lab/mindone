import numpy as np

import mindspore as ms

from mindone.diffusers import StableDiffusionImg2ImgPipeline
from mindone.diffusers.utils.testing_utils import enable_full_determinism, load_downloaded_image_from_hf_hub, slow

from .single_file_testing_utils import SDSingleFileTesterMixin

enable_full_determinism()


@slow
class TestStableDiffusionImg2ImgPipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionImg2ImgPipeline
    ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
    original_config = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    repo_id = "runwayml/stable-diffusion-v1-5"

    def get_inputs(self, dtype=ms.float32, seed=0):
        generator = np.random.default_rng(seed)
        init_image = load_downloaded_image_from_hf_hub(
            repo_id="diffusers/test-arrays",
            subfolder="stable_diffusion_img2img",
            filename="sketch-mountains-input.png",
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)


@slow
class TestStableDiffusion21Img2ImgPipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionImg2ImgPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
    repo_id = "stabilityai/stable-diffusion-2-1"

    def get_inputs(self, dtype=ms.float32, seed=0):
        generator = np.random.default_rng(seed)
        init_image = load_downloaded_image_from_hf_hub(
            repo_id="diffusers/test-arrays",
            subfolder="stable_diffusion_img2img",
            filename="sketch-mountains-input.png",
        )
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "image": init_image,
            "generator": generator,
            "num_inference_steps": 3,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
