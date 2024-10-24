import numpy as np

import mindspore as ms

from mindone.diffusers import DDIMScheduler, StableDiffusionXLImg2ImgPipeline
from mindone.diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_downloaded_image_from_hf_hub,
    numpy_cosine_similarity_distance,
    slow,
)

from .single_file_testing_utils import SDXLSingleFileTesterMixin

enable_full_determinism()


@slow
class TestStableDiffusionXLImg2ImgPipelineSingleFileSlow(SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )

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
class TestStableDiffusionXLImg2ImgRefinerPipelineSingleFileSlow:
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    ckpt_path = (
        "https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors"
    )
    repo_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_refiner.yaml"
    )

    def test_single_file_format_inference_is_same_as_pretrained(self):
        init_image = load_downloaded_image_from_hf_hub(
            repo_id="diffusers/test-arrays",
            subfolder="stable_diffusion_img2img",
            filename="sketch-mountains-input.png",
        )

        pipe = self.pipeline_class.from_pretrained(self.repo_id, mindspore_dtype=ms.float16)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_default_attn_processor()

        generator = np.random.default_rng(0)
        image = pipe(
            prompt="mountains", image=init_image, num_inference_steps=5, generator=generator, output_type="np"
        )[0][0]

        pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path, mindspore_dtype=ms.float16)
        pipe_single_file.scheduler = DDIMScheduler.from_config(pipe_single_file.scheduler.config)
        pipe_single_file.unet.set_default_attn_processor()

        generator = np.random.default_rng(0)
        image_single_file = pipe_single_file(
            prompt="mountains", image=init_image, num_inference_steps=5, generator=generator, output_type="np"
        )[0][0]

        max_diff = numpy_cosine_similarity_distance(image.flatten(), image_single_file.flatten())

        assert max_diff < 5e-4
