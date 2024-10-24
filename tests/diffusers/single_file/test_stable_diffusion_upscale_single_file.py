import numpy as np

from mindone.diffusers import StableDiffusionUpscalePipeline
from mindone.diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_downloaded_image_from_hf_hub,
    numpy_cosine_similarity_distance,
    slow,
)

from .single_file_testing_utils import SDSingleFileTesterMixin

enable_full_determinism()


@slow
class TestStableDiffusionUpscalePipelineSingleFileSlow(SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionUpscalePipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/blob/main/x4-upscaler-ema.safetensors"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/x4-upscaling.yaml"
    )
    repo_id = "stabilityai/stable-diffusion-x4-upscaler"

    def test_single_file_format_inference_is_same_as_pretrained(self):
        image = load_downloaded_image_from_hf_hub(
            repo_id="hf-internal-testing/diffusers-images",
            subfolder="sd2-upscale",
            filename="low_res_cat.png",
        )

        prompt = "a cat sitting on a park bench"
        pipe = StableDiffusionUpscalePipeline.from_pretrained(self.repo_id)

        generator = np.random.default_rng(0)
        output = pipe(prompt=prompt, image=image, generator=generator, output_type="np", num_inference_steps=3)
        image_from_pretrained = output[0][0]

        pipe_from_single_file = StableDiffusionUpscalePipeline.from_single_file(self.ckpt_path)

        generator = np.random.default_rng(0)
        output_from_single_file = pipe_from_single_file(
            prompt=prompt, image=image, generator=generator, output_type="np", num_inference_steps=3
        )
        image_from_single_file = output_from_single_file[0][0]

        assert image_from_pretrained.shape == (512, 512, 3)
        assert image_from_single_file.shape == (512, 512, 3)
        assert (
            numpy_cosine_similarity_distance(image_from_pretrained.flatten(), image_from_single_file.flatten()) < 1e-3
        )
