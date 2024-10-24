import tempfile

import numpy as np

import mindspore as ms

from mindone.diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from mindone.diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_downloaded_image_from_hf_hub,
    numpy_cosine_similarity_distance,
    slow,
)

from .single_file_testing_utils import (
    SDXLSingleFileTesterMixin,
    download_diffusers_config,
    download_original_config,
    download_single_file_checkpoint,
)

enable_full_determinism()


@slow
class TestStableDiffusionXLAdapterPipelineSingleFileSlow(SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLAdapterPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )

    def get_inputs(self):
        prompt = "toy"
        generator = np.random.default_rng(0)
        image = load_downloaded_image_from_hf_hub(
            repo_id="hf-internal-testing/diffusers-images",
            subfolder="t2i_adapter",
            filename="toy_canny.png",
        )

        inputs = {
            "prompt": prompt,
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 7.5,
            "output_type": "np",
        }

        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe_single_file = StableDiffusionXLAdapterPipeline.from_single_file(
            self.ckpt_path,
            adapter=adapter,
            mindspore_dtype=ms.float16,
            safety_checker=None,
        )
        pipe_single_file.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        images_single_file = pipe_single_file(**inputs)[0][0]

        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            self.repo_id,
            adapter=adapter,
            mindspore_dtype=ms.float16,
            safety_checker=None,
        )

        inputs = self.get_inputs()
        images = pipe(**inputs)[0][0]

        assert images_single_file.shape == (768, 512, 3)
        assert images.shape == (768, 512, 3)

        max_diff = numpy_cosine_similarity_distance(images.flatten(), images_single_file.flatten())
        assert max_diff < 5e-3

    def test_single_file_components(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            adapter=adapter,
            mindspore_dtype=ms.float16,
        )

        pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path, safety_checker=None, adapter=adapter)
        super().test_single_file_components(pipe, pipe_single_file)

    def test_single_file_components_local_files_only(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            adapter=adapter,
            mindspore_dtype=ms.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)

            single_file_pipe = self.pipeline_class.from_single_file(
                local_ckpt_path, adapter=adapter, safety_checker=None, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_diffusers_config(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            adapter=adapter,
            mindspore_dtype=ms.float16,
            safety_checker=None,
        )

        pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path, config=self.repo_id, adapter=adapter)
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config_local_files_only(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            adapter=adapter,
            mindspore_dtype=ms.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_diffusers_config = download_diffusers_config(self.repo_id, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                config=local_diffusers_config,
                adapter=adapter,
                safety_checker=None,
                local_files_only=True,
            )
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_original_config(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            adapter=adapter,
            mindspore_dtype=ms.float16,
            safety_checker=None,
        )

        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, original_config=self.original_config, adapter=adapter
        )
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_original_config_local_files_only(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            adapter=adapter,
            mindspore_dtype=ms.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_original_config = download_original_config(self.original_config, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                original_config=local_original_config,
                adapter=adapter,
                safety_checker=None,
                local_files_only=True,
            )
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_setting_pipeline_dtype_to_fp16(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)

        single_file_pipe = self.pipeline_class.from_single_file(
            self.ckpt_path, adapter=adapter, mindspore_dtype=ms.float16
        )
        super().test_single_file_setting_pipeline_dtype_to_fp16(single_file_pipe)
