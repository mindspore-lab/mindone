import tempfile

import numpy as np

import mindspore as ms

from mindone.diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from mindone.diffusers.utils.testing_utils import (
    enable_full_determinism,
    load_downloaded_image_from_hf_hub,
    numpy_cosine_similarity_distance,
    slow,
)

from .single_file_testing_utils import (
    SDXLSingleFileTesterMixin,
    download_diffusers_config,
    download_single_file_checkpoint,
)

enable_full_determinism()


@slow
class TestStableDiffusionXLControlNetPipelineSingleFileSlow(SDXLSingleFileTesterMixin):
    pipeline_class = StableDiffusionXLControlNetPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors"
    repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
    original_config = (
        "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"
    )

    def get_inputs(self, dtype=ms.float32, seed=0):
        generator = np.random.default_rng(seed)
        image = load_downloaded_image_from_hf_hub(
            repo_id="hf-internal-testing/diffusers-images",
            subfolder="sd_controlnet",
            filename="stormtrooper_depth.png",
        )
        inputs = {
            "prompt": "Stormtrooper's lecture",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }

        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, mindspore_dtype=ms.float16
        )
        pipe_single_file.unet.set_default_attn_processor()
        pipe_single_file.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        single_file_images = pipe_single_file(**inputs)[0][0]

        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet, mindspore_dtype=ms.float16)
        pipe.unet.set_default_attn_processor()

        inputs = self.get_inputs()
        images = pipe(**inputs)[0][0]

        assert images.shape == (512, 512, 3)
        assert single_file_images.shape == (512, 512, 3)

        max_diff = numpy_cosine_similarity_distance(images[0].flatten(), single_file_images[0].flatten())
        assert max_diff < 5e-2

    def test_single_file_components(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            mindspore_dtype=ms.float16,
        )

        pipe_single_file = self.pipeline_class.from_single_file(self.ckpt_path, controlnet=controlnet)
        super().test_single_file_components(pipe, pipe_single_file)

    def test_single_file_components_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            mindspore_dtype=ms.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)

            single_file_pipe = self.pipeline_class.from_single_file(
                local_ckpt_path, controlnet=controlnet, safety_checker=None, local_files_only=True
            )

        self._compare_component_configs(pipe, single_file_pipe)

    def test_single_file_components_with_original_config(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            mindspore_dtype=ms.float16,
        )

        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path,
            original_config=self.original_config,
            controlnet=controlnet,
        )
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_original_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            variant="fp16",
            controlnet=controlnet,
            mindspore_dtype=ms.float16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                safety_checker=None,
                controlnet=controlnet,
                local_files_only=True,
            )
        self._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(self.repo_id, controlnet=controlnet)
        pipe_single_file = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, config=self.repo_id
        )

        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_components_with_diffusers_config_local_files_only(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
        )
        pipe = self.pipeline_class.from_pretrained(
            self.repo_id,
            controlnet=controlnet,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_diffusers_config = download_diffusers_config(self.repo_id, tmpdir)

            pipe_single_file = self.pipeline_class.from_single_file(
                local_ckpt_path,
                config=local_diffusers_config,
                safety_checker=None,
                controlnet=controlnet,
                local_files_only=True,
            )
        super()._compare_component_configs(pipe, pipe_single_file)

    def test_single_file_setting_pipeline_dtype_to_fp16(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
        )
        single_file_pipe = self.pipeline_class.from_single_file(
            self.ckpt_path, controlnet=controlnet, safety_checker=None, mindspore_dtype=ms.float16
        )
        super().test_single_file_setting_pipeline_dtype_to_fp16(single_file_pipe)
