"""Adapted from https://github.com/huggingface/diffusers/tree/main/tests//pipelines/unidiffuser/test_unidiffuser.py."""

import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
    get_module,
    get_pipeline_components,
)

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@ddt
class UniDiffuserPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_config = [
        [
            "vae",
            "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="vae",
                revision="refs/pr/1",
            ),
        ],
        [
            "text_encoder",
            "transformers.models.clip.modeling_clip.CLIPTextModel",
            "mindone.transformers.models.clip.modeling_clip.CLIPTextModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="text_encoder",
                revision="refs/pr/1",
            ),
        ],
        [
            "image_encoder",
            "transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            "mindone.transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="image_encoder",
                revision="refs/pr/1",
            ),
        ],
        [
            "clip_image_processor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            "transformers.models.clip.image_processing_clip.CLIPImageProcessor",
            dict(
                crop_size=32,
                size=32,
            ),
        ],
        [
            "clip_tokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            "transformers.models.clip.tokenization_clip.CLIPTokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="clip_tokenizer",
                revision="refs/pr/1",
            ),
        ],
        [
            "text_decoder",
            "diffusers.pipelines.unidiffuser.modeling_text_decoder.UniDiffuserTextDecoder",
            "mindone.diffusers.pipelines.unidiffuser.modeling_text_decoder.UniDiffuserTextDecoder",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="text_decoder",
                revision="refs/pr/1",
            ),
        ],
        [
            "text_tokenizer",
            "transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer",
            "transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="text_tokenizer",
                revision="refs/pr/1",
            ),
        ],
        [
            "unet",
            "diffusers.pipelines.unidiffuser.modeling_uvit.UniDiffuserModel",
            "mindone.diffusers.pipelines.unidiffuser.modeling_uvit.UniDiffuserModel",
            dict(
                pretrained_model_name_or_path="hf-internal-testing/unidiffuser-diffusers-test",
                subfolder="unet",
                revision="refs/pr/1",
            ),
        ],
        [
            "scheduler",
            "diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            "mindone.diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler",
            dict(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                solver_order=3,
            ),
        ],
    ]

    def get_dummy_components(self):
        components = {
            key: None
            for key in [
                "vae",
                "text_encoder",
                "image_encoder",
                "clip_image_processor",
                "clip_tokenizer",
                "text_decoder",
                "text_tokenizer",
                "unet",
                "scheduler",
            ]
        }
        return get_pipeline_components(components, self.pipeline_config)

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.permute(0, 2, 3, 1)[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")

        inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def get_fixed_latents(self, seed=0):
        generator = torch.manual_seed(seed)
        # Hardcode the shapes for now.
        prompt_latents = randn_tensor((1, 77, 32), generator=generator, device=torch.device("cpu"), dtype=torch.float32)
        vae_latents = randn_tensor((1, 4, 16, 16), generator=generator, device=torch.device("cpu"), dtype=torch.float32)
        clip_latents = randn_tensor((1, 1, 32), generator=generator, device=torch.device("cpu"), dtype=torch.float32)

        pt_prompt_latents, pt_vae_latents, pt_clip_latents = prompt_latents, vae_latents, clip_latents
        ms_prompt_latents, ms_vae_latents, ms_clip_latents = (
            ms.tensor(prompt_latents.numpy()),
            ms.tensor(vae_latents.numpy()),
            ms.tensor(clip_latents.numpy()),
        )

        pt_latents = {
            "prompt_latents": pt_prompt_latents,
            "vae_latents": pt_vae_latents,
            "clip_latents": pt_clip_latents,
        }
        ms_latents = {
            "prompt_latents": ms_prompt_latents,
            "vae_latents": ms_vae_latents,
            "clip_latents": ms_clip_latents,
        }
        return pt_latents, ms_latents

    def get_dummy_inputs_with_latents(self, seed=0):
        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "unidiffuser_example_image.jpg",
            subfolder="unidiffuser",
        )

        image = image.resize((32, 32))
        pt_latents, ms_latents = self.get_fixed_latents(seed=seed)

        pt_inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "prompt_latents": pt_latents.get("pt_prompt_latents"),
            "vae_latents": pt_latents.get("pt_vae_latents"),
            "clip_latents": pt_latents.get("pt_clip_latents"),
        }
        ms_inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
            "prompt_latents": ms_latents.get("ms_prompt_latents"),
            "vae_latents": ms_latents.get("ms_vae_latents"),
            "clip_latents": ms_latents.get("ms_clip_latents"),
        }
        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_unidiffuser_default_joint(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()

        pt_pipe_cls = get_module("diffusers.pipelines.unidiffuser.UniDiffuserPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.unidiffuser.UniDiffuserPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)

        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)
        if pt_pipe.text_decoder.transformer.transformer.wte.weight.is_meta:
            pt_pipe.text_decoder.transformer.transformer.wte.weight = torch.randn(
                pt_pipe.text_decoder.transformer.transformer.wte.weight.shape,
                dtype=pt_pipe.text_decoder.transformer.transformer.wte.weight.dtype,
            )
        pt_pipe.text_decoder.transformer.lm_head.weight = pt_pipe.text_decoder.transformer.transformer.wte.weight
        weight = ms.Tensor(pt_pipe.text_decoder.transformer.lm_head.weight.detach().numpy())
        ms_pipe.text_decoder.transformer.lm_head.weight = weight
        ms_pipe.text_decoder.transformer.transformer.wte.embedding_table = weight

        # inputs = self.get_dummy_inputs(device)
        pt_inputs, ms_inputs = self.get_dummy_inputs_with_latents()
        # Delete prompt and image for joint inference.
        del pt_inputs["prompt"]
        del pt_inputs["image"]
        del ms_inputs["prompt"]
        del ms_inputs["image"]

        # Set mode to 'joint'
        pt_pipe.set_joint_mode()
        assert pt_pipe.mode == "joint"

        torch.manual_seed(42)
        pt_sample = pt_pipe(**pt_inputs)
        pt_image = pt_sample.images
        pt_text = pt_sample.text
        assert pt_image.shape == (1, 32, 32, 3)

        ms_pipe.set_joint_mode()
        assert ms_pipe.mode == "joint"

        torch.manual_seed(42)
        ms_sample = ms_pipe(**ms_inputs)
        ms_image = ms_sample.images
        ms_text = ms_sample.text
        assert ms_image.shape == (1, 32, 32, 3)

        pt_image_slice = pt_image[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0, -3:, -3:, -1]
        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold

        assert pt_text[0][:10] == ms_text[0][:10]

    @data(*test_cases)
    @unpack
    def test_unidiffuser_default_text2img(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()

        pt_pipe_cls = get_module("diffusers.pipelines.unidiffuser.UniDiffuserPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.unidiffuser.UniDiffuserPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)

        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        # inputs = self.get_dummy_inputs(device)
        pt_inputs, ms_inputs = self.get_dummy_inputs_with_latents()
        # Delete prompt and image for joint inference.
        del pt_inputs["image"]
        del ms_inputs["image"]

        # Set mode to 'text2img'
        pt_pipe.set_text_to_image_mode()
        assert pt_pipe.mode == "text2img"

        torch.manual_seed(42)
        pt_image = pt_pipe(**pt_inputs).images
        assert pt_image.shape == (1, 32, 32, 3)

        ms_pipe.set_text_to_image_mode()
        assert ms_pipe.mode == "text2img"

        torch.manual_seed(42)
        ms_image = ms_pipe(**ms_inputs).images
        assert ms_image.shape == (1, 32, 32, 3)

        pt_image_slice = pt_image[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0, -3:, -3:, -1]
        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold

    @data(*test_cases)
    @unpack
    def test_unidiffuser_default_img2text(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()

        pt_pipe_cls = get_module("diffusers.pipelines.unidiffuser.UniDiffuserPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.unidiffuser.UniDiffuserPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)

        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        if pt_pipe.text_decoder.transformer.transformer.wte.weight.is_meta:
            pt_pipe.text_decoder.transformer.transformer.wte.weight = torch.randn(
                pt_pipe.text_decoder.transformer.transformer.wte.weight.shape,
                dtype=pt_pipe.text_decoder.transformer.transformer.wte.weight.dtype,
            )
        pt_pipe.text_decoder.transformer.lm_head.weight = pt_pipe.text_decoder.transformer.transformer.wte.weight
        weight = ms.Tensor(pt_pipe.text_decoder.transformer.lm_head.weight.detach().numpy())
        ms_pipe.text_decoder.transformer.lm_head.weight = weight
        ms_pipe.text_decoder.transformer.transformer.wte.embedding_table = weight

        # inputs = self.get_dummy_inputs(device)
        pt_inputs, ms_inputs = self.get_dummy_inputs_with_latents()
        # Delete prompt and image for joint inference.
        del pt_inputs["prompt"]
        del ms_inputs["prompt"]

        # Set mode to 'img2text'
        pt_pipe.set_image_to_text_mode()
        assert pt_pipe.mode == "img2text"
        torch.manual_seed(42)
        pt_text = pt_pipe(**pt_inputs).text

        ms_pipe.set_image_to_text_mode()
        assert ms_pipe.mode == "img2text"
        torch.manual_seed(42)
        ms_text = ms_pipe(**pt_inputs).text

        assert pt_text[0][:10] == ms_text[0][:10]


@slow
@ddt
class UniDiffuserPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_unidiffuser_default_joint_v1(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.unidiffuser.UniDiffuserPipeline")

        pipe = pipe_cls.from_pretrained("thu-ml/unidiffuser-v1", mindspore_dtype=ms_dtype, revision="refs/pr/6")

        torch.manual_seed(32)
        sample = pipe(num_inference_steps=20, guidance_scale=8.0)
        image = sample.images[0]
        text = sample.text[0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"joint_{dtype}.npy",
            subfolder="unidiffuser",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

        expected_text_prefix = (
            "A group of motorcycles parked next to each other" if dtype == "float32" else "A railway track in the woods"
        )
        assert text[: len(expected_text_prefix)] == expected_text_prefix

    @data(*test_cases)
    @unpack
    def test_unidiffuser_default_text2img_v1(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.unidiffuser.UniDiffuserPipeline")

        pipe = pipe_cls.from_pretrained("thu-ml/unidiffuser-v1", mindspore_dtype=ms_dtype, revision="refs/pr/6")

        torch.manual_seed(42)

        prompt = "an elephant under the sea"
        sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
        image = sample.images[0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"t2i_{dtype}.npy",
            subfolder="unidiffuser",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

    @data(*test_cases)
    @unpack
    def test_unidiffuser_default_img2text_v1(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe_cls = get_module("mindone.diffusers.pipelines.unidiffuser.UniDiffuserPipeline")

        pipe = pipe_cls.from_pretrained("thu-ml/unidiffuser-v1", mindspore_dtype=ms_dtype, revision="refs/pr/6")

        image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "unidiffuser_example_image.jpg",
            subfolder="unidiffuser",
        )
        image = image.resize((512, 512))
        torch.manual_seed(42)
        sample = pipe(image=image, num_inference_steps=20, guidance_scale=8.0)
        text = sample.text[0]

        expected_text_prefix = (
            "An astronaut flying in space with the Earth in the background"
            if dtype == "float32"
            else "An image of an astronaut flying over the earth"
        )
        assert text[: len(expected_text_prefix)] == expected_text_prefix
