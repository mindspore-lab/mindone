# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import unittest

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import CLIPTokenizer

import mindspore as ms

from mindone.diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    StableDiffusionPipeline,
)
from mindone.diffusers.utils.testing_utils import load_image, numpy_cosine_similarity_distance, slow
from mindone.transformers import CLIPTextModel
from tests.diffusers_tests.pipelines.pipeline_test_utils import PipelineTesterMixin

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set  # noqa: E402


class StableDiffusionLoRATests(PeftLoraLoaderMixinTests, unittest.TestCase):
    pipeline_class = StableDiffusionPipeline
    scheduler_cls = DDIMScheduler
    scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "clip_sample": False,
        "set_alpha_to_one": False,
        "steps_offset": 1,
    }
    unet_kwargs = {
        "block_out_channels": (32, 64),
        "layers_per_block": 2,
        "sample_size": 32,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D"),
        "up_block_types": ("CrossAttnUpBlock2D", "UpBlock2D"),
        "cross_attention_dim": 32,
    }
    vae_kwargs = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
    }
    text_encoder_cls, text_encoder_id = CLIPTextModel, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"

    @property
    def output_shape(self):
        return (1, 64, 64, 3)


@slow
class LoraIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def test_integration_logits_with_scale(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, mindspore_dtype=ms.float32)
        pipe.load_lora_weights(lora_id, revision="refs/pr/1")

        self.assertTrue(
            check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder",
        )

        prompt = "a red sks dog"

        images = pipe(
            prompt=prompt,
            num_inference_steps=15,
            cross_attention_kwargs={"scale": 0.5},
            generator=torch.manual_seed(1),
            output_type="np",
        )[0]

        expected_slice_scale = np.array([0.387, 0.388, 0.350, 0.327, 0.330, 0.270, 0.358, 0.328, 0.354])
        predicted_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_integration_logits_no_scale(self):
        path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        lora_id = "takuma104/lora-test-text-encoder-lora-target"

        pipe = StableDiffusionPipeline.from_pretrained(path, mindspore_dtype=ms.float32)
        pipe.load_lora_weights(lora_id, revision="refs/pr/1")

        self.assertTrue(
            check_if_lora_correctly_set(pipe.text_encoder),
            "Lora not correctly set in text encoder",
        )

        prompt = "a red sks dog"

        images = pipe(prompt=prompt, num_inference_steps=30, generator=torch.manual_seed(0), output_type="np")[0]

        expected_slice_scale = np.array([0.074, 0.064, 0.073, 0.0842, 0.069, 0.0641, 0.0794, 0.076, 0.084])
        predicted_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)

        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_dreambooth_old_format(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/lora_dreambooth_dog_example"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe.load_lora_weights(lora_model_id, revision="refs/pr/1")

        images = pipe(
            "A photo of a sks dog floating in the river", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7207, 0.6787, 0.6010, 0.7478, 0.6838, 0.6064, 0.6984, 0.6443, 0.5785])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_dreambooth_text_encoder_new_format(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/lora-trained"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe.load_lora_weights(lora_model_id, revision="refs/pr/1")

        images = pipe("A photo of a sks dog", output_type="np", generator=generator, num_inference_steps=2)[0]

        images = images[0, -3:, -3:, -1].flatten()

        expected = np.array([0.6628, 0.6138, 0.5390, 0.6625, 0.6130, 0.5463, 0.6166, 0.5788, 0.5359])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_a1111(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/Counterfeit-V2.5", safety_checker=None, revision="refs/pr/1"
        )
        lora_model_id = "hf-internal-testing/civitai-light-shadow-lora"
        lora_filename = "light_and_shadow.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.3636, 0.3708, 0.3694, 0.3679, 0.3829, 0.3677, 0.3692, 0.3688, 0.3292])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_lycoris(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/Amixx", safety_checker=None, use_safetensors=True, variant="fp16"
        )
        lora_model_id = "hf-internal-testing/edgLycorisMugler-light"
        lora_filename = "edgLycorisMugler-light.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.6463, 0.658, 0.599, 0.6542, 0.6512, 0.6213, 0.658, 0.6485, 0.6017])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_kohya_sd_v15_with_higher_dimensions(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        )
        lora_model_id = "hf-internal-testing/urushisato-lora"
        lora_filename = "urushisato_v15.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.7165, 0.6616, 0.5833, 0.7504, 0.6718, 0.587, 0.6871, 0.6361, 0.5694])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_vanilla_funetuning(self):
        generator = torch.Generator().manual_seed(0)

        lora_model_id = "hf-internal-testing/sd-model-finetuned-lora-t4"

        base_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        pipe = StableDiffusionPipeline.from_pretrained(base_model_id, safety_checker=None)
        pipe.load_lora_weights(lora_model_id, revision="refs/pr/1")

        images = pipe("A pokemon with blue eyes.", output_type="np", generator=generator, num_inference_steps=2)[0]

        image_slice = images[0, -3:, -3:, -1].flatten()

        expected_slice = np.array([0.6542, 0.61253, 0.5396, 0.6843, 0.6044, 0.5468, 0.6349, 0.5905, 0.5381])

        max_diff = numpy_cosine_similarity_distance(expected_slice, image_slice)
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_unload_kohya_lora(self):
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        )
        initial_images = pipe(prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps)[0]
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps)[0]
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        )[0]
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=1e-3))

    def test_load_unload_load_kohya_lora(self):
        # This test ensures that a Kohya-style LoRA can be safely unloaded and then loaded
        # without introducing any side-effects. Even though the test uses a Kohya-style
        # LoRA, the underlying adapter handling mechanism is format-agnostic.
        generator = torch.manual_seed(0)
        prompt = "masterpiece, best quality, mountain"
        num_inference_steps = 2

        pipe = StableDiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", safety_checker=None
        )
        initial_images = pipe(prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps)[0]
        initial_images = initial_images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/civitai-colored-icons-lora"
        lora_filename = "Colored_Icons_by_vizsumit.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images = pipe(prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps)[0]
        lora_images = lora_images[0, -3:, -3:, -1].flatten()

        pipe.unload_lora_weights()
        generator = torch.manual_seed(0)
        unloaded_lora_images = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        )[0]
        unloaded_lora_images = unloaded_lora_images[0, -3:, -3:, -1].flatten()

        self.assertFalse(np.allclose(initial_images, lora_images))
        self.assertTrue(np.allclose(initial_images, unloaded_lora_images, atol=1e-3))

        # make sure we can load a LoRA again after unloading and they don't have
        # any undesired effects.
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        generator = torch.manual_seed(0)
        lora_images_again = pipe(
            prompt, output_type="np", generator=generator, num_inference_steps=num_inference_steps
        )[0]
        lora_images_again = lora_images_again[0, -3:, -3:, -1].flatten()

        self.assertTrue(np.allclose(lora_images, lora_images_again, atol=1e-3))

    def test_not_empty_state_dict(self):
        # Makes sure https://github.com/huggingface/diffusers/issues/7054 does not happen again
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        cached_file = hf_hub_download("hf-internal-testing/lcm-lora-test-sd-v1-5", "test_lora.safetensors")
        lcm_lora = ms.load_checkpoint(cached_file, format="safetensors")

        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        self.assertTrue(lcm_lora != {})

    def test_load_unload_load_state_dict(self):
        # Makes sure https://github.com/huggingface/diffusers/issues/7054 does not happen again
        pipe = AutoPipelineForText2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        cached_file = hf_hub_download("hf-internal-testing/lcm-lora-test-sd-v1-5", "test_lora.safetensors")
        lcm_lora = ms.load_checkpoint(cached_file, format="safetensors")
        previous_state_dict = lcm_lora.copy()

        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        self.assertDictEqual(lcm_lora, previous_state_dict)

        pipe.unload_lora_weights()
        pipe.load_lora_weights(lcm_lora, adapter_name="lcm")
        self.assertDictEqual(lcm_lora, previous_state_dict)

    def test_sdv1_5_lcm_lora(self):
        pipe = DiffusionPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        generator = torch.Generator().manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdv1-5"
        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "masterpiece, best quality, mountain", generator=generator, num_inference_steps=4, guidance_scale=0.5
        )[0][0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdv15_lcm_lora.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_sdv1_5_lcm_lora_img2img(self):
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", mindspore_dtype=ms.float16
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/fantasy_landscape.png"
        )

        generator = torch.Generator().manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdv1-5"
        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "snowy mountain",
            generator=generator,
            image=init_image,
            strength=0.5,
            num_inference_steps=4,
            guidance_scale=0.5,
        )[0][0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdv15_lcm_lora_img2img.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 1e-4

        pipe.unload_lora_weights()

    def test_sd_load_civitai_empty_network_alpha(self):
        """
        This test simply checks that loading a LoRA with an empty network alpha works fine
        See: https://github.com/huggingface/diffusers/issues/5606
        """
        pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        civitai_path = hf_hub_download("ybelkada/test-ahi-civitai", "ahi_lora_weights.safetensors")
        pipeline.load_lora_weights(civitai_path, adapter_name="ahri")

        images = pipeline(
            "ahri, masterpiece, league of legends",
            output_type="np",
            generator=torch.Generator().manual_seed(156),
            num_inference_steps=5,
        )[0]
        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.0, 0.0, 0.0, 0.002557, 0.020954, 0.001792, 0.006581, 0.00591, 0.002995])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipeline.unload_lora_weights()
