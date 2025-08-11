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
import copy
import importlib
import sys
import time
import unittest

import numpy as np
import torch
from packaging import version
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore import mint

from mindone.diffusers import (
    ControlNetModel,
    EulerDiscreteScheduler,
    LCMScheduler,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    T2IAdapter,
)
from mindone.diffusers.utils import logging
from mindone.diffusers.utils.testing_utils import (
    CaptureLogger,
    is_flaky,
    load_image,
    numpy_cosine_similarity_distance,
    slow,
)
from mindone.peft import __version__ as _mindone_peft_version
from mindone.transformers import CLIPTextModel, CLIPTextModelWithProjection
from tests.diffusers_tests.pipelines.pipeline_test_utils import PipelineTesterMixin

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests, check_if_lora_correctly_set, state_dicts_almost_equal  # noqa: E402


class StableDiffusionXLLoRATests(PeftLoraLoaderMixinTests, unittest.TestCase):
    has_two_text_encoders = True
    pipeline_class = StableDiffusionXLPipeline
    scheduler_cls = EulerDiscreteScheduler
    scheduler_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "timestep_spacing": "leading",
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
        "attention_head_dim": (2, 4),
        "use_linear_projection": True,
        "addition_embed_type": "text_time",
        "addition_time_embed_dim": 8,
        "transformer_layers_per_block": (1, 2),
        "projection_class_embeddings_input_dim": 80,  # 6 * 8 + 32
        "cross_attention_dim": 64,
    }
    vae_kwargs = {
        "block_out_channels": [32, 64],
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
        "latent_channels": 4,
        "sample_size": 128,
    }
    text_encoder_cls, text_encoder_id = CLIPTextModel, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"
    text_encoder_2_cls, text_encoder_2_id = CLIPTextModelWithProjection, "peft-internal-testing/tiny-clip-text-2"
    tokenizer_2_cls, tokenizer_2_id = CLIPTokenizer, "peft-internal-testing/tiny-clip-text-2"

    @property
    def output_shape(self):
        return (1, 64, 64, 3)

    @is_flaky
    def test_multiple_wrong_adapter_name_raises_error(self):
        super().test_multiple_wrong_adapter_name_raises_error()

    def test_simple_inference_with_text_denoiser_lora_unfused(self):
        expected_atol = 9e-2
        expected_rtol = 9e-2

        super().test_simple_inference_with_text_denoiser_lora_unfused(
            expected_atol=expected_atol, expected_rtol=expected_rtol
        )

    def test_simple_inference_with_text_lora_denoiser_fused_multi(self):
        expected_atol = 9e-2
        expected_rtol = 9e-2

        super().test_simple_inference_with_text_lora_denoiser_fused_multi(
            expected_atol=expected_atol, expected_rtol=expected_rtol
        )

    def test_lora_scale_kwargs_match_fusion(self):
        expected_atol = 9e-2
        expected_rtol = 9e-2

        super().test_lora_scale_kwargs_match_fusion(expected_atol=expected_atol, expected_rtol=expected_rtol)


@slow
class LoraSDXLIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    def test_sdxl_1_0_lora(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4061, 0.4134, 0.3637, 0.3202, 0.365, 0.3786, 0.3725, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 5e-4

        pipe.unload_lora_weights()

    def test_sdxl_1_0_blockwise_lora(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, adapter_name="offset")
        scales = {
            "unet": {
                "down": {"block_1": [1.0, 1.0], "block_2": [1.0, 1.0]},
                "mid": 1.0,
                "up": {"block_0": [1.0, 1.0, 1.0], "block_1": [1.0, 1.0, 1.0]},
            },
        }
        pipe.set_adapters(["offset"], [scales])

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([00.4468, 0.4061, 0.4134, 0.3637, 0.3202, 0.365, 0.3786, 0.3725, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 5e-4

        pipe.unload_lora_weights()

    def test_sdxl_lcm_lora(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16
        )
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        generator = torch.Generator("cpu").manual_seed(0)

        lora_model_id = "latent-consistency/lcm-lora-sdxl"

        pipe.load_lora_weights(lora_model_id)

        image = pipe(
            "masterpiece, best quality, mountain", generator=generator, num_inference_steps=4, guidance_scale=0.5
        )[0][0]

        expected_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_lora/sdxl_lcm_lora.png"
        )

        image_np = pipe.image_processor.pil_to_numpy(image)
        expected_image_np = pipe.image_processor.pil_to_numpy(expected_image)

        max_diff = numpy_cosine_similarity_distance(image_np.flatten(), expected_image_np.flatten())
        assert max_diff < 5e-4

        pipe.unload_lora_weights()

    def test_sdxl_1_0_lora_fusion(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        pipe.fuse_lora()
        # We need to unload the lora weights since in the previous API `fuse_lora` led to lora weights being
        # silently deleted - otherwise this will CPU OOM
        pipe.unload_lora_weights()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        # This way we also test equivalence between LoRA fusion and the non-fusion behaviour.
        expected = np.array([0.4468, 0.4061, 0.4134, 0.3637, 0.3202, 0.365, 0.3786, 0.3725, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 5e-4

    def test_sdxl_1_0_lora_unfusion(self):
        generator = torch.Generator("cpu").manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=3
        )[0]
        images_with_fusion = images.flatten()

        pipe.unfuse_lora()
        generator = torch.Generator("cpu").manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=3
        )[0]
        images_without_fusion = images.flatten()

        max_diff = numpy_cosine_similarity_distance(images_with_fusion, images_without_fusion)
        assert max_diff < 5e-4

    def test_sdxl_1_0_lora_unfusion_effectivity(self):
        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]
        original_image_slice = images[0, -3:, -3:, -1].flatten()

        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        _ = pipe("masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2)[0]

        pipe.unfuse_lora()

        # We need to unload the lora weights - in the old API unfuse led to unloading the adapter weights
        pipe.unload_lora_weights()

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]
        images_without_fusion_slice = images[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(images_without_fusion_slice, original_image_slice)
        assert max_diff < 1e-3

    def test_sdxl_1_0_lora_fusion_efficiency(self):
        generator = torch.Generator().manual_seed(0)
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16
        )
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, mindspore_dtype=ms.float16)

        start_time = time.time()
        for _ in range(3):
            pipe("masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2)[0]
        end_time = time.time()
        elapsed_time_non_fusion = end_time - start_time

        del pipe

        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16
        )
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename, mindspore_dtype=ms.float16)
        pipe.fuse_lora()

        # We need to unload the lora weights since in the previous API `fuse_lora` led to lora weights being
        # silently deleted - otherwise this will CPU OOM
        pipe.unload_lora_weights()

        generator = torch.Generator().manual_seed(0)
        start_time = time.time()
        for _ in range(3):
            pipe("masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2)[0]
        end_time = time.time()
        elapsed_time_fusion = end_time - start_time

        self.assertTrue(elapsed_time_fusion < elapsed_time_non_fusion)

    def test_sdxl_1_0_last_ben(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "TheLastBen/Papercut_SDXL"
        lora_filename = "papercut.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe("papercut.safetensors", output_type="np", generator=generator, num_inference_steps=2)[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.5244, 0.4347, 0.4312, 0.4246, 0.4398, 0.4409, 0.4884, 0.4938, 0.4094])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_sdxl_1_0_fuse_unfuse_all(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16
        )
        text_encoder_1_sd = copy.deepcopy(pipe.text_encoder.state_dict())
        text_encoder_2_sd = copy.deepcopy(pipe.text_encoder_2.state_dict())
        unet_sd = copy.deepcopy(pipe.unet.state_dict())

        pipe.load_lora_weights(
            "davizca87/sun-flower", weight_name="snfw3rXL-000004.safetensors", mindspore_dtype=ms.float16
        )

        fused_te_state_dict = pipe.text_encoder.state_dict()
        fused_te_2_state_dict = pipe.text_encoder_2.state_dict()
        unet_state_dict = pipe.unet.state_dict()

        peft_ge_070 = version.parse(_mindone_peft_version) >= version.parse("0.7.0")

        def remap_key(key, sd):
            # some keys have moved around for PEFT >= 0.7.0, but they should still be loaded correctly
            if (key in sd) or (not peft_ge_070):
                return key

            # instead of linear.weight, we now have linear.base_layer.weight, etc.
            if key.endswith(".weight"):
                key = key[:-7] + ".base_layer.weight"
            elif key.endswith(".bias"):
                key = key[:-5] + ".base_layer.bias"
            return key

        for key, value in text_encoder_1_sd.items():
            key = remap_key(key, fused_te_state_dict)
            self.assertTrue(mint.allclose(fused_te_state_dict[key], value))

        for key, value in text_encoder_2_sd.items():
            key = remap_key(key, fused_te_2_state_dict)
            self.assertTrue(mint.allclose(fused_te_2_state_dict[key], value))

        for key, value in unet_state_dict.items():
            self.assertTrue(mint.allclose(unet_state_dict[key], value))

        pipe.fuse_lora()
        pipe.unload_lora_weights()

        assert not state_dicts_almost_equal(text_encoder_1_sd, pipe.text_encoder.state_dict())
        assert not state_dicts_almost_equal(text_encoder_2_sd, pipe.text_encoder_2.state_dict())
        assert not state_dicts_almost_equal(unet_sd, pipe.unet.state_dict())

        del unet_sd, text_encoder_1_sd, text_encoder_2_sd

    def test_sdxl_1_0_lora_with_sequential_cpu_offloading(self):
        generator = torch.Generator().manual_seed(0)

        pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        lora_model_id = "hf-internal-testing/sdxl-1.0-lora"
        lora_filename = "sd_xl_offset_example-lora_1.0.safetensors"

        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]

        images = images[0, -3:, -3:, -1].flatten()
        expected = np.array([0.4468, 0.4087, 0.4134, 0.366, 0.3202, 0.3505, 0.3786, 0.387, 0.3535])

        max_diff = numpy_cosine_similarity_distance(expected, images)
        assert max_diff < 1e-3

        pipe.unload_lora_weights()

    def test_controlnet_canny_lora(self):
        controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet
        )
        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors")

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "corgi"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3)[0]

        assert images[0].shape == (768, 512, 3)

        original_image = images[0, -3:, -3:, -1].flatten()
        expected_image = np.array([0.4574, 0.4487, 0.4435, 0.446, 0.4396, 0.4474, 0.4474, 0.4465, 0.4333])

        max_diff = numpy_cosine_similarity_distance(expected_image, original_image)
        assert max_diff < 5e-4

        pipe.unload_lora_weights()

    def test_sdxl_t2i_adapter_canny_lora(self):
        adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-lineart-sdxl-1.0", mindspore_dtype=ms.float16)
        pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            adapter=adapter,
            mindspore_dtype=ms.float16,
            variant="fp16",
        )
        pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors")
        pipe.set_progress_bar_config(disable=None)

        generator = torch.Generator(device="cpu").manual_seed(0)
        prompt = "toy"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png"
        )

        images = pipe(prompt, image=image, generator=generator, output_type="np", num_inference_steps=3)[0]

        assert images[0].shape == (768, 512, 3)

        image_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice = np.array([0.4284, 0.4337, 0.4319, 0.4255, 0.4329, 0.4280, 0.4338, 0.4420, 0.4226])
        assert numpy_cosine_similarity_distance(image_slice, expected_slice) < 5e-4

    def test_sequential_fuse_unfuse(self):
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", mindspore_dtype=ms.float16
        )

        # 1. round
        pipe.load_lora_weights("Pclanglais/TintinIA", mindspore_dtype=ms.float16)
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        images = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]
        image_slice = images[0, -3:, -3:, -1].flatten()

        pipe.unfuse_lora()

        # 2. round
        pipe.load_lora_weights("ProomptEngineer/pe-balloon-diffusion-style", mindspore_dtype=ms.float16)
        pipe.fuse_lora()
        pipe.unfuse_lora()

        # 3. round
        pipe.load_lora_weights("ostris/crayon_style_lora_sdxl", mindspore_dtype=ms.float16)
        pipe.fuse_lora()
        pipe.unfuse_lora()

        # 4. back to 1st round
        pipe.load_lora_weights("Pclanglais/TintinIA", mindspore_dtype=ms.float16)
        pipe.fuse_lora()

        generator = torch.Generator().manual_seed(0)
        images_2 = pipe(
            "masterpiece, best quality, mountain", output_type="np", generator=generator, num_inference_steps=2
        )[0]
        image_slice_2 = images_2[0, -3:, -3:, -1].flatten()

        max_diff = numpy_cosine_similarity_distance(image_slice, image_slice_2)
        assert max_diff < 1e-3
        pipe.unload_lora_weights()

    def test_integration_logits_multi_adapter(self):
        path = "stabilityai/stable-diffusion-xl-base-1.0"
        lora_id = "CiroN2022/toy-face"

        pipe = StableDiffusionXLPipeline.from_pretrained(path, mindspore_dtype=ms.float16)
        pipe.load_lora_weights(lora_id, weight_name="toy_face_sdxl.safetensors", adapter_name="toy")

        self.assertTrue(check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet")

        prompt = "toy_face of a hacker with a hoodie"

        lora_scale = 0.9

        images = pipe(
            prompt=prompt,
            num_inference_steps=30,
            generator=torch.manual_seed(0),
            cross_attention_kwargs={"scale": lora_scale},
            output_type="np",
        )[0]
        expected_slice_scale = np.array([0.538, 0.539, 0.540, 0.540, 0.542, 0.539, 0.538, 0.541, 0.539])

        predicted_slice = images[0, -3:, -3:, -1].flatten()
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipe.set_adapters("pixel")

        prompt = "pixel art, a hacker with a hoodie, simple, flat colors"
        images = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": lora_scale},
            generator=torch.manual_seed(0),
            output_type="np",
        )[0]

        predicted_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice_scale = np.array(
            [0.61973065, 0.62018543, 0.62181497, 0.61933696, 0.6208608, 0.620576, 0.6200281, 0.62258327, 0.6259889]
        )
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        # multi-adapter inference
        pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
        images = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": 1.0},
            generator=torch.manual_seed(0),
            output_type="np",
        )[0]
        predicted_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice_scale = np.array([0.5888, 0.5897, 0.5946, 0.5888, 0.5935, 0.5946, 0.5857, 0.5891, 0.5909])
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3

        # Lora disabled
        pipe.disable_lora()
        images = pipe(
            prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            cross_attention_kwargs={"scale": lora_scale},
            generator=torch.manual_seed(0),
            output_type="np",
        )[0]
        predicted_slice = images[0, -3:, -3:, -1].flatten()
        expected_slice_scale = np.array([0.5456, 0.5466, 0.5487, 0.5458, 0.5469, 0.5454, 0.5446, 0.5479, 0.5487])
        max_diff = numpy_cosine_similarity_distance(expected_slice_scale, predicted_slice)
        assert max_diff < 1e-3
