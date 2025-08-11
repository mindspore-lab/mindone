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
from transformers import AutoTokenizer, CLIPTokenizer

import mindspore as ms

from mindone.diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3Pipeline,
)
from mindone.diffusers.utils import load_image
from mindone.diffusers.utils.testing_utils import is_flaky, numpy_cosine_similarity_distance, slow
from mindone.transformers import CLIPTextModelWithProjection, T5EncoderModel
from tests.diffusers_tests.pipelines.pipeline_test_utils import PipelineTesterMixin

sys.path.append(".")

from utils import PeftLoraLoaderMixinTests  # noqa: E402

ms.set_deterministic(True)
ms.manual_seed(0)
np.random.seed(0)


class SD3LoRATests(unittest.TestCase, PeftLoraLoaderMixinTests):
    pipeline_class = StableDiffusion3Pipeline
    scheduler_cls = FlowMatchEulerDiscreteScheduler
    scheduler_kwargs = {}
    scheduler_classes = [FlowMatchEulerDiscreteScheduler]
    transformer_kwargs = {
        "sample_size": 32,
        "patch_size": 1,
        "in_channels": 4,
        "num_layers": 1,
        "attention_head_dim": 8,
        "num_attention_heads": 4,
        "caption_projection_dim": 32,
        "joint_attention_dim": 32,
        "pooled_projection_dim": 64,
        "out_channels": 4,
    }
    transformer_cls = SD3Transformer2DModel
    vae_kwargs = {
        "sample_size": 32,
        "in_channels": 3,
        "out_channels": 3,
        "block_out_channels": (4,),
        "layers_per_block": 1,
        "latent_channels": 4,
        "norm_num_groups": 1,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "shift_factor": 0.0609,
        "scaling_factor": 1.5035,
    }
    has_three_text_encoders = True
    tokenizer_cls, tokenizer_id = CLIPTokenizer, "hf-internal-testing/tiny-random-clip"
    tokenizer_2_cls, tokenizer_2_id = CLIPTokenizer, "hf-internal-testing/tiny-random-clip"
    tokenizer_3_cls, tokenizer_3_id = AutoTokenizer, "hf-internal-testing/tiny-random-t5"
    text_encoder_cls, text_encoder_id = CLIPTextModelWithProjection, "hf-internal-testing/tiny-sd3-text_encoder"
    text_encoder_2_cls, text_encoder_2_id = CLIPTextModelWithProjection, "hf-internal-testing/tiny-sd3-text_encoder-2"
    text_encoder_3_cls, text_encoder_3_id, text_encoder_3_revision = (
        T5EncoderModel,
        "hf-internal-testing/tiny-random-t5",
        "refs/pr/1",
    )

    @property
    def output_shape(self):
        return (1, 32, 32, 3)

    def test_sd3_lora(self):
        """
        Test loading the loras that are saved with the diffusers and peft formats.
        Related PR: https://github.com/huggingface/diffusers/pull/8584
        """
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components[0])
        pipe.set_progress_bar_config(disable=None)

        lora_model_id = "hf-internal-testing/tiny-sd3-loras"

        lora_filename = "lora_diffusers_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
        pipe.unload_lora_weights()

        lora_filename = "lora_peft_format.safetensors"
        pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)

    @unittest.skip("Not supported in SD3.")
    def test_simple_inference_with_text_denoiser_block_scale(self):
        pass

    @unittest.skip("Not supported in SD3.")
    def test_simple_inference_with_text_denoiser_multi_adapter_block_lora(self):
        pass

    @unittest.skip("Not supported in SD3.")
    def test_simple_inference_with_text_denoiser_block_scale_for_all_dict_options(self):
        pass

    @unittest.skip("Not supported in SD3.")
    def test_modify_padding_mode(self):
        pass

    @is_flaky
    def test_multiple_wrong_adapter_name_raises_error(self):
        super().test_multiple_wrong_adapter_name_raises_error()


@slow
class SD3LoraIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusion3Img2ImgPipeline
    repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    def get_inputs(self, seed=0):
        init_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        generator = torch.manual_seed(seed)

        return {
            "prompt": "corgi",
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "generator": generator,
            "image": init_image,
        }

    def test_sd3_img2img_lora(self):
        pipe = self.pipeline_class.from_pretrained(self.repo_id, mindspore_dtype=ms.float16)
        pipe.load_lora_weights("zwloong/sd3-lora-training-rank16-v2")
        pipe.fuse_lora()
        pipe.unload_lora_weights()

        inputs = self.get_inputs()

        image = pipe(**inputs)[0][0]
        image_slice = image[0, -3:, -3:]
        expected_slice = np.array([0.5649, 0.5405, 0.5488, 0.5688, 0.5449, 0.5513, 0.5337, 0.5107, 0.5059])

        max_diff = numpy_cosine_similarity_distance(expected_slice.flatten(), image_slice.flatten())

        assert max_diff < 1e-4, f"Outputs are not close enough, got {max_diff}"
        pipe.unload_lora_weights()
