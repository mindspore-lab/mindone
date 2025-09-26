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

import random
import unittest
from typing import Any, Dict

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import mint

from mindone.diffusers import ClassifierFreeGuidance, StableDiffusionXLAutoBlocks, StableDiffusionXLModularPipeline
from mindone.diffusers.loaders import ModularIPAdapterMixin
from mindone.diffusers.models.attention_processor import IPAdapterAttnProcessor
from mindone.diffusers.models.embeddings import ImageProjection
from mindone.diffusers.utils.testing_utils import enable_full_determinism, floats_tensor

from ..test_modular_pipelines_common import ModularPipelineTesterMixin

enable_full_determinism()


def create_ip_adapter_state_dict(model):
    # "ip_adapter" (cross-attention weights)
    ip_cross_attn_state_dict = {}
    key_id = 1

    for name in model.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") or "motion_module" in name else model.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]

        if cross_attention_dim is not None:
            sd = IPAdapterAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
            ).state_dict()
            ip_cross_attn_state_dict.update(
                {
                    f"{key_id}.to_k_ip.weight": sd["to_k_ip.0.weight"],
                    f"{key_id}.to_v_ip.weight": sd["to_v_ip.0.weight"],
                }
            )

            key_id += 2

    # "image_proj" (ImageProjection layer weights)
    cross_attention_dim = model.config["cross_attention_dim"]
    image_projection = ImageProjection(
        cross_attention_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, num_image_text_embeds=4
    )

    ip_image_projection_state_dict = {}
    sd = image_projection.state_dict()
    ip_image_projection_state_dict.update(
        {
            "proj.weight": sd["image_embeds.weight"],
            "proj.bias": sd["image_embeds.bias"],
            "norm.weight": sd["norm.weight"],
            "norm.bias": sd["norm.bias"],
        }
    )

    del sd
    ip_state_dict = {}
    ip_state_dict.update({"image_proj": ip_image_projection_state_dict, "ip_adapter": ip_cross_attn_state_dict})
    return ip_state_dict


class SDXLModularTests:
    """
    This mixin defines method to create pipeline, base input and base test across all SDXL modular tests.
    """

    pipeline_class = StableDiffusionXLModularPipeline
    pipeline_blocks_class = StableDiffusionXLAutoBlocks
    repo = "hf-internal-testing/tiny-sdxl-modular"
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "negative_prompt",
            "cross_attention_kwargs",
            "image",
            "mask_image",
        ]
    )
    batch_params = frozenset(["prompt", "negative_prompt", "image", "mask_image"])

    def get_pipeline(self, components_manager=None, mindspore_dtype=ms.float32):
        pipeline = self.pipeline_blocks_class().init_pipeline(self.repo, components_manager=components_manager)
        pipeline.load_default_components(mindspore_dtype=mindspore_dtype)
        return pipeline

    def get_dummy_inputs(self, seed=0):
        generator = np.random.default_rng(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def _test_stable_diffusion_xl_euler(self, expected_image_shape, expected_slice, expected_max_diff=1e-2):
        sd_pipe = self.get_pipeline()
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs, output="images")
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == expected_image_shape

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < expected_max_diff
        ), "Image Slice does not match expected slice"


class SDXLModularIPAdapterTests:
    """
    This mixin is designed to test IP Adapter.
    """

    def test_pipeline_inputs_and_blocks(self):
        blocks = self.pipeline_blocks_class()
        parameters = blocks.input_names

        assert issubclass(self.pipeline_class, ModularIPAdapterMixin)
        assert (
            "ip_adapter_image" in parameters
        ), "`ip_adapter_image` argument must be supported by the `__call__` method"
        assert "ip_adapter" in blocks.sub_blocks, "pipeline must contain an IPAdapter block"

        _ = blocks.sub_blocks.pop("ip_adapter")
        parameters = blocks.input_names
        assert (
            "ip_adapter_image" not in parameters
        ), "`ip_adapter_image` argument must be removed from the `__call__` method"

    def _get_dummy_image_embeds(self, cross_attention_dim: int = 32):
        return mint.randn((1, 1, cross_attention_dim))

    def _get_dummy_faceid_image_embeds(self, cross_attention_dim: int = 32):
        return mint.randn((1, 1, 1, cross_attention_dim))

    def _get_dummy_masks(self, input_size: int = 64):
        _masks = mint.zeros((1, 1, input_size, input_size))
        _masks[0, :, :, : int(input_size / 2)] = 1
        return _masks

    def _modify_inputs_for_ip_adapter_test(self, inputs: Dict[str, Any]):
        blocks = self.pipeline_blocks_class()
        _ = blocks.sub_blocks.pop("ip_adapter")
        parameters = blocks.input_names
        if "image" in parameters and "strength" in parameters:
            inputs["num_inference_steps"] = 4

        inputs["output_type"] = "np"
        return inputs

    def test_ip_adapter(self, expected_max_diff: float = 1e-4, expected_pipe_slice=None):
        r"""Tests for IP-Adapter.

        The following scenarios are tested:
          - Single IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Multi IP-Adapter with scale=0 should produce same output as no IP-Adapter.
          - Single IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
          - Multi IP-Adapter with scale!=0 should produce different output compared to no IP-Adapter.
        """
        # Raising the tolerance for this test when it's run on a CPU because we
        # compare against static slices and that can be shaky (with a VVVV low probability).
        expected_max_diff = 9e-4

        blocks = self.pipeline_blocks_class()
        _ = blocks.sub_blocks.pop("ip_adapter")
        pipe = blocks.init_pipeline(self.repo)
        pipe.load_default_components(mindspore_dtype=ms.float32)
        pipe.set_progress_bar_config(disable=None)
        cross_attention_dim = pipe.unet.config.get("cross_attention_dim")

        # forward pass without ip adapter
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        if expected_pipe_slice is None:
            output_without_adapter = pipe(**inputs, output="images")
        else:
            output_without_adapter = expected_pipe_slice

        # 1. Single IP-Adapter test cases
        adapter_state_dict = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights(adapter_state_dict)

        # forward pass with single ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(0.0)
        output_without_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_without_adapter_scale = output_without_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)]
        pipe.set_ip_adapter_scale(42.0)
        output_with_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_with_adapter_scale = output_with_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_adapter_scale = np.abs(output_without_adapter_scale - output_without_adapter).max()
        max_diff_with_adapter_scale = np.abs(output_with_adapter_scale - output_without_adapter).max()

        assert (
            max_diff_without_adapter_scale < expected_max_diff
        ), "Output without ip-adapter must be same as normal inference"
        assert max_diff_with_adapter_scale > 1e-2, "Output with ip-adapter must be different from normal inference"

        # 2. Multi IP-Adapter test cases
        adapter_state_dict_1 = create_ip_adapter_state_dict(pipe.unet)
        adapter_state_dict_2 = create_ip_adapter_state_dict(pipe.unet)
        pipe.unet._load_ip_adapter_weights([adapter_state_dict_1, adapter_state_dict_2])

        # forward pass with multi ip adapter, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([0.0, 0.0])
        output_without_multi_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_without_multi_adapter_scale = output_without_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        # forward pass with multi ip adapter, but with scale of adapter weights
        inputs = self._modify_inputs_for_ip_adapter_test(self.get_dummy_inputs())
        inputs["ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        inputs["negative_ip_adapter_embeds"] = [self._get_dummy_image_embeds(cross_attention_dim)] * 2
        pipe.set_ip_adapter_scale([42.0, 42.0])
        output_with_multi_adapter_scale = pipe(**inputs, output="images")
        if expected_pipe_slice is not None:
            output_with_multi_adapter_scale = output_with_multi_adapter_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_multi_adapter_scale = np.abs(output_without_multi_adapter_scale - output_without_adapter).max()
        max_diff_with_multi_adapter_scale = np.abs(output_with_multi_adapter_scale - output_without_adapter).max()
        assert (
            max_diff_without_multi_adapter_scale < expected_max_diff
        ), "Output without multi-ip-adapter must be same as normal inference"
        assert (
            max_diff_with_multi_adapter_scale > 1e-2
        ), "Output with multi-ip-adapter scale must be different from normal inference"


class SDXLModularControlNetTests:
    """
    This mixin is designed to test ControlNet.
    """

    def test_pipeline_inputs(self):
        blocks = self.pipeline_blocks_class()
        parameters = blocks.input_names

        assert "control_image" in parameters, "`control_image` argument must be supported by the `__call__` method"
        assert (
            "controlnet_conditioning_scale" in parameters
        ), "`controlnet_conditioning_scale` argument must be supported by the `__call__` method"

    def _modify_inputs_for_controlnet_test(self, inputs: Dict[str, Any]):
        controlnet_embedder_scale_factor = 2
        image = mint.randn(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
        )
        inputs["control_image"] = image
        return inputs

    def test_controlnet(self, expected_max_diff: float = 1e-4, expected_pipe_slice=None):
        r"""Tests for ControlNet.

        The following scenarios are tested:
          - Single ControlNet with scale=0 should produce same output as no ControlNet.
          - Single ControlNet with scale!=0 should produce different output compared to no ControlNet.
        """
        # Raising the tolerance for this test when it's run on a CPU because we
        # compare against static slices and that can be shaky (with a VVVV low probability).
        expected_max_diff = 9e-4

        pipe = self.get_pipeline()
        pipe.set_progress_bar_config(disable=None)

        # forward pass without controlnet
        inputs = self.get_dummy_inputs()
        output_without_controlnet = pipe(**inputs, output="images")
        output_without_controlnet = output_without_controlnet[0, -3:, -3:, -1].flatten()

        # forward pass with single controlnet, but scale=0 which should have no effect
        inputs = self._modify_inputs_for_controlnet_test(self.get_dummy_inputs())
        inputs["controlnet_conditioning_scale"] = 0.0
        output_without_controlnet_scale = pipe(**inputs, output="images")
        output_without_controlnet_scale = output_without_controlnet_scale[0, -3:, -3:, -1].flatten()

        # forward pass with single controlnet, but with scale of adapter weights
        inputs = self._modify_inputs_for_controlnet_test(self.get_dummy_inputs())
        inputs["controlnet_conditioning_scale"] = 42.0
        output_with_controlnet_scale = pipe(**inputs, output="images")
        output_with_controlnet_scale = output_with_controlnet_scale[0, -3:, -3:, -1].flatten()

        max_diff_without_controlnet_scale = np.abs(output_without_controlnet_scale - output_without_controlnet).max()
        max_diff_with_controlnet_scale = np.abs(output_with_controlnet_scale - output_without_controlnet).max()

        assert (
            max_diff_without_controlnet_scale < expected_max_diff
        ), "Output without controlnet must be same as normal inference"
        assert max_diff_with_controlnet_scale > 1e-2, "Output with controlnet must be different from normal inference"

    def test_controlnet_cfg(self):
        pipe = self.get_pipeline()
        pipe.set_progress_bar_config(disable=None)

        # forward pass with CFG not applied
        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self._modify_inputs_for_controlnet_test(self.get_dummy_inputs())
        out_no_cfg = pipe(**inputs, output="images")

        # forward pass with CFG applied
        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)
        inputs = self._modify_inputs_for_controlnet_test(self.get_dummy_inputs())
        out_cfg = pipe(**inputs, output="images")

        assert out_cfg.shape == out_no_cfg.shape
        max_diff = np.abs(out_cfg - out_no_cfg).max()
        assert max_diff > 1e-2, "Output with CFG must be different from normal inference"


class SDXLModularGuiderTests:
    def test_guider_cfg(self):
        pipe = self.get_pipeline()
        pipe.set_progress_bar_config(disable=None)

        # forward pass with CFG not applied
        guider = ClassifierFreeGuidance(guidance_scale=1.0)
        pipe.update_components(guider=guider)

        inputs = self.get_dummy_inputs()
        out_no_cfg = pipe(**inputs, output="images")

        # forward pass with CFG applied
        guider = ClassifierFreeGuidance(guidance_scale=7.5)
        pipe.update_components(guider=guider)
        inputs = self.get_dummy_inputs()
        out_cfg = pipe(**inputs, output="images")

        assert out_cfg.shape == out_no_cfg.shape
        max_diff = np.abs(out_cfg - out_no_cfg).max()
        assert max_diff > 1e-2, "Output with CFG must be different from normal inference"


class SDXLModularPipelineFastTests(
    SDXLModularTests,
    SDXLModularIPAdapterTests,
    SDXLModularControlNetTests,
    SDXLModularGuiderTests,
    ModularPipelineTesterMixin,
    unittest.TestCase,
):
    """Test cases for Stable Diffusion XL modular pipeline fast tests."""

    def test_stable_diffusion_xl_euler(self):
        self._test_stable_diffusion_xl_euler(
            expected_image_shape=(1, 64, 64, 3),
            expected_slice=[
                0.4396658,
                0.4829341,
                0.48474038,
                0.43009943,
                0.43683282,
                0.5610152,
                0.42787912,
                0.48445842,
                0.5704539,
            ],
            expected_max_diff=1e-2,
        )

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


class SDXLImg2ImgModularPipelineFastTests(
    SDXLModularTests,
    SDXLModularIPAdapterTests,
    SDXLModularControlNetTests,
    SDXLModularGuiderTests,
    ModularPipelineTesterMixin,
    unittest.TestCase,
):
    """Test cases for Stable Diffusion XL image-to-image modular pipeline fast tests."""

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed))
        image = image / 2 + 0.5
        inputs["image"] = image
        inputs["strength"] = 0.8

        return inputs

    def test_stable_diffusion_xl_euler(self):
        self._test_stable_diffusion_xl_euler(
            expected_image_shape=(1, 64, 64, 3),
            expected_slice=[
                0.47611386,
                0.48132724,
                0.49074963,
                0.38083073,
                0.4189427,
                0.5158315,
                0.44459057,
                0.5438763,
                0.47947535,
            ],
            expected_max_diff=1e-2,
        )

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)


class SDXLInpaintingModularPipelineFastTests(
    SDXLModularTests,
    SDXLModularIPAdapterTests,
    SDXLModularControlNetTests,
    SDXLModularGuiderTests,
    ModularPipelineTesterMixin,
    unittest.TestCase,
):
    """Test cases for Stable Diffusion XL inpainting modular pipeline fast tests."""

    def get_dummy_inputs(self, seed=0):
        inputs = super().get_dummy_inputs(seed)
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        # create mask
        image[8:, 8:, :] = 255
        mask_image = Image.fromarray(np.uint8(image)).convert("L").resize((64, 64))

        inputs["image"] = init_image
        inputs["mask_image"] = mask_image
        inputs["strength"] = 1.0

        return inputs

    def test_stable_diffusion_xl_euler(self):
        self._test_stable_diffusion_xl_euler(
            expected_image_shape=(1, 64, 64, 3),
            expected_slice=[
                0.754773,
                0.7224033,
                0.4416766,
                0.5406277,
                0.57709944,
                0.5159552,
                0.5753975,
                0.30414963,
                0.44755602,
            ],
            expected_max_diff=1e-2,
        )

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)
