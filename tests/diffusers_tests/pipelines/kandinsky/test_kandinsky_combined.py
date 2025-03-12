# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import (
    load_downloaded_image_from_hf_hub,
    load_downloaded_numpy_from_hf_hub,
    slow,
)

from ..pipeline_test_utils import THRESHOLD_FP16, THRESHOLD_FP32, THRESHOLD_PIXEL, PipelineTesterMixin, get_module
from .test_kandinsky import Dummies
from .test_kandinsky_img2img import Dummies as Img2ImgDummies
from .test_kandinsky_inpaint import Dummies as InpaintDummies
from .test_kandinsky_prior import Dummies as PriorDummies

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class KandinskyPipelineCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        dummy = Dummies()
        prior_dummy = PriorDummies()
        pt_components, ms_components = dummy.get_dummy_components()
        pt_prior_components, ms_prior_components = prior_dummy.get_dummy_components()

        pt_components.update({f"prior_{k}": v for k, v in pt_prior_components.items()})
        ms_components.update({f"prior_{k}": v for k, v in ms_prior_components.items()})
        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        prior_dummy = PriorDummies()
        inputs = prior_dummy.get_dummy_inputs(seed=seed)
        inputs.update(
            {
                "height": 64,
                "width": 64,
            }
        )
        return inputs

    @data(*test_cases)
    @unpack
    def test_kandinsky(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky.KandinskyCombinedPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyCombinedPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@ddt
class KandinskyPipelineImg2ImgCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        dummy = Img2ImgDummies()
        prior_dummy = PriorDummies()
        pt_components, ms_components = dummy.get_dummy_components()
        pt_prior_components, ms_prior_components = prior_dummy.get_dummy_components()

        pt_components.update({f"prior_{k}": v for k, v in pt_prior_components.items()})
        ms_components.update({f"prior_{k}": v for k, v in ms_prior_components.items()})
        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        prior_dummy = PriorDummies()
        dummy = Img2ImgDummies()
        pt_prior_inputs = prior_dummy.get_dummy_inputs(seed=seed)
        ms_prior_inputs = prior_dummy.get_dummy_inputs(seed=seed)
        pt_inputs, ms_inputs = dummy.get_dummy_inputs(seed=seed)

        pt_prior_inputs.update(pt_inputs)
        ms_prior_inputs.update(ms_inputs)

        pt_prior_inputs.pop("image_embeds")
        pt_prior_inputs.pop("negative_image_embeds")
        ms_prior_inputs.pop("image_embeds")
        ms_prior_inputs.pop("negative_image_embeds")

        return pt_prior_inputs, ms_prior_inputs

    @data(*test_cases)
    @unpack
    def test_kandinsky(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky.KandinskyImg2ImgCombinedPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyImg2ImgCombinedPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@ddt
class KandinskyPipelineInpaintCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        dummy = InpaintDummies()
        prior_dummy = PriorDummies()
        pt_components, ms_components = dummy.get_dummy_components()
        pt_prior_components, ms_prior_components = prior_dummy.get_dummy_components()

        pt_components.update({f"prior_{k}": v for k, v in pt_prior_components.items()})
        ms_components.update({f"prior_{k}": v for k, v in ms_prior_components.items()})
        return pt_components, ms_components

    def get_dummy_inputs(self, seed=0):
        prior_dummy = PriorDummies()
        dummy = InpaintDummies()
        pt_prior_inputs = prior_dummy.get_dummy_inputs(seed=seed)
        ms_prior_inputs = prior_dummy.get_dummy_inputs(seed=seed)
        pt_inputs, ms_inputs = dummy.get_dummy_inputs(seed=seed)

        pt_prior_inputs.update(pt_inputs)
        ms_prior_inputs.update(ms_inputs)

        pt_prior_inputs.pop("image_embeds")
        pt_prior_inputs.pop("negative_image_embeds")
        ms_prior_inputs.pop("image_embeds")
        ms_prior_inputs.pop("negative_image_embeds")

        return pt_prior_inputs, ms_prior_inputs

    @data(*test_cases)
    @unpack
    def test_kandinsky(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky.KandinskyInpaintCombinedPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyInpaintCombinedPipeline")

        pt_pipe = pt_pipe_cls(**pt_components)
        ms_pipe = ms_pipe_cls(**ms_components)

        pt_pipe.set_progress_bar_config(disable=None)
        ms_pipe.set_progress_bar_config(disable=None)

        ms_dtype, pt_dtype = getattr(ms, dtype), getattr(torch, dtype)
        pt_pipe = pt_pipe.to(pt_dtype)
        ms_pipe = ms_pipe.to(ms_dtype)

        pt_inputs, ms_inputs = self.get_dummy_inputs()

        torch.manual_seed(0)
        pt_image = pt_pipe(**pt_inputs)
        torch.manual_seed(0)
        ms_image = ms_pipe(**ms_inputs)

        pt_image_slice = pt_image.images[0, -3:, -3:, -1]
        ms_image_slice = ms_image[0][0, -3:, -3:, -1]

        threshold = THRESHOLD_FP32 if dtype == "float32" else THRESHOLD_FP16
        assert np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice) < threshold


@slow
@ddt
class KandinskyPipelineCombinedIntegrationTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_kandinsky(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyCombinedPipeline")
        pipe = pipe_cls.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms_dtype)

        prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"

        torch.manual_seed(0)
        image = pipe(prompt=prompt, num_inference_steps=25)[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"combined_t2i_{dtype}.npy",
            subfolder="kandinsky",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

    @data(*test_cases)
    @unpack
    def test_kandinsky_img2img(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyImg2ImgCombinedPipeline")
        pipe = pipe_cls.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms_dtype)

        prompt = "A fantasy landscape, Cinematic lighting"
        negative_prompt = "low quality, bad quality"

        original_image = load_downloaded_image_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            "combined_i2i_input.jpg",
            subfolder="kandinsky2_2",
        )
        original_image.thumbnail((768, 768))

        torch.manual_seed(0)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original_image,
            num_inference_steps=25,
        )[
            0
        ][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"combined_i2i_{dtype}.npy",
            subfolder="kandinsky",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL

    @data(*test_cases)
    @unpack
    def test_kandinsky_inpaint(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky.KandinskyInpaintCombinedPipeline")
        pipe = pipe_cls.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", mindspore_dtype=ms_dtype)

        prompt = "A fantasy landscape, Cinematic lighting"
        negative_prompt = "low quality, bad quality"

        original_image = load_downloaded_image_from_hf_hub(
            "hf-internal-testing/diffusers-images",
            "cat.png",
            subfolder="kandinsky",
        )

        mask = np.zeros((768, 768), dtype=np.float32)
        # Let's mask out an area above the cat's head
        mask[:250, 250:-250] = 1

        torch.manual_seed(0)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=original_image,
            mask_image=mask,
            num_inference_steps=25,
        )[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"combined_inpaint_{dtype}.npy",
            subfolder="kandinsky",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
