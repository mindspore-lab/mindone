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

from ..pipeline_test_utils import THRESHOLD_FP16, THRESHOLD_FP32, PipelineTesterMixin, get_module
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
class KandinskyV22PipelineCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
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
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky2_2.KandinskyV22CombinedPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky2_2.KandinskyV22CombinedPipeline")

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
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@ddt
class KandinskyV22PipelineImg2ImgCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
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
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky2_2.KandinskyV22Img2ImgCombinedPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky2_2.KandinskyV22Img2ImgCombinedPipeline")

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
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold


@ddt
class KandinskyV22PipelineInpaintCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
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
        pt_pipe_cls = get_module("diffusers.pipelines.kandinsky2_2.KandinskyV22InpaintCombinedPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.kandinsky2_2.KandinskyV22InpaintCombinedPipeline")

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
        assert np.max(np.linalg.norm(pt_image_slice - ms_image_slice) / np.linalg.norm(pt_image_slice)) < threshold
