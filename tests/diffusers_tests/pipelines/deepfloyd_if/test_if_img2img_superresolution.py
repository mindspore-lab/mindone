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

import random
import unittest

import numpy as np
import torch
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers.utils.testing_utils import load_downloaded_numpy_from_hf_hub, slow

from ..pipeline_test_utils import (
    THRESHOLD_FP16,
    THRESHOLD_FP32,
    THRESHOLD_PIXEL,
    PipelineTesterMixin,
    floats_tensor,
    get_module,
)
from . import IFPipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@ddt
class IFImg2ImgSuperResolutionPipelineFastTests(PipelineTesterMixin, IFPipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        return self._get_superresolution_dummy_components()

    def get_dummy_inputs(self, seed=0):
        pt_original_image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        pt_image = floats_tensor((1, 3, 16, 16), rng=random.Random(seed))
        ms_original_image = ms.Tensor(pt_original_image.numpy())
        ms_image = ms.Tensor(pt_image.numpy())

        pt_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": pt_image,
            "original_image": pt_original_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        ms_inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": ms_image,
            "original_image": ms_original_image,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        return pt_inputs, ms_inputs

    @data(*test_cases)
    @unpack
    def test_inference(self, mode, dtype):
        ms.set_context(mode=mode)

        pt_components, ms_components = self.get_dummy_components()
        pt_pipe_cls = get_module("diffusers.pipelines.deepfloyd_if.IFImg2ImgSuperResolutionPipeline")
        ms_pipe_cls = get_module("mindone.diffusers.pipelines.deepfloyd_if.IFImg2ImgSuperResolutionPipeline")

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
class IFImg2ImgSuperResolutionPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_if_img2img_superresolution(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)

        pipe_cls = get_module("mindone.diffusers.pipelines.deepfloyd_if.IFImg2ImgSuperResolutionPipeline")
        pipe = pipe_cls.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            variant="fp16",
            mindspore_dtype=ms_dtype,
        )

        original_image = ms.Tensor(floats_tensor((1, 3, 256, 256), rng=random.Random(0)).numpy())
        image = ms.Tensor(floats_tensor((1, 3, 64, 64), rng=random.Random(0)).numpy())

        torch.manual_seed(0)
        output = pipe(
            prompt="anime turtle",
            image=image,
            original_image=original_image,
            num_inference_steps=2,
        )

        image = output[0][0]

        expected_image = load_downloaded_numpy_from_hf_hub(
            "The-truth/mindone-testing-arrays",
            f"img2img_superresolution_{dtype}.npy",
            subfolder="deepfloyd_if",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
