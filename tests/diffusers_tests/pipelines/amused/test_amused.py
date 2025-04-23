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
from ddt import data, ddt, unpack

import mindspore as ms

from mindone.diffusers import AmusedPipeline
from mindone.diffusers.utils.testing_utils import slow

from ..pipeline_test_utils import PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
    {"mode": ms.GRAPH_MODE, "dtype": "float32"},
    {"mode": ms.GRAPH_MODE, "dtype": "float16"},
]


@slow
@ddt
class AmusedPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_amused_512(self, mode, dtype):
        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe = AmusedPipeline.from_pretrained("amused/amused-512", mindspore_dtype=ms_dtype)
        image = pipe("dog", generator=ms.Generator().manual_seed(0), num_inference_steps=2, output_type="np")[0]
        image_slice = image[0, -3:, -3:, -1].flatten()
        if dtype == "float32":
            expected_slice = np.array([0.1970, 0.1935, 0.1952, 0.1937, 0.1933, 0.1886, 0.2016, 0.2101, 0.1864])
        else:
            expected_slice = np.array([0.2334, 0.2289, 0.2288, 0.2342, 0.2324, 0.2268, 0.2444, 0.2529, 0.2336])
        assert np.abs(image_slice - expected_slice).max() < 0.003
