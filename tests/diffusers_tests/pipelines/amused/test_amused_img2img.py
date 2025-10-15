# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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
import pytest
from ddt import data, ddt, unpack
from packaging.version import parse

import mindspore as ms

from mindone.diffusers import AmusedImg2ImgPipeline, __version__
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, load_numpy_from_local_file, slow

from ..pipeline_test_utils import THRESHOLD_PIXEL, PipelineTesterMixin

test_cases = [
    {"mode": ms.PYNATIVE_MODE, "dtype": "float32"},
    {"mode": ms.PYNATIVE_MODE, "dtype": "float16"},
]


@slow
@ddt
class AmusedPipelineSlowTests(PipelineTesterMixin, unittest.TestCase):
    @data(*test_cases)
    @unpack
    def test_amused_512(self, mode, dtype):
        if parse(__version__) > parse("0.33.1"):
            pytest.skip("Skipping this case in diffusers > 0.33.1")

        ms.set_context(mode=mode)
        ms_dtype = getattr(ms, dtype)
        pipe = AmusedImg2ImgPipeline.from_pretrained("amused/amused-512", mindspore_dtype=ms_dtype)
        image = (
            load_downloaded_image_from_hf_hub(
                "diffusers/docs-images",
                "mountains.jpg",
                subfolder="open_muse",
            )
            .resize((512, 512))
            .convert("RGB")
        )
        image = pipe(
            "winter mountains",
            image,
            generator=ms.Generator().manual_seed(0),
        )[
            0
        ][0]

        expected_image = load_numpy_from_local_file(
            "mindone-testing-arrays",
            f"i2i_{dtype}.npy",
            subfolder="amused",
        )
        assert np.mean(np.abs(np.array(image, dtype=np.float32) - expected_image)) < THRESHOLD_PIXEL
