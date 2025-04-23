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

from mindone.diffusers import AmusedInpaintPipeline
from mindone.diffusers.utils.testing_utils import load_downloaded_image_from_hf_hub, slow

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
        pipe = AmusedInpaintPipeline.from_pretrained("amused/amused-512", mindspore_dtype=ms_dtype)
        image = (
            load_downloaded_image_from_hf_hub(
                "diffusers/docs-images",
                "mountains_1.jpg",
                subfolder="open_muse",
            )
            .resize((512, 512))
            .convert("RGB")
        )
        mask_image = (
            load_downloaded_image_from_hf_hub(
                "diffusers/docs-images",
                "mountains_1_mask.png",
                subfolder="open_muse",
            )
            .resize((512, 512))
            .convert("L")
        )
        image = pipe(
            "winter mountains",
            image,
            mask_image,
            generator=ms.Generator().manual_seed(0),
            num_inference_steps=2,
            output_type="np",
        )[0]
        image_slice = image[0, -3:, -3:, -1].flatten()

        if dtype == "float32":
            expected_slice = np.array([0.0254, 0.0188, 0.0131, 0.0241, 0.0278, 0.0153, 0.0304, 0.0409, 0.0121])
            assert np.abs(image_slice - expected_slice).max() < 0.05
        else:
            expected_slice = np.array([0.0400, 0.0340, 0.0267, 0.0382, 0.0423, 0.0292, 0.0456, 0.0553, 0.0271])
            assert np.abs(image_slice - expected_slice).max() < 0.003
