# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
"""Testing suite for the MindSpore EfficientNet model."""

import numpy as np
import pytest
import torch
from transformers import EfficientNetConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "bf16": 5e-2}
MODES = [1]  # not support graph mode yet


class EfficientNetModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=32,
        num_channels=3,
        kernel_sizes=[3, 3, 5],
        in_channels=[32, 16, 24],
        out_channels=[16, 24, 20],
        strides=[1, 1, 2],
        num_block_repeats=[1, 1, 2],
        expand_ratios=[1, 6, 6],
        is_training=True,
        use_labels=True,
        intermediate_size=37,
        hidden_act="gelu",
        num_labels=10,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.num_block_repeats = num_block_repeats
        self.expand_ratios = expand_ratios
        self.is_training = is_training
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.use_labels = use_labels

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size], self.num_labels)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self):
        return EfficientNetConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            kernel_sizes=self.kernel_sizes,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            strides=self.strides,
            num_block_repeats=self.num_block_repeats,
            expand_ratios=self.expand_ratios,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


model_tester = EfficientNetModelTester()
config, pixel_values, labels = model_tester.prepare_config_and_inputs()


_CASES = [
    [
        "EfficientNetForImageClassification",
        "transformers.EfficientNetForImageClassification",
        "mindone.transformers.EfficientNetForImageClassification",
        (config,),
        {},
        (pixel_values,),
        {},
        {
            "logits": 0,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in _CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
