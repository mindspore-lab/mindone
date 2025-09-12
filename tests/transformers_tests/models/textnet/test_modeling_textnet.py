# coding=utf-8
# Copyright 2024 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the MindSpore TextNet model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import TextNetConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class TextNetModelTester:
    def __init__(
        self,
        stem_kernel_size=3,
        stem_stride=2,
        stem_in_channels=3,
        stem_out_channels=32,
        stem_act_func="relu",
        dropout_rate=0,
        ops_order="weight_bn_act",
        conv_layer_kernel_sizes=[
            [[3, 3]],
            [[3, 3]],
            [[3, 3]],
            [[3, 3]],
        ],
        conv_layer_strides=[
            [2],
            [2],
            [2],
            [2],
        ],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        out_indices=[1, 2, 3, 4],
        batch_size=3,
        num_channels=3,
        image_size=[32, 32],
        is_training=True,
        use_labels=True,
        num_labels=3,
        hidden_sizes=[32, 32, 32, 32, 32],
    ):
        self.stem_kernel_size = stem_kernel_size
        self.stem_stride = stem_stride
        self.stem_in_channels = stem_in_channels
        self.stem_out_channels = stem_out_channels
        self.act_func = stem_act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order
        self.conv_layer_kernel_sizes = conv_layer_kernel_sizes
        self.conv_layer_strides = conv_layer_strides

        self.out_features = out_features
        self.out_indices = out_indices

        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.hidden_sizes = hidden_sizes

        self.num_stages = 5

    def get_config(self):
        return TextNetConfig(
            stem_kernel_size=self.stem_kernel_size,
            stem_stride=self.stem_stride,
            stem_num_channels=self.stem_in_channels,
            stem_out_channels=self.stem_out_channels,
            act_func=self.act_func,
            dropout_rate=self.dropout_rate,
            ops_order=self.ops_order,
            conv_layer_kernel_sizes=self.conv_layer_kernel_sizes,
            conv_layer_strides=self.conv_layer_strides,
            out_features=self.out_features,
            out_indices=self.out_indices,
            hidden_sizes=self.hidden_sizes,
            image_size=self.image_size,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels


model_tester = TextNetModelTester()
config, pixel_values, labels = model_tester.prepare_config_and_inputs()
TEXTNET_CASES = [
    [
        "TextNetModel",
        "transformers.TextNetModel",
        "mindone.transformers.TextNetModel",
        (config,),
        {},
        (pixel_values,),
        {},
        {
            "last_hidden_state": 0,
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
        for case in TEXTNET_CASES
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

    # set `hidden_dtype` if requiring, for some modules always compute in float
    # precision and require specific `hidden_dtype` to cast before return
    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
    # print("ms:", ms_outputs)
    # print("pt:", pt_outputs)
    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            # print("===map", pt_key, ms_idx)
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
