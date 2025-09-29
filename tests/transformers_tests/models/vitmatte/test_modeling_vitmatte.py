# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the MindSpore VitMatte model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import VitDetConfig, VitMatteConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

# interpolate bilinear mode raise error, aclnnUpsampleBilinear2d operator does not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3}  # , "bf16": 7e-3}
MODES = [1]


class VitMatteModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=32,
        patch_size=16,
        num_channels=4,
        is_training=True,
        use_labels=False,
        hidden_size=2,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_act="gelu",
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
        out_features=["stage1"],
        fusion_hidden_sizes=[128, 64, 32, 16],
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.out_features = out_features
        self.fusion_hidden_sizes = fusion_hidden_sizes

        self.seq_length = (self.image_size // self.patch_size) ** 2

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            raise NotImplementedError("Training is not yet supported")

        config = self.get_config()

        return config, pixel_values, labels

    def get_backbone_config(self):
        return VitDetConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            is_training=self.is_training,
            hidden_act=self.hidden_act,
            out_features=self.out_features,
        )

    def get_config(self):
        return VitMatteConfig(
            backbone_config=self.get_backbone_config(),
            backbone=None,
            hidden_size=self.hidden_size,
            fusion_hidden_sizes=self.fusion_hidden_sizes,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


model_tester = VitMatteModelTester()
(
    config,
    inputs_dict,
) = model_tester.prepare_config_and_inputs_for_common()


VIT_MATTE_CASES = [
    [
        "VitMatteForImageMatting",
        "transformers.VitMatteForImageMatting",
        "mindone.transformers.VitMatteForImageMatting",
        (config,),
        {},
        (),
        {
            "pixel_values": inputs_dict["pixel_values"],
        },
        {
            "alphas": 0,
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
        for case in VIT_MATTE_CASES
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
