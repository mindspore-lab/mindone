# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the MindSpore EoMT model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import EomtConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class EomtForUniversalSegmentationTester:
    def __init__(
        self,
        batch_size=2,
        is_training=True,
        image_size=40,
        patch_size=2,
        num_queries=5,
        num_register_tokens=19,
        num_labels=4,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=4,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_queries = num_queries
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_register_tokens = num_register_tokens

        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def get_config(self):
        config = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_register_tokens": self.num_register_tokens,
            "num_queries": self.num_queries,
            "num_blocks": 1,
        }
        return EomtConfig(**config)

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, 3, self.image_size, self.image_size])

        mask_labels = (np.random.rand(self.batch_size, self.num_labels, self.image_size, self.image_size) > 0.5).astype(
            np.float32
        )
        class_labels = (np.random.rand(self.batch_size, self.num_labels) > 0.5).astype(np.int64)

        config = self.get_config()
        return config, pixel_values, mask_labels, class_labels


model_tester = EomtForUniversalSegmentationTester()
config, pixel_values, mask_labels, class_labels = model_tester.prepare_config_and_inputs()
EOMT_CASES = [
    [
        "EomtForUniversalSegmentation",
        "transformers.EomtForUniversalSegmentation",
        "mindone.transformers.EomtForUniversalSegmentation",
        (config,),
        {},
        (pixel_values,),
        {
            "mask_labels": mask_labels,
            "class_labels": class_labels,
        },
        {
            "last_hidden_state": 3,
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
        for case in EOMT_CASES
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
